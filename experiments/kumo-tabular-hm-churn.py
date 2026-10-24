"""Experiment 1, T1: Kumo Tabular reads the data scientist's 71 hand-built features (H&M churn).

The flattening route without training: Kumo Tabular (size large, 8 estimators, the default
recipe, fp16 autocast) reads the feature tables lgbm-hm-churn.py --build_only cached in
data/ds-features/ (the RelBench user study's feats.sql, rendered in DuckDB). Inputs: the
study's feature columns with the study's stypes (numerical or categorical), without
customer_id and timestamp; target: churn.

Context, the same selection rule as Kumo Relational's K1: the most recent
num_estimators x context_size rows of the train and val feature tables (by timestamp), shuffled
with torch.randperm under --seed and split into one block per estimator (8 x 20,000 = 160,000),
passed to fit() with num_estimators=None so each estimator gets its own block, as
nvidia-relbench-example.py does through the shared ICLModel API. If fit() rejects that form,
the script falls back to the most recent 20,000 rows with num_estimators=8 and records it.
Queries: every test feature row, in batches of --batch_size.

Writes results/runs/t1-s<seed>.json and results/preds/t1-s<seed>-test.parquet.

Run (spark):  HF_HOME=~/kumo-relational-tds/cache/hf taskset -c 5-9,15-19 \
    ~/kumo-relational-tds/.venv/bin/python -I kumo-tabular-hm-churn.py --seed 0
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_guard  # noqa: E402

memory_guard.start()

import pandas as pd  # noqa: E402
import relbench  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import sdm  # noqa: E402

ROOT = Path.home() / "kumo-relational-tds"
FEATS = ROOT / "data" / "ds-features"
STYPES_PATH = ROOT / "data" / "relbench-user-study-445bb7a" / "inferred_stypes.py"
ID_COLS = ["customer_id", "timestamp"]
TARGET = "churn"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--context_size", type=int, default=20_000)
parser.add_argument("--num_estimators", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=10_000)
parser.add_argument("--run_id", type=str, default=None)
args = parser.parse_args()
RUN_ID = args.run_id or f"t1-s{args.seed}"
torch.set_num_threads(10)

t0 = time.perf_counter()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = STYPES_PATH.read_text()
block = text[text.index("'rel-hm-user-churn'"):]
block = block[block.index("{") + 1:block.index("}")]
study = dict(re.findall(r"'([^']+)':\s*stype\.(\w+)", block))
features = {c: s for c, s in study.items() if c not in ID_COLS + [TARGET]}
assert set(features.values()) <= {"numerical", "categorical"}, features

manifest = json.loads((FEATS / "manifest.json").read_text())
frames = {"train": pd.read_parquet(FEATS / manifest["train_file"]),
          "val": pd.read_parquet(FEATS / "val.parquet"),
          "test": pd.read_parquet(FEATS / "test.parquet")}
stypes = {**features, TARGET: "categorical"}
cols = list(features) + [TARGET]

pool = pd.concat([frames["train"], frames["val"]], ignore_index=True)
width = args.context_size * args.num_estimators
# K1's rule: newest rows first (argsort on the timestamp, descending), then one random permutation.
stamps = torch.from_numpy(pool["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64"))
order = stamps.argsort(descending=True)[:width]
perm = torch.randperm(len(order))
context_df = pool.iloc[order[perm].numpy()][cols].reset_index(drop=True)
labels = relbench.load_dataset("rel-hm").load_task("user-churn").get_table(
    "test", mask_input_cols=False).df
# The study's SQL leaves the label out of the test table; attach it by (customer_id, timestamp).
query_df = frames["test"].drop(columns=[TARGET], errors="ignore").merge(
    labels[ID_COLS + [TARGET]], on=ID_COLS, how="left", validate="one_to_one")
assert query_df[TARGET].notna().all() and len(query_df) == len(labels)
context = sdm.TableTensor.from_pandas(df=context_df, stypes=stypes)
query = sdm.TableTensor.from_pandas(df=query_df[cols], stypes=stypes)
y_query = query[TARGET]
query = query.drop_columns(TARGET)
t_data = time.perf_counter()

model = sdm.models.KumoTabular(size="large", device=device)
t_model = time.perf_counter()
context_form = "per_estimator_blocks"
try:
    blocks = context.unflatten(0, (args.num_estimators, args.context_size))
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        model.fit(x=blocks.drop_columns(TARGET).to(device), y=blocks[TARGET].to(device),
                  num_estimators=None, recipe=model.default_recipe())
    num_members = args.num_estimators
except (TypeError, ValueError, RuntimeError) as err:
    if "out of memory" in str(err).lower():
        raise
    print(f"[t1] per-estimator context rejected ({err!r}); falling back", flush=True)
    context_form = f"most_recent_{args.context_size}_rows_num_estimators_{args.num_estimators}"
    single = sdm.TableTensor.from_pandas(
        df=pool.iloc[stamps.argsort(descending=True)[:args.context_size].numpy()][cols], stypes=stypes)
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        model.fit(x=single.drop_columns(TARGET).to(device), y=single[TARGET].to(device),
                  num_estimators=args.num_estimators, recipe=model.default_recipe())
    num_members = None
if device.type == "cuda":
    torch.cuda.synchronize()
t_fit = time.perf_counter()

scores = []
for start in range(0, query.size(0), args.batch_size):
    x = query[start:start + args.batch_size].to(device)
    y = y_query[start:start + args.batch_size].to(device)
    if num_members is not None:
        x = x.expand(num_members, *x.size())
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        out = model.predict(x)
    pred, _ = sdm.evaluation.to_binary_class(out, y, 1)
    scores.append(pred.float().cpu().reshape(-1))
if device.type == "cuda":
    torch.cuda.synchronize()
t_pred = time.perf_counter()

keyed = query_df[ID_COLS].assign(score=torch.cat(scores).numpy())
preds = labels[ID_COLS + [TARGET]].merge(keyed, on=ID_COLS, how="left", validate="one_to_one")
assert len(preds) == len(labels) and preds["score"].notna().all()
preds = preds.rename(columns={TARGET: "label"})
preds.to_parquet(ROOT / "results" / "preds" / f"{RUN_ID}-test.parquet", index=False)
auroc = float(roc_auc_score(preds["label"], preds["score"]))
t_end = time.perf_counter()

result = {
    "script": "kumo-tabular-hm-churn.py",
    "run_id": RUN_ID,
    "args": vars(args),
    "context_form": context_form,
    "context_rows": len(context_df),
    "context_newest_timestamp": str(pool["timestamp"].max()),
    "context_oldest_timestamp": str(pool.iloc[order.numpy()]["timestamp"].min()),
    "context_churn_rate": round(float(context_df[TARGET].mean()), 4),
    "train_file": manifest["train_file"],
    "num_features": len(features),
    "feature_stypes": {s: sum(v == s for v in features.values()) for s in ("numerical", "categorical")},
    "num_query_rows": int(query.size(0)),
    "value": auroc,
    "test_auroc": auroc,
    "seconds": {
        "load_data": round(t_data - t0, 2), "load_model": round(t_model - t_data, 2),
        "fit": round(t_fit - t_model, 2), "predict": round(t_pred - t_fit, 2),
        "total": round(t_end - t0, 2),
    },
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "torch": torch.__version__,
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
    "peak_rss_mb": round(memory_guard.vm_hwm_mb(), 1),
    **memory_guard.stats(),
}
(ROOT / "results" / "runs" / f"{RUN_ID}.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ("run_id", "context_form", "test_auroc", "seconds")}, indent=2))
