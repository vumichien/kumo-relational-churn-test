"""Experiment 1 supervised baselines on H&M churn (RelBench rel-hm/user-churn).

L1 (--features ds): the data scientist's features. The RelBench user study's
hm/user-churn/feats.sql at commit 445bb7a3b1230f49f8e5890ae81754d3e365680f is downloaded at run
time into ~/kumo-relational-tds/data/ (the study's repo has no licence file, so it is never
copied into the article folder) and rendered per split in DuckDB exactly as the study's
train_gbdt.py renders it. Column stypes come from the study's inferred_stypes.py (downloaded
the same way and parsed as text, not executed).
L2 (--features auto): automatic inputs only: the customer table's columns plus the same 20 past
churn labels Kumo Relational's K1 builds, read from the task table the gate run dumped
(data/hm-task-table-lags20.parquet); stypes inferred with torch_frame's infer_df_stype.
L3 (--features ds_lags): L1's features plus the same 20 past churn labels, joined from that task
table on (customer_id, timestamp) in every split; the lag columns are categorical, as in L2.

Tuning follows the study's train_gbdt.py: torch_frame's LightGBM (pytorch-frame 0.3.0,
torch_frame/gbdt/tuned_lightgbm.py) with tune(num_trials=10) on validation AUROC, then predict
test with the tuned model. As in the study, every column in the stype map except the target is
an input: customer_id as a number, and timestamp, which torch_frame's LightGBM ignores (it only
reads categorical, numerical and embedding columns). Article change: the seed. torch_frame
seeds neither Optuna nor LightGBM, so this script passes TPESampler(seed=--seed) to
optuna.create_study and adds seed=--seed and num_threads=10 to every lightgbm.train call; the
search space and the early-stopping rule are torch_frame's, unchanged.

--build_only builds L1's feature tables (train, val, test) once and caches them as parquet in
data/ds-features/ (stage ds-features); the L1 runs read the cache, so every seed uses the same
rows. --subsample N (the fallback when the full training table doesn't fit) uses the study's own
`using sample` clause, made repeatable: `using sample reservoir(N rows) repeatable (0)`.

Writes results/runs/<id>.json and results/preds/<id>-test.parquet and -val.parquet, where <id>
is l1-s<seed>, l2-s<seed> or l3-s<seed>.

Run (spark):  HF_HOME=~/kumo-relational-tds/cache/hf taskset -c 5-9,15-19 \
    ~/kumo-relational-tds/.venv/bin/python -I lgbm-hm-churn.py --features ds --seed 0
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_guard  # noqa: E402

memory_guard.start()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path.home() / "kumo-relational-tds"
DATA = ROOT / "data"
COMMIT = "445bb7a3b1230f49f8e5890ae81754d3e365680f"
BASE = f"https://raw.githubusercontent.com/snap-stanford/relbench-user-study/{COMMIT}"
STUDY = DATA / "relbench-user-study-445bb7a"
SQL_PATH = STUDY / "hm-user-churn-feats.sql"
STYPES_PATH = STUDY / "inferred_stypes.py"
FEATS = DATA / "ds-features"
TASK_TABLE = DATA / "hm-task-table-lags20.parquet"
ID_COLS = ["customer_id", "timestamp"]
TARGET = "churn"

parser = argparse.ArgumentParser()
parser.add_argument("--features", choices=["ds", "auto", "ds_lags"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--subsample", type=int, default=0)
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--build_only", action="store_true")
parser.add_argument("--run_id", type=str, default=None)
args = parser.parse_args()
RUN_ID = args.run_id or f"{ {'ds': 'l1', 'auto': 'l2', 'ds_lags': 'l3'}[args.features] }-s{args.seed}"

T0 = time.perf_counter()
log: dict = {"script": "lgbm-hm-churn.py", "args": vars(args), "run_id": RUN_ID, "stages": {}}


def mem() -> dict:
    return {"peak_rss_mb": round(memory_guard.vm_hwm_mb(), 1)}


def stage(name: str, t: float, **extra) -> None:
    log["stages"][name] = {"seconds": round(time.perf_counter() - t, 2), **mem(), **extra}
    print(f"[{name}] {log['stages'][name]}", flush=True)


def fetch(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())


def train_file(subsample: int) -> Path:
    return FEATS / ("train.parquet" if subsample == 0 else f"train-sub{subsample}.parquet")


# ---------------------------------------------------------------------------------------------
# L1 features in DuckDB

def build_ds_features(splits: list[str], subsample: int) -> None:
    import duckdb
    import relbench
    from jinja2 import Template

    fetch(f"{BASE}/hm/user-churn/feats.sql", SQL_PATH)
    sql = SQL_PATH.read_text()
    if subsample:
        old = "using sample {{ subsample }}"
        assert old in sql, "the study's subsample clause changed"
        sql = sql.replace(old, "using sample reservoir({{ subsample }} rows) repeatable (0)")
    template = Template(sql)
    t = time.perf_counter()
    dataset = relbench.load_dataset("rel-hm")
    task = dataset.load_task("user-churn")
    db = dataset.get_db()  # rows after the test timestamp are hidden, as in the study
    (DATA / "duckdb-tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '10GB'; SET threads = 10; "
                f"SET temp_directory = '{DATA / 'duckdb-tmp'}';")
    for name in ["article", "customer", "transactions"]:
        df = db.table_dict[name].df  # noqa: F841  (read by DuckDB's replacement scan)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    label_rows = {}
    for split in ["train", "val", "test"]:
        df = task.get_table(split, mask_input_cols=False).df  # noqa: F841
        label_rows[split] = len(df)
        con.execute(f"CREATE TABLE user_churn_{split} AS SELECT * FROM df")
    del db, df
    stage("load_tables", t, label_rows=label_rows,
          transactions_rows_upto_test=int(con.execute("select count(*) from transactions").fetchone()[0]))
    FEATS.mkdir(parents=True, exist_ok=True)
    for split in splits:
        t = time.perf_counter()
        con.execute(template.render(set=split, subsample=subsample if split == "train" else 0))
        out = train_file(subsample) if split == "train" else FEATS / f"{split}.parquet"
        con.execute(f"COPY user_churn_{split}_feats TO '{out}' (FORMAT parquet)")
        con.execute(f"DROP TABLE user_churn_{split}_feats")
        n, c = con.execute(f"select count(*) from '{out}'").fetchone()[0], \
            len(con.execute(f"select * from '{out}' limit 1").description)
        stage(f"features_{split}", t, rows=int(n), columns=int(c), file=out.name)
    con.close()
    log["sql_lines"] = len(SQL_PATH.read_text().splitlines())
    log["duckdb"] = duckdb.__version__


def study_stypes() -> dict:
    """The study's stype map for rel-hm-user-churn, parsed from inferred_stypes.py as text."""
    fetch(f"{BASE}/inferred_stypes.py", STYPES_PATH)
    text = STYPES_PATH.read_text()
    block = text[text.index("'rel-hm-user-churn'"):]
    block = block[block.index("{") + 1:block.index("}")]
    return dict(re.findall(r"'([^']+)':\s*stype\.(\w+)", block))


# ---------------------------------------------------------------------------------------------
# Tuning with torch_frame's LightGBM, seeded

def tune_and_predict(train_df, val_df, test_df, col_to_stype: dict) -> tuple:
    import lightgbm
    import optuna
    import torch
    import torch_frame
    from torch_frame import TaskType
    from torch_frame.data import Dataset
    from torch_frame.gbdt import LightGBM
    from torch_frame.typing import Metric

    torch.set_num_threads(10)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _create_study = optuna.create_study
    optuna.create_study = lambda *a, **k: _create_study(
        *a, **{**k, "sampler": optuna.samplers.TPESampler(seed=args.seed)})
    _train = lightgbm.train
    trials = []

    def seeded_train(params, *a, **k):
        params = {**params, "seed": args.seed, "num_threads": 10}
        t = time.perf_counter()
        booster = _train(params, *a, **k)
        trials.append({"params": {p: params[p] for p in sorted(params)},
                       "best_iteration": booster.best_iteration,
                       "val_auc": booster.best_score.get("valid_0", {}).get("auc"),
                       "seconds": round(time.perf_counter() - t, 1)})
        print(f"[lightgbm.train] {trials[-1]}", flush=True)
        return booster

    lightgbm.train = seeded_train

    t = time.perf_counter()
    train_dset = Dataset(train_df, col_to_stype=col_to_stype, target_col=TARGET).materialize()
    val_tf = train_dset.convert_to_tensor_frame(val_df)
    test_tf = train_dset.convert_to_tensor_frame(test_df)
    stage("materialize", t, train_rows=train_dset.tensor_frame.num_rows,
          train_cols=train_dset.tensor_frame.num_cols)
    t = time.perf_counter()
    gbdt = LightGBM(TaskType.BINARY_CLASSIFICATION, num_classes=2, metric=Metric.ROCAUC)
    gbdt.tune(tf_train=train_dset.tensor_frame, tf_val=val_tf, num_trials=args.trials)
    stage("tune", t)
    t = time.perf_counter()
    val_pred = gbdt.predict(tf_test=val_tf).numpy()
    test_pred = gbdt.predict(tf_test=test_tf).numpy()
    stage("predict", t)
    log["engine"] = f"torch_frame {torch_frame.__version__} LightGBM, lightgbm {lightgbm.__version__}, " \
                    f"optuna {optuna.__version__}"
    log["best_params"] = gbdt.params
    log["final_best_iteration"] = gbdt.model.best_iteration
    log["tuning_trials"] = trials
    log["feature_stypes_used_by_lightgbm"] = {
        str(k): v for k, v in train_dset.tensor_frame.col_names_dict.items()}
    return val_pred, test_pred


def aligned(feats_df: pd.DataFrame, labels: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Predictions in the label table's row order, matched by (customer_id, timestamp)."""
    keyed = feats_df[ID_COLS].assign(score=pred)
    out = labels[ID_COLS + [TARGET]].merge(keyed, on=ID_COLS, how="left", validate="one_to_one")
    assert len(out) == len(labels) and out["score"].notna().all(), "features don't cover the labels"
    return out.rename(columns={TARGET: "label"})


# ---------------------------------------------------------------------------------------------

def main() -> None:
    import relbench

    if args.build_only:
        build_ds_features(["val", "test", "train"], args.subsample)
        log["train_file"] = train_file(args.subsample).name
        (FEATS / "manifest.json").write_text(json.dumps(
            {"train_file": train_file(args.subsample).name, "subsample": args.subsample}))
        log["total_seconds"] = round(time.perf_counter() - T0, 2)
        log.update(mem(), **memory_guard.stats())
        (ROOT / "results" / "runs" / "ds-features.json").write_text(json.dumps(log, indent=2))
        return

    task = relbench.load_dataset("rel-hm").load_task("user-churn")
    labels = {s: task.get_table(s, mask_input_cols=False).df for s in ["val", "test"]}
    t = time.perf_counter()
    if args.features in ("ds", "ds_lags"):
        subsample = args.subsample
        if subsample == 0 and (FEATS / "manifest.json").exists():
            subsample = json.loads((FEATS / "manifest.json").read_text())["subsample"]
        need = [s for s in ["val", "test"] if not (FEATS / f"{s}.parquet").exists()]
        need += [] if train_file(subsample).exists() else ["train"]
        if need:
            build_ds_features(need, subsample)
        frames = {"train": pd.read_parquet(train_file(subsample)),
                  "val": pd.read_parquet(FEATS / "val.parquet"),
                  "test": pd.read_parquet(FEATS / "test.parquet")}
        stypes = study_stypes()
        log["train_subsample"] = subsample
        log["stype_source"] = f"relbench-user-study {COMMIT[:7]} inferred_stypes.py"
        if args.features == "ds_lags":
            tt = pd.read_parquet(TASK_TABLE)
            lag_cols = [c for c in tt.columns if c.startswith(f"{TARGET}_lag_")]
            for s_ in frames:
                part = tt.loc[tt["split"] == s_, ["customer_id", "timestamp", *lag_cols]]
                part = part.astype({"timestamp": frames[s_]["timestamp"].dtype,
                                    "customer_id": frames[s_]["customer_id"].dtype})
                before = len(frames[s_])
                merged = frames[s_].merge(part, on=["customer_id", "timestamp"], how="left",
                                          validate="one_to_one", indicator=True)
                found = int((merged.pop("_merge") == "both").sum())
                assert len(merged) == before and found == before, (s_, before, len(merged), found)
                frames[s_] = merged
                log.setdefault("lag_rows_matched", {})[s_] = found
            stypes = {**stypes, **dict.fromkeys(lag_cols, "categorical")}
            log["lag_columns"] = len(lag_cols)
    else:
        import torch_frame
        from torch_frame.utils import infer_df_stype

        tt = pd.read_parquet(TASK_TABLE)
        lag_cols = [c for c in tt.columns if c.startswith(f"{TARGET}_lag_")]
        customer = relbench.load_dataset("rel-hm").get_db().table_dict["customer"].df
        tt = tt.merge(customer, on="customer_id", how="left", validate="many_to_one")
        frames = {s: tt[tt["split"] == s].drop(columns=["split"]).reset_index(drop=True)
                  for s in ["train", "val", "test"]}
        inferred = infer_df_stype(frames["train"].sample(100_000, random_state=0))
        stypes = {c: str(s.value) if hasattr(s, "value") else str(s) for c, s in inferred.items()}
        stypes.update({"customer_id": "numerical", "timestamp": "timestamp", TARGET: "categorical"})
        dropped = {c: s for c, s in stypes.items()
                   if s not in ("numerical", "categorical", "timestamp")}
        for c in dropped:
            del stypes[c]
            for s in frames:
                frames[s] = frames[s].drop(columns=[c])
        log["lag_columns"] = len(lag_cols)
        log["customer_columns"] = [c for c in customer.columns]
        log["stypes_dropped_unsupported"] = dropped
        log["stype_source"] = f"torch_frame {torch_frame.__version__} infer_df_stype on 100,000 train rows"
        log["train_subsample"] = 0
    stage("load_features", t, rows={s: len(f) for s, f in frames.items()})

    from torch_frame import stype as tf_stype
    col_to_stype = {c: getattr(tf_stype, s) for c, s in stypes.items() if c in frames["train"].columns}
    missing = sorted(set(stypes) - set(col_to_stype))
    log["stypes"] = stypes
    log["stypes_missing_from_frames"] = missing
    log["num_input_columns"] = len(col_to_stype) - 1
    log["train_rows"] = len(frames["train"])
    log["train_churn_rate"] = round(float(frames["train"][TARGET].mean()), 4)

    val_pred, test_pred = tune_and_predict(frames["train"], frames["val"], frames["test"], col_to_stype)
    preds_dir = ROOT / "results" / "preds"
    out = {}
    for split, pred in (("val", val_pred), ("test", test_pred)):
        df = aligned(frames[split], labels[split], pred)
        df.to_parquet(preds_dir / f"{RUN_ID}-{split}.parquet", index=False)
        out[split] = float(roc_auc_score(df["label"], df["score"]))
    log["val_auroc"] = out["val"]
    log["test_auroc"] = out["test"]
    log["value"] = out["test"]
    log["seconds_to_preds_written"] = round(time.perf_counter() - T0, 2)
    log.update(mem(), **memory_guard.stats())
    (ROOT / "results" / "runs" / f"{RUN_ID}.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: log[k] for k in ("run_id", "train_rows", "val_auroc", "test_auroc")}, indent=2))


main()
