# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copied from NVIDIA/structured-data-models examples/relational/relbench/main.py
# (commit f8d054cbc589c9ecd2c35634434f76c6e4aae79a, Apache-2.0). Changes for the article,
# none of which touches the modelling code:
# - timing of each stage, peak GPU and process memory, and a JSON result file (--out);
# - the target column is dropped before the query is expanded across estimators (needed
#   under sdm 0.2.1; see the comment at the change);
# - --context_splits: "train,val" (default, NVIDIA's behaviour) or "train" (context drawn from
#   the training weeks only; the lag columns are still built from all three splits);
# - --save_preds: test predictions as parquet (entity id, timestamp, the score and label the
#   AUROC uses, which are for the script's positive_class, and P(target = 1) with the target);
# - --dump_task_table: the task table with its lag columns, all three splits, as parquet;
# - the memory guard (memory_guard.py), 10 torch CPU threads, and freeing the pandas frames
#   once the task table is a tensor;
# - --upto_test_timestamp (default off, as NVIDIA's example): load the database with
#   task.get_db(upto_test_timestamp=True). In relbench 3.0.2, Dataset.get_db then calls
#   Database.upto(test_timestamp), and Table.upto keeps rows with time_col <= the test
#   timestamp (inclusive) in every table that has a time column; tables without one are
#   returned unchanged. For rel-hm only `transactions` has a time column. The row count of
#   every table is written to the result file (db_rows) either way;
# - --eval_split: "test" (default, NVIDIA's behaviour) or "val" (context from the training
#   weeks only, queries are the validation week; for tuning without touching the test week).

import argparse
import gc
import json
import sys
import math
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import relbench
import torch
import torchmetrics
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_guard  # noqa: E402

memory_guard.start()

import sdm  # noqa: E402
import sdm.processing as sp


def embed_entity_text(
    entity: sdm.TableTensor,
    task_ids: pd.Series,
    pkey: str,
    *,
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor] | None:
    """Embed task entities and return CPU embeddings."""
    if len(entity.text_columns) == 0:
        return None
    entity_ids = entity.select_columns(pkey).to_pandas()[pkey]
    in_task = torch.from_numpy(
        entity_ids.isin(pd.unique(task_ids)).to_numpy(dtype=bool)
    )
    text = entity.select_stypes(sdm.Stype.text)[in_task]
    if text.size(0) == 0:
        return None
    encoder = sp.SentenceTransformer(
        model_name=model_name,
        batch_size=32,
        max_seq_length=2048,
    )
    row_chunk = 1024
    parts = []
    columns: tuple[str, ...] = ()
    with torch.inference_mode():
        for start in range(0, text.size(0), row_chunk):
            embedded = encoder(text[start : start + row_chunk].to(device))
            columns = embedded.numerical_columns
            parts.append(embedded.numerical.cpu())
    del encoder, embedded, text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0), columns, in_task


def attach_text(
    entity: sdm.TableTensor,
    embeddings: torch.Tensor,
    columns: tuple[str, ...],
    in_task: torch.Tensor,
    context_ids: pd.Series,
    pkey: str,
    text_dim: int,
) -> sdm.TableTensor:
    """Project embeddings with a PCA fit on the context entities."""
    entity_ids = entity.select_columns(pkey).to_pandas()[pkey]
    in_context = torch.from_numpy(
        entity_ids.isin(pd.unique(context_ids)).to_numpy(dtype=bool)
    )
    pca = sp.PCA(text_dim)
    pca.fit(
        sdm.TableTensor(
            columns={"numerical": columns},
            numerical=embeddings[in_context[in_task]],
        )
    )
    reduced = pca.transform(
        sdm.TableTensor(
            columns={"numerical": columns},
            numerical=embeddings,
        )
    )
    values = torch.full(
        (entity.size(0), reduced.numerical.size(-1)),
        torch.nan,
        dtype=reduced.numerical.dtype,
    )
    values[in_task] = reduced.numerical
    return cast(
        sdm.TableTensor,
        torch.cat(
            [
                entity.drop_stypes(sdm.Stype.text),
                sdm.TableTensor(
                    columns={
                        "numerical": reduced.numerical_columns,
                    },
                    numerical=values,
                ),
            ],
            dim=-1,
        ),
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument(
    "--context_mode", choices=("latest", "random"), default="latest"
)
parser.add_argument("--batch_size", type=int, default=10_000)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--num_lags", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text", action="store_true")
parser.add_argument("--text_dim", type=int, default=64)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--context_splits", choices=("train,val", "train"), default="train,val")
parser.add_argument("--save_preds", type=str, default=None)
parser.add_argument("--dump_task_table", type=str, default=None)
parser.add_argument("--upto_test_timestamp", action="store_true")
parser.add_argument("--eval_split", choices=("test", "val"), default="test")
args = parser.parse_args()
torch.set_num_threads(10)


def _vm_hwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def _mem(stage: str) -> None:
    status = Path("/proc/self/status").read_text().splitlines()
    rss = next(int(l.split()[1]) for l in status if l.startswith("VmRSS:"))
    gpu = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    print(f"[mem] {stage}: rss {rss / 1024:,.0f} MB, peak rss {_vm_hwm_mb():,.0f} MB, "
          f"peak gpu {gpu:,.0f} MB", flush=True)


_t = {"start": time.perf_counter()}
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
dataset = relbench.load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = task.get_db(upto_test_timestamp=args.upto_test_timestamp)
_db_rows = {name: len(table.df) for name, table in db.table_dict.items()}
print(f"[db] upto_test_timestamp={args.upto_test_timestamp}: {_db_rows}", flush=True)
tables = {
    name: sdm.TableTensor.from_pandas(
        df=table.df,
        stypes=sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                cast(str, table.pkey_col): "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text=(
                "infer" if args.text and name == task.entity_table else "drop"
            ),
            unsupported="drop",
        ),
    )
    for name, table in db.table_dict.items()
}
data = sdm.RelationalData(
    tables=tables,
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": cast(str, db.table_dict[right_table].pkey_col),
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)
time_columns = {
    name: table.time_col
    for name, table in db.table_dict.items()
    if table.time_col is not None
}
_mem("database_tensorized")

# Collect Task Table ##########################################################
if task.task_type not in {
    relbench.base.TaskType.BINARY_CLASSIFICATION,
    relbench.base.TaskType.MULTICLASS_CLASSIFICATION,
    relbench.base.TaskType.REGRESSION,
}:
    quit(f"Unsupported task {task.task_type.value!r}")

dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
df = pd.concat(dfs, ignore_index=True)

# Add lag target features to task table:
right = df.sort_values(task.time_col)
left = pd.DataFrame(
    {
        task.entity_col: df[task.entity_col],
        "__lookup_time__": df[task.time_col],
        "__row__": range(len(df)),
    }
)
for lag in range(1, args.num_lags + 1):
    column = f"{task.target_col}_lag_{lag}"
    if task.task_type == relbench.base.TaskType.REGRESSION:
        df[column] = float("NaN")
    else:
        df[column] = pd.Series(pd.NA, index=df.index, dtype="object")

    merged = pd.merge_asof(
        left.sort_values("__lookup_time__"),
        right,
        by=task.entity_col,
        left_on="__lookup_time__",
        right_on=task.time_col,
        direction="backward",
        allow_exact_matches=False,
    )
    mask = merged[task.time_col].notna()
    rows = merged.loc[mask, "__row__"].to_numpy()
    df.loc[rows, column] = merged.loc[mask, task.target_col].to_numpy()

    # Find the next lag strictly before the previous observation.
    left = merged.loc[mask, [task.entity_col, "__row__", task.time_col]]
    left = left.rename(columns={task.time_col: "__lookup_time__"})

if task.task_type == relbench.base.TaskType.REGRESSION:
    target_stype = "numerical"
else:
    target_stype = "categorical"

task_table = sdm.TableTensor.from_pandas(
    df=df,
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: target_stype,
        **{
            f"{task.target_col}_lag_{lag}": target_stype
            for lag in range(1, args.num_lags + 1)
        },
    },
)
_n_train, _n_val, _n_test = len(dfs[0]), len(dfs[1]), len(dfs[2])
if args.dump_task_table:
    _dump = df.copy()
    for lag in range(1, args.num_lags + 1):
        column = f"{task.target_col}_lag_{lag}"
        _dump[column] = _dump[column].astype("Int8")
    _dump["split"] = ["train"] * _n_train + ["val"] * _n_val + ["test"] * _n_test
    _dump.to_parquet(args.dump_task_table, index=False)
    del _dump
if args.eval_split == "val":
    if args.context_splits != "train":
        quit("--eval_split val needs --context_splits train")
    context, query, _ = task_table.split([_n_train, _n_val, _n_test])
elif args.context_splits == "train":
    context, _, query = task_table.split([_n_train, _n_val, _n_test])
else:
    context, query = task_table.split([_n_train + _n_val, _n_test])
positive_class = dfs[0][task.target_col].unique()[-1].item()
_query_keys = dfs[1 if args.eval_split == "val" else 2][
    [task.entity_col, task.time_col]].reset_index(drop=True)
if not args.text:  # Free the pandas frames; the task table is a tensor now.
    for _name in ("df", "dfs", "left", "right", "merged"):  # merged is unset when num_lags is 0
        globals().pop(_name, None)
    gc.collect()

num_estimators = args.num_estimators
if len(context) > args.context_size:  # Sample different context per estimator:
    width = args.context_size * num_estimators
    if args.context_mode == "latest" and len(context) > width:
        order = context.datetime.reshape(-1).argsort(descending=True)
        context = context[order[:width]]
    repeats = math.ceil(width / len(context))
    perm = torch.cat([torch.randperm(len(context)) for _ in range(repeats)])
    context = context[perm[:width]]
    if num_estimators > 1:
        context = context.unflatten(0, (num_estimators, args.context_size))
        query = query.expand(num_estimators, *query.size())
        num_estimators = None

# Article change: drop the target before batching. Dropping a column from the
# expanded query copies it, so the sampler no longer sees a shared (stride-0)
# query and fails with a view error under sdm 0.2.1. NVIDIA's RelArena script
# never has the target column in its query, so this matches its path.
_query_y = query[task.target_col][0] if query.dim() == 3 else query[task.target_col]
query = query.drop_columns(task.target_col)
if query.dim() == 3:
    query = query[0].expand(query.size(0), *query.size()[1:])
_mem("task_table_ready")

# Execute Model ###############################################################
if args.text:
    entity_pkey = cast(str, db.table_dict[task.entity_table].pkey_col)
    entity = tables[task.entity_table]
    encoded = embed_entity_text(
        entity=entity,
        task_ids=df[task.entity_col],
        pkey=entity_pkey,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )
    if encoded is not None:
        embeddings, columns, in_task = encoded
        tables[task.entity_table] = attach_text(
            entity=entity,
            embeddings=embeddings,
            columns=columns,
            in_task=in_task,
            context_ids=pd.concat(
                [dfs[0][task.entity_col], dfs[1][task.entity_col]],
                ignore_index=True,
            ),
            pkey=entity_pkey,
            text_dim=args.text_dim,
        )

_t["data_ready"] = time.perf_counter()
_num_context_rows = int(context.size(-2))
_num_query_rows = int(query.size(-2))
sampler = data.sampler(time_columns=time_columns)
model = sdm.models.KumoRelational(device=device)
_t["model_loaded"] = time.perf_counter()
recipe = model.default_recipe()
kwargs: dict[str, Any] = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.float16, enabled=True):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=num_estimators,
        recipe=recipe,
    )
if device.type == "cuda":
    torch.cuda.synchronize()
_t["fit_done"] = time.perf_counter()

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError()
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    metric = torchmetrics.classification.BinaryAUROC()
else:
    metric = torchmetrics.classification.MulticlassAccuracy(average="micro")
metric = metric.to(device)
_mem("fit_done")
_preds: list[torch.Tensor] = []
_targets: list[torch.Tensor] = []
_churn_scores: list[torch.Tensor] = []  # P(target = 1) and the 0/1 target, for binary tasks
_churn_labels: list[torch.Tensor] = []
_batches = list(zip(
    query.split(args.batch_size, -2), _query_y.split(args.batch_size, -2)
))[: args.max_test_steps]
for x_query, y_query in tqdm(_batches):
    y_query = y_query.to(device)
    with torch.amp.autocast(device.type, torch.float16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))

    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["q500"].numerical.squeeze(-1)  # Median prediction.
        target = y_query.numerical.squeeze(-1)
    elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
        pred, target = sdm.evaluation.to_binary_class(
            out, y_query, positive_class
        )
        _p1, _y1 = sdm.evaluation.to_binary_class(out, y_query, 1)
        _churn_scores.append(_p1.float().cpu())
        _churn_labels.append(_y1.float().cpu())
    else:
        pred, target = sdm.evaluation.to_class_indices(out, y_query)
    metric.update(pred, target)
    _preds.append(pred.float().cpu())
    _targets.append(target.float().cpu())

if device.type == "cuda":
    torch.cuda.synchronize()
_t["predict_done"] = time.perf_counter()

_recomputed = None
if args.save_preds:
    _p = torch.cat(_preds).reshape(-1).numpy()
    _y = torch.cat(_targets).reshape(-1).numpy()
    _keys = _query_keys.iloc[: len(_p)]
    pd.DataFrame({
        task.entity_col: _keys[task.entity_col].to_numpy(),
        task.time_col: _keys[task.time_col].to_numpy(),
        "score": _p,  # the score the script's AUROC uses: P(target = positive_class)
        "label": _y,  # 1 where target = positive_class
        "positive_class": positive_class,
        **({"score_target_1": torch.cat(_churn_scores).reshape(-1).numpy(),
            "target": torch.cat(_churn_labels).reshape(-1).numpy()} if _churn_scores else {}),
    }).to_parquet(args.save_preds, index=False)
    if task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
        from sklearn.metrics import roc_auc_score
        _recomputed = float(roc_auc_score(_y, _p))
_t["preds_written"] = time.perf_counter()

if task.task_type == relbench.base.TaskType.REGRESSION:
    _name, _value = "nMAE", float(metric.compute() / task.nmae_std)
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    _name, _value = "AUROC", float(metric.compute())
else:
    _name, _value = "ACC", float(metric.compute())
print(f"{_name}: {_value:.4f}")

_result = {
    "script": "nvidia-relbench-example.py",
    "args": vars(args),
    "eval_split": args.eval_split,
    "metric": _name if args.eval_split == "test" else f"val_{_name}",
    "value": _value,
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "torch": torch.__version__,
    "relbench": getattr(relbench, "__version__", None),
    "num_context_rows": _num_context_rows,
    "num_query_rows": _num_query_rows,
    "seconds": {
        "load_and_prepare_data": round(_t["data_ready"] - _t["start"], 2),
        "load_model": round(_t["model_loaded"] - _t["data_ready"], 2),
        "sample_context_and_fit": round(_t["fit_done"] - _t["model_loaded"], 2),
        "sample_queries_and_predict": round(_t["predict_done"] - _t["fit_done"], 2),
        "total": round(_t["predict_done"] - _t["start"], 2),
        "to_preds_written": round(_t["preds_written"] - _t["start"], 2),
    },
    "auroc_sklearn_from_preds": _recomputed,
    "num_train_rows": _n_train,
    "num_val_rows": _n_val,
    "num_test_rows": _n_test,
    "upto_test_timestamp": args.upto_test_timestamp,
    "db_rows": _db_rows,
    "positive_class": positive_class,
    "cuda_version": torch.version.cuda,
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if device.type == "cuda" else None,
    "peak_rss_mb": round(_vm_hwm_mb(), 1),
    **memory_guard.stats(),
}
print(json.dumps(_result, indent=2))
if args.out:
    Path(args.out).write_text(json.dumps(_result, indent=2))
