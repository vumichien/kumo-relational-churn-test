"""Experiment 2: can Kumo Relational see a pattern that per-column summaries hide?

The pilot's generator (pilot-cooccurrence.py), unchanged. Synthetic two-table shop: 4,000
customers (the first 2,000 labelled as context, the rest predicted), 4-12 orders each, yes/no
flags `late` and `refunded`. A customer churns if and only if one single order was both late
and refunded; the numbers of late and refunded orders come from the same distribution in both
classes, so per-column summaries carry no signal by construction. The co-occurrence test of the
KumoRFM-2 paper (Fig. 4), rebuilt for the open checkpoint, and rigged on purpose.

Rows (phase-1-experiments.md, definition 5), in the pilot's order for C1, C2, C3, C5:
  C1  Kumo Relational, every order visible (num_neighbors [16]; nobody has more than 12)
  C2  Kumo Relational, one sampled order per customer (num_neighbors [1])
  C3  LightGBM (300 trees, learning rate 0.05) on per-column summaries
  C5  C3 plus one hand-made feature: the number of orders both late and refunded
  C4  Kumo Tabular (size large, 8 estimators) on the same per-column summaries
Also writes the fairness check (per class: orders, late orders, refunded orders) and which
order the sampler picks in C2 (no time columns, so the "last" rule can't apply).

Data seed = model seed = --seed. Writes results/runs/c-s<seed>.json and
results/preds/c-s<seed>-test.parquet (one score column per row C1-C5).

Run (spark):  HF_HOME=~/kumo-relational-tds/cache/hf taskset -c 5-9,15-19 \
    ~/kumo-relational-tds/.venv/bin/python -I cooccurrence-test.py --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_guard  # noqa: E402

memory_guard.start()

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import sdm  # noqa: E402

NUM_CUSTOMERS = 4_000  # half context, half query
ROOT = Path.home() / "kumo-relational-tds"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
SEED = args.seed
torch.set_num_threads(10)


def make_shop(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers, orders, labels = [], [], []
    for cid in range(NUM_CUSTOMERS):
        n = int(rng.integers(4, 13))
        k_late = int(rng.integers(1, n // 2 + 1))
        k_refunded = int(rng.integers(1, n // 2 + 1))
        churn = int(rng.integers(0, 2))
        rows = rng.permutation(n)
        late = np.zeros(n, dtype=int)
        refunded = np.zeros(n, dtype=int)
        late[rows[:k_late]] = 1
        # Churners share exactly one order between the two flags; the others share none.
        start = k_late - 1 if churn else k_late
        refunded[rows[start:start + k_refunded]] = 1
        customers.append({"customer_id": cid, "age": int(rng.integers(18, 80))})
        labels.append({"customer_id": cid, "churn": churn})
        orders += [{"customer_id": cid, "late": int(a), "refunded": int(b)}
                   for a, b in zip(late, refunded)]
    return pd.DataFrame(customers), pd.DataFrame(orders), pd.DataFrame(labels)


def column_summaries(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    agg = orders.groupby("customer_id").agg(
        num_orders=("late", "size"),
        **{f"{c}_{f}": (c, f) for c in ["late", "refunded"]
           for f in ["sum", "mean", "min", "max", "std"]},
    )
    return customers.set_index("customer_id").join(agg).reset_index()


def relational_data(customers, orders):
    tables = {
        "customers": sdm.TableTensor.from_pandas(
            df=customers, stypes={"customer_id": "id", "age": "numerical"}),
        "orders": sdm.TableTensor.from_pandas(
            df=orders, stypes={"customer_id": "id", "late": "numerical", "refunded": "numerical"}),
    }
    return sdm.RelationalData(tables=tables, relationships=[{
        "left_table": "orders", "left_column": "customer_id",
        "right_table": "customers", "right_column": "customer_id"}])


KWARGS = {"task_link": {"task_column": "customer_id", "table": "customers",
                        "table_column": "customer_id"}}


def kumo_relational(customers, orders, labels, num_neighbors, device) -> np.ndarray:
    data = relational_data(customers, orders)
    task = sdm.TableTensor.from_pandas(
        df=labels, stypes={"customer_id": "id", "churn": "categorical"})
    half = NUM_CUSTOMERS // 2
    context, query = task[:half], task[half:]
    y_query = query["churn"]
    query = query.drop_columns("churn")
    sampler = data.sampler()
    kwargs = {**KWARGS, "num_neighbors": num_neighbors}
    model = sdm.models.KumoRelational(device=device)
    ctx, related = sampler(context, **kwargs).to(device)
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        model.fit(x=ctx.drop_columns("churn"), y=ctx["churn"], related_tables=related,
                  num_estimators=8, recipe=model.default_recipe())
        out = model.predict(*sampler(query, **kwargs).to(device))
    pred, _ = sdm.evaluation.to_binary_class(out, y_query.to(device), 1)
    return pred.float().cpu().numpy().reshape(-1)


def lightgbm_scores(flat: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    half = NUM_CUSTOMERS // 2
    X = flat.drop(columns=["customer_id"])
    y = labels["churn"]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED,
                               verbose=-1, n_jobs=10)
    model.fit(X[:half], y[:half])
    return model.predict_proba(X[half:])[:, 1]


def kumo_tabular(flat: pd.DataFrame, labels: pd.DataFrame, device) -> np.ndarray:
    half = NUM_CUSTOMERS // 2
    X = flat.drop(columns=["customer_id"])
    table = sdm.TableTensor.from_pandas(
        df=X.assign(churn=labels["churn"].to_numpy()),
        stypes={**{c: "numerical" for c in X.columns}, "churn": "categorical"})
    context, query = table[:half], table[half:]
    y_query = query["churn"]
    model = sdm.models.KumoTabular(size="large", device=device)
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        model.fit(x=context.drop_columns("churn").to(device), y=context["churn"].to(device),
                  num_estimators=8, recipe=model.default_recipe())
        out = model.predict(query.drop_columns("churn").to(device))
    pred, _ = sdm.evaluation.to_binary_class(out, y_query.to(device), 1)
    return pred.float().cpu().numpy().reshape(-1)


def sampler_check(customers, orders, labels) -> dict:
    """Which order does num_neighbors [1] pick when the shop has no timestamps?

    Same graph as C2, with an `order_id` id column added only here (id columns are not model
    features), so each sampled order can be compared with the customer's first order.
    """
    with_ids = orders.reset_index().rename(columns={"index": "order_id"})
    data = sdm.RelationalData(
        tables={
            "customers": sdm.TableTensor.from_pandas(
                df=customers, stypes={"customer_id": "id", "age": "numerical"}),
            "orders": sdm.TableTensor.from_pandas(
                df=with_ids, stypes={"order_id": "id", "customer_id": "id",
                                     "late": "numerical", "refunded": "numerical"}),
        },
        relationships=[{"left_table": "orders", "left_column": "customer_id",
                        "right_table": "customers", "right_column": "customer_id"}])
    task = sdm.TableTensor.from_pandas(df=labels[["customer_id"]], stypes={"customer_id": "id"})
    picks = {}
    for tag, s in (("a", 100), ("a_again", 100), ("b", 101)):
        torch.manual_seed(s)
        _, related = data.sampler()(task, **KWARGS, num_neighbors=[1])
        o = related.tables["orders"].to_pandas()
        picks[tag] = o.set_index("customer_id")["order_id"].sort_index()
    first = with_ids.groupby("customer_id")["order_id"].min()
    n_orders = with_ids.groupby("customer_id").size()
    a = picks["a"]
    return {
        "orders_sampled_per_customer": sorted(set(a.groupby(level=0).size().tolist())),
        "same_torch_seed_same_orders": bool(a.equals(picks["a_again"])),
        "different_torch_seed_different_orders": bool(not a.equals(picks["b"])),
        "share_sampled_order_is_first_in_table": round(float((a == first.loc[a.index]).mean()), 4),
        "share_expected_if_uniform": round(float((1 / n_orders).mean()), 4),
        "note": "pyg-lib hetero_neighbor_sample, replace=False, no node times (seed_time = int64 max)",
    }


t0 = time.perf_counter()
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customers, orders, labels = make_shop(SEED)
flat = column_summaries(customers, orders)
both = (orders["late"] & orders["refunded"]).groupby(orders["customer_id"]).sum()
flat_plus = flat.assign(late_and_refunded=flat["customer_id"].map(both))
half = NUM_CUSTOMERS // 2
y_test = labels["churn"].to_numpy()[half:]

timings, scores = {}, {}
for key, fn in [
    ("c1", lambda: kumo_relational(customers, orders, labels, [16], device)),
    ("c2", lambda: kumo_relational(customers, orders, labels, [1], device)),
    ("c3", lambda: lightgbm_scores(flat, labels)),
    ("c5", lambda: lightgbm_scores(flat_plus, labels)),
    ("c4", lambda: kumo_tabular(flat, labels, device)),
]:
    t = time.perf_counter()
    scores[key] = fn()
    timings[key] = round(time.perf_counter() - t, 2)
t_preds = time.perf_counter()

preds_path = ROOT / "results" / "preds" / f"c-s{SEED}-test.parquet"
pd.DataFrame({"customer_id": labels["customer_id"].to_numpy()[half:], "label": y_test,
              **{f"score_{k}": v for k, v in sorted(scores.items())}}).to_parquet(preds_path, index=False)

per_order = orders.assign(late_and_refunded=orders["late"] & orders["refunded"])
counts = per_order.groupby("customer_id").agg(
    orders=("late", "size"), late_orders=("late", "sum"), refunded_orders=("refunded", "sum"),
    late_and_refunded_orders=("late_and_refunded", "sum")).join(labels.set_index("customer_id"))
fairness = {}
for churn, grp in counts.groupby("churn"):
    fairness[f"churn_{churn}"] = {
        "customers": len(grp),
        **{f"mean_{c}": round(float(grp[c].mean()), 4) for c in
           ["orders", "late_orders", "refunded_orders", "late_and_refunded_orders"]},
        **{f"distribution_{c}": {str(k): int(v) for k, v in grp[c].value_counts().sort_index().items()}
           for c in ["orders", "late_orders", "refunded_orders"]},
    }

result = {
    "script": "cooccurrence-test.py",
    "seed": SEED,
    "num_customers": NUM_CUSTOMERS,
    "num_context": half,
    "num_query": NUM_CUSTOMERS - half,
    "num_orders": len(orders),
    "churn_rate": round(float(labels["churn"].mean()), 4),
    "auroc": {k: float(roc_auc_score(y_test, v)) for k, v in sorted(scores.items())},
    "rows": {"c1": "Kumo Relational, every order visible (num_neighbors [16])",
             "c2": "Kumo Relational, one sampled order per customer (num_neighbors [1])",
             "c3": "LightGBM (300 trees, lr 0.05) on per-column summaries",
             "c4": "Kumo Tabular large, 8 estimators, on per-column summaries",
             "c5": "C3 plus the number of orders both late and refunded"},
    "summary_columns": [c for c in flat.columns if c != "customer_id"],
    "fairness": fairness,
    "seconds_per_row": timings,
    "seconds_to_preds_written": round(t_preds - t0, 2),
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "torch": torch.__version__,
    "lightgbm": lgb.__version__,
}
result["sampler_check_c2"] = sampler_check(customers, orders, labels)
result["seconds_total"] = round(time.perf_counter() - t0, 2)
result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None
result["peak_rss_mb"] = round(memory_guard.vm_hwm_mb(), 1)
result.update(memory_guard.stats())
result["value"] = result["auroc"]["c1"]
(ROOT / "results" / "runs" / f"c-s{SEED}.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ("seed", "auroc", "sampler_check_c2", "seconds_total")}, indent=2))
