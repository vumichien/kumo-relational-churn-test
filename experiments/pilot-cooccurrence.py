"""Pilot: can Kumo Relational see a pattern that per-column summaries hide?

Synthetic two-table shop. Each customer has 4-12 orders; every order carries two yes/no flags,
`late` and `refunded`. A customer churns if and only if at least one single order was both late
and refunded. Both classes get the same number of late orders and the same number of refunded
orders (drawn from one distribution), so per-column summaries of the orders (count, sum, mean,
min, max, spread of each flag) carry no signal. This is the co-occurrence test of the KumoRFM-2
paper (Fig. 4), rebuilt for the open checkpoint.

Compares, on the same customers: Kumo Relational seeing all of a customer's orders, Kumo
Relational seeing one sampled order, and LightGBM on the per-column summaries. Writes
results/pilot-cooccurrence.json. Planning evidence only: phase 1 reruns it formally.

Run (WSL):  HF_HOME=~/kumo-relational-tds/cache/hf ~/kumo-relational-tds/.venv/bin/python -I pilot-cooccurrence.py
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import sdm

NUM_CUSTOMERS = 4_000  # half context, half query
SEED = 0


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


def kumo_relational(customers, orders, labels, num_neighbors, device) -> float:
    tables = {
        "customers": sdm.TableTensor.from_pandas(
            df=customers, stypes={"customer_id": "id", "age": "numerical"}),
        "orders": sdm.TableTensor.from_pandas(
            df=orders, stypes={"customer_id": "id", "late": "numerical", "refunded": "numerical"}),
    }
    data = sdm.RelationalData(tables=tables, relationships=[{
        "left_table": "orders", "left_column": "customer_id",
        "right_table": "customers", "right_column": "customer_id"}])
    task = sdm.TableTensor.from_pandas(
        df=labels, stypes={"customer_id": "id", "churn": "categorical"})
    half = NUM_CUSTOMERS // 2
    context, query = task[:half], task[half:]
    y_query = query["churn"]
    query = query.drop_columns("churn")
    sampler = data.sampler()
    kwargs = {"task_link": {"task_column": "customer_id", "table": "customers",
                            "table_column": "customer_id"},
              "num_neighbors": num_neighbors}
    model = sdm.models.KumoRelational(device=device)
    ctx, related = sampler(context, **kwargs).to(device)
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        model.fit(x=ctx.drop_columns("churn"), y=ctx["churn"], related_tables=related,
                  num_estimators=8, recipe=model.default_recipe())
        out = model.predict(*sampler(query, **kwargs).to(device))
    pred, target = sdm.evaluation.to_binary_class(out, y_query.to(device), 1)
    return float(roc_auc_score(target.cpu().numpy(), pred.float().cpu().numpy()))


def lightgbm_auroc(flat: pd.DataFrame, labels: pd.DataFrame) -> float:
    half = NUM_CUSTOMERS // 2
    X = flat.drop(columns=["customer_id"])
    y = labels["churn"]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
    model.fit(X[:half], y[:half])
    return float(roc_auc_score(y[half:], model.predict_proba(X[half:])[:, 1]))


t0 = time.perf_counter()
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customers, orders, labels = make_shop(SEED)
flat = column_summaries(customers, orders)
both = (orders["late"] & orders["refunded"]).groupby(orders["customer_id"]).sum()
flat_plus = flat.assign(late_and_refunded=flat["customer_id"].map(both))

result = {
    "script": "pilot-cooccurrence.py",
    "seed": SEED,
    "num_customers": NUM_CUSTOMERS,
    "num_orders": len(orders),
    "churn_rate": round(float(labels["churn"].mean()), 4),
    "auroc": {
        "kumo_relational_all_orders": kumo_relational(customers, orders, labels, [16], device),
        "kumo_relational_one_order": kumo_relational(customers, orders, labels, [1], device),
        "lightgbm_column_summaries": lightgbm_auroc(flat, labels),
        "lightgbm_column_summaries_plus_cooccurrence": lightgbm_auroc(flat_plus, labels),
    },
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "total_seconds": round(time.perf_counter() - t0, 1),
}
out_dir = Path.home() / "kumo-relational-tds" / "results"
(out_dir / "pilot-cooccurrence.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
