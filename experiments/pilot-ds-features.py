"""Pilot: does the RelBench user study's hand-written H&M churn SQL still run, and how heavy is it?

Loads rel-hm through relbench 3.x, puts the three tables and the user-churn label tables into an
in-memory DuckDB, renders the data scientist's feats.sql (snap-stanford/relbench-user-study,
hm/user-churn/feats.sql, commit 445bb7a) for val and test in full and for a subsample of train,
then fits one untuned LightGBM as a sanity check. Writes results/pilot-ds-features.json.

The SQL file is downloaded at run time into ~/kumo-relational-tds/data/ (the repo has no licence
file, so it is not copied into the article folder).

Run (WSL):  HF_HOME=~/kumo-relational-tds/cache/hf ~/kumo-relational-tds/.venv/bin/python -I pilot-ds-features.py
"""

import json
import time
import urllib.request
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import relbench
from jinja2 import Template
from sklearn.metrics import roc_auc_score

ROOT = Path.home() / "kumo-relational-tds"
SQL_URL = ("https://raw.githubusercontent.com/snap-stanford/relbench-user-study/"
           "445bb7a3b1230f49f8e5890ae81754d3e365680f/hm/user-churn/feats.sql")
SQL_PATH = ROOT / "data" / "relbench-user-study-445bb7a" / "hm-user-churn-feats.sql"
TRAIN_SUBSAMPLE = 50_000


def mem() -> dict:
    status = Path("/proc/self/status").read_text().splitlines()
    get = lambda key: next(int(l.split()[1]) for l in status if l.startswith(key)) / 1024
    return {"rss_mb": round(get("VmRSS:")), "peak_rss_mb": round(get("VmHWM:"))}


t0 = time.perf_counter()
log = {"stages": {}}

if not SQL_PATH.exists():
    SQL_PATH.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SQL_URL, headers={"User-Agent": "Mozilla/5.0"})
    SQL_PATH.write_bytes(urllib.request.urlopen(req, timeout=60).read())
template = Template(SQL_PATH.read_text())

dataset = relbench.load_dataset("rel-hm")
task = dataset.load_task("user-churn")
db = dataset.get_db()  # rows after the test timestamp are hidden
con = duckdb.connect()
con.execute("SET memory_limit = '8GB'; SET threads = 12;")
for name in ["article", "customer", "transactions"]:
    df = db.table_dict[name].df  # noqa: F841  (read by DuckDB's replacement scan)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
labels = {}
for split in ["train", "val", "test"]:
    labels[split] = task.get_table(split, mask_input_cols=False).df
    df = labels[split]  # noqa: F841
    con.execute(f"CREATE TABLE user_churn_{split} AS SELECT * FROM df")
log["stages"]["load"] = {"seconds": round(time.perf_counter() - t0, 1), **mem(),
                         "label_rows": {s: len(d) for s, d in labels.items()},
                         "transactions_rows": len(db.table_dict["transactions"].df)}

feats = {}
for split in ["val", "test", "train"]:
    t = time.perf_counter()
    sub = TRAIN_SUBSAMPLE if split == "train" else 0
    con.execute(template.render(set=split, subsample=sub))
    feats[split] = con.execute(f"SELECT * FROM user_churn_{split}_feats").df()
    log["stages"][f"features_{split}"] = {
        "seconds": round(time.perf_counter() - t, 1), "rows": len(feats[split]),
        "columns": feats[split].shape[1], **mem()}

id_cols = ["customer_id", "timestamp"]
target = "churn"
cat_cols = [c for c in feats["train"].columns  # pandas 3 reads text as the str dtype
            if not pd.api.types.is_numeric_dtype(feats["train"][c])
            and not pd.api.types.is_datetime64_any_dtype(feats["train"][c])
            and c not in id_cols + [target]]
for split in feats:
    for c in cat_cols:
        feats[split][c] = feats[split][c].astype("category")
X = {s: feats[s].drop(columns=[c for c in id_cols + [target] if c in feats[s].columns])
     for s in feats}
cats = {c: feats["train"][c].cat.categories for c in cat_cols}
for s in ["val", "test"]:
    for c in cat_cols:
        X[s][c] = X[s][c].cat.set_categories(cats[c])

t = time.perf_counter()
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           random_state=0, verbose=-1)
model.fit(X["train"], feats["train"][target])
log["stages"]["lightgbm_untuned_fit"] = {"seconds": round(time.perf_counter() - t, 1)}

# Score in the label tables' row order, as relbench evaluates.
for s in ["val", "test"]:
    p = model.predict_proba(X[s])[:, 1]
    key = feats[s][id_cols].astype(str).agg("|".join, axis=1)
    order = labels[s][id_cols].astype(str).agg("|".join, axis=1)
    lookup = dict(zip(key, p))  # (customer_id, timestamp) -> probability
    aligned = np.array([lookup[v] for v in order])
    log[f"{s}_auroc_untuned_subsample"] = round(float(roc_auc_score(labels[s][target], aligned)), 4)

log["feature_columns"] = int(X["train"].shape[1])
log["sql_lines"] = len(SQL_PATH.read_text().splitlines())
log["total_seconds"] = round(time.perf_counter() - t0, 1)
log["relbench"] = relbench.__version__ if hasattr(relbench, "__version__") else None
log["duckdb"] = duckdb.__version__
log["lightgbm"] = lgb.__version__
(ROOT / "results" / "pilot-ds-features.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))
