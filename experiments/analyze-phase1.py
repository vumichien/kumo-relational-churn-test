"""Phase 1 analysis: recompute every AUROC from the saved predictions and write the summaries.

One function, `auroc()` (sklearn.metrics.roc_auc_score), recomputes every test AUROC from
results/preds/; each script's own value must agree within 1e-4 (definition 2), and any
disagreement is listed under "agreement_failures". Wall-clock and memory come from the
driver's state/runs/<id>.done markers and the scripts' result JSONs; a run flagged for swap or
a busy tenant has "clean_timing": false.

Writes (results/): exp1-summary.json, context-sweep.json, exp2-summary.json,
figure-data/figure-exp1.json, figure-data/figure-exp2.json, figure-data/figure-context-sweep.json,
prose-facts.json.

Run (spark, stage analyze):  ~/kumo-relational-tds/.venv/bin/python -I analyze-phase1.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_guard  # noqa: E402

memory_guard.start()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path.home() / "kumo-relational-tds"
RES = ROOT / "results"
STATE = ROOT / "state" / "runs"
TASK_TABLE = ROOT / "data" / "hm-task-table-lags20.parquet"
AGREE_TOL = 1e-4
SEEDS = (0, 1, 2)
LABELLED_TRAIN_VAL = 3_832_692 + 76_556  # train + val label rows (definition 1)

EXP1 = {
    "k1": "Kumo Relational, NVIDIA's settings",
    "k2": "Kumo Relational, context from the training weeks only",
    "k3": "Kumo Relational, no past churn labels",
    "k4": "Kumo Relational, no related tables",
    "k5": "Kumo Relational, the library's defaults",
    "t1": "Kumo Tabular on the hand-built features",
    "l1": "LightGBM on the hand-built features (tuned)",
    "l2": "LightGBM on the customer table and past labels (tuned)",
}
EXP1_ROLE = {"k1": "kumo_relational", "k2": "kumo_relational", "k3": "kumo_relational",
             "k4": "kumo_relational", "k5": "kumo_relational", "t1": "kumo_tabular",
             "l1": "flat_table", "l2": "flat_table"}
SWEEP = {"s1": 1_000, "s2": 5_000, "k1": 20_000}
EXP2 = {
    "c1": ("Kumo Relational, sees every order", "kumo_relational"),
    "c2": ("Kumo Relational, sees one sampled order", "kumo_relational"),
    "c3": ("LightGBM on per-column summaries", "flat_table"),
    "c4": ("Kumo Tabular on per-column summaries", "kumo_tabular"),
    "c5": ("LightGBM on summaries plus one hand-made feature", "flat_table"),
}
ROLE_COLOUR = {"kumo_relational": "BLUE #1971c2", "kumo_tabular": "BLUE #1971c2, hollow",
               "flat_table": "ORANGE #e8590c"}

agreement_failures: list[dict] = []


def auroc(label: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(label, score))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": round(float(np.mean(values)), 4), "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4), "n": len(values)}


def check(run_id: str, script_value: float | None, recomputed: float) -> None:
    if script_value is not None and abs(script_value - recomputed) > AGREE_TOL:
        agreement_failures.append({"run": run_id, "script": script_value, "recomputed": recomputed})


def timing(run_id: str) -> dict:
    done, result = load(STATE / f"{run_id}.done"), load(RES / "runs" / f"{run_id}.json")
    if done is None:
        return {"status": "missing"}
    out = {"status": done.get("status"), "wall_s_process": done.get("wall_s"),
           "flags": done.get("flags"), "clean_timing": not done.get("flags"),
           "peak_rss_mb": done.get("time_max_rss_mb") or done.get("peak_rss_mb"),
           "peak_gpu_mb": done.get("peak_gpu_mb"), "fallback_used": done.get("fallback_used"),
           "infra_stops": len(done.get("infra_stops") or []),
           "start": done.get("start"), "mem_available_min_gib": done.get("mem_available_min_gib_driver")}
    if result:
        secs = result.get("seconds") or {}
        out["seconds_to_preds_written"] = (secs.get("to_preds_written") or secs.get("total")
                                           or result.get("seconds_to_preds_written"))
        out["seconds_by_stage"] = secs or result.get("stages")
    return out


def single_run(run_id: str) -> dict | None:
    preds = RES / "preds" / f"{run_id}-test.parquet"
    if not preds.exists():
        return None
    df = pd.read_parquet(preds)
    value = auroc(df["label"].to_numpy(), df["score"].to_numpy())
    result = load(RES / "runs" / f"{run_id}.json") or {}
    check(run_id, result.get("value"), value)
    out = {"test_auroc": round(value, 6), "script_value": result.get("value"), "rows": len(df)}
    if "score_target_1" in df:  # Kumo Relational runs: the same AUROC from P(churn = 1)
        out["test_auroc_from_churn_scores"] = round(auroc(df["target"], df["score_target_1"]), 6)
        check(run_id + " (churn scores)", value, out["test_auroc_from_churn_scores"])
    val = RES / "preds" / f"{run_id}-val.parquet"
    if val.exists():
        v = pd.read_parquet(val)
        out["val_auroc"] = round(auroc(v["label"].to_numpy(), v["score"].to_numpy()), 6)
        check(run_id + " (val)", result.get("val_auroc"), out["val_auroc"])
    if run_id.startswith(("l1", "l2")):
        out["train_rows"] = result.get("train_rows")
        out["train_subsample"] = result.get("train_subsample")
        out["best_params"] = result.get("best_params")
    if run_id.startswith("t1"):
        out["context_form"] = result.get("context_form")
        out["context_rows"] = result.get("context_rows")
    return out


def exp1() -> dict:
    summary = {}
    for config, label in EXP1.items():
        runs = {f"s{s}": single_run(f"{config}-s{s}") for s in SEEDS}
        values = [r["test_auroc"] for r in runs.values() if r]
        entry = {"label": label, "role": EXP1_ROLE[config], "seeds": runs,
                 "test_auroc": stats(values), "seed0_timing": timing(f"{config}-s0"),
                 "missing_seeds": [s for s, r in runs.items() if r is None]}
        for s in SEEDS:
            t = timing(f"{config}-s{s}")
            if t.get("status") not in ("done", "missing"):
                entry.setdefault("not_done", {})[f"s{s}"] = t.get("status")
        if config in ("l1", "l2"):
            entry["val_auroc"] = stats([r["val_auroc"] for r in runs.values() if r and "val_auroc" in r])
            entry["train_rows"] = sorted({r["train_rows"] for r in runs.values() if r})
        summary[config] = entry
    return summary


def main() -> None:
    (RES / "figure-data").mkdir(parents=True, exist_ok=True)
    e1 = exp1()
    gate = load(RES / "gate.json")
    (RES / "exp1-summary.json").write_text(json.dumps({
        "task": "RelBench rel-hm/user-churn, test split (74,575 rows, t = 2020-09-14)",
        "metric": "test AUROC recomputed from results/preds with sklearn roc_auc_score",
        "machine": "spark: NVIDIA GB10 (20-core Arm CPU, 121 GiB shared CPU/GPU memory), shared "
                   "with another tenant's model servers",
        "gate": gate, "configs": e1, "agreement_failures": agreement_failures,
        "lgbm_budget": load(ROOT / "state" / "lgbm-budget.json"),
        "ds_features_build": (load(RES / "runs" / "ds-features.json") or {}).get("stages"),
    }, indent=2))

    sweep = {}
    for config, size in SWEEP.items():
        st = e1["k1"]["test_auroc"] if config == "k1" else stats(
            [r["test_auroc"] for r in (single_run(f"{config}-s{s}") for s in SEEDS) if r])
        total = size * 8
        sweep[config] = {"rows_per_estimator": size, "estimators": 8, "total_context_rows": total,
                         "share_of_labelled_train_val_rows": round(total / LABELLED_TRAIN_VAL, 5),
                         "test_auroc": st}
    sweep_doc = {"labelled_train_val_rows": LABELLED_TRAIN_VAL, "points": sweep,
                 "note": "context rows drawn by recency (NVIDIA's 'latest' mode), 8 estimators, "
                         "other settings as K1"}
    (RES / "context-sweep.json").write_text(json.dumps(sweep_doc, indent=2))

    e2_runs = {s: load(RES / "runs" / f"c-s{s}.json") for s in SEEDS}
    rows = {}
    for key, (label, role) in EXP2.items():
        per_seed = {}
        for s in SEEDS:
            p = RES / "preds" / f"c-s{s}-test.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                value = auroc(df["label"].to_numpy(), df[f"score_{key}"].to_numpy())
                check(f"c-s{s} {key}", (e2_runs[s] or {}).get("auroc", {}).get(key), value)
                per_seed[f"s{s}"] = round(value, 6)
        rows[key] = {"label": label, "role": role, "seeds": per_seed,
                     "test_auroc": stats(list(per_seed.values()))}
    (RES / "exp2-summary.json").write_text(json.dumps({
        "design": "synthetic shop, 4,000 customers per data seed (2,000 context, 2,000 predicted), "
                  "4-12 orders each; churn iff one order was both late and refunded",
        "rows": rows,
        "fairness": {f"s{s}": (r or {}).get("fairness") for s, r in e2_runs.items()},
        "churn_rate": {f"s{s}": (r or {}).get("churn_rate") for s, r in e2_runs.items()},
        "num_orders": {f"s{s}": (r or {}).get("num_orders") for s, r in e2_runs.items()},
        "sampler_check_c2": {f"s{s}": (r or {}).get("sampler_check_c2") for s, r in e2_runs.items()},
        "seconds_per_row_seed0": (e2_runs[0] or {}).get("seconds_per_row"),
        "agreement_failures": [f for f in agreement_failures if f["run"].startswith("c-")],
    }, indent=2))

    def bar(key, entry):
        st = entry["test_auroc"]
        return {"id": key.upper(), "label": entry["label"], "role": entry["role"],
                "colour": ROLE_COLOUR[entry["role"]], "mean": st["mean"], "min": st["min"],
                "max": st["max"], "n_seeds": st["n"]}

    (RES / "figure-data" / "figure-exp1.json").write_text(json.dumps({
        "title": "H&M churn: test AUROC (mean of three seeds, bars show min-max)",
        "chance_level": 0.5, "bars": [bar(k, v) for k, v in e1.items()],
        "reference_lines": {"NVIDIA README, same settings": 0.6992,
                            "RelBench user study data scientist (KumoRFM whitepaper Table 2)": 0.6900},
    }, indent=2))
    (RES / "figure-data" / "figure-exp2.json").write_text(json.dumps({
        "title": "Synthetic shop: can the model see 'one order both late and refunded'? "
                 "(test AUROC, mean of three data seeds)",
        "chance_level": 0.5, "bars": [bar(k, v) for k, v in rows.items()],
    }, indent=2))
    (RES / "figure-data" / "figure-context-sweep.json").write_text(json.dumps({
        "title": "More labelled examples in the context: H&M churn test AUROC",
        "x_label": "labelled rows per estimator (8 estimators)",
        "points": [{"rows_per_estimator": v["rows_per_estimator"],
                    "total_context_rows": v["total_context_rows"],
                    "share_of_labelled_rows": v["share_of_labelled_train_val_rows"],
                    **{k: v["test_auroc"][k] for k in ("mean", "min", "max")}}
                   for v in sweep.values()],
    }, indent=2))

    import relbench
    tt = pd.read_parquet(TASK_TABLE)
    lags = [c for c in tt.columns if c.startswith("churn_lag_")]
    test = tt[tt["split"] == "test"]
    n_past = test[lags].notna().sum(axis=1)
    db = relbench.load_dataset("rel-hm").load_task("user-churn").get_db(upto_test_timestamp=False)
    ds = load(RES / "runs" / "ds-features.json") or {}
    prose = {
        "churn_rate": {s: round(float(tt.loc[tt["split"] == s, "churn"].mean()), 4)
                       for s in ("train", "val", "test")},
        "label_rows": {s: int((tt["split"] == s).sum()) for s in ("train", "val", "test")},
        "context_rows_k1": 160_000, "labelled_rows_available_train_val": LABELLED_TRAIN_VAL,
        "context_share_k1": round(160_000 / LABELLED_TRAIN_VAL, 4),
        "test_rows_with_at_least_one_past_label_share": round(float((n_past > 0).mean()), 4),
        "test_rows_past_labels_median": float(n_past.median()),
        "test_rows_past_labels_mean": round(float(n_past.mean()), 2),
        "test_customers_distinct": int(test["customer_id"].nunique()),
        "num_lags": len(lags),
        "tables": {name: len(t.df) for name, t in db.table_dict.items()},
        "tables_note": "full tables (upto_test_timestamp=False), as Kumo Relational reads them",
        "transactions_up_to_test_timestamp": (ds.get("stages", {}).get("load_tables", {})
                                              .get("transactions_rows_upto_test")),
        "transactions_up_to_test_note": "what the data scientist's SQL sees (dataset.get_db())",
    }
    (RES / "prose-facts.json").write_text(json.dumps(prose, indent=2))
    print(json.dumps({"agreement_failures": agreement_failures,
                      "exp1": {k: v["test_auroc"] for k, v in e1.items()},
                      "exp2": {k: v["test_auroc"] for k, v in rows.items()}}, indent=2))
    if agreement_failures:
        sys.exit(1)


main()
