"""Phase 5 driver: two checks on spark, resumably, with the phase 1 driver's run machinery.

K6: K1 on the database cut at the test timestamp (--upto_test_timestamp), seeds 0-2.
K7: K3 (no past labels) with its neighbour fan-out retuned on the validation week. Seed 0 of
K3 at --num_neighbors 1 1, 4 4, 16 16 and 32 32 with --eval_split val (context from the training weeks,
queries are the validation week); the best validation AUROC (sklearn, from the saved
predictions) is the pick. If the pick is 1 1, K3's three test seeds already are K7 and nothing
reruns; otherwise K3 with the picked fan-out runs on the test week, seeds 0-2.

Memory handling, decided before the runs:
- every run waits for 20 GiB available and is stopped by memory_guard under 8 GiB, as in phase 1;
- an out-of-memory error with memory still available: one retry with --batch_size 2000 (query
  batches only; the context is never shrunk);
- a guard stop or kernel kill: wait for memory and rerun. A validation candidate stopped twice is
  recorded as "did_not_fit" and left out of the pick; a K6 or picked K7 run stopped three times
  writes blocked_reason to state/phase5-status.json and the driver exits 3 (reported, not shrunk).

Every run's marker goes to state/runs/<run-id>.done, like phase 1's. Start it with
    setsid nohup taskset -c 5-9,15-19 .venv/bin/python -I experiments/run-phase5.py \
        >> logs/phase5.log 2>&1 < /dev/null & echo $!
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_phase1", HERE / "run-phase1.py")
rp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp1)

RES, STATE, LOGS = rp1.RES, rp1.STATE, rp1.LOGS
K3 = rp1.KUMO_CONFIGS["k3"]
FANOUTS = (1, 4, 16, 32)  # 32 added after 16 won at the edge of the first grid
BATCH_FALLBACK = ["--batch_size", "2000"]


def spec_for(rid: str, config: str, args: list[str], split: str) -> dict:
    preds = RES / "preds" / f"{rid}-{split}.parquet"
    return {
        "id": rid, "config": config, "script": "nvidia-relbench-example.py",
        "args": args + ["--out", str(RES / "runs" / f"{rid}.json"), "--save_preds", str(preds)],
        "result": RES / "runs" / f"{rid}.json", "preds": [preds],
    }


def status(**kw) -> None:
    path = STATE / "phase5-status.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw, updated=rp1.now())
    path.write_text(json.dumps(cur, indent=2))


rp1.write_status = status  # keep phase 1's state/status.json as phase 1 left it


def run(spec: dict, max_stops: int, candidate: bool) -> dict:
    marker = STATE / "runs" / f"{spec['id']}.done"
    if marker.exists():
        return json.loads(marker.read_text())
    extra: list[str] = []
    stops, attempts = [], []
    while True:
        rp1.wait_for_memory()
        status(run=spec["id"])
        info = rp1.execute(spec, extra)
        attempts.append({k: info[k] for k in ("start", "end", "exit_code", "wall_s", "args")})
        rc = info["exit_code"]
        if rc == 0 and spec["result"].exists() and all(p.exists() for p in spec["preds"]):
            result = json.loads(spec["result"].read_text())
            info.update(status="done", result_file=str(spec["result"].relative_to(rp1.ROOT)),
                        preds_files=[str(p.relative_to(rp1.ROOT)) for p in spec["preds"]],
                        auroc=result["value"], auroc_sklearn=result.get("auroc_sklearn_from_preds", result["value"]),
                        eval_split=result.get("eval_split"), db_rows=result.get("db_rows"),
                        peak_rss_mb=result.get("peak_rss_mb"), peak_gpu_mb=result.get("peak_gpu_mb"),
                        mem_available_min_gib=result.get("mem_available_min_gib"),
                        fallback_used=extra or None, attempts=attempts, infra_stops=stops)
            marker.write_text(json.dumps(info, indent=2))
            rp1.log(f"done {spec['id']}: {result.get('metric', 'test AUROC')} {result['value']:.6f} "
                    f"in {info['wall_s']:.0f} s, flags {info['flags']}")
            return info
        if info["start_floor_refusal"]:
            continue
        if (info["oom_in_log"] and info["mem_available_min_gib_driver"] >= rp1.STOP_FLOOR_GIB and not extra
                and spec["script"] == "nvidia-relbench-example.py"):
            rp1.log(f"{spec['id']}: out of memory with memory available; retry with {BATCH_FALLBACK}")
            extra = BATCH_FALLBACK
            continue
        infra = (rc in (75, 137, -9) or info.get("time_signal") == 9 or info["guard"] is not None
                 or info["mem_available_min_gib_driver"] < rp1.STOP_FLOOR_GIB or info["oom_in_log"])
        if infra:
            stops.append({k: info[k] for k in ("start", "end", "exit_code", "wall_s", "guard",
                                               "mem_available_min_gib_driver", "args", "before", "after")})
            rp1.log(f"{spec['id']}: memory stop {len(stops)} (exit {rc}, lowest "
                    f"{info['mem_available_min_gib_driver']} GiB)")
            if len(stops) >= max_stops:
                if candidate:
                    info.update(status="did_not_fit", attempts=attempts, infra_stops=stops,
                                error_tail=(LOGS / f"{spec['id']}.log").read_text(errors="replace")[-3000:])
                    marker.write_text(json.dumps(info, indent=2))
                    rp1.log(f"{spec['id']}: did not fit under the memory guard; left out of the pick")
                    return info
                status(blocked_reason=f"{spec['id']} stopped {len(stops)} times for memory")
                sys.exit(3)
            continue
        info.update(status="failed", attempts=attempts)
        (STATE / "runs" / f"{spec['id']}.failed").write_text(json.dumps(info, indent=2))
        status(failed=spec["id"])
        rp1.log(f"{spec['id']}: failed with exit {rc}; see logs/{spec['id']}.log")
        sys.exit(2)


def main() -> None:
    Path("/proc/self/oom_score_adj").write_text("1000")
    status(started=rp1.now(), blocked_reason=None, failed=None)
    rp1.log("phase 5 driver start")

    k6 = [run(spec_for(f"k6-s{s}", "k6", rp1.K1 + ["--upto_test_timestamp", "--seed", str(s)], "test"),
              max_stops=3, candidate=False) for s in (0, 1, 2)]

    tuning = {}
    for n in FANOUTS:
        args = rp1.override(K3, context_splits=["train"], num_neighbors=[str(n), str(n)])
        info = run(spec_for(f"k7val-nn{n}-s0", "k7val", args + ["--eval_split", "val", "--seed", "0"], "val"),
                   max_stops=2, candidate=True)
        tuning[f"{n} {n}"] = {"status": info["status"], "val_auroc_sklearn": info.get("auroc_sklearn"),
                              "val_auroc_script": info.get("auroc"), "run": info["id"],
                              "peak_rss_mb": info.get("peak_rss_mb"), "peak_gpu_mb": info.get("peak_gpu_mb"),
                              "wall_s": info.get("wall_s")}
    fitted = {k: v for k, v in tuning.items() if v["status"] == "done"}
    pick = max(fitted, key=lambda k: fitted[k]["val_auroc_sklearn"])
    n = int(pick.split()[0])
    decision = {
        "rule": "best validation AUROC (sklearn from saved predictions) among FANOUTS (1 1, 4 4, 16 16, 32 32); seed 0; "
                "K3 settings otherwise, context from the training weeks only, queries the validation week",
        "candidates": tuning, "pick": pick, "time": rp1.now(),
        "k7_runs": ("k3-s0, k3-s1, k3-s2 (the pick is K3's own fan-out; nothing reran)" if n == 1
                    else "k7-s0, k7-s1, k7-s2" if n == 16 else f"k7-nn{n}-s0, k7-nn{n}-s1, k7-nn{n}-s2"),
    }
    (RES / "phase5-k7-tuning.json").write_text(json.dumps(decision, indent=2))
    rp1.log(f"K7 pick: {pick}")

    if n != 1:
        for s in (0, 1, 2):
            args = rp1.override(K3, num_neighbors=[str(n), str(n)])
            rid = f"k7-s{s}" if n == 16 else f"k7-nn{n}-s{s}"  # the first pick, 16 16, ran as k7-s*
            run(spec_for(rid, "k7", args + ["--seed", str(s)], "test"), max_stops=3, candidate=False)
    status(run=None, done=rp1.now())
    rp1.log("phase 5 driver finished")


def main_l3() -> None:
    """L3: LightGBM on the data scientist's 71 features plus the 20 past churn labels, seeds 0-2.

    Added after the third gate's first round asked for the control no earlier run had; registered
    in docs/run-log.md before it ran. Same tuning as L1 (10 trials on validation AUROC)."""
    Path("/proc/self/oom_score_adj").write_text("1000")
    rp1.log("phase 5 driver start (L3)")
    for s in (0, 1, 2):
        rid = f"l3-s{s}"
        run({"id": rid, "config": "l3", "script": "lgbm-hm-churn.py",
             "args": ["--features", "ds_lags", "--seed", str(s), "--trials", "10"],
             "result": RES / "runs" / f"{rid}.json",
             "preds": [RES / "preds" / f"{rid}-test.parquet", RES / "preds" / f"{rid}-val.parquet"]},
            max_stops=3, candidate=False)
    status(run=None, done=rp1.now())
    rp1.log("phase 5 driver finished (L3)")


if __name__ == "__main__":
    main_l3() if "--l3" in sys.argv else main()
