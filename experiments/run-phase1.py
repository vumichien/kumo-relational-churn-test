"""Phase 1 driver: runs every experiment on spark, one heavy job at a time, resumably.

Stages, in order: env-check, gate, exp1-kumo, exp2, ds-features, exp1-tabular, exp1-lgbm,
analyze. `--until <stage>` stops after that stage; with no option it runs whatever isn't done.
Each finished run writes state/runs/<run-id>.done (JSON) and each finished stage writes
state/<stage>.done, so starting the driver again skips finished work. state/status.json
always says where it is.

Every child runs as
    /usr/bin/time -v -o logs/<run>.time taskset -c 5-9,15-19 .venv/bin/python -I <script> ...
and imports memory_guard (20 GiB start floor, 8 GiB stop floor, oom_score_adj 1000).
The driver itself also sets oom_score_adj 1000 and polls MemAvailable every 2 s while a run is
going, so a run killed by the kernel still gets its lowest reading.

Failure handling (decided in advance, phase-1-experiments.md "Running it unattended"):
- under 20 GiB available: wait, check every 5 minutes, carry on by itself; a 24-hour wait
  writes blocked_reason to status.json and exits 3;
- a run stopped by the guard (exit 75), killed by the kernel (signal 9), or whose lowest
  MemAvailable fell under 8 GiB: an infrastructure stop; wait for memory and rerun from the
  start; the third stop of the same run writes blocked_reason and exits 3;
- a run that fails for its own memory need (an out-of-memory error while available memory
  stayed above 8 GiB): the run's pre-decided fallbacks in order; when none is left the run is
  recorded as "did_not_fit" and the driver moves on;
- any other failure: state/runs/<id>.failed and exit 2 (the session diagnoses it);
- gate miss (K1 seed 0 more than 0.005 from 0.6992, or sklearn's AUROC from the saved
  predictions more than 1e-4 from the script's): state/gate.json says so and the driver exits 4.

Child mode `--child env-check` writes results/env-check.json and results/model-facts.json.

Start it with run-phase1.sh (setsid nohup, PID printed).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path.home() / "kumo-relational-tds"
EXP = ROOT / "experiments"
RES = ROOT / "results"
STATE = ROOT / "state"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
PY = ROOT / ".venv" / "bin" / "python"
JST = timezone(timedelta(hours=9))

START_FLOOR_GIB = 20.0
STOP_FLOOR_GIB = 8.0
WAIT_CHECK_S = 300
WAIT_LIMIT_S = 24 * 3600
MAX_INFRA_STOPS = 3
GATE_TARGET, GATE_TOL, AGREE_TOL = 0.6992, 0.005, 1e-4
SWAP_FLAG_BYTES = 100 * 2**20
TENANT_BUSY_UTIL = 50
LGBM_BUDGET_S = 8 * 3600
CPUS = "5-9,15-19"
OOM_PATTERN = re.compile(r"out of memory|MemoryError|OutOfMemoryError|std::bad_alloc|"
                         r"Out of Memory Error|failed to allocate", re.I)

STAGES = ["env-check", "gate", "exp1-kumo", "exp2", "ds-features", "exp1-tabular",
          "exp1-lgbm", "analyze"]

K1 = ["--dataset", "rel-hm", "--task", "user-churn", "--context_size", "20000",
      "--num_estimators", "8", "--context_mode", "latest", "--num_neighbors", "1", "1",
      "--num_lags", "20", "--context_splits", "train,val"]


def override(base: list[str], **flags: list[str]) -> list[str]:
    """K1's flags with some replaced (each flag's values are everything up to the next --)."""
    out, i = [], 0
    while i < len(base):
        flag = base[i]
        j = i + 1
        while j < len(base) and not base[j].startswith("--"):
            j += 1
        key = flag[2:]
        out += [flag] + (flags.pop(key) if key in flags else base[i + 1:j])
        i = j
    for key, values in flags.items():
        out += [f"--{key}"] + values
    return out


KUMO_CONFIGS = {
    "k1": K1,
    "k2": override(K1, context_splits=["train"]),
    "k3": override(K1, num_lags=["0"]),
    "k4": override(K1, num_neighbors=[]),
    "k5": override(K1, context_size=["10000"], num_estimators=["1"], num_neighbors=["16", "16"]),
    "s1": override(K1, context_size=["1000"]),
    "s2": override(K1, context_size=["5000"]),
}


def kumo_run(config: str, seed: int, extra: list[str] | None = None) -> dict:
    rid = f"{config}-s{seed}"
    return {
        "id": rid, "config": config, "script": "nvidia-relbench-example.py",
        "args": KUMO_CONFIGS[config] + ["--seed", str(seed),
                                        "--out", str(RES / "runs" / f"{rid}.json"),
                                        "--save_preds", str(RES / "preds" / f"{rid}-test.parquet")]
                + (extra or []),
        "result": RES / "runs" / f"{rid}.json",
        "preds": [RES / "preds" / f"{rid}-test.parquet"],
        "fallbacks": [["--batch_size", "2000"]],
    }


def plan() -> dict[str, list[dict]]:
    gate = kumo_run("k1", 0, ["--dump_task_table", str(DATA / "hm-task-table-lags20.parquet")])
    exp1 = [kumo_run("k1", s) for s in (1, 2)]
    for config in ("k2", "k3", "k4", "s1", "s2", "k5"):
        exp1 += [kumo_run(config, s) for s in (0, 1, 2)]
    exp2 = [{
        "id": f"c-s{s}", "config": "c", "script": "cooccurrence-test.py",
        "args": ["--seed", str(s)],
        "result": RES / "runs" / f"c-s{s}.json",
        "preds": [RES / "preds" / f"c-s{s}-test.parquet"], "fallbacks": [],
    } for s in (0, 1, 2)]
    ds = [{
        "id": "ds-features", "config": "ds-features", "script": "lgbm-hm-churn.py",
        "args": ["--features", "ds", "--build_only"],
        "result": RES / "runs" / "ds-features.json", "preds": [],
        "fallbacks": [["--subsample", "1000000"], ["--subsample", "500000"]],
    }]
    tab = [{
        "id": f"t1-s{s}", "config": "t1", "script": "kumo-tabular-hm-churn.py",
        "args": ["--seed", str(s)],
        "result": RES / "runs" / f"t1-s{s}.json",
        "preds": [RES / "preds" / f"t1-s{s}-test.parquet"],
        "fallbacks": [["--context_size", "10000"], ["--context_size", "10000", "--batch_size", "2000"]],
    } for s in (0, 1, 2)]
    lgbm = [{
        "id": f"{c}-s{s}", "config": c, "script": "lgbm-hm-churn.py",
        "args": ["--features", feats, "--seed", str(s), "--trials", "10"],
        "result": RES / "runs" / f"{c}-s{s}.json",
        "preds": [RES / "preds" / f"{c}-s{s}-test.parquet", RES / "preds" / f"{c}-s{s}-val.parquet"],
        "fallbacks": ([["--subsample", "1000000"], ["--subsample", "500000"]] if c == "l1" else []),
    } for c, feats in (("l1", "ds"), ("l2", "auto")) for s in (0, 1, 2)]
    return {"gate": [gate], "exp1-kumo": exp1, "exp2": exp2, "ds-features": ds,
            "exp1-tabular": tab, "exp1-lgbm": lgbm}


# ---------------------------------------------------------------------------------------------
# Machine readings

def now() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("no MemAvailable")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except subprocess.TimeoutExpired:
        return "timeout"


def snapshot() -> dict:
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    util = sh("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    return {
        "time": now(), "uptime": sh("uptime"),
        "mem_available_gib": round(mem_available_gib(), 2),
        "gpu_util_pct": int(util) if util.isdigit() else util,
        "gpu_apps": apps.splitlines(),
        "pswpin": int(vm["pswpin"]), "pswpout": int(vm["pswpout"]),
    }


# ---------------------------------------------------------------------------------------------
# Status

STATUS: dict = {}


def write_status(**kw) -> None:
    STATUS.update(kw, updated=now())
    tmp = STATE / "status.json.tmp"
    tmp.write_text(json.dumps(STATUS, indent=2))
    tmp.replace(STATE / "status.json")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def done_runs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((STATE / "runs").glob("*.done"))]


def update_progress(stage: str, run_id: str | None, all_runs: list[str]) -> None:
    done = done_runs()
    done_ids = {d["id"] for d in done}
    left = [r for r in all_runs if r not in done_ids]
    walls = [d["wall_s"] for d in done if d.get("wall_s")]
    avg = sum(walls) / len(walls) if walls else None
    last = max(done, key=lambda d: d["end"]) if done else None
    write_status(
        stage=stage, run=run_id, runs_done=len(done_ids & set(all_runs)), runs_left=len(left),
        left=left,
        last_run=None if last is None else {k: last.get(k) for k in ("id", "auroc", "wall_s", "status")},
        projected_finish=None if avg is None else
        (datetime.now(JST) + timedelta(seconds=avg * len(left))).strftime("%Y-%m-%d %H:%M JST"),
        projection_note="mean wall-clock of finished runs x runs left; stages differ, so rough",
    )


# ---------------------------------------------------------------------------------------------
# Waiting for memory

def wait_for_memory() -> None:
    gib = mem_available_gib()
    if gib >= START_FLOOR_GIB:
        return
    since = time.time()
    since_s = now()
    waits = json.loads((STATE / "memory-waits.json").read_text()) if (STATE / "memory-waits.json").exists() else []
    lowest = gib
    while gib < START_FLOOR_GIB:
        lowest = min(lowest, gib)
        write_status(waiting_for_memory={"since": since_s, "last_mem_available_gib": round(gib, 2),
                                         "lowest_gib": round(lowest, 2)})
        log(f"waiting for memory: {gib:.2f} GiB available (< {START_FLOOR_GIB})")
        if time.time() - since > WAIT_LIMIT_S:
            write_status(blocked_reason=f"waited for memory more than 24 h since {since_s}")
            log("blocked: 24-hour memory wait")
            sys.exit(3)
        time.sleep(WAIT_CHECK_S)
        gib = mem_available_gib()
    waits.append({"from": since_s, "to": now(), "lowest_gib": round(lowest, 2)})
    (STATE / "memory-waits.json").write_text(json.dumps(waits, indent=2))
    write_status(waiting_for_memory=None)
    log(f"memory back: {gib:.2f} GiB")


# ---------------------------------------------------------------------------------------------
# One run

def fallback_index(config: str) -> int:
    p = STATE / f"fallback-{config}.json"
    return json.loads(p.read_text())["index"] if p.exists() else 0


def child_env(run_id: str) -> dict:
    env = dict(os.environ)
    env.update(HF_HOME=str(ROOT / "cache" / "hf"), XDG_CACHE_HOME=str(ROOT / "cache" / "xdg"),
               OMP_NUM_THREADS="10", MKL_NUM_THREADS="10", OPENBLAS_NUM_THREADS="10",
               GUARD_FILE=str(STATE / "runs" / f"{run_id}.guard"),
               HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_MININTERVAL="30")
    return env


def parse_time_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size"):
            out["time_max_rss_mb"] = round(int(line.split(":")[-1]) / 1024, 1)
        elif line.startswith("Elapsed (wall clock) time"):
            val = line.split("):")[-1].strip()
            parts = [float(x) for x in val.split(":")]
            secs = 0.0
            for x in parts:
                secs = secs * 60 + x
            out["time_elapsed_s"] = round(secs, 2)
        elif line.startswith("Exit status"):
            out["time_exit_status"] = int(line.split(":")[-1])
        elif line.startswith("Command terminated by signal"):
            out["time_signal"] = int(line.split()[-1])
    return out


def execute(spec: dict, extra: list[str]) -> dict:
    rid = spec["id"]
    log_path = LOGS / f"{rid}.log"
    time_path = LOGS / f"{rid}.time"
    guard_path = STATE / "runs" / f"{rid}.guard"
    guard_path.unlink(missing_ok=True)
    for p in [spec["result"], *spec["preds"]]:
        p.unlink(missing_ok=True)
    cmd = ["/usr/bin/time", "-v", "-o", str(time_path), "taskset", "-c", CPUS,
           str(PY), "-I", str(EXP / spec["script"]), *spec["args"], *extra]
    before = snapshot()
    log(f"start {rid}: {' '.join(cmd[7:])}")
    t0 = time.time()
    min_gib = before["mem_available_gib"]
    with open(log_path, "a") as fh:
        fh.write(f"\n===== {now()} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=EXP, env=child_env(rid), stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            min_gib = min(min_gib, mem_available_gib())
            time.sleep(2)
    wall = time.time() - t0
    after = snapshot()
    rc = proc.returncode
    swap_bytes = (after["pswpin"] - before["pswpin"] + after["pswpout"] - before["pswpout"]) * os.sysconf("SC_PAGE_SIZE")
    util0 = before["gpu_util_pct"]
    flags = []
    if swap_bytes > SWAP_FLAG_BYTES:
        flags.append("swap_moved_over_100MB")
    if isinstance(util0, int) and util0 > TENANT_BUSY_UTIL:
        flags.append("tenant_gpu_busy_at_start")
    tail = log_path.read_text(errors="replace")[-20000:]
    return {
        "id": rid, "config": spec["config"], "script": spec["script"],
        "args": spec["args"] + extra, "start": before["time"], "end": after["time"],
        "exit_code": rc, "wall_s": round(wall, 2), "before": before, "after": after,
        "swap_moved_mb": round(swap_bytes / 2**20, 1), "mem_available_min_gib_driver": round(min_gib, 2),
        "flags": flags, "guard": json.loads(guard_path.read_text()) if guard_path.exists() else None,
        "start_floor_refusal": rc == 75 and "start floor" in tail[-2000:],
        "oom_in_log": bool(OOM_PATTERN.search(tail)), **parse_time_file(time_path),
    }


def run_one(spec: dict) -> dict:
    rid = spec["id"]
    marker = STATE / "runs" / f"{rid}.done"
    if marker.exists():
        return json.loads(marker.read_text())
    stops_path = STATE / "runs" / f"{rid}.stops.json"
    stops = json.loads(stops_path.read_text()) if stops_path.exists() else []
    fb = fallback_index(spec["config"])
    attempts = []
    while True:
        wait_for_memory()
        extra = [] if fb == 0 else spec["fallbacks"][fb - 1]
        info = execute(spec, extra)
        attempts.append({k: info[k] for k in ("start", "end", "exit_code", "wall_s", "args")})
        rc = info["exit_code"]
        ok = rc == 0 and spec["result"].exists() and all(p.exists() for p in spec["preds"])
        if ok:
            result = json.loads(spec["result"].read_text())
            info.update(status="done", result_file=str(spec["result"].relative_to(ROOT)),
                        preds_files=[str(p.relative_to(ROOT)) for p in spec["preds"]],
                        auroc=result.get("value", result.get("test_auroc")),
                        peak_rss_mb=result.get("peak_rss_mb"), peak_gpu_mb=result.get("peak_gpu_mb"),
                        mem_available_min_gib=result.get("mem_available_min_gib"),
                        fallback_used=extra or None, attempts=attempts, infra_stops=stops)
            marker.write_text(json.dumps(info, indent=2))
            log(f"done {rid}: auroc {info['auroc']} in {info['wall_s']:.0f} s, flags {info['flags']}")
            return info
        if info["start_floor_refusal"]:
            log(f"{rid}: refused at the start floor; waiting")
            continue
        infra = (rc in (75, 137, -9) or info.get("time_signal") == 9
                 or info["mem_available_min_gib_driver"] < STOP_FLOOR_GIB or info["guard"] is not None)
        if infra:
            stops.append({k: info[k] for k in ("start", "end", "exit_code", "wall_s", "guard",
                                               "mem_available_min_gib_driver", "args")})
            stops_path.write_text(json.dumps(stops, indent=2))
            log(f"{rid}: infrastructure stop {len(stops)} (exit {rc}, lowest "
                f"{info['mem_available_min_gib_driver']} GiB)")
            if len(stops) >= MAX_INFRA_STOPS:
                write_status(blocked_reason=f"{rid} stopped {len(stops)} times for memory")
                sys.exit(3)
            continue
        if info["oom_in_log"]:
            log(f"{rid}: out of memory with {info['mem_available_min_gib_driver']} GiB still available")
            if fb < len(spec["fallbacks"]):
                fb += 1
                (STATE / f"fallback-{spec['config']}.json").write_text(json.dumps(
                    {"index": fb, "args": spec["fallbacks"][fb - 1], "decided_by": rid, "time": now()}))
                log(f"{rid}: fallback {fb}: {spec['fallbacks'][fb - 1]}")
                continue
            info.update(status="did_not_fit", attempts=attempts, infra_stops=stops,
                        error_tail=(LOGS / f"{rid}.log").read_text(errors="replace")[-3000:])
            marker.write_text(json.dumps(info, indent=2))
            log(f"{rid}: did not fit")
            return info
        info.update(status="failed", attempts=attempts)
        (STATE / "runs" / f"{rid}.failed").write_text(json.dumps(info, indent=2))
        write_status(failed=rid)
        log(f"{rid}: failed with exit {rc}; see logs/{rid}.log")
        sys.exit(2)


# ---------------------------------------------------------------------------------------------
# Stages

def stage_env_check() -> None:
    spec = {"id": "env-check", "config": "env-check", "script": "run-phase1.py",
            "args": ["--child", "env-check"], "result": RES / "env-check.json",
            "preds": [RES / "model-facts.json"], "fallbacks": []}
    info = run_one(spec)
    env = json.loads((RES / "env-check.json").read_text())
    if not env["ok"]:
        log(f"env-check failed: {env['problems']}")
        sys.exit(2)


def stage_gate(specs: list[dict]) -> None:
    info = run_one(specs[0])
    result = json.loads(specs[0]["result"].read_text())
    value, sk = result["value"], result["auroc_sklearn_from_preds"]
    passed = abs(value - GATE_TARGET) <= GATE_TOL and sk is not None and abs(value - sk) <= AGREE_TOL
    gate = {"run": "k1-s0", "auroc": value, "auroc_sklearn_from_preds": sk, "target": GATE_TARGET,
            "tolerance": GATE_TOL, "agreement_tolerance": AGREE_TOL, "passed": passed, "time": now()}
    (STATE / "gate.json").write_text(json.dumps(gate, indent=2))
    (RES / "gate.json").write_text(json.dumps(gate, indent=2))
    log(f"gate: {gate}")
    if not passed:
        write_status(blocked_reason="gate miss")
        sys.exit(4)


def stage_lgbm(specs: list[dict]) -> None:
    l1 = [s for s in specs if s["config"] == "l1"]
    l2 = [s for s in specs if s["config"] == "l2"]
    first = run_one(l1[0])
    budget_path = STATE / "lgbm-budget.json"
    if not budget_path.exists():
        projected = (len(l1) - 1 + len(l2)) * first["wall_s"]
        seed0_only = projected > LGBM_BUDGET_S
        budget_path.write_text(json.dumps({
            "l1_s0_wall_s": first["wall_s"], "projected_remaining_s": round(projected),
            "rule": "remaining L1 and L2 runs each take as long as L1 seed 0; over 8 h -> seed 0 only",
            "seed0_only": seed0_only, "time": now()}, indent=2))
    budget = json.loads(budget_path.read_text())
    todo = [l1[0], l2[0]] if budget["seed0_only"] else l1[1:] + l2
    for spec in todo:
        run_one(spec)


def stage_analyze() -> None:
    spec = {"id": "analyze", "config": "analyze", "script": "analyze-phase1.py", "args": [],
            "result": RES / "exp1-summary.json",
            "preds": [RES / "context-sweep.json", RES / "exp2-summary.json", RES / "prose-facts.json"],
            "fallbacks": []}
    run_one(spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", choices=STAGES, default=None)
    parser.add_argument("--child", choices=["env-check"], default=None)
    args = parser.parse_args()
    if args.child == "env-check":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import memory_guard
        memory_guard.start()
        env_check(memory_guard)
        return

    Path("/proc/self/oom_score_adj").write_text("1000")
    for d in (RES / "runs", RES / "preds", RES / "figure-data", STATE / "runs", LOGS, DATA):
        d.mkdir(parents=True, exist_ok=True)
    runs = plan()
    all_ids = ["env-check"] + [s["id"] for st in runs.values() for s in st] + ["analyze"]
    write_status(pid=os.getpid(), started=now(), blocked_reason=None, failed=None,
                 waiting_for_memory=None, until=args.until)
    log(f"driver pid {os.getpid()}, until {args.until}")
    for stage in STAGES:
        marker = STATE / f"{stage}.done"
        if marker.exists():
            log(f"stage {stage}: done earlier")
        else:
            update_progress(stage, None, all_ids)
            log(f"stage {stage}: start")
            if stage == "env-check":
                stage_env_check()
            elif stage == "gate":
                stage_gate(runs["gate"])
            elif stage == "exp1-lgbm":
                stage_lgbm(runs[stage])
            elif stage == "analyze":
                stage_analyze()
            else:
                for spec in runs[stage]:
                    update_progress(stage, spec["id"], all_ids)
                    run_one(spec)
            marker.write_text(json.dumps({"stage": stage, "done": now()}))
            log(f"stage {stage}: done")
        update_progress(stage, None, all_ids)
        if args.until == stage:
            break
    write_status(stage="done" if args.until in (None, "analyze") else f"stopped after {args.until}",
                 run=None)
    log("driver finished")


# ---------------------------------------------------------------------------------------------
# env-check child (runs under the memory guard like every other run)

def env_check(memory_guard) -> None:
    import importlib.metadata as md
    import inspect

    import torch

    problems = []
    pilot = {}
    for line in (EXP / "pip-freeze-pilot.txt").read_text().splitlines():
        if "==" in line:
            name, ver = line.split("==")
            pilot[name.lower()] = ver
    versions = {}
    for pkg in ["torch", "structured-data-models", "pyg-lib", "relbench", "lightgbm", "duckdb",
                "optuna", "pandas", "numpy", "scikit-learn", "torchmetrics", "pyarrow"]:
        ver = md.version(pkg)
        versions[pkg] = {"spark": ver, "pilot": pilot.get(pkg)}
        if ver != pilot.get(pkg):
            problems.append(f"{pkg} {ver} != pilot {pilot.get(pkg)}")
    import duckdb, lightgbm, pyg_lib, relbench, sdm  # noqa: F401,E401  (import check on aarch64)

    cuda_ok = torch.cuda.is_available()
    matmul_ok = False
    if cuda_ok:
        a = torch.randn(4096, 4096, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        matmul_ok = b == b  # not NaN
        del a
    if not (cuda_ok and matmul_ok):
        problems.append("CUDA check failed")

    hub = ROOT / "cache" / "hf" / "hub"
    rel_snap = hub / "models--nvidia--Kumo-Relational" / "snapshots" / "2bd603d3d8f25f67a7aaa8579908595e20567e22"
    data_snap = hub / "datasets--stanford-star--relbench-v1" / "snapshots" / "21d7284df2a9621c71f21bc4a1a30b7e8effb8dd"
    for p in (rel_snap, data_snap):
        if not p.exists():
            problems.append(f"missing cache {p.name}")
    meminfo = {line.split(":")[0]: line.split(":")[1].strip()
               for line in Path("/proc/meminfo").read_text().splitlines()
               if line.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
    env = {
        "time": now(), "host": os.uname().nodename, "machine": os.uname().machine,
        "python": sys.version.split()[0], "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu": torch.cuda.get_device_name(0) if cuda_ok else None,
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "torch_cuda": torch.version.cuda, "cuda_available": cuda_ok, "matmul_4096_ok": matmul_ok,
        "meminfo": meminfo, "gpu_apps": sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                                           "--format=csv,noheader").splitlines(),
        "versions": versions,
        "model_snapshot": rel_snap.name if rel_snap.exists() else None,
        "dataset_snapshot": data_snap.name if data_snap.exists() else None,
        "problems": problems, "ok": not problems, **memory_guard.stats(),
    }
    (RES / "env-check.json").write_text(json.dumps(env, indent=2))

    from sdm import Task
    from sdm.models import KumoRelational, KumoTabular
    from sdm.models.kumo.relational.model import _KumoRelational

    def heads(model) -> dict:
        return {str(t): sum(p.numel() for p in m.parameters()) for t, m in model.models.items()}

    def files(snap: Path) -> dict:
        return {p.name: p.stat().st_size for p in sorted(snap.rglob("*")) if p.is_file()}

    rel = KumoRelational(device="cpu")
    defaults = {k: v.default for k, v in inspect.signature(_KumoRelational.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty}
    tab = KumoTabular(size="large", device="cpu")
    tab_snaps = list((hub / "models--nvidia--Kumo-Tabular" / "snapshots").glob("*"))
    import sdm.models.kumo.tabular.model as tabmod
    facts = {
        "time": now(),
        "kumo_relational": {
            "revision": "v1.0.1", "commit": rel_snap.name,
            "parameters_per_head": heads(rel), "parameters_total": sum(p.numel() for p in rel.parameters()),
            "checkpoint_files_bytes": files(rel_snap),
            "config_json": json.loads((rel_snap / "config.json").read_text()),
            "_KumoRelational_defaults": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                                         for k, v in defaults.items()},
            "default_recipe": repr(rel.default_recipe()),
        },
        "kumo_tabular_large": {
            "revision": "v1.0.1", "snapshots": [p.name for p in tab_snaps],
            "parameters_per_head": heads(tab), "parameters_total": sum(p.numel() for p in tab.parameters()),
            "checkpoint_files_bytes": {s.name: files(s) for s in tab_snaps},
            "model_kwargs": tabmod.MODEL_KWARGS["large"],
            "default_recipe": repr(tab.default_recipe()),
        },
        "task_classification": str(Task.classification),
    }
    (RES / "model-facts.json").write_text(json.dumps(facts, indent=2, default=str))
    print(json.dumps({"env_ok": env["ok"], "problems": problems,
                      "rel_params": facts["kumo_relational"]["parameters_per_head"],
                      "tab_params": facts["kumo_tabular_large"]["parameters_per_head"]}, indent=2))


if __name__ == "__main__":
    main()
