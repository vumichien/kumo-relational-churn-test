"""Memory guard for every phase 1 process on spark (a GB10 shared with another tenant).

Import it first and call `start()`:
- writes 1000 to /proc/self/oom_score_adj, so if memory runs out the kernel stops our job
  (children inherit it), never the tenant's;
- refuses to start (exit 75) unless MemAvailable is at least START_FLOOR_GIB;
- runs a watchdog thread that reads /proc/meminfo every second and stops the process
  cleanly (exit 75, reason written to GUARD_FILE if set) when MemAvailable falls under
  STOP_FLOOR_GIB, before the kernel has to act.

/proc/meminfo reports kB that are KiB, so 20 GiB = 20 * 1024**2 kB.
`stats()` returns the start reading and the lowest reading seen by the watchdog.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

START_FLOOR_GIB = 20.0
STOP_FLOOR_GIB = 8.0
EXIT_MEMORY = 75

_state = {"start_gib": None, "min_gib": None, "stopped": False}


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def set_oom_score_adj() -> None:
    Path("/proc/self/oom_score_adj").write_text("1000")


def _watchdog(stop_floor: float, guard_file: str | None) -> None:
    while True:
        gib = mem_available_gib()
        if _state["min_gib"] is None or gib < _state["min_gib"]:
            _state["min_gib"] = gib
        if gib < stop_floor:
            _state["stopped"] = True
            reason = {"guard_stop": True, "mem_available_gib": round(gib, 2),
                      "stop_floor_gib": stop_floor, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
            print(f"[memory_guard] MemAvailable {gib:.2f} GiB < {stop_floor} GiB: stopping",
                  file=sys.stderr, flush=True)
            if guard_file:
                try:
                    Path(guard_file).write_text(json.dumps(reason))
                except OSError:
                    pass
            os._exit(EXIT_MEMORY)
        time.sleep(1.0)


def start(start_floor: float = START_FLOOR_GIB, stop_floor: float = STOP_FLOOR_GIB) -> None:
    set_oom_score_adj()
    gib = mem_available_gib()
    _state["start_gib"] = _state["min_gib"] = gib
    if gib < start_floor:
        print(f"[memory_guard] MemAvailable {gib:.2f} GiB < start floor {start_floor} GiB",
              file=sys.stderr, flush=True)
        sys.exit(EXIT_MEMORY)
    guard_file = os.environ.get("GUARD_FILE")
    threading.Thread(target=_watchdog, args=(stop_floor, guard_file), daemon=True).start()


def stats() -> dict:
    return {"mem_available_start_gib": round(_state["start_gib"] or float("nan"), 2),
            "mem_available_min_gib": round(_state["min_gib"] or float("nan"), 2)}


def vm_hwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")
