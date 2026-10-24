#!/usr/bin/env bash
# Start the phase 1 driver on spark so it outlives the ssh call; prints the driver's PID.
# Usage (from the PC):  ssh spark 'bash ~/kumo-relational-tds/experiments/run-phase1.sh [--until <stage>]'
set -euo pipefail
ROOT="$HOME/kumo-relational-tds"
mkdir -p "$ROOT/logs" "$ROOT/state"
cd "$ROOT/experiments"
HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache/xdg" \
    setsid nohup taskset -c 5-9,15-19 "$ROOT/.venv/bin/python" -I "$ROOT/experiments/run-phase1.py" "$@" \
    >> "$ROOT/logs/phase1.log" 2>&1 < /dev/null &
echo $!
