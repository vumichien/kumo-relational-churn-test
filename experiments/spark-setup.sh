#!/usr/bin/env bash
# Build the phase 1 environment in ~/kumo-relational-tds on spark (NVIDIA GB10, aarch64).
# A port of wsl-setup.sh with the pilot's version pins (results/pip-freeze.txt).
# Re-running it is safe: it reuses the venv and the caches.
# Start (from the PC):
#   scp experiments/spark-setup.sh experiments/pip-freeze-pilot.txt spark:~/kumo-relational-tds/experiments/
#   ssh spark 'cd ~/kumo-relational-tds && setsid nohup bash experiments/spark-setup.sh > logs/spark-setup.log 2>&1 < /dev/null & echo $!'
set -euo pipefail

# The kernel's first out-of-memory victim is us, never the other tenant (children inherit it).
echo 1000 > /proc/self/oom_score_adj

ROOT="$HOME/kumo-relational-tds"
UV="$HOME/.local/bin/uv"
mkdir -p "$ROOT"/experiments "$ROOT"/results "$ROOT"/logs "$ROOT"/cache/hf "$ROOT"/data "$ROOT"/state
cd "$ROOT"
export HF_HOME="$ROOT/cache/hf"
# Keep every cache (uv, its Pythons, relbench, others) inside the article folder.
export UV_CACHE_DIR="$ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/cache/uv-python" XDG_CACHE_HOME="$ROOT/cache/xdg"
T0=$(date +%s)

avail_gib=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
echo "start $(date '+%F %T %Z'), MemAvailable ${avail_gib} GiB"
if awk -v a="$avail_gib" 'BEGIN {exit !(a < 20)}'; then
    echo "MemAvailable under 20 GiB: not starting"; exit 75
fi

# Python 3.12.13 (the pilot's) if uv can provide it, else the system 3.12.3.
if [ ! -x .venv/bin/python ]; then
    if "$UV" python install 3.12.13 && "$UV" venv --python 3.12.13 .venv; then
        echo "python: uv-managed 3.12.13"
    else
        "$UV" venv --python /usr/bin/python3.12 .venv
        echo "python: system $(/usr/bin/python3.12 --version)"
    fi
fi
PY="$ROOT/.venv/bin/python"
"$PY" --version

# Constraints from the pilot's freeze: local tags dropped; torch's own CUDA/triton wheels and
# pyg-lib are left to torch and the pyg index, because their aarch64 builds differ from x86's.
grep -vE '^(torch|pyg-lib|triton|nvidia-|cuda-)' experiments/pip-freeze-pilot.txt \
    | sed -E 's/\+[a-z0-9]+$//' > experiments/constraints.txt

# torch and pyg-lib must match (pyg-lib 0.9.0 is built against torch 2.14.0 + CUDA 13.0).
"$UV" pip install --python "$PY" "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install --python "$PY" "pyg-lib==0.9.0" -f https://data.pyg.org/whl/torch-2.14.0+cu130.html
"$UV" pip install --python "$PY" -c experiments/constraints.txt "torch==2.14.0" \
    "structured-data-models==0.2.1" "relbench==3.0.2" pandas pyarrow \
    torchmetrics tqdm lightgbm duckdb jinja2 optuna scikit-learn \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

"$PY" - <<'EOF'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
for pkg in ["structured-data-models", "relbench", "pyg-lib", "lightgbm", "duckdb", "optuna", "pandas"]:
    print(pkg, md.version(pkg))
import sdm, relbench, pyg_lib, lightgbm, duckdb  # noqa: F401  (import check)
print("imports ok")
EOF

# Weights at the revision the library pins, and the RelBench rel-hm snapshot.
"$PY" - <<'EOF'
import huggingface_hub as hh
import relbench
p = hh.snapshot_download("nvidia/Kumo-Relational", revision="v1.0.1")
print("weights", p)
dataset = relbench.load_dataset("rel-hm")
task = dataset.load_task("user-churn")
db = task.get_db(upto_test_timestamp=False)
print("rel-hm tables", {k: len(v.df) for k, v in db.table_dict.items()})
for s in ["train", "val", "test"]:
    print(s, len(task.get_table(s, mask_input_cols=False).df))
EOF
find "$HF_HOME" -maxdepth 4 -path '*snapshots/*' -type d | sed "s|$HOME|~|"
du -sh "$HF_HOME"

"$UV" pip freeze --python "$PY" > "$ROOT/results/pip-freeze-spark.txt"
echo "environment ready: $ROOT in $(( $(date +%s) - T0 )) s"
