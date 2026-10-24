#!/usr/bin/env bash
# Build the experiment environment in ~/kumo-relational-tds on the author's laptop
# (WSL Ubuntu 20.04, RTX 3080 10 GB). Re-running it is safe: it reuses the venv.
# Run from Windows:  wsl.exe -d Ubuntu --exec bash -lc "bash <path to this repo>/experiments/wsl-setup.sh"
set -euo pipefail

ROOT="$HOME/kumo-relational-tds"
UV="$HOME/.local/bin/uv"
mkdir -p "$ROOT"/experiments "$ROOT"/results "$ROOT"/logs "$ROOT"/cache/hf "$ROOT"/data
cd "$ROOT"

[ -x .venv/bin/python ] || "$UV" venv --python 3.12 .venv
PY="$ROOT/.venv/bin/python"

# torch and pyg-lib must match (pyg-lib 0.9.0 is built against torch 2.14.0 + CUDA 13.0).
"$UV" pip install --python "$PY" "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
"$UV" pip install --python "$PY" "pyg-lib==0.9.0" -f https://data.pyg.org/whl/torch-2.14.0+cu130.html
"$UV" pip install --python "$PY" "structured-data-models==0.2.1" relbench pandas pyarrow \
    torchmetrics tqdm lightgbm duckdb jinja2 optuna scikit-learn

HF_HOME="$ROOT/cache/hf" "$PY" - <<'EOF'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
for pkg in ["structured-data-models", "relbench", "pyg-lib", "lightgbm", "duckdb", "optuna", "pandas"]:
    print(pkg, md.version(pkg))
import sdm, relbench, pyg_lib  # noqa: F401  (import check)
print("imports ok")
EOF

"$UV" pip freeze --python "$PY" > "$ROOT/results/pip-freeze.txt"
echo "environment ready: $ROOT"
