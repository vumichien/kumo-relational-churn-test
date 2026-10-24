"""Smoke test: load Kumo Relational and run the four-user example from its docstring.

Checks that the weights download, the model runs on the GPU (or CPU), and prints
the load time, the prediction time and the peak memory. Writes results/pilot-smoke.json.

Run (WSL):  HF_HOME=~/kumo-relational-tds/cache/hf ~/kumo-relational-tds/.venv/bin/python -I smoke-kumo-relational.py
"""

import json
import platform
import time
from pathlib import Path

import torch

import sdm
from sdm import RelatedTables, TableTensor
from sdm.models import KumoRelational


def vm_hwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.perf_counter()

task_table = TableTensor.from_columns(
    {"user_id": [0, 1, 2, 3], "churn": [True, False, True, False]},
    stypes={"user_id": "id", "churn": "categorical"},
    device=device,
)
related = RelatedTables(
    tables={
        "users": TableTensor.from_columns(
            {"user_id": [0, 1, 2, 3], "age": [42, 23, 31, 26]},
            stypes={"user_id": "id", "age": "numerical"},
            device=device,
        ),
        "orders": TableTensor.from_columns(
            {"user_id": [0, 0, 1, 3, 3, 3], "amount": [9.99, 4.99, 12.99, 7.99, 3.99, 5.99]},
            stypes={"user_id": "id", "amount": "numerical"},
            device=device,
        ),
    },
    relationships=[{"left_table": "orders", "left_columns": "user_id",
                    "right_table": "users", "right_columns": "user_id"}],
    task_links=[{"task_columns": "user_id", "table": "users", "table_columns": "user_id"}],
)
context_related = related.replace_tables({
    "users": related.tables["users"][:2], "orders": related.tables["orders"][:3]})
query_related = related.replace_tables({
    "users": related.tables["users"][2:], "orders": related.tables["orders"][3:]})

t_load = time.perf_counter()
model = KumoRelational(device=device)
load_s = time.perf_counter() - t_load

t_pred = time.perf_counter()
with torch.inference_mode():
    out = model(
        x_context=task_table[:2].drop_columns("churn"),
        y_context=task_table[:2, "churn"],
        x_query=task_table[2:].drop_columns("churn"),
        related_context_tables=context_related,
        related_query_tables=query_related,
        num_hops=1,
    )
if device == "cuda":
    torch.cuda.synchronize()
pred_s = time.perf_counter() - t_pred

result = {
    "script": "smoke-kumo-relational.py",
    "device": device,
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    "torch": torch.__version__,
    "sdm": getattr(sdm, "__version__", None),
    "python": platform.python_version(),
    "num_parameters": sum(p.numel() for p in model.parameters()),
    "output_columns": list(out.column_names),
    "probabilities": out.numerical.float().cpu().tolist(),
    "load_seconds": round(load_s, 2),
    "predict_seconds": round(pred_s, 3),
    "total_seconds": round(time.perf_counter() - t0, 2),
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
    "peak_rss_mb": round(vm_hwm_mb(), 1),
}
out_dir = Path.home() / "kumo-relational-tds" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "pilot-smoke.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
