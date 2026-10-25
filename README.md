# Kumo Relational churn test

This is the companion code for the Towards Data Science article *Can a Pretrained Model Predict Customer Churn Across Your Tables Without Training? NVIDIA Kumo Relational, Tested* (link added when the article is published). The article tests NVIDIA's open Kumo Relational checkpoint, which predicts from labelled example rows handed to it as context instead of being trained on the task. It is run on the H&M customer-churn task from RelBench (`rel-hm/user-churn`), next to Kumo Tabular and a data scientist's tuned LightGBM pipeline, plus a synthetic test that is rigged on purpose to show what per-column summaries can't see.

Everything here ran on one NVIDIA GB10 machine (aarch64, unified memory) that was shared with another tenant's model servers, so the timings in `results/` come from a shared machine.

## What's in the repo

```
experiments/   every script the article's numbers came from, the setup scripts and the version pins
results/       the aggregate results: one JSON per run, the summaries, and the data behind the figures
```

| Script | What it does |
|---|---|
| `experiments/nvidia-relbench-example.py` | NVIDIA's RelBench example for Kumo Relational, copied from [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (`examples/relational/relbench/main.py`, commit `f8d054c`, Apache-2.0) with timing, memory, a result file, saved predictions and the `--context_splits`, `--upto_test_timestamp` (tables cut at the test date) and `--eval_split val` (score the validation week, for tuning) options added; the modelling code is unchanged. Runs K1–K7 and the context sweep S1–S2. |
| `experiments/kumo-tabular-hm-churn.py` | T1: Kumo Tabular reads the data scientist's 71 hand-built features. |
| `experiments/lgbm-hm-churn.py` | L1, L2 and L3: tuned LightGBM on the hand-built features, on the customer table plus the 20 past churn labels Kumo Relational builds, and on the hand-built features plus those labels (`--features ds_lags`). `--build_only` renders the features. |
| `experiments/cooccurrence-test.py` | Experiment 2: a synthetic two-table shop where a customer churns only if one single order was both late and refunded (C1–C5). |
| `experiments/analyze-phase1.py` | Recomputes every AUROC from the saved predictions with scikit-learn and writes the summaries in `results/`; `--phase5` does the same for K6, K7 and L3 (`results/phase5-summary.json`). |
| `experiments/run-phase5.py` | The follow-up checks: K6 (K1 on tables cut at the test date) and K7 (K3 with its neighbour count tuned on the validation week, then three test seeds); `--l3` runs L3's three seeds. Reuses `run-phase1.py`'s run machinery. |
| `experiments/run-phase1.py`, `run-phase1.sh` | The driver: runs every stage in order, one heavy job at a time, and can be restarted (finished runs are skipped). |
| `experiments/memory_guard.py` | Refuses to start under 20 GiB of free memory and stops a run cleanly under 8 GiB, so a shared machine's other users are never pushed out. |
| `experiments/spark-setup.sh`, `wsl-setup.sh` | Build the environment (GB10 aarch64; or WSL with a CUDA GPU) with the pinned versions in `experiments/pip-freeze-pilot.txt`. |
| `experiments/smoke-kumo-relational.py`, `pilot-*.py` | The smoke test and the pilots run before the experiments were planned. |

## The results

Test AUROC on H&M churn, mean of three seeds (min–max in brackets), from `results/exp1-summary.json`:

| Run | What it is | Test AUROC |
|---|---|---|
| K1 | Kumo Relational, NVIDIA's settings (8 × 20,000 context rows, 20 past labels) | 0.6988 (0.6983–0.6992) |
| K2 | K1 with context from the training weeks only | 0.6998 (0.6994–0.7003) |
| K3 | K1 without past churn labels | 0.5904 (0.5902–0.5904) |
| K4 | K1 without related tables | 0.6939 (0.6937–0.6941) |
| K5 | the library's defaults | 0.6965 (0.6959–0.6972) |
| T1 | Kumo Tabular on the hand-built features | 0.6927 (0.6925–0.6930) |
| L1 | tuned LightGBM on the hand-built features, all 3,832,692 training rows | 0.6935 (0.6926–0.6941) |
| L2 | tuned LightGBM on the customer table and the past labels | 0.6945 (0.6944–0.6946) |
| L3 | tuned LightGBM on L1's inputs plus the past labels | 0.7020 (0.7017–0.7022) |
| K6 | K1 on tables cut at the test date | 0.6988 (0.6983–0.6993) |
| K7 | K3 with 16 sampled neighbours per hop, tuned on the validation week (1, 4 and 16 tried; 32 did not fit under the memory guard) | 0.6786 (0.6783–0.6790) |

The first K1 run reproduces NVIDIA's published 0.6992 (`results/gate.json`: 0.69925). The synthetic test is in `results/exp2-summary.json` and the context sweep in `results/context-sweep.json`.

### Details the article leaves here

- **Tuning.** NVIDIA picked K1's settings on the validation week; I didn't retune them. Each LightGBM (L1, L2, L3) got the user study's 10 tuning trials on the validation week (`args.trials` in `results/runs/l*-s*.json`).
- **K5, the library's defaults** (16 sampled neighbours per hop, 10,000 context rows, one estimator), still scored 0.6965, about 0.002 below NVIDIA's tuned settings.
- **K2, context from the training weeks only,** scored 0.6998, slightly above K1, which also reads the validation week as examples. I expected the opposite and can't explain it yet.
- **K7's neighbour search** on the validation week (`results/phase5-k7-tuning.json`): 1 per hop 0.5964, 4 per hop 0.6396, 16 per hop 0.6790; 32 per hop did not fit in the machine's free memory, so the score was still rising at the largest setting tried.

## Rerun it

The scripts expect the repo at `~/kumo-relational-tds` and write to `results/`, `data/`, `logs/` and `cache/` inside it. They pin themselves to CPUs 5–9 and 15–19 (the GB10's performance cores); change `CPUS` in `run-phase1.py` and the `taskset` line in `run-phase1.sh` for another machine.

```bash
git clone https://github.com/vumichien/kumo-relational-churn-test ~/kumo-relational-tds
cd ~/kumo-relational-tds
bash experiments/spark-setup.sh            # or experiments/wsl-setup.sh on WSL with a CUDA GPU
bash experiments/run-phase1.sh             # all stages; --until <stage> stops early
HF_HOME=~/kumo-relational-tds/cache/hf .venv/bin/python -I experiments/run-phase5.py   # K6 and K7, after phase 1
.venv/bin/python -I experiments/run-phase5.py --l3     # L3, after phase 1
.venv/bin/python -I experiments/analyze-phase1.py --phase5
```

To run one configuration by hand, for example K1 with seed 0:

```bash
cd ~/kumo-relational-tds/experiments
HF_HOME=~/kumo-relational-tds/cache/hf ~/kumo-relational-tds/.venv/bin/python -I nvidia-relbench-example.py \
  --dataset rel-hm --task user-churn --context_size 20000 --num_estimators 8 --context_mode latest \
  --num_neighbors 1 1 --num_lags 20 --context_splits train,val --seed 0 --out ../results/runs/k1-s0.json
```

A full rerun overwrites the files in `results/`, so `git diff` shows how your numbers differ from mine. The synthetic test needs no downloads: `python -I cooccurrence-test.py --seed 0`.

## Data and licences

- **The H&M data is not in this repo and never will be.** RelBench downloads it on first use. It comes from Kaggle's H&M Personalized Fashion Recommendations competition, whose rules allow non-commercial use only (competition, academic research, education) and forbid redistribution. By rerunning experiment 1 you accept [Kaggle's rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules). That's why `results/` holds aggregate scores only, never rows, predictions or derived feature files. RelBench's own conversion, schemas and splits are CDLA-Permissive-2.0.
- **The data scientist's features** come from the RelBench user study's `hm/user-churn/feats.sql` (snap-stanford/relbench-user-study, commit `445bb7a`). That repo has no licence file, so the SQL isn't copied here: `lgbm-hm-churn.py` downloads it at run time.
- **The model weights** download from Hugging Face (`nvidia/Kumo-Relational`, OpenMDW-1.1); the `structured-data-models` library is Apache-2.0.
- **This repo's code** is MIT (see `LICENSE`), except `experiments/nvidia-relbench-example.py`, which keeps NVIDIA's Apache-2.0 licence and copyright header (see `NOTICE`).

Some comments in the scripts refer to the article's planning notes ("definition 5", "phase 1"); those notes aren't published, and the comments say what each rule is.
