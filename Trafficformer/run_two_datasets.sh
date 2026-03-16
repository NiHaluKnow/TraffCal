#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/nihal/Desktop/ML_50/.venv/bin/python}"

echo "Using Python: $PYTHON_BIN"
echo "=== 1/2: METR-LA run (same settings) ==="
"$PYTHON_BIN" train_metrla.py \
  --h5_path data/metr-la/METR-LA.h5 \
  --n_days 20 \
  --mode mean \
  --epochs 20 \
  --batch_size 128 \
  --save_name metrla_mean

echo "=== 2/2: newdataset x.pt/y.pt run (same settings) ==="
"$PYTHON_BIN" train_ptdataset.py \
  --x_path ../newdataset/x.pt \
  --y_path ../newdataset/y.pt \
  --aggregate nodewise \
  --feature_index 0 \
  --train_size 23970 \
  --val_size 5141 \
  --test_size 5141 \
  --epochs 20 \
  --batch_size 128 \
  --save_name ptdataset_nodewise_23970_5141_5141

echo "=== Done: both runs completed ==="
