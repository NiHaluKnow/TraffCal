#!/usr/bin/env bash
set -euo pipefail

# Local inference launcher for supervisors.
# Supports:
#   1) METR-LA base model inference
#   2) Weather-integrated model inference
#   3) Model selection and multi-sample inference
#   4) Automatic inference image generation
#
# Usage examples:
#   ./run_local_inference.sh --dataset metrla weather --model transformer --sample-index 0 --num-samples 5
#   ./run_local_inference.sh --dataset weather --model cnn --sample-index 10 --num-samples 3
#   ./run_local_inference.sh --dataset metrla --model transformer --custom-seq-file /path/to/20_values.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/nihal/Desktop/ML_50/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN env var, e.g.:"
  echo "  PYTHON_BIN=/path/to/python ./run_local_inference.sh --dataset metrla"
  exit 1
fi

DATASETS=()
MODEL="transformer"
SAMPLE_INDEX=0
NUM_SAMPLES=1
CUSTOM_SEQ_FILE=""
METRLA_WEIGHTS_DEFAULT="$PROJECT_ROOT/Brain/metrla_full_run.weights.h5"
WEATHER_WEIGHTS_DEFAULT="$PROJECT_ROOT/Brain/weather_integrated_full_run_v2.weights.h5"
METRLA_WEIGHTS="$METRLA_WEIGHTS_DEFAULT"
WEATHER_WEIGHTS="$WEATHER_WEIGHTS_DEFAULT"

# Optional local checkpoints for non-transformer models.
CNN_MODEL_PATH="${CNN_MODEL_PATH:-$PROJECT_ROOT/best_cnn_model.h5}"
BILSTM_MODEL_PATH="${BILSTM_MODEL_PATH:-$PROJECT_ROOT/best_bilstm_model.h5}"
HYBRID_MODEL_PATH="${HYBRID_MODEL_PATH:-$PROJECT_ROOT/best_hybrid_cnn_lstm_model.h5}"
GNN_MODEL_PATH="${GNN_MODEL_PATH:-$PROJECT_ROOT/best_gnn_model.h5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "[ERROR] --dataset requires at least one value: metrla and/or weather"
        exit 1
      fi
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --sample-index)
      SAMPLE_INDEX="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --custom-seq-file)
      CUSTOM_SEQ_FILE="$2"
      shift 2
      ;;
    --metrla-weights)
      METRLA_WEIGHTS="$2"
      shift 2
      ;;
    --weather-weights)
      WEATHER_WEIGHTS="$2"
      shift 2
      ;;
    -h|--help)
      cat << 'EOF'
run_local_inference.sh

Required:
  --dataset metrla [weather]

Optional:
  --model NAME               Model to run: transformer|cnn|bilstm|hybrid_cnn_lstm|gnn
  --sample-index N            Test sample start index (default: 0)
  --num-samples N             Number of consecutive test samples (default: 1)
  --custom-seq-file PATH      METR-LA only: text file with exactly 20 values
  --metrla-weights PATH       Override METR-LA weights file
  --weather-weights PATH      Override weather-integrated weights file

Examples:
  ./run_local_inference.sh --dataset metrla weather --model transformer --sample-index 0 --num-samples 5
  ./run_local_inference.sh --dataset weather --model cnn --sample-index 25 --num-samples 3
  ./run_local_inference.sh --dataset metrla --model transformer --custom-seq-file ./my_20_values.txt
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] --dataset is required (metrla and/or weather)"
  exit 1
fi

declare -A seen_dataset=()
VALID_DATASETS=()
for d in "${DATASETS[@]}"; do
  if [[ "$d" != "metrla" && "$d" != "weather" ]]; then
    echo "[ERROR] Invalid dataset: $d (allowed: metrla, weather)"
    exit 1
  fi
  if [[ -z "${seen_dataset[$d]+x}" ]]; then
    VALID_DATASETS+=("$d")
    seen_dataset[$d]=1
  fi
done
DATASETS=("${VALID_DATASETS[@]}")

if [[ "$MODEL" != "transformer" && "$MODEL" != "cnn" && "$MODEL" != "bilstm" && "$MODEL" != "hybrid_cnn_lstm" && "$MODEL" != "gnn" ]]; then
  echo "[ERROR] --model must be one of: transformer, cnn, bilstm, hybrid_cnn_lstm, gnn"
  exit 1
fi

if ! [[ "$SAMPLE_INDEX" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --sample-index must be a non-negative integer"
  exit 1
fi

if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [[ "$NUM_SAMPLES" == "0" ]]; then
  echo "[ERROR] --num-samples must be a positive integer"
  exit 1
fi

if [[ -n "$CUSTOM_SEQ_FILE" && ! -f "$CUSTOM_SEQ_FILE" ]]; then
  echo "[ERROR] --custom-seq-file not found: $CUSTOM_SEQ_FILE"
  exit 1
fi

if [[ -n "$CUSTOM_SEQ_FILE" ]]; then
  if [[ ${#DATASETS[@]} -ne 1 || "${DATASETS[0]}" != "metrla" ]]; then
    echo "[ERROR] --custom-seq-file can only be used with: --dataset metrla"
    exit 1
  fi
fi

DATASETS_CSV="$(IFS=,; echo "${DATASETS[*]}")"

export PROJECT_ROOT DATASETS_CSV MODEL SAMPLE_INDEX NUM_SAMPLES CUSTOM_SEQ_FILE METRLA_WEIGHTS WEATHER_WEIGHTS
export CNN_MODEL_PATH BILSTM_MODEL_PATH HYBRID_MODEL_PATH GNN_MODEL_PATH

"$PYTHON_BIN" - << 'PY'
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

project_root = Path(os.environ["PROJECT_ROOT"])
datasets = [d.strip().lower() for d in os.environ["DATASETS_CSV"].split(",") if d.strip()]
model_name = os.environ["MODEL"].strip().lower()
sample_index = int(os.environ["SAMPLE_INDEX"])
num_samples = int(os.environ["NUM_SAMPLES"])
custom_seq_file = os.environ.get("CUSTOM_SEQ_FILE", "").strip()
metrla_weights = Path(os.environ["METRLA_WEIGHTS"])
weather_weights = Path(os.environ["WEATHER_WEIGHTS"])
cnn_model_path = Path(os.environ["CNN_MODEL_PATH"])
bilstm_model_path = Path(os.environ["BILSTM_MODEL_PATH"])
hybrid_model_path = Path(os.environ["HYBRID_MODEL_PATH"])
gnn_model_path = Path(os.environ["GNN_MODEL_PATH"])

import sys
sys.path.insert(0, str(project_root))
from All_model.transformer import transformer


def load_graph_convolution_class():
    import importlib.util

    gnn_file = project_root / "Others Models/gnn_model.py"
    spec = importlib.util.spec_from_file_location("gnn_model_module", str(gnn_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphConvolution


def build_legacy_gnn_inference_model(
    graph_conv_cls,
    input_len: int,
    num_nodes: int,
    feature_dim: int,
    gcn_units: tuple[int, int],
    output_len: int,
    dropout: float,
) -> tf.keras.Model:
    layers = tf.keras.layers
    keras = tf.keras

    feature_input = keras.Input(shape=(input_len, num_nodes, feature_dim), name="features")
    adjacency_input = keras.Input(shape=(num_nodes, num_nodes), name="adjacency")

    outputs = []
    for t in range(input_len):
        x_t = layers.Lambda(
            lambda x, t=t: x[:, t, :, :],
            output_shape=(num_nodes, feature_dim),
        )(feature_input)

        x_t = graph_conv_cls(gcn_units[0], activation="relu")([x_t, adjacency_input])
        x_t = layers.Dropout(dropout)(x_t)
        x_t = graph_conv_cls(gcn_units[1], activation="relu")([x_t, adjacency_input])
        x_t = layers.Dropout(dropout)(x_t)
        outputs.append(x_t)

    x = layers.Lambda(
        lambda xs: tf.stack(xs, axis=1),
        output_shape=(input_len, num_nodes, gcn_units[1]),
    )(outputs)
    x = layers.Reshape((input_len, num_nodes * gcn_units[1]))(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_nodes * output_len)(x)
    out = layers.Reshape((output_len, num_nodes))(x)

    model = keras.Model(inputs=[feature_input, adjacency_input], outputs=out)
    return model


def load_gnn_model_compat(gnn_path: Path) -> tf.keras.Model:
    graph_conv_cls = load_graph_convolution_class()

    try:
        return tf.keras.models.load_model(
            str(gnn_path),
            custom_objects={"GraphConvolution": graph_conv_cls},
            safe_mode=False,
        )
    except Exception:
        # Fallback for legacy Keras-H5 models that contain Lambda layers whose
        # output shape can't be inferred during deserialization in newer Keras.
        model = build_legacy_gnn_inference_model(
            graph_conv_cls=graph_conv_cls,
            input_len=20,
            num_nodes=1,
            feature_dim=1,
            gcn_units=(16, 8),
            output_len=1,
            dropout=0.3,
        )
        model.load_weights(str(gnn_path))
        return model


def build_transformer(input_steps: int) -> tf.keras.Model:
    model = transformer(
        input_shape=[input_steps, 1],
        head_size=64,
        num_heads=3,
        ff_dim=3,
        num_transformer_blocks=3,
        mlp_units=[64],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    return model


def _adapt_sequence_steps(x_2d: np.ndarray, required_steps: int | None) -> np.ndarray:
    if required_steps is None or x_2d.shape[1] == required_steps:
        return x_2d
    if x_2d.shape[1] > required_steps:
        return x_2d[:, -required_steps:]
    out = np.zeros((x_2d.shape[0], required_steps), dtype=x_2d.dtype)
    out[:, -x_2d.shape[1]:] = x_2d
    return out


def predict_with_loaded_sequence_model(model: tf.keras.Model, x_2d: np.ndarray) -> np.ndarray:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not isinstance(input_shape, tuple):
        raise ValueError(f"Unsupported model input shape: {input_shape}")

    if len(input_shape) == 3:
        # Typical shape: (batch, steps, channels)
        required_steps = input_shape[1]
        required_channels = input_shape[2] if input_shape[2] is not None else 1
        x_adj = _adapt_sequence_steps(x_2d, required_steps)
        if required_channels == 1:
            x_in = x_adj.reshape(len(x_adj), x_adj.shape[1], 1)
        else:
            x_in = np.repeat(x_adj.reshape(len(x_adj), x_adj.shape[1], 1), repeats=required_channels, axis=2)
        return model.predict(x_in, verbose=0).reshape(-1)

    if len(input_shape) == 2:
        # Shape: (batch, features)
        required_steps = input_shape[1]
        x_in = _adapt_sequence_steps(x_2d, required_steps)
        return model.predict(x_in, verbose=0).reshape(-1)

    raise ValueError(f"Unsupported model rank for sequence inference: {input_shape}")


def predict_with_loaded_gnn_model(model: tf.keras.Model, x_2d: np.ndarray) -> np.ndarray:
    input_shapes = model.input_shape
    if not isinstance(input_shapes, list) or len(input_shapes) < 2:
        raise ValueError(f"Unexpected GNN input shape config: {input_shapes}")

    feat_shape = input_shapes[0]
    adj_shape = input_shapes[1]
    if not isinstance(feat_shape, tuple) or len(feat_shape) != 4:
        raise ValueError(f"Unsupported GNN feature input shape: {feat_shape}")

    required_steps = feat_shape[1]
    required_nodes = feat_shape[2] if feat_shape[2] is not None else 1
    required_channels = feat_shape[3] if feat_shape[3] is not None else 1

    x_adj_steps = _adapt_sequence_steps(x_2d, required_steps)
    x_in = np.repeat(x_adj_steps[:, :, None], repeats=required_nodes, axis=2)
    x_in = x_in[:, :, :, None]
    if required_channels != 1:
        x_in = np.repeat(x_in, repeats=required_channels, axis=3)

    if isinstance(adj_shape, tuple) and len(adj_shape) == 3:
        adj_nodes = adj_shape[1] if adj_shape[1] is not None else required_nodes
        adj = np.ones((len(x_adj_steps), adj_nodes, adj_nodes), dtype=np.float32)
    else:
        adj = np.ones((len(x_adj_steps), required_nodes, required_nodes), dtype=np.float32)

    return model.predict([x_in, adj], verbose=0).reshape(-1)


def run_metrla() -> dict:
    h5_path = project_root / "data/metr-la/METR-LA.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"METR-LA file missing: {h5_path}")
    if not metrla_weights.exists():
        raise FileNotFoundError(f"METR-LA weights missing: {metrla_weights}")

    with h5py.File(h5_path, "r") as f:
        values = f["df"]["block0_values"][:]

    series = values.mean(axis=1).astype(np.float32)
    n_days = 20

    train_end = int(len(series) * 0.7)
    val_end = int(len(series) * 0.85)

    train_mean = float(series[:train_end].mean())
    train_std = float(series[:train_end].std())
    if train_std == 0.0:
        train_std = 1.0

    if model_name == "transformer":
        model = build_transformer(n_days)
        model.load_weights(str(metrla_weights))
    elif model_name == "cnn":
        if not cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model file missing: {cnn_model_path}")
        model = tf.keras.models.load_model(str(cnn_model_path))
    elif model_name == "bilstm":
        if not bilstm_model_path.exists():
            raise FileNotFoundError(f"BiLSTM model file missing: {bilstm_model_path}")
        model = tf.keras.models.load_model(str(bilstm_model_path))
    elif model_name == "hybrid_cnn_lstm":
        if not hybrid_model_path.exists():
            raise FileNotFoundError(f"Hybrid CNN-LSTM model file missing: {hybrid_model_path}")
        model = tf.keras.models.load_model(str(hybrid_model_path))
    elif model_name == "gnn":
        if not gnn_model_path.exists():
            raise FileNotFoundError(f"GNN model file missing: {gnn_model_path}")
        model = load_gnn_model_compat(gnn_model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if custom_seq_file:
        vals = []
        with open(custom_seq_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    vals.append(float(line))
        if len(vals) != n_days:
            raise ValueError(f"Custom sequence must contain exactly {n_days} values, got {len(vals)}")

        seq = np.array(vals, dtype=np.float32)
        seq_norm = (seq - train_mean) / train_std
        x = seq_norm.reshape(1, n_days, 1)
        if model_name == "gnn":
            pred_norm = float(predict_with_loaded_gnn_model(model, seq_norm.reshape(1, n_days))[0])
        elif model_name in {"cnn", "bilstm", "hybrid_cnn_lstm"}:
            pred_norm = float(predict_with_loaded_sequence_model(model, seq_norm.reshape(1, n_days))[0])
        else:
            pred_norm = float(model.predict(x, verbose=0).reshape(-1)[0])
        pred = pred_norm * train_std + train_mean

        print("=== METR-LA CUSTOM INFERENCE ===")
        print(f"Model: {model_name}")
        print(f"Input file: {custom_seq_file}")
        print(f"Prediction (next step, original scale): {pred:.4f}")
        print("Note: actual target is unknown for custom input.")
        return {
          "dataset": "metrla",
          "mode": "custom",
          "indices": np.array([sample_index], dtype=np.int64),
          "pred": np.array([pred], dtype=np.float32),
          "actual": np.array([], dtype=np.float32),
          "abs_err": np.array([], dtype=np.float32),
        }

    # Use exact split logic from training script and predict on test sample
    normalized = (series - train_mean) / train_std
    features = []
    labels = []
    target_indices = []
    for i in range(len(normalized) - n_days):
        features.append(normalized[i : i + n_days])
        labels.append(normalized[i + n_days])
        target_indices.append(i + n_days)

    x_all = np.array(features, dtype=np.float32).reshape(-1, n_days, 1)
    y_all = np.array(labels, dtype=np.float32)
    target_indices = np.array(target_indices, dtype=np.int64)

    test_mask = target_indices >= val_end
    x_test = x_all[test_mask]
    y_test = y_all[test_mask]

    if sample_index >= len(x_test):
        raise IndexError(f"sample-index out of range for METR-LA test set: {sample_index} >= {len(x_test)}")

    end_index = min(sample_index + num_samples, len(x_test))
    x_batch = x_test[sample_index:end_index]
    y_batch = y_test[sample_index:end_index]

    if model_name == "gnn":
        pred_norm = predict_with_loaded_gnn_model(model, x_batch.reshape(len(x_batch), n_days))
    elif model_name in {"cnn", "bilstm", "hybrid_cnn_lstm"}:
        pred_norm = predict_with_loaded_sequence_model(model, x_batch.reshape(len(x_batch), n_days))
    else:
        pred_norm = model.predict(x_batch, verbose=0).reshape(-1)

    y_true_norm = y_batch.reshape(-1)

    pred = pred_norm * train_std + train_mean
    y_true = y_true_norm * train_std + train_mean
    abs_err = np.abs(pred - y_true)

    print("=== METR-LA TEST INFERENCE ===")
    print(f"Model: {model_name}")
    print(f"Split (train/val/test): {np.sum(target_indices < train_end)}/{np.sum((target_indices >= train_end) & (target_indices < val_end))}/{np.sum(test_mask)}")
    print(f"Sample index range (test): {sample_index}..{end_index - 1}")
    print(f"Num samples: {len(x_batch)}")
    for i in range(len(x_batch)):
        idx_show = sample_index + i
        print(f"[{idx_show}] pred={pred[i]:.4f} actual={y_true[i]:.4f} abs_err={abs_err[i]:.4f}")
    print(f"Mean absolute error (selected): {float(abs_err.mean()):.4f}")
    return {
        "dataset": "metrla",
        "mode": "test",
        "indices": np.arange(sample_index, end_index, dtype=np.int64),
        "pred": pred,
        "actual": y_true,
        "abs_err": abs_err,
    }


def run_weather() -> dict:
    x_path = project_root / "newdatset/x.pt"
    y_path = project_root / "newdatset/y.pt"
    meta_path = project_root / "newdatset/metadata.json"

    for p in [x_path, y_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")
    if model_name == "transformer" and not weather_weights.exists():
        raise FileNotFoundError(f"Weather weights missing: {weather_weights}")

    x = torch.load(x_path, map_location="cpu").float().numpy()  # [S, in, N, F]
    y = torch.load(y_path, map_location="cpu").float().numpy()  # [S, out, N, 1]

    # Match training alignment in train_ptdataset.py used in this project
    x_series = x[:, :, :, 0].mean(axis=2).astype(np.float32)      # [S, in]
    y_target = y[:, 0, :, 0].mean(axis=1).astype(np.float32)      # [S]

    n_samples = len(x_series)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    x_test = x_series[val_end:]
    y_test = y_target[val_end:]

    if sample_index >= len(x_test):
        raise IndexError(f"sample-index out of range for weather test set: {sample_index} >= {len(x_test)}")

    end_index = min(sample_index + num_samples, len(x_test))
    x_batch = x_test[sample_index:end_index]
    y_batch = y_test[sample_index:end_index]

    if model_name == "transformer":
        model = build_transformer(input_steps=x_series.shape[1])
        model.load_weights(str(weather_weights))
        pred_norm = model.predict(x_batch.reshape(len(x_batch), x_series.shape[1], 1), verbose=0).reshape(-1)
    elif model_name == "cnn":
        if not cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model file missing: {cnn_model_path}")
        model = tf.keras.models.load_model(str(cnn_model_path))
        pred_norm = predict_with_loaded_sequence_model(model, x_batch.reshape(len(x_batch), x_series.shape[1]))
    elif model_name == "bilstm":
        if not bilstm_model_path.exists():
            raise FileNotFoundError(f"BiLSTM model file missing: {bilstm_model_path}")
        model = tf.keras.models.load_model(str(bilstm_model_path))
        pred_norm = predict_with_loaded_sequence_model(model, x_batch.reshape(len(x_batch), x_series.shape[1]))
    elif model_name == "hybrid_cnn_lstm":
        if not hybrid_model_path.exists():
            raise FileNotFoundError(f"Hybrid CNN-LSTM model file missing: {hybrid_model_path}")
        model = tf.keras.models.load_model(str(hybrid_model_path))
        pred_norm = predict_with_loaded_sequence_model(model, x_batch.reshape(len(x_batch), x_series.shape[1]))
    elif model_name == "gnn":
        if not gnn_model_path.exists():
            raise FileNotFoundError(f"GNN model file missing: {gnn_model_path}")
        model = load_gnn_model_compat(gnn_model_path)
        pred_norm = predict_with_loaded_gnn_model(model, x_batch.reshape(len(x_batch), x_series.shape[1]))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    y_true_norm = y_batch.reshape(-1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    speed_stats = meta["scaler"]["stats"]["speed"]
    mean = float(speed_stats["mean"])
    std = float(speed_stats["std"])

    pred = pred_norm.reshape(-1) * std + mean
    y_true = y_true_norm * std + mean
    abs_err = np.abs(pred - y_true)

    print("=== WEATHER-INTEGRATED TEST INFERENCE ===")
    print(f"Model: {model_name}")
    print(f"Split (train/val/test): {train_end}/{val_end - train_end}/{n_samples - val_end}")
    print(f"Sample index range (test): {sample_index}..{end_index - 1}")
    print(f"Num samples: {len(x_batch)}")
    for i in range(len(x_batch)):
        idx_show = sample_index + i
        print(f"[{idx_show}] pred={pred[i]:.4f} actual={y_true[i]:.4f} abs_err={abs_err[i]:.4f}")
    print(f"Mean absolute error (selected): {float(abs_err.mean()):.4f}")
    return {
        "dataset": "weather",
        "mode": "test",
        "indices": np.arange(sample_index, end_index, dtype=np.int64),
        "pred": pred,
        "actual": y_true,
        "abs_err": abs_err,
    }


def save_inference_plot(results: list[dict]) -> Path:
    out_dir = project_root / "results /presentation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_tag = "-".join([r["dataset"] for r in results])
    fname = f"inference_{datasets_tag}_{model_name}_idx{sample_index}_n{num_samples}.png"
    out_file = out_dir / fname

    if len(results) == 1:
        r = results[0]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
        if r["mode"] == "custom":
            axes[0].bar(["Prediction"], [float(r["pred"][0])], color="#f28e2b")
            axes[0].set_title(f"{r['dataset'].upper()} Custom Inference ({model_name})")
            axes[0].set_ylabel("Value")
            axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
            axes[1].axis("off")
            axes[1].text(0.02, 0.5, "No ground truth for custom input", fontsize=11)
        else:
            axes[0].plot(r["indices"], r["actual"], marker="o", label="Actual", linewidth=2)
            axes[0].plot(r["indices"], r["pred"], marker="s", label="Predicted", linewidth=2)
            axes[0].set_title(f"{r['dataset'].upper()} Inference ({model_name})")
            axes[0].set_ylabel("Value")
            axes[0].grid(True, linestyle="--", alpha=0.35)
            axes[0].legend()

            axes[1].bar(r["indices"], r["abs_err"], color="#e15759")
            axes[1].set_title("Absolute Error per Sample")
            axes[1].set_xlabel("Test Sample Index")
            axes[1].set_ylabel("Abs Error")
            axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), sharex=False)
        if len(results) == 1:
            axes = [axes]
        for ax, r in zip(axes, results):
            ax.plot(r["indices"], r["actual"], marker="o", label="Actual", linewidth=2)
            ax.plot(r["indices"], r["pred"], marker="s", label="Predicted", linewidth=2)
            mae = float(np.mean(r["abs_err"]))
            ax.set_title(f"{r['dataset'].upper()} Inference ({model_name}) | mean abs err={mae:.4f}")
            ax.set_xlabel("Test Sample Index")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
        plt.tight_layout()

    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    return out_file

results = []
for ds in datasets:
    if ds == "metrla":
        results.append(run_metrla())
    elif ds == "weather":
        results.append(run_weather())
    else:
        raise ValueError(f"Unsupported dataset: {ds}")

plot_file = save_inference_plot(results)
print(f"Inference image saved: {plot_file}")
PY
