import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from model import transformer
from train_metrla import load_metrla_series, make_dataset


@dataclass
class DatasetSplits:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    target_scaler: Dict[str, float]
    num_features: int
    input_len: int
    total_samples: int
    output_len: int = 1
    timestamps: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair comparison: METR-LA baseline vs METR-LA + weather/time features")

    parser.add_argument("--metrla-h5", type=str, default="data/metr-la/METR-LA.h5")
    parser.add_argument("--baseline-mode", type=str, default="mean", choices=["mean", "sensor"])
    parser.add_argument("--baseline-sensor-index", type=int, default=0)

    parser.add_argument("--aug-x-path", type=str, default="../newdatset/x.pt")
    parser.add_argument("--aug-y-path", type=str, default="../newdatset/y.pt")
    parser.add_argument("--aug-metadata", type=str, default="../newdatset/metadata.json")
    parser.add_argument("--aug-aggregate", type=str, default="mean", choices=["mean", "sensor0"])

    parser.add_argument("--n-days", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--results-dir", type=str, default="visualization/experiment_comparison")
    parser.add_argument("--save-prefix", type=str, default="comparison")
    parser.add_argument(
        "--no-align-baseline-with-augmented",
        action="store_true",
        help="Disable timestamp/sample alignment of baseline windows to augmented metadata.",
    )

    return parser.parse_args()


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")


def _split_indices(num_samples: int, train_ratio: float, val_ratio: float) -> Dict[str, int]:
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(num_samples - 1, val_end)
    return {"train_end": train_end, "val_end": val_end}


def _extract_speed_scaler_from_metadata(metadata: Dict) -> Dict[str, float]:
    scaler = metadata.get("scaler", {})
    method = scaler.get("method", "identity")
    speed_stats = scaler.get("stats", {}).get("speed")

    if speed_stats is None:
        return {"method": "identity"}

    if method == "zscore":
        std = float(speed_stats.get("std", 1.0))
        if std == 0.0:
            std = 1.0
        return {
            "method": "zscore",
            "mean": float(speed_stats.get("mean", 0.0)),
            "std": std,
        }

    if method == "minmax":
        min_v = float(speed_stats.get("min", 0.0))
        max_v = float(speed_stats.get("max", 1.0))
        if max_v == min_v:
            max_v = min_v + 1.0
        return {
            "method": "minmax",
            "min": min_v,
            "max": max_v,
        }

    return {"method": "identity"}


def _inverse_target_scale(values: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
    method = scaler.get("method", "identity")
    if method == "zscore":
        return values * scaler["std"] + scaler["mean"]
    if method == "minmax":
        return values * (scaler["max"] - scaler["min"]) + scaler["min"]
    return values


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _load_metrla_series_with_optional_alignment(
    h5_path: str,
    mode: str,
    sensor_index: int,
    aligned_timestamps: Optional[List[str]],
) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5_file:
        values = h5_file["df"]["block0_values"][:].astype(np.float32)
        ts_raw = h5_file["df"]["axis1"][:]

    if mode == "mean":
        series_values = values.mean(axis=1).astype(np.float32)
    elif mode == "sensor":
        if sensor_index < 0 or sensor_index >= values.shape[1]:
            raise ValueError(f"sensor_index must be in [0, {values.shape[1] - 1}]")
        series_values = values[:, sensor_index].astype(np.float32)
    else:
        raise ValueError("mode must be either 'mean' or 'sensor'")

    if aligned_timestamps is None:
        return series_values

    source_ts = pd.to_datetime(pd.Series(ts_raw), utc=True, errors="coerce")
    if source_ts.isna().any():
        raise ValueError("Source METR-LA timestamps contain invalid entries")

    source_series = pd.Series(series_values, index=pd.DatetimeIndex(source_ts))
    source_series = source_series[~source_series.index.duplicated(keep="last")].sort_index()

    ts = pd.to_datetime(pd.Series(aligned_timestamps), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Aligned timestamps include invalid entries")

    aligned = source_series.reindex(pd.DatetimeIndex(ts))
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(f"Baseline alignment failed: {missing} timestamps missing in METR-LA series")

    return aligned.values.astype(np.float32)


def load_baseline_dataset(
    h5_path: str,
    n_days: int,
    mode: str,
    sensor_index: int,
    train_ratio: float,
    val_ratio: float,
    aligned_timestamps: Optional[List[str]] = None,
    reserve_future_steps: int = 0,
    expected_samples: Optional[int] = None,
) -> DatasetSplits:
    series = _load_metrla_series_with_optional_alignment(
        h5_path=h5_path,
        mode=mode,
        sensor_index=sensor_index,
        aligned_timestamps=aligned_timestamps,
    )

    train_end = int(len(series) * train_ratio)

    train_mean = float(series[:train_end].mean())
    train_std = float(series[:train_end].std())
    if train_std == 0.0:
        train_std = 1.0

    normalized = (series - train_mean) / train_std
    x_all, y_all, _ = make_dataset(normalized, window_size=n_days)

    if reserve_future_steps > 0:
        if reserve_future_steps >= len(x_all):
            raise ValueError("reserve_future_steps is too large for baseline sample count")
        x_all = x_all[:-reserve_future_steps]
        y_all = y_all[:-reserve_future_steps]

    if expected_samples is not None and len(x_all) != expected_samples:
        raise ValueError(
            f"Sample alignment mismatch: baseline windows={len(x_all)}, augmented windows={expected_samples}. "
            "Check n_days/output_len alignment."
        )

    split = _split_indices(len(x_all), train_ratio, val_ratio)
    train_split, val_split = split["train_end"], split["val_end"]

    x_train, y_train = x_all[:train_split], y_all[:train_split]
    x_val, y_val = x_all[train_split:val_split], y_all[train_split:val_split]
    x_test, y_test = x_all[val_split:], y_all[val_split:]

    return DatasetSplits(
        name="baseline_metrla",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        target_scaler={"method": "zscore", "mean": train_mean, "std": train_std},
        num_features=int(x_train.shape[-1]),
        input_len=int(x_train.shape[1]),
        total_samples=int(len(x_all)),
        output_len=1,
        timestamps=aligned_timestamps,
    )


def load_augmented_dataset(
    x_path: str,
    y_path: str,
    metadata_path: str,
    aggregate: str,
    n_days: int,
    train_ratio: float,
    val_ratio: float,
) -> DatasetSplits:
    x = torch.load(x_path, map_location="cpu").float().numpy()
    y = torch.load(y_path, map_location="cpu").float().numpy()

    if x.ndim != 4:
        raise ValueError(f"Expected x shape [samples, in_steps, nodes, features], got {x.shape}")
    if y.ndim != 4:
        raise ValueError(f"Expected y shape [samples, out_steps, nodes, 1], got {y.shape}")

    if aggregate == "mean":
        x_input = x.mean(axis=2)
        y_target = y[:, 0, :, 0].mean(axis=1)
    elif aggregate == "sensor0":
        x_input = x[:, :, 0, :]
        y_target = y[:, 0, 0, 0]
    else:
        raise ValueError("aggregate must be one of: mean, sensor0")

    if x_input.shape[1] != n_days:
        raise ValueError(
            f"Window length mismatch: n_days={n_days}, augmented input_len={x_input.shape[1]}. "
            "Use matching n_days for fair comparison."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [])
    if feature_names and len(feature_names) != x_input.shape[-1]:
        raise ValueError(
            f"Feature alignment mismatch: metadata has {len(feature_names)} features, "
            f"tensor has {x_input.shape[-1]}"
        )

    split = _split_indices(len(x_input), train_ratio, val_ratio)
    train_end, val_end = split["train_end"], split["val_end"]

    x_train, y_train = x_input[:train_end], y_target[:train_end]
    x_val, y_val = x_input[train_end:val_end], y_target[train_end:val_end]
    x_test, y_test = x_input[val_end:], y_target[val_end:]

    scaler = _extract_speed_scaler_from_metadata(metadata)
    output_len = int(metadata.get("output_len", y.shape[1]))
    timestamps = metadata.get("timestamps")

    return DatasetSplits(
        name="weather_time_augmented_metrla",
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        target_scaler=scaler,
        num_features=int(x_train.shape[-1]),
        input_len=int(x_train.shape[1]),
        total_samples=int(len(x_input)),
        output_len=output_len,
        timestamps=timestamps,
    )


def train_and_evaluate(
    dataset: DatasetSplits,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    model_dir: Path,
) -> Dict[str, float]:
    _set_seed(seed)
    tf.keras.backend.clear_session()

    model = transformer(
        input_shape=[dataset.input_len, dataset.num_features],
        head_size=64,
        num_heads=3,
        ff_dim=3,
        num_transformer_blocks=3,
        mlp_units=[64],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    checkpoint_path = model_dir / f"{dataset.name}.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred_norm = model.predict(dataset.x_test, verbose=0).reshape(-1)
    y_true_norm = dataset.y_test.reshape(-1)

    y_pred = _inverse_target_scale(y_pred_norm, dataset.target_scaler)
    y_true = _inverse_target_scale(y_true_norm, dataset.target_scaler)

    metrics = _compute_metrics(y_true, y_pred)

    metrics.update(
        {
            "experiment": dataset.name,
            "input_len": dataset.input_len,
            "num_features": dataset.num_features,
            "train_samples": int(len(dataset.x_train)),
            "val_samples": int(len(dataset.x_val)),
            "test_samples": int(len(dataset.x_test)),
            "best_val_loss": float(np.min(history.history["val_loss"])),
            "weights_path": str(checkpoint_path),
        }
    )
    return metrics


def _pct_change(new_value: float, base_value: float) -> float:
    if base_value == 0:
        return float("nan")
    return ((new_value - base_value) / base_value) * 100.0


def print_comparison_table(baseline: Dict[str, float], augmented: Dict[str, float]) -> None:
    mae_change = _pct_change(augmented["mae"], baseline["mae"])
    rmse_change = _pct_change(augmented["rmse"], baseline["rmse"])
    mape_change = _pct_change(augmented["mape"], baseline["mape"])

    print("\n======= FINAL COMPARISON =======")
    print(f"{'Experiment':36s} {'MAE':>12s} {'RMSE':>12s} {'MAPE(%)':>12s}")
    print("-" * 78)
    print(f"{baseline['experiment']:36s} {baseline['mae']:12.4f} {baseline['rmse']:12.4f} {baseline['mape']:12.4f}")
    print(f"{augmented['experiment']:36s} {augmented['mae']:12.4f} {augmented['rmse']:12.4f} {augmented['mape']:12.4f}")
    print("-" * 78)
    print(f"{'% change (augmented vs baseline)':36s} {mae_change:11.2f}% {rmse_change:11.2f}% {mape_change:11.2f}%")


def save_metric_bar_charts(
    baseline: Dict[str, float],
    augmented: Dict[str, float],
    output_dir: Path,
    save_prefix: str,
) -> Dict[str, str]:
    metric_specs = [
        ("mae", "MAE", "MAE"),
        ("rmse", "RMSE", "RMSE"),
        ("mape", "MAPE", "MAPE (%)"),
    ]

    labels = ["Baseline METR-LA", "METR-LA + Weather/Time"]
    colors = ["#4C78A8", "#F58518"]

    saved_paths: Dict[str, str] = {}

    for metric_key, metric_title, y_label in metric_specs:
        values = [float(baseline[metric_key]), float(augmented[metric_key])]

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{metric_title} Comparison")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        y_max = max(values)
        y_offset = max(y_max * 0.02, 0.01)
        ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1.0)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        fig.tight_layout()

        out_path = output_dir / f"{save_prefix}_{metric_key}_bar.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths[metric_key] = str(out_path)

    return saved_paths


def save_reports(
    baseline: Dict[str, float],
    augmented: Dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.save_prefix}_comparison.csv"
    json_path = output_dir / f"{args.save_prefix}_comparison.json"

    rows: List[Dict[str, float]] = [baseline, augmented]
    fields = [
        "experiment",
        "mae",
        "rmse",
        "mape",
        "input_len",
        "num_features",
        "train_samples",
        "val_samples",
        "test_samples",
        "best_val_loss",
        "weights_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    chart_paths = save_metric_bar_charts(
        baseline=baseline,
        augmented=augmented,
        output_dir=output_dir,
        save_prefix=args.save_prefix,
    )

    comparison_summary = {
        "baseline": baseline,
        "augmented": augmented,
        "percent_change_augmented_vs_baseline": {
            "mae": _pct_change(augmented["mae"], baseline["mae"]),
            "rmse": _pct_change(augmented["rmse"], baseline["rmse"]),
            "mape": _pct_change(augmented["mape"], baseline["mape"]),
        },
        "shared_training_settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "loss": "mean_squared_error",
            "optimizer": "Adam",
            "patience": args.patience,
            "seed": args.seed,
        },
        "visualization_files": chart_paths,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nSaved CSV report: {csv_path}")
    print(f"Saved JSON report: {json_path}")
    print("Saved bar charts:")
    print(f"  - MAE: {chart_paths['mae']}")
    print(f"  - RMSE: {chart_paths['rmse']}")
    print(f"  - MAPE: {chart_paths['mape']}")


def main() -> None:
    args = parse_args()
    _validate_split_ratios(args.train_ratio, args.val_ratio)

    output_dir = Path(args.results_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    augmented = load_augmented_dataset(
        x_path=args.aug_x_path,
        y_path=args.aug_y_path,
        metadata_path=args.aug_metadata,
        aggregate=args.aug_aggregate,
        n_days=args.n_days,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    align_baseline = not args.no_align_baseline_with_augmented
    aligned_timestamps = augmented.timestamps if align_baseline else None
    reserve_future_steps = max(0, augmented.output_len - 1) if align_baseline else 0
    expected_samples = augmented.total_samples if align_baseline else None

    baseline = load_baseline_dataset(
        h5_path=args.metrla_h5,
        n_days=args.n_days,
        mode=args.baseline_mode,
        sensor_index=args.baseline_sensor_index,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        aligned_timestamps=aligned_timestamps,
        reserve_future_steps=reserve_future_steps,
        expected_samples=expected_samples,
    )

    print("======= RUN SETTINGS =======")
    print(f"Loss: mean_squared_error")
    print(f"Optimizer: Adam(lr={args.learning_rate})")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Patience: {args.patience}")
    print(f"Baseline aligned to augmented timeline: {align_baseline}")
    print(f"Baseline input shape: ({baseline.input_len}, {baseline.num_features})")
    print(f"Augmented input shape: ({augmented.input_len}, {augmented.num_features})")
    print(f"Baseline samples: {baseline.total_samples}")
    print(f"Augmented samples: {augmented.total_samples}")

    baseline_result = train_and_evaluate(
        dataset=baseline,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        model_dir=model_dir,
    )

    augmented_result = train_and_evaluate(
        dataset=augmented,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        model_dir=model_dir,
    )

    print_comparison_table(baseline_result, augmented_result)
    save_reports(baseline_result, augmented_result, args, output_dir)


if __name__ == "__main__":
    main()