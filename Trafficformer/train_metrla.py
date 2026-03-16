import argparse
import numpy as np
import tensorflow as tf
import h5py

from model import transformer


def make_dataset(data, window_size):
    features = []
    labels = []
    target_indices = []

    for i in range(len(data) - window_size):
        features.append(data[i : i + window_size])
        labels.append(data[i + window_size])
        target_indices.append(i + window_size)

    x = np.array(features, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.array(labels, dtype=np.float32)
    idx = np.array(target_indices, dtype=np.int64)
    return x, y, idx


def load_metrla_series(h5_path, mode="mean", sensor_index=0):
    with h5py.File(h5_path, "r") as h5_file:
        values = h5_file["df"]["block0_values"][:]

    if mode == "mean":
        return values.mean(axis=1).astype(np.float32)

    if mode == "sensor":
        if sensor_index < 0 or sensor_index >= values.shape[1]:
            raise ValueError(f"sensor_index must be in [0, {values.shape[1] - 1}]")
        return values[:, sensor_index].astype(np.float32)

    raise ValueError("mode must be either 'mean' or 'sensor'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="data/metr-la/METR-LA.h5")
    parser.add_argument("--n_days", type=int, default=20)
    parser.add_argument("--mode", type=str, default="mean", choices=["mean", "sensor"])
    parser.add_argument("--sensor_index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_name", type=str, default="metrla_trafficformer")
    args = parser.parse_args()

    series = load_metrla_series(args.h5_path, mode=args.mode, sensor_index=args.sensor_index)

    train_end = int(len(series) * 0.7)
    val_end = int(len(series) * 0.85)

    train_mean = float(series[:train_end].mean())
    train_std = float(series[:train_end].std())
    if train_std == 0.0:
        train_std = 1.0

    normalized = (series - train_mean) / train_std
    x_all, y_all, target_indices = make_dataset(normalized, window_size=args.n_days)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    x_test, y_test = x_all[test_mask], y_all[test_mask]

    model = transformer(
        input_shape=[args.n_days, 1],
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model/{args.save_name}.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    pred_test_norm = model.predict(x_test, verbose=0).reshape(-1)
    y_test_norm = y_test.reshape(-1)

    pred_test = pred_test_norm * train_std + train_mean
    y_test_real = y_test_norm * train_std + train_mean

    mae = float(np.mean(np.abs(pred_test - y_test_real)))

    print("=======METR-LA EVALUATION=======")
    print(f"Mode: {args.mode}")
    if args.mode == "sensor":
        print(f"Sensor index: {args.sensor_index}")
    print(f"Train/Val/Test samples: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"Test MAE (original scale): {mae:.4f}")


if __name__ == "__main__":
    main()