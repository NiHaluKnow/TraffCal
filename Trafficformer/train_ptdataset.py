import argparse
import numpy as np
import tensorflow as tf
import torch

from model import transformer


def load_pt_dataset(x_path, y_path, feature_index=0, aggregate="mean"):
    x = torch.load(x_path, map_location="cpu").float().numpy()
    y = torch.load(y_path, map_location="cpu").float().numpy()

    if x.ndim != 4:
        raise ValueError(f"x must be 4D [samples, in_steps, nodes, features], got shape {x.shape}")
    if y.ndim != 4:
        raise ValueError(f"y must be 4D [samples, out_steps, nodes, 1], got shape {y.shape}")

    if feature_index < 0 or feature_index >= x.shape[-1]:
        raise ValueError(f"feature_index must be in [0, {x.shape[-1] - 1}]")

    x_feature = x[:, :, :, feature_index]
    y_step0 = y[:, 0, :, 0]

    if aggregate == "mean":
        x_series = x_feature.mean(axis=2)
        y_target = y_step0.mean(axis=1)
    elif aggregate == "sensor0":
        x_series = x_feature[:, :, 0]
        y_target = y_step0[:, 0]
    elif aggregate == "nodewise":
        x_series = x_feature.transpose(0, 2, 1).reshape(-1, x_feature.shape[1])
        y_target = y_step0.reshape(-1)
    else:
        raise ValueError("aggregate must be one of: 'mean', 'sensor0', 'nodewise'")

    x_series = x_series.astype(np.float32)
    y_target = y_target.astype(np.float32)

    x_input = x_series.reshape(x_series.shape[0], x_series.shape[1], 1)
    return x_input, y_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", type=str, default="../newdatset/x.pt")
    parser.add_argument("--y_path", type=str, default="../newdatset/y.pt")
    parser.add_argument("--feature_index", type=int, default=0)
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "sensor0", "nodewise"])
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_name", type=str, default="ptdataset_trafficformer")
    args = parser.parse_args()

    x_all, y_all = load_pt_dataset(
        args.x_path,
        args.y_path,
        feature_index=args.feature_index,
        aggregate=args.aggregate,
    )

    num_samples = len(x_all)
    if args.train_size > 0 and args.val_size > 0 and args.test_size > 0:
        requested_total = args.train_size + args.val_size + args.test_size
        if requested_total > num_samples:
            raise ValueError(
                f"Requested split total ({requested_total}) exceeds available samples ({num_samples})."
            )
        train_end = args.train_size
        val_end = args.train_size + args.val_size
        test_end = requested_total
        x_all = x_all[:test_end]
        y_all = y_all[:test_end]
    else:
        train_end = int(num_samples * 0.7)
        val_end = int(num_samples * 0.85)
        test_end = num_samples

    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_val, y_val = x_all[train_end:val_end], y_all[train_end:val_end]
    x_test, y_test = x_all[val_end:test_end], y_all[val_end:test_end]

    input_steps = x_all.shape[1]

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

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
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

    pred_test = model.predict(x_test, verbose=0).reshape(-1)
    mae = float(np.mean(np.abs(pred_test - y_test)))

    print("=======PT DATASET EVALUATION=======")
    print(f"Aggregate mode: {args.aggregate}")
    print(f"Feature index: {args.feature_index}")
    print(f"Train/Val/Test samples: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"Test MAE (original scale): {mae:.4f}")


if __name__ == "__main__":
    main()