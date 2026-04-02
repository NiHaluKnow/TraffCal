import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Importing Libraries
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_sequence(data_path: Path) -> np.ndarray:
    data = []
    with open(data_path, "r") as f:
        for d in f.readlines():
            data.append(int(d))
    return np.array(data).reshape([1, -1, 1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default ="predict.traff", help="data directory"
    )
    parser.add_argument(
        "--save_name", type=str, help="data directory"
    )

    args = parser.parse_args()
    DATA_PATH = args.filename
    SAVE_NAME = args.save_name

    data = _read_sequence(PROJECT_ROOT / "data" / DATA_PATH)
    print(data)

    model = tf.keras.models.load_model(str(PROJECT_ROOT / "All_models" / f"{SAVE_NAME}.h5"))
    pred = model.predict(np.array(data))[0][0]

    print("=======MODEL OUTPUT=======")
    print(f"Prediction of {DATA_PATH} data is {int(pred)}")
    print()


def get_predict(DATA_PATH, SAVE_NAME):
    data = _read_sequence(PROJECT_ROOT / "data" / DATA_PATH)
    print(data)

    model = tf.keras.models.load_model(str(PROJECT_ROOT / "All_models" / f"{SAVE_NAME}.h5"))
    pred = model.predict(np.array(data))[0][0]
    return pred
