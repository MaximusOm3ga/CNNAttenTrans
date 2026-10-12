import pandas as pd
import numpy as np
import torch
from pathlib import Path

WINDOW = 160
STRIDE = 16
NUM_FEATURES = 6
TARGET_COL = 0

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electricityloaddiagrams20112014" / "LD2011_2014.txt"
OUTPUT_PATH = BASE_DIR / "electricity_windows.pt"


def make_windows(data, window, stride, target_col):
    X, Y = [], []
    T = data.shape[0]

    for i in range(0, T - window + 1, stride):
        X.append(data[i:i + window])
        Y.append(data[i:i + window, target_col])

    return np.stack(X), np.stack(Y)


def main():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        index_col=0,
        low_memory=False,
    )

    data = df.values.astype(np.float32)[:, :NUM_FEATURES]
    X, Y = make_windows(data, WINDOW, STRIDE, TARGET_COL)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    torch.save(
        {
            "X_train": torch.from_numpy(X_train),
            "Y_train": torch.from_numpy(Y_train),
            "X_test": torch.from_numpy(X_test),
            "Y_test": torch.from_numpy(Y_test),
            "meta": {
                "dataset": "ElectricityLoadDiagrams",
                "window": WINDOW,
                "stride": STRIDE,
                "features": NUM_FEATURES,
                "target_col": TARGET_COL,
                "normalization": "train_only",
            },
        },
        OUTPUT_PATH,
    )

    print("Electricity dataset prepared with train-only normalization")
    print("Train:", X_train.shape, "Test:", X_test.shape)


if __name__ == "__main__":
    main()