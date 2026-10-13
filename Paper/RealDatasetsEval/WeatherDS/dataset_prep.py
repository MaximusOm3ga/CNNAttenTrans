import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Dataset" / "weather.csv"
OUTPUT_PATH = BASE_DIR / "weather_windows.pt"
WINDOW = 1024
WINDOW_STRIDE = 16
TARGET_NAME = "OT"


def build_weather_windows() -> dict:
    df = pd.read_csv(CSV_PATH)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df.dropna().reset_index(drop=True)

    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found in weather dataset.")

    feature_cols = [c for c in df.columns if c != TARGET_NAME]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[TARGET_NAME].to_numpy(dtype=np.float32)

    X_windows, Y_windows = [], []
    for i in range(0, X.shape[0] - WINDOW + 1, WINDOW_STRIDE):
        X_windows.append(X[i:i + WINDOW])
        Y_windows.append(Y[i:i + WINDOW])

    X_windows = np.stack(X_windows)
    Y_windows = np.stack(Y_windows)

    train_end = int(0.7 * len(X_windows))
    val_end = int(0.85 * len(X_windows))

    data = {
        "X_train": torch.from_numpy(X_windows[:train_end]),
        "Y_train": torch.from_numpy(Y_windows[:train_end]),
        "X_val": torch.from_numpy(X_windows[train_end:val_end]),
        "Y_val": torch.from_numpy(Y_windows[train_end:val_end]),
        "X_test": torch.from_numpy(X_windows[val_end:]),
        "Y_test": torch.from_numpy(Y_windows[val_end:]),
        "feature_names": feature_cols,
        "target_name": TARGET_NAME,
        "window": WINDOW,
        "stride": WINDOW_STRIDE,
    }

    torch.save(data, OUTPUT_PATH)
    print(f"Saved weather windows to {OUTPUT_PATH}")
    print(f"train={data['X_train'].shape}, val={data['X_val'].shape}, test={data['X_test'].shape}")
    return data


if __name__ == "__main__":
    build_weather_windows()
