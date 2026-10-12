import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baselineTCN.model import TCN
from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_windows.pt"
OUTPUT_PATH = BASE_DIR / "weather_tcn.pt"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_tcn_weather(horizon: int = 96, epochs: int = 20, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather dataset: {DATA_PATH}. Run dataset_prep.py first.")

    data = torch.load(DATA_PATH, map_location="cpu")
    X_train = data["X_train"].float()
    Y_train = data["Y_train"].float()

    X_f, Y_f_full = make_forecast_pairs(X_train, Y_train, horizon=horizon)
    P_in = X_f.size(1) // PATCH_SIZE
    if P_in <= 0:
        raise ValueError(f"Input too short for horizon={horizon} and patch_size={PATCH_SIZE}")
    Y_f = reduce_to_patches(Y_f_full[:, :P_in * PATCH_SIZE], PATCH_SIZE, reduction="mean")

    train_loader = DataLoader(TensorDataset(X_f, Y_f), batch_size=batch_size, shuffle=True)
    model = TCN(input_dim=X_train.size(-1)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="mean")
    best = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            pred_patch = reduce_to_patches(pred, PATCH_SIZE, reduction="mean")
            loss = criterion(pred_patch, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                pred_patch = reduce_to_patches(pred, PATCH_SIZE, reduction="mean")
                val_loss += criterion(pred_patch, yb).item() * xb.size(0)
            val_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            os.makedirs(str(OUTPUT_PATH.parent), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH.with_name(f"weather_tcn_h{horizon}.pt"))

    print(f"Saved trained TCN to {OUTPUT_PATH.with_name(f'weather_tcn_h{horizon}.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN on WeatherDS for a forecast horizon.")
    parser.add_argument("--horizon", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_tcn_weather(horizon=args.horizon, epochs=args.epochs, batch_size=args.batch_size)
