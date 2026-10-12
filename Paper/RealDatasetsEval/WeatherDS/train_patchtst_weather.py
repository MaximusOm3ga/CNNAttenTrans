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

from Paper.baselinePatchTST.model import PatchTST
from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_windows.pt"
OUTPUT_PATH = BASE_DIR / "weather_patchtst_h96.pt"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_patchtst_weather(horizon: int = 96, epochs: int = 20, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather dataset: {DATA_PATH}. Run dataset_prep.py first.")

    data = torch.load(DATA_PATH, map_location="cpu")
    X_train = data["X_train"].float()
    Y_train = data["Y_train"].float()

    X_f, Y_f_full = make_forecast_pairs(X_train, Y_train, horizon=horizon)
    P_in = X_f.size(1) // PATCH_SIZE
    Y_f = reduce_to_patches(Y_f_full[:, :P_in * PATCH_SIZE], PATCH_SIZE, reduction="mean")

    train_loader = DataLoader(TensorDataset(X_f, Y_f), batch_size=batch_size, shuffle=True)
    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=X_train.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            if preds.size(-1) > yb.size(-1):
                preds = preds[:, : yb.size(-1)]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                if preds.size(-1) > yb.size(-1):
                    preds = preds[:, : yb.size(-1)]
                val_loss += criterion(preds, yb).item() * xb.size(0)
            val_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            out_path = BASE_DIR / f"weather_patchtst_h{horizon}.pt"
            os.makedirs(str(out_path.parent), exist_ok=True)
            torch.save(model.state_dict(), out_path)
            print(f"Saved best PatchTST for horizon {horizon} -> {out_path}")

    print(f"Training complete. Best val: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchTST on WeatherDS for a forecast horizon.")
    parser.add_argument("--horizon", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_patchtst_weather(horizon=args.horizon, epochs=args.epochs, batch_size=args.batch_size)
