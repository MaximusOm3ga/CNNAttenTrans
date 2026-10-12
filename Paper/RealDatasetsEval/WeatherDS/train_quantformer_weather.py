import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.transformerModel.transformer import QuantFormer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_tokens.pt"
OUTPUT_PATH = BASE_DIR / "quantformer_weather.pth"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reduce_y_to_patches(y: torch.Tensor, P: int, patch_size: int, reduction: str = "mean") -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f"Expected y to be 2D (B,T), got {tuple(y.shape)}")
    B, T = y.shape
    T_eff = P * patch_size
    if T < T_eff:
        raise ValueError(f"y too short: T={T} < T_eff={T_eff} (P={P}, patch_size={patch_size})")
    y_eff = y[:, :T_eff].contiguous().view(B, P, patch_size)
    if reduction == "mean":
        return y_eff.mean(dim=-1)
    if reduction == "last":
        return y_eff[:, :, -1]
    raise ValueError(f"Unknown reduction={reduction}")


def _make_forecast_pairs(tokens: torch.Tensor, y_patched: torch.Tensor, horizon: int):
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be (N,P,E), got {tuple(tokens.shape)}")
    if y_patched.dim() != 2:
        raise ValueError(f"y_patched must be (N,P), got {tuple(y_patched.shape)}")
    if tokens.size(0) != y_patched.size(0) or tokens.size(1) != y_patched.size(1):
        raise ValueError(f"tokens/y_patched mismatch: {tuple(tokens.shape)} vs {tuple(y_patched.shape)}")
    P = int(tokens.size(1))
    if P <= horizon:
        raise ValueError(f"Not enough patches P={P} for horizon={horizon}")
    x_f = tokens[:, :P - horizon, :]
    y_f = y_patched[:, horizon:]
    return x_f, y_f


def train_quantformer_weather(horizon: int = 96, epochs: int = 30, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather tokens file: {DATA_PATH}. Run extract_tokens_weather.py first.")

    obj = torch.load(DATA_PATH, map_location="cpu")
    tokens = obj["tokens_train"].float()
    Y = obj["Y_train"].float()

    P = int(tokens.size(1))
    Y_patched = _reduce_y_to_patches(Y, P=P, patch_size=PATCH_SIZE, reduction="mean")
    X_f, Y_f = _make_forecast_pairs(tokens, Y_patched, horizon=horizon)

    split = int(0.8 * X_f.size(0))
    train_ds = TensorDataset(X_f[:split], Y_f[:split])
    val_ds = TensorDataset(X_f[split:], Y_f[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = QuantFormer(
        input_dim=tokens.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(str(OUTPUT_PATH.parent), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"Saved best QuantFormer for weather horizon {horizon} -> {OUTPUT_PATH}")

    print(f"Training complete. Best val: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuantFormer on the WeatherDS benchmark.")
    parser.add_argument("--horizon", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_quantformer_weather(horizon=args.horizon, epochs=args.epochs, batch_size=args.batch_size)
