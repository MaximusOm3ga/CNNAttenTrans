import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Paper.proposed.modelCNN.model import AutoencoderTokenExtractor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_windows.pt"
OUTPUT_PATH = BASE_DIR / "electricity_encoder.pth"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_encoder(epochs: int = 2000, batch_size: int = 32):
    data = torch.load(DATA_PATH, map_location="cpu")
    X = data["X_train"].float()
    split = int(0.9 * X.size(0))
    train_loader = DataLoader(TensorDataset(X[:split]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[split:]), batch_size=batch_size)
    model = AutoencoderTokenExtractor(
        input_size=X.size(-1), patch_size=PATCH_SIZE, embed_dim=256
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()
    best = float("inf")
    best_encoder = None

    for epoch in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            effective = xb[:, :(xb.size(1) // PATCH_SIZE) * PATCH_SIZE]
            target = effective.reshape(
                xb.size(0), effective.size(1) // PATCH_SIZE, PATCH_SIZE, xb.size(2)
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(effective)
            loss = criterion(reconstruction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = count = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                effective = xb[:, :(xb.size(1) // PATCH_SIZE) * PATCH_SIZE]
                target = effective.reshape(
                    xb.size(0), effective.size(1) // PATCH_SIZE, PATCH_SIZE, xb.size(2)
                )
                reconstruction, _ = model(effective)
                total += criterion(reconstruction, target).item() * xb.size(0)
                count += xb.size(0)
        val_loss = total / count
        scheduler.step()
        if val_loss < best and math.isfinite(val_loss):
            best = val_loss
            best_encoder = {
                key: value.detach().cpu().clone()
                for key, value in model.encoder.state_dict().items()
            }
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Val {val_loss:.6f}", flush=True)

    model.encoder.load_state_dict(best_encoder)
    torch.save(model.encoder.state_dict(), OUTPUT_PATH)
    print(f"Saved ElecDS encoder -> {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_encoder(args.epochs, args.batch_size)
