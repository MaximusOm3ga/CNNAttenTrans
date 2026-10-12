import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.modelCNN.model import AutoencoderTokenExtractor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_windows.pt"
OUTPUT_PATH = BASE_DIR / "weather_encoder.pth"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_weather_encoder(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather windows file: {DATA_PATH}. Run dataset_prep.py first.")

    data = torch.load(DATA_PATH, map_location="cpu")
    X_train = data["X_train"].float()
    X_val = data["X_val"].float()
    feature_dim = X_train.size(-1)

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(X_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = AutoencoderTokenExtractor(input_size=feature_dim, patch_size=PATCH_SIZE, embed_dim=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()
    best_val = float("inf")

    print(
        f"Starting weather encoder training on {DEVICE} | "
        f"X_train={tuple(X_train.shape)} X_val={tuple(X_val.shape)} "
        f"epochs={epochs} batch_size={batch_size}"
    )
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_elems = 0
        first_batch_logged = False
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            if not first_batch_logged:
                print(f"Epoch {epoch+1:03d} | first train batch shape={tuple(xb.shape)} device={xb.device}")
                first_batch_logged = True
            B, T, D = xb.shape
            num_patches = T // PATCH_SIZE
            if num_patches <= 0:
                continue
            x_eff = xb[:, :num_patches * PATCH_SIZE, :]
            patches_target = x_eff.reshape(B, num_patches, PATCH_SIZE, D)
            reconstructed, _ = model(x_eff)
            if reconstructed.shape != patches_target.shape:
                raise RuntimeError(f"shape mismatch: reconstructed {tuple(reconstructed.shape)} != target {tuple(patches_target.shape)}")
            loss = criterion(reconstructed, patches_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * reconstructed.numel()
            train_elems += reconstructed.numel()

        if train_elems > 0:
            train_loss /= train_elems

        model.eval()
        val_loss = 0.0
        val_elems = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                B, T, D = xb.shape
                num_patches = T // PATCH_SIZE
                if num_patches <= 0:
                    continue
                x_eff = xb[:, :num_patches * PATCH_SIZE, :]
                patches_target = x_eff.reshape(B, num_patches, PATCH_SIZE, D)
                reconstructed, _ = model(x_eff)
                loss = criterion(reconstructed, patches_target)
                val_loss += loss.item() * reconstructed.numel()
                val_elems += reconstructed.numel()
        if val_elems > 0:
            val_loss /= val_elems

        scheduler.step()
        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.encoder.state_dict(), OUTPUT_PATH)
            print(f"Saved best encoder -> {OUTPUT_PATH}")

    print(f"Weather encoder training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WeatherDS patch encoder.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_weather_encoder(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
