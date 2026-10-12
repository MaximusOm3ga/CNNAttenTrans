import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches
from Paper.proposed.transformerModel.transformer import QuantFormer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_tokens.pt"
OUTPUT_PATH = BASE_DIR / "quantformer_electricity.pth"

PATCH = 8
HORIZON = 1
EPOCHS = 300
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


def train_quantformer(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing token file: {DATA_PATH}. Run extract_tokens_electricity.py first.")

    DATA = torch.load(DATA_PATH, map_location="cpu")

    tokens = DATA["tokens_train"].float()
    Y = DATA["Y_train"].float()
    Y_patch = reduce_to_patches(Y, patch_size=PATCH, reduction="mean")
    X_f, Y_f = make_forecast_pairs(tokens, Y_patch, horizon=HORIZON)

    split = int(0.8 * X_f.size(0))
    train_ds = TensorDataset(X_f[:split], Y_f[:split])
    val_ds = TensorDataset(X_f[split:], Y_f[split:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = QuantFormer(
        input_dim=tokens.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: pred {tuple(pred.shape)} vs yb {tuple(yb.shape)}")
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)

        val_loss /= len(val_dl.dataset)
        scheduler.step()

        if val_loss < best and math.isfinite(val_loss):
            best = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"Saved best model -> {OUTPUT_PATH}")

        print(f"Epoch {epoch + 1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the proposed QuantFormer on ElecDS.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train_quantformer(epochs=args.epochs, batch_size=args.batch_size)
