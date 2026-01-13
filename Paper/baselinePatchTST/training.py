import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import reduce_to_patches
from model import PatchTST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DATA_PATH = "../dataset/paper_dataset.pt"
OUTPUT_PATH = "baseline_patchtst.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON = 1  # in patches

EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3


def train_patchtst():
    data = torch.load(DATA_PATH)
    X = data["X"].float()  # (N, T, F)
    Y = data["Y"].float()  # (N, T)

    Y_patch = reduce_to_patches(Y, PATCH_SIZE, PATCH_REDUCTION)

    X_f = X
    Y_f = Y_patch[:, FORECAST_HORIZON:]

    N = X_f.size(0)
    split = int(0.8 * N)

    train_ds = TensorDataset(X_f[:split], Y_f[:split])
    val_ds = TensorDataset(X_f[split:], Y_f[split:])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=X.size(-1),
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)[:, :-FORECAST_HORIZON]

            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)[:, :-FORECAST_HORIZON]
                val_loss += criterion(preds, yb).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("Saved best PatchTST baseline")

    print("PatchTST training complete.")


if __name__ == "__main__":
    train_patchtst()
