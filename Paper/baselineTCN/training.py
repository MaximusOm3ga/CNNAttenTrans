import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

from baseline_utils import reduce_to_patches, make_forecast_pairs
from model import TCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DATA_PATH = "../dataset/paper_dataset.pt"
OUTPUT_PATH = "baseline_tcn.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON = 1

EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3


def train_tcn():
    data = torch.load(DATA_PATH)
    X = data["X"].float()
    Y = data["Y"].float()

    Y_patch = reduce_to_patches(
        Y,
        patch_size=PATCH_SIZE,
        reduction=PATCH_REDUCTION
    )


    X_f, Y_f = make_forecast_pairs(X, Y, horizon=PATCH_SIZE * FORECAST_HORIZON)

    N = X_f.size(0)
    split = int(0.8 * N)

    train_ds = TensorDataset(X_f[:split], Y_f[:split])
    val_ds = TensorDataset(X_f[split:], Y_f[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TCN(input_dim=X.size(-1)).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            preds_t = model(xb)
            preds_patch = reduce_to_patches(preds_t, PATCH_SIZE)
            yb_patch = reduce_to_patches(yb, PATCH_SIZE)
            loss = criterion(preds_patch, yb_patch)

            # preds_patch = reduce_to_patches(
            #     preds_t,
            #     patch_size=PATCH_SIZE,
            #     reduction=PATCH_REDUCTION
            # )[:, FORECAST_HORIZON:]
            #
            # if preds_patch.shape != yb.shape:
            #     raise RuntimeError(
            #         f"Shape mismatch: preds {preds_patch.shape} vs y {yb.shape}"
            #     )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds_t = model(xb)
                preds_patch = reduce_to_patches(preds_t, PATCH_SIZE)
                yb_patch = reduce_to_patches(yb, PATCH_SIZE)

                val_loss += criterion(preds_patch, yb_patch).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | Val {val_loss:.6f}"
        )

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("Saved best TCN baseline")

    print("TCN training complete.")


if __name__ == "__main__":
    train_tcn()
