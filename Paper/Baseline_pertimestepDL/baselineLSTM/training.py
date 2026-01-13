import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline_utils import make_forecast_pairs
from baselineLSTM import BaselineLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DATASET_PATH = "../../dataset/paper_dataset.pt"
OUTPUT_PATH = "baseline_lstm.pt"

FORECAST_HORIZON = 1


def train_lstm():
    data = torch.load(DATASET_PATH)

    X = data["X"].float()
    Y = data["Y"].float()

    X, Y = make_forecast_pairs(X, Y, FORECAST_HORIZON)

    N = X.size(0)
    split = int(0.8 * N)

    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = BaselineLSTM(
        input_dim=X.size(-1),
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)

    best_val = float("inf")

    for epoch in range(600):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            if preds.shape != yb.shape:
                raise RuntimeError(
                    f"Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(yb.shape)}"
                )
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
                preds = model(xb)
                if preds.shape != yb.shape:
                    raise RuntimeError(
                        f"(val) Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(yb.shape)}"
                    )
                val_loss += criterion(preds, yb).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | Val {val_loss:.6f}"
        )

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("Saved best LSTM baseline")

    print("LSTM training complete.")


if __name__ == "__main__":
    train_lstm()
