import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Transformer.proposed.transformerModel.transformer import QuantFormer

D_MODEL = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DATASET_PATH = "../dataset/paper_dataset.pt"
OUTPUT_PATH = "quantformer_raw.pt"


def train_quantformer_on_raw_synthetic():
    data = torch.load(DATASET_PATH)

    X = data["X"].float()
    Y = data["Y"].float()

    assert X.dim() == 3
    assert Y.dim() == 2
    assert X.size(0) == Y.size(0)
    assert X.size(1) == Y.size(1)

    N = X.size(0)
    split = int(0.8 * N)

    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val     = X[split:], Y[split:]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = QuantFormer(
        input_dim=X.size(-1),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        output_dim=1,
    ).to(DEVICE)

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
            preds = model(xb)
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
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("Saved best RAW-X model")

    print("Raw-X training complete.")


if __name__ == "__main__":
    train_quantformer_on_raw_synthetic()
