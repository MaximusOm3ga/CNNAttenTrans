import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from baseTrans import BaselineTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

DATASET_PATH = "../../dataset/paper_dataset.pt"
OUTPUT_PATH = "baseline_transformer.pt"


def train_baseline():
    data = torch.load(DATASET_PATH)

    X = data["X"].float()
    Y = data["Y"].float()

    N = X.size(0)
    split = int(0.8 * N)

    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val     = X[split:], Y[split:]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)

    model = BaselineTransformer(
        input_dim=X.size(-1),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")

    for epoch in range(300):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("Saved best baseline model")

    print("Baseline Transformer training complete.")


if __name__ == "__main__":
    train_baseline()
