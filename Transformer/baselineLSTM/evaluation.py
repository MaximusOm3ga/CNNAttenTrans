# File: eval_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselineLSTM import BaselineLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "baseline_lstm.pt"
DATA_PATH = "../dataset/paper_new_test.pt"
BATCH_SIZE = 32


def evaluate_lstm():
    data = torch.load(DATA_PATH)

    X = data["X"].float()
    Y = data["Y"].float()

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineLSTM(
        input_dim=X.size(-1),
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)

            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    rmse = mse ** 0.5
    mae = total_mae / total_count

    print("LSTM baseline (RAW X → LSTM)")
    print(f"MSE :  {mse:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE :  {mae:.6f}")


if __name__ == "__main__":
    evaluate_lstm()
