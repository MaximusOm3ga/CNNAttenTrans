# eval_gru.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import BaselineGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_gru.pt"
DATA_PATH = "../dataset/paper_new_test.pt"

def eval_gru():
    data = torch.load(DATA_PATH)
    X, Y = data["X"].float(), data["Y"].float()

    dl = DataLoader(TensorDataset(X, Y), batch_size=32)

    model = BaselineGRU(input_dim=X.size(-1)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mse, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            mse += ((preds - yb) ** 2).sum().item()
            mae += (preds - yb).abs().sum().item()
            n += yb.numel()

    print("GRU baseline (RAW X → GRU)")
    print(f"MSE : {mse / n:.6f}")
    print(f"MAE : {mae / n:.6f}")

if __name__ == "__main__":
    eval_gru()
