import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import make_forecast_pairs
from model import BaselineGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_gru.pt"
DATA_PATH = "../../dataset/paper_new_test.pt"

FORECAST_HORIZON = 1


def eval_gru():
    data = torch.load(DATA_PATH)
    X, Y = data["X"].float(), data["Y"].float()

    X, Y = make_forecast_pairs(X, Y, FORECAST_HORIZON)

    dl = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=False)

    model = BaselineGRU(input_dim=X.size(-1)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mse, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            if preds.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(yb.shape)}")
            mse += ((preds - yb) ** 2).sum().item()
            mae += (preds - yb).abs().sum().item()
            n += yb.numel()

    print("GRU baseline (1-step forecast, RAW X → GRU)")
    print(f"MSE : {mse / n:.6f}")
    print(f"MAE : {mae / n:.6f}")


if __name__ == "__main__":
    eval_gru()
