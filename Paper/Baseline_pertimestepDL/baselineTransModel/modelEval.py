import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import make_forecast_pairs
from baseTrans import BaselineTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "baseline_transformer.pt"
DATA_PATH = "../../dataset/paper_new_test.pt"
BATCH_SIZE = 32

FORECAST_HORIZON = 1


def evaluate_baseline():
    data = torch.load(DATA_PATH)

    X = data["X"].float()
    Y = data["Y"].float()

    X, Y = make_forecast_pairs(X, Y, FORECAST_HORIZON)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineTransformer(
        input_dim=X.size(-1),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
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
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            if preds.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(yb.shape)}")

            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    rmse = mse ** 0.5
    mae = total_mae / total_count

    print("Baseline Transformer (1-step forecast, RAW X → Transformer)")
    print(f"MSE :  {mse:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE :  {mae:.6f}")


if __name__ == "__main__":
    evaluate_baseline()
