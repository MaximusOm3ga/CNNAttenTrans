import math

from baseline_utils import make_forecast_pairs,reduce_to_patches
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import TCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../dataset/paper_new_test.pt"
MODEL_PATH = "baseline_tcn.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON = 1
BATCH_SIZE = 32


def evaluate_tcn():
    data = torch.load(DATA_PATH)
    X = data["X"].float()
    Y = data["Y"].float()

    X_f, Y_f = make_forecast_pairs(
        X, Y, horizon=PATCH_SIZE * FORECAST_HORIZON
    )

    ds = TensorDataset(X_f, Y_f)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TCN(input_dim=X.size(-1)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds_t = model(xb)

            preds_patch = reduce_to_patches(preds_t, PATCH_SIZE)
            yb_patch = reduce_to_patches(yb, PATCH_SIZE)

            total_mse += mse_loss(preds_patch, yb_patch).item()
            total_mae += mae_loss(preds_patch, yb_patch).item()
            total_count += yb_patch.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count
    rmse = math.sqrt(mse)

    print("TCN baseline (raw X → patch-level forecasting)")
    print(f"patch_size: {PATCH_SIZE}")
    print(f"patch_reduction: {PATCH_REDUCTION}")
    print(f"forecast_horizon_patches: {FORECAST_HORIZON}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")

if __name__ == "__main__":
    evaluate_tcn()
