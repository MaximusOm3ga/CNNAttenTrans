import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import reduce_to_patches, make_forecast_pairs
from model import PatchTST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../dataset/paper_new_test.pt"
MODEL_PATH = "baseline_patchtst.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON = 1
BATCH_SIZE = 32


def evaluate_patchtst():
    data = torch.load(DATA_PATH)
    X = data["X"].float()
    Y = data["Y"].float()

    horizon_steps = PATCH_SIZE * FORECAST_HORIZON
    X_f, Y_f_full = make_forecast_pairs(X, Y, horizon=horizon_steps)

    T_in = int(X_f.size(1))
    P_in = T_in // PATCH_SIZE
    if P_in <= FORECAST_HORIZON:
        raise ValueError(
            f"Not enough input patches P_in={P_in} for FORECAST_HORIZON={FORECAST_HORIZON}"
        )

    P_target = P_in - FORECAST_HORIZON
    T_target = P_target * PATCH_SIZE
    Y_f_full = Y_f_full[:, :T_target]

    Y_f = reduce_to_patches(Y_f_full, PATCH_SIZE, PATCH_REDUCTION)

    ds = TensorDataset(X_f, Y_f)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=X.size(-1),
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
            preds = model(xb)[:, :-FORECAST_HORIZON]

            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    rmse = math.sqrt(total_mse)

    print("PatchTST baseline (patch-level causal forecasting)")
    print(f"MSE : {total_mse / total_count:.6f}")
    print(f"MAE : {total_mae / total_count:.6f}")
    print(f"RMSE : {rmse:.6f}")


if __name__ == "__main__":
    evaluate_patchtst()
