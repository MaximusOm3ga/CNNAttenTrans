# baselineTCN/eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import reduce_to_patches
from model import TCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../dataset/paper_new_test.pt"
MODEL_PATH = "baseline_tcn.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON = 1  # in patches
BATCH_SIZE = 32


def evaluate_tcn():
    data = torch.load(DATA_PATH)
    X = data["X"].float()  # (N, T, F)
    Y = data["Y"].float()  # (N, T)

    # ---- Patch-level targets ----
    Y_patch = reduce_to_patches(
        Y,
        patch_size=PATCH_SIZE,
        reduction=PATCH_REDUCTION
    )  # (N, K)

    if Y_patch.size(1) <= FORECAST_HORIZON:
        raise ValueError("Not enough patches for forecast horizon")

    # ---- Patch-level forecast shift (Y only) ----
    Y_f = Y_patch[:, FORECAST_HORIZON:]  # (N, K - h)
    X_f = X                              # raw input

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

            # ---- Timestep predictions ----
            preds_t = model(xb)  # (B, T)

            # ---- Reduce to patch level ----
            preds_patch = reduce_to_patches(
                preds_t,
                patch_size=PATCH_SIZE,
                reduction=PATCH_REDUCTION
            )[:, FORECAST_HORIZON:]  # (B, K - h)

            if preds_patch.shape != yb.shape:
                raise RuntimeError(
                    f"Shape mismatch: preds {preds_patch.shape} vs y {yb.shape}"
                )

            total_mse += mse_loss(preds_patch, yb).item()
            total_mae += mae_loss(preds_patch, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count

    print("TCN baseline (raw X → patch-level forecasting)")
    print(f"patch_size: {PATCH_SIZE}")
    print(f"patch_reduction: {PATCH_REDUCTION}")
    print(f"forecast_horizon_patches: {FORECAST_HORIZON}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")


if __name__ == "__main__":
    evaluate_tcn()
