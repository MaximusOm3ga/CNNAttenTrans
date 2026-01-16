import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_utils import reduce_to_patches
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

    Y_patch = reduce_to_patches(Y, PATCH_SIZE, PATCH_REDUCTION)
    Y_f = Y_patch[:, FORECAST_HORIZON:]
    X_f = X

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

    print("PatchTST baseline (patch-level forecasting)")
    print(f"MSE : {total_mse / total_count:.6f}")
    print(f"MAE : {total_mae / total_count:.6f}")


if __name__ == "__main__":
    evaluate_patchtst()
