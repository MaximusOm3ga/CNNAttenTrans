import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline_utils import make_forecast_pairs, reduce_to_patches, make_patch_forecast_pairs
from model import BaselineGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_gru.pt"
DATA_PATH = "../../dataset/paper_new_test.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON_TIMESTEPS = 1
FORECAST_HORIZON_PATCHES = 1

BATCH_SIZE = 32


def main():
    data = torch.load(DATA_PATH)
    X, Y = data["X"].float(), data["Y"].float()

    # timestep-level 1-step forecast
    X_f, Y_f = make_forecast_pairs(X, Y, FORECAST_HORIZON_TIMESTEPS)

    dl = DataLoader(TensorDataset(X_f, Y_f), batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineGRU(input_dim=X_f.size(-1)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    preds_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            preds = model(xb).detach().cpu()
            preds_all.append(preds)
            y_all.append(yb.detach().cpu())

    preds_t = torch.cat(preds_all, dim=0)
    y_t = torch.cat(y_all, dim=0)

    # reduce to patch resolution (comparable to proposed)
    y_patch = reduce_to_patches(y_t, patch_size=PATCH_SIZE, reduction=PATCH_REDUCTION)
    pred_patch = reduce_to_patches(preds_t, patch_size=PATCH_SIZE, reduction=PATCH_REDUCTION)

    # patch-level 1-patch-ahead forecast (match proposed FORECAST_HORIZON)
    pred_patch_f, y_patch_f = make_patch_forecast_pairs(pred_patch, y_patch, horizon_patches=FORECAST_HORIZON_PATCHES)

    mse = ((pred_patch_f - y_patch_f) ** 2).mean().item()
    mae = (pred_patch_f - y_patch_f).abs().mean().item()

    print("GRU baseline (patch-level comparable eval)")
    print(f"patch_size: {PATCH_SIZE} | patch_reduction: {PATCH_REDUCTION} | horizon_patches: {FORECAST_HORIZON_PATCHES}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")


if __name__ == "__main__":
    main()

