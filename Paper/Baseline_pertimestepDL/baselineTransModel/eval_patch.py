import os
import sys
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline_utils import make_forecast_pairs, reduce_to_patches, make_patch_forecast_pairs
from baseTrans import BaselineTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_transformer.pt"
DATA_PATH = "../../dataset/paper_new_test.pt"

PATCH_SIZE = 8
PATCH_REDUCTION = "mean"
FORECAST_HORIZON_TIMESTEPS = 1
FORECAST_HORIZON_PATCHES = 1

BATCH_SIZE = 32


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset at: {os.path.abspath(DATA_PATH)}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing weights at: {os.path.abspath(MODEL_PATH)}")

    data = torch.load(DATA_PATH, map_location="cpu")
    X, Y = data["X"].float(), data["Y"].float()

    X_f, Y_f = make_forecast_pairs(X, Y, FORECAST_HORIZON_TIMESTEPS)

    model = BaselineTransformer(
        input_dim=int(X_f.size(-1)),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
    model.eval()

    preds_all = []
    y_all = []
    with torch.no_grad():
        for i in range(0, X_f.size(0), BATCH_SIZE):
            xb = X_f[i : i + BATCH_SIZE].to(DEVICE)
            preds = model(xb).detach().cpu()
            preds_all.append(preds)
            y_all.append(Y_f[i : i + BATCH_SIZE].detach().cpu())

    preds_t = torch.cat(preds_all, dim=0)
    y_t = torch.cat(y_all, dim=0)

    y_patch = reduce_to_patches(y_t, patch_size=PATCH_SIZE, reduction=PATCH_REDUCTION)
    pred_patch = reduce_to_patches(preds_t, patch_size=PATCH_SIZE, reduction=PATCH_REDUCTION)

    pred_patch_f, y_patch_f = make_patch_forecast_pairs(
        pred_patch, y_patch, horizon_patches=FORECAST_HORIZON_PATCHES
    )

    mse = ((pred_patch_f - y_patch_f) ** 2).mean().item()
    mae = (pred_patch_f - y_patch_f).abs().mean().item()

    print("Baseline Transformer (patch-level comparable eval)")
    print(
        f"patch_size: {PATCH_SIZE} | patch_reduction: {PATCH_REDUCTION} | horizon_patches: {FORECAST_HORIZON_PATCHES}"
    )
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")


if __name__ == "__main__":
    main()

