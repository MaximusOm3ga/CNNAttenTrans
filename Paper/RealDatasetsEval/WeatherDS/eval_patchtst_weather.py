import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baselinePatchTST.model import PatchTST
from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_windows.pt"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_patchtst_weather(horizon: int = 96, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather dataset: {DATA_PATH}. Run dataset_prep.py first.")
    model_path = BASE_DIR / f"weather_patchtst_h{horizon}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PatchTST weights: {model_path}. Run train_patchtst_weather.py first.")

    data = torch.load(DATA_PATH, map_location="cpu")
    X_test = data["X_test"].float()
    Y_test = data["Y_test"].float()

    X_f, Y_f_full = make_forecast_pairs(X_test, Y_test, horizon=horizon)
    P_in = X_f.size(1) // PATCH_SIZE
    Y_f = reduce_to_patches(Y_f_full[:, :P_in * PATCH_SIZE], PATCH_SIZE, reduction="mean")
    dl = DataLoader(TensorDataset(X_f, Y_f), batch_size=batch_size, shuffle=False)

    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=X_test.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            if preds.size(-1) > yb.size(-1):
                preds = preds[:, : yb.size(-1)]
            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            count += yb.numel()

    mse = total_mse / count
    mae = total_mae / count
    rmse = math.sqrt(mse)
    print(f"PatchTST on WeatherDS | horizon={horizon}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    return mse, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PatchTST on WeatherDS for a selected horizon.")
    parser.add_argument("--horizon", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_patchtst_weather(horizon=args.horizon, batch_size=args.batch_size)
