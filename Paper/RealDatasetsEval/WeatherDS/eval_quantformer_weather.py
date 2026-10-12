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

from Paper.proposed.transformerModel.transformer import QuantFormer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_tokens.pt"
MODEL_PATH = BASE_DIR / "quantformer_weather.pth"
PATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reduce_y_to_patches(y: torch.Tensor, P: int, patch_size: int, reduction: str = "mean") -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f"Expected y to be 2D (B,T), got {tuple(y.shape)}")
    B, T = y.shape
    T_eff = P * patch_size
    if T < T_eff:
        raise ValueError(f"y too short: T={T} < T_eff={T_eff} (P={P}, patch_size={patch_size})")
    y_eff = y[:, :T_eff].contiguous().view(B, P, patch_size)
    if reduction == "mean":
        return y_eff.mean(dim=-1)
    if reduction == "last":
        return y_eff[:, :, -1]
    raise ValueError(f"Unknown reduction={reduction}")


def _make_forecast_pairs(tokens: torch.Tensor, y_patched: torch.Tensor, horizon: int):
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    P = int(tokens.size(1))
    if P <= horizon:
        raise ValueError(f"Not enough patches P={P} for horizon={horizon}")
    return tokens[:, :P - horizon, :], y_patched[:, horizon:]


def evaluate_quantformer_weather(horizon: int = 96, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather tokens: {DATA_PATH}. Run extract_tokens_weather.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing QuantFormer weights: {MODEL_PATH}. Run train_quantformer_weather.py first.")

    obj = torch.load(DATA_PATH, map_location="cpu")
    tokens = obj["tokens_test"].float()
    Y = obj["Y_test"].float()

    P = int(tokens.size(1))
    Y_patched = _reduce_y_to_patches(Y, P=P, patch_size=PATCH_SIZE, reduction="mean")
    X_f, Y_f = _make_forecast_pairs(tokens, Y_patched, horizon=horizon)

    dl = DataLoader(TensorDataset(X_f, Y_f), batch_size=batch_size, shuffle=False)
    model = QuantFormer(
        input_dim=tokens.size(-1),
        d_model=256,
        num_heads=16,
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
    count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            total_mse += mse_loss(pred, yb).item()
            total_mae += mae_loss(pred, yb).item()
            count += yb.numel()

    mse = total_mse / count
    mae = total_mae / count
    rmse = math.sqrt(mse)
    print(f"QuantFormer on WeatherDS | horizon={horizon}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    return mse, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the QuantFormer on WeatherDS for a selected forecast horizon.")
    parser.add_argument("--horizon", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_quantformer_weather(horizon=args.horizon, batch_size=args.batch_size)
