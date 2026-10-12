import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.transformerModel.transformer import QuantFormer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_tokens.pt"
MODEL_PATH = BASE_DIR / "quantformer_electricity.pth"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing token file: {DATA_PATH}. Run extract_tokens_electricity.py first.")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}. Run train_quantformer_electricity.py first.")

DATA = torch.load(DATA_PATH, map_location="cpu")

tokens = DATA["tokens_test"].float()
Y = DATA["Y_test"].float()

PATCH = 8
HORIZON = 1

P = tokens.size(1)
Y = Y[:, :P * PATCH].view(Y.size(0), P, PATCH).mean(dim=-1)

X = tokens[:, :-HORIZON]
Y = Y[:, HORIZON:]

dl = DataLoader(TensorDataset(X, Y), batch_size=32)

model = QuantFormer(
    input_dim=256,
    d_model=256,
    num_heads=16,
    num_layers=4,
    d_ff=512,
    output_dim=1,
).cuda()

model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"))
model.eval()

mse = nn.MSELoss(reduction="sum")
mae = nn.L1Loss(reduction="sum")

tot_mse = tot_mae = n = 0

with torch.no_grad():
    for xb, yb in dl:
        pred = model(xb.cuda())
        tot_mse += mse(pred, yb.cuda()).item()
        tot_mae += mae(pred, yb.cuda()).item()
        n += yb.numel()

rmse = math.sqrt(tot_mse / n)

print("Electricity Load Forecasting Results")
print("MSE:", tot_mse / n)
print("MAE:", tot_mae / n)
print("RMSE:", rmse)
