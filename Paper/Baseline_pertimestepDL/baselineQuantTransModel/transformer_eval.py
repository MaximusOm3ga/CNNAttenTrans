import os
import sys
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline_utils import make_forecast_pairs
from proposed.transformerModel.transformer import QuantFormer

D_MODEL = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

FORECAST_HORIZON = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_quantformer_on_raw_synthetic(model_path=None):
    data = torch.load("../../dataset/paper_new_test.pt")

    X = data["X"].float()
    Y = data["Y"].float()

    X, Y = make_forecast_pairs(X, Y, FORECAST_HORIZON)

    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    model = QuantFormer(
        input_dim=X.size(-1),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        output_dim=1,
    ).to(DEVICE)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()
    with torch.no_grad():
        preds = model(X)

    if preds.shape != Y.shape:
        raise RuntimeError(f"Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(Y.shape)}")

    mse = ((preds - Y) ** 2).mean().item()
    mae = (preds - Y).abs().mean().item()

    print("RAW X → Transformer (1-step forecast)")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")


if __name__=="__main__":
    eval_quantformer_on_raw_synthetic(
        model_path="quantformer_raw_new.pt"
    )