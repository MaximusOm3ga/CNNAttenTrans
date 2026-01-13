import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Paper.proposed.modelCNN.model import DenseNetTokenEncoder
from Paper.proposed.transformerModel.transformer import QuantFormer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../../dataset/paper_new_test.pt"
ENCODER_WEIGHTS = "../modelCNN/trained_encoder_new.pth"
MODEL_WEIGHTS = "./quantformer_price_new.pt"

PATCH_SIZE = 8
EMBED_DIM = 256
BATCH_SIZE = 32
FORECAST_HORIZON = 1


def _reduce_y_to_patches(y: torch.Tensor, P: int, patch_size: int, reduction: str = "mean") -> torch.Tensor:

    if y.dim() != 2:
        raise ValueError(f"Expected y to be 2D (N,T), got {tuple(y.shape)}")
    if P <= 0 or patch_size <= 0:
        raise ValueError(f"Invalid P={P}, patch_size={patch_size}")

    N, T = y.shape
    T_eff = P * patch_size
    if T < T_eff:
        raise ValueError(f"Y too short: T={T} < T_eff={T_eff} (P={P}, patch_size={patch_size})")

    y_eff = y[:, :T_eff].contiguous().view(N, P, patch_size)

    if reduction == "mean":
        return y_eff.mean(dim=-1)
    if reduction == "last":
        return y_eff[:, :, -1]
    raise ValueError(f"Unknown reduction={reduction}")


def _make_forecast_pairs(tokens: torch.Tensor, y_patches: torch.Tensor, horizon: int):
    if horizon <= 0:
        raise ValueError(f"Expected horizon > 0, got {horizon}")
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens to be 3D (N,P,E), got {tuple(tokens.shape)}")
    if y_patches.dim() != 2:
        raise ValueError(f"Expected y_patches to be 2D (N,P), got {tuple(y_patches.shape)}")
    if tokens.size(0) != y_patches.size(0) or tokens.size(1) != y_patches.size(1):
        raise ValueError(
            f"Expected tokens (N,P,*) and y_patches (N,P) to match, got "
            f"{tuple(tokens.shape)} vs {tuple(y_patches.shape)}"
        )

    P = int(tokens.size(1))
    if P <= horizon:
        raise ValueError(f"Not enough patches P={P} for horizon={horizon}")

    x_f = tokens[:, : P - horizon, :]
    y_f = y_patches[:, horizon:]
    return x_f, y_f


@torch.no_grad()
def _encode_tokens_batched(
    encoder: nn.Module,
    X: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    encoder.eval()
    outs = []
    dl = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        outs.append(encoder(xb).detach().cpu())
    return torch.cat(outs, dim=0)


def main():
    data = torch.load(DATA_PATH, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()

    if X.dim() != 3:
        raise ValueError(f"Expected X to be 3D (N,T,F), got {tuple(X.shape)}")
    if Y.dim() != 2:
        raise ValueError(f"Expected Y to be 2D (N,T), got {tuple(Y.shape)}")
    if X.size(0) != Y.size(0):
        raise ValueError(f"N mismatch: X {tuple(X.shape)} vs Y {tuple(Y.shape)}")

    encoder = DenseNetTokenEncoder(input_size=X.size(-1), patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=DEVICE))

    tokens = _encode_tokens_batched(encoder, X, batch_size=BATCH_SIZE, device=DEVICE)

    P_tokens = int(tokens.size(1))
    Yp = _reduce_y_to_patches(Y, P=P_tokens, patch_size=PATCH_SIZE, reduction="mean")

    tokens_f, Yp_f = _make_forecast_pairs(tokens, Yp, horizon=FORECAST_HORIZON)

    model = QuantFormer(
        input_dim=tokens_f.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    dl = DataLoader(TensorDataset(tokens_f, Yp_f), batch_size=BATCH_SIZE, shuffle=False)

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_mse, total_mae, total_count = 0.0, 0.0, 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            preds = model(xb)
            if preds.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: preds {tuple(preds.shape)} vs y {tuple(yb.shape)}")

            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count

    print("CNN tokens (patched) -> QuantFormer, per-patch *forecast* evaluation")
    print(f"patch_size: {PATCH_SIZE}")
    print(f"forecast_horizon_patches: {FORECAST_HORIZON}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")


if __name__ == "__main__":
    main()
