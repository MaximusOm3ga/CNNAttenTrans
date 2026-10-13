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

from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches
from Paper.proposed.modelCNN.model import DenseNetTokenEncoder
from Paper.proposed.transformerModel.transformer import QuantFormer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "Paper" / "dataset" / "paper_new_test.pt"
TOKEN_PATH = REPO_ROOT / "Paper" / "proposed" / "transformerModel" / "synthetic_tokens_new.pt"
ENCODER_PATH = REPO_ROOT / "Paper" / "proposed" / "modelCNN" / "trained_encoder_new.pth"

PATCH_SIZE = 8
HORIZON_PATCHES = 1
EMBED_DIM = 256
BATCH_SIZE = 32
DEFAULT_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RawPatchEmbedder(nn.Module):
    def __init__(self, input_dim: int, patch_size: int = PATCH_SIZE, d_model: int = EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(input_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, features = x.shape
        patches = steps // self.patch_size
        x = x[:, :patches * self.patch_size].reshape(
            batch, patches, self.patch_size * features
        )
        return self.projection(x)


class LinearForecastHead(nn.Module):
    def __init__(self, d_model: int = EMBED_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x)).squeeze(-1)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing synthetic dataset: {DATA_PATH}. Run Paper/dataset/data.py first."
        )
    data = torch.load(DATA_PATH, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()
    if X.dim() != 3 or Y.dim() != 2 or X.size(0) != Y.size(0) or X.size(1) != Y.size(1):
        raise ValueError(f"Expected X (N,T,F) and Y (N,T), got {tuple(X.shape)} and {tuple(Y.shape)}")
    split = int(0.8 * X.size(0))
    return X[:split], Y[:split], X[split:], Y[split:]


@torch.no_grad()
def encode(encoder: nn.Module, X: torch.Tensor) -> torch.Tensor:
    encoder.eval()
    outputs = []
    for (xb,) in DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False):
        outputs.append(encoder(xb.to(DEVICE)).cpu())
    return torch.cat(outputs, dim=0)


def load_tokens(X_train, X_test):
    if TOKEN_PATH.exists():
        token_data = torch.load(TOKEN_PATH, map_location="cpu")
        tokens = token_data["tokens"].float()
        if tokens.size(0) == X_train.size(0) + X_test.size(0):
            return tokens[:X_train.size(0)], tokens[X_train.size(0):]

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Missing encoder weights: {ENCODER_PATH}. Run Paper/proposed/modelCNN/training.py first."
        )
    encoder = DenseNetTokenEncoder(
        input_size=X_train.size(-1),
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    all_tokens = encode(encoder, torch.cat([X_train, X_test], dim=0))
    return all_tokens[:X_train.size(0)], all_tokens[X_train.size(0):]


def patch_targets(Y: torch.Tensor, patches: int) -> torch.Tensor:
    return reduce_to_patches(Y[:, :patches * PATCH_SIZE], PATCH_SIZE, reduction="mean")


def pairs(X, Y):
    return make_forecast_pairs(X, patch_targets(Y, X.size(1)), HORIZON_PATCHES)


def train_and_evaluate(variant, X_train, Y_train, X_test, Y_test, epochs):
    if variant in {"full", "no_pos", "no_transformer"}:
        train_tokens, test_tokens = load_tokens(X_train, X_test)
        X_tr, y_tr = pairs(train_tokens, Y_train)
        X_te, y_te = pairs(test_tokens, Y_test)
        if variant == "no_transformer":
            model = LinearForecastHead(X_tr.size(-1)).to(DEVICE)
        else:
            model = QuantFormer(
                input_dim=X_tr.size(-1),
                d_model=EMBED_DIM,
                num_heads=16,
                num_layers=4,
                d_ff=512,
                dropout=0.1,
                output_dim=1,
                use_positional_encoding=variant == "full",
            ).to(DEVICE)
    elif variant == "no_cnn":
        patcher = RawPatchEmbedder(input_dim=X_train.size(-1)).to(DEVICE)
        with torch.no_grad():
            train_tokens = patcher(X_train.to(DEVICE)).cpu()
            test_tokens = patcher(X_test.to(DEVICE)).cpu()
        X_tr, y_tr = pairs(train_tokens, Y_train)
        X_te, y_te = pairs(test_tokens, Y_test)
        model = QuantFormer(
            input_dim=EMBED_DIM,
            d_model=EMBED_DIM,
            num_heads=16,
            num_layers=4,
            d_ff=512,
            dropout=0.1,
            output_dim=1,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=0.5)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    mse_sum = mae_sum = count = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(DEVICE))
            target = yb.to(DEVICE)
            mse_sum += nn.functional.mse_loss(pred, target, reduction="sum").item()
            mae_sum += nn.functional.l1_loss(pred, target, reduction="sum").item()
            count += target.numel()
    mse = mse_sum / count
    mae = mae_sum / count
    return mse, mae, math.sqrt(mse)


def main():
    parser = argparse.ArgumentParser(description="Synthetic-data component ablation study.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Synthetic ablation on {DATA_PATH}")
    print(f"Device: {DEVICE} | train={tuple(X_train.shape)} test={tuple(X_test.shape)}")
    print("Variants: full, no_pos, no_transformer, no_cnn")
    for variant in ("full", "no_pos", "no_transformer", "no_cnn"):
        mse, mae, rmse = train_and_evaluate(
            variant, X_train, Y_train, X_test, Y_test, args.epochs
        )
        print(f"{variant:>18} | MSE={mse:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
