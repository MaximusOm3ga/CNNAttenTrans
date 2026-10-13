import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baselinePatchTST.model import PatchTST
from Paper.baselineTCN.model import TCN
from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches
from Paper.proposed.modelCNN.model import DenseNetTokenEncoder
from Paper.proposed.transformerModel.transformer import QuantFormer

DATA_TRAIN = REPO_ROOT / "Paper" / "dataset" / "paper_dataset.pt"
DATA_TEST = REPO_ROOT / "Paper" / "dataset" / "paper_new_test.pt"
ENCODER_WEIGHTS = REPO_ROOT / "Paper" / "proposed" / "modelCNN" / "trained_encoder_new.pth"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"

PATCH_SIZE = 8
EMBED_DIM = 256
BATCH_SIZE = 32
DEFAULT_EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = {3: 24, 5: 40}


def set_seed(seed: int = 43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_xy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    data = torch.load(path, map_location="cpu")
    X = data["X"].float()
    Y = data["Y"].float()
    if X.dim() != 3 or Y.dim() != 2 or X.shape[:2] != Y.shape:
        raise ValueError(f"Expected X (N,T,F) and Y (N,T), got {tuple(X.shape)} and {tuple(Y.shape)}")
    return X, Y


@torch.no_grad()
def encode_tokens(encoder, X):
    encoder.eval()
    outputs = []
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    for (xb,) in loader:
        outputs.append(encoder(xb.to(DEVICE)).float().cpu())
    return torch.cat(outputs, dim=0)


def load_encoder_tokens(X_train, X_test):
    if not ENCODER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Missing encoder weights: {ENCODER_WEIGHTS}. "
            "Train Paper/proposed/modelCNN/training.py first."
        )
    encoder = DenseNetTokenEncoder(
        input_size=X_train.size(-1),
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=DEVICE))
    return encode_tokens(encoder, X_train), encode_tokens(encoder, X_test)


def patch_targets(Y, patch_count):
    return reduce_to_patches(
        Y[:, :patch_count * PATCH_SIZE],
        PATCH_SIZE,
        reduction="mean",
    )


def token_pairs(tokens, Y, horizon_patches):
    y_patches = patch_targets(Y, tokens.size(1))
    return make_forecast_pairs(tokens, y_patches, horizon=horizon_patches)


def raw_pairs(X, Y, horizon_steps):
    X_forecast, Y_forecast = make_forecast_pairs(X, Y, horizon=horizon_steps)
    patch_count = X_forecast.size(1) // PATCH_SIZE
    target = patch_targets(Y_forecast[:, :patch_count * PATCH_SIZE], patch_count)
    return X_forecast[:, :patch_count * PATCH_SIZE], target


def metric_summary(predictions, targets):
    mse = nn.functional.mse_loss(predictions, targets).item()
    mae = nn.functional.l1_loss(predictions, targets).item()
    return mse, mae, math.sqrt(mse)


class TCNPatchForecast(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.tcn = TCN(input_dim=input_dim)

    def forward(self, x):
        return reduce_to_patches(self.tcn(x), PATCH_SIZE, reduction="mean")


def train_model(model, train_loader, epochs):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=0.5)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            if prediction.shape != yb.shape:
                raise RuntimeError(
                    f"Training shape mismatch: prediction={tuple(prediction.shape)} "
                    f"target={tuple(yb.shape)}"
                )
            loss = criterion(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"    epoch {epoch + 1}/{epochs}", flush=True)
    return model


def split_loader(X, Y):
    split = int(0.8 * X.size(0))
    train_loader = DataLoader(
        TensorDataset(X[:split], Y[:split]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[split:], Y[split:]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader


def train_model_with_validation(model, X, Y, epochs):
    train_loader, val_loader = split_loader(X, Y)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=0.5)
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            if prediction.shape != yb.shape:
                raise RuntimeError(
                    f"Training shape mismatch: prediction={tuple(prediction.shape)} "
                    f"target={tuple(yb.shape)}"
                )
            loss = criterion(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_sum += criterion(model(xb), yb).item() * xb.size(0)
                val_count += xb.size(0)
        val_loss = val_sum / val_count
        scheduler.step()

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"    epoch {epoch + 1}/{epochs} | val={val_loss:.6f}", flush=True)

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    for xb, yb in test_loader:
        predictions.append(model(xb.to(DEVICE)).cpu())
        targets.append(yb)
    return metric_summary(torch.cat(predictions), torch.cat(targets))


def train_and_evaluate_horizon(horizon_patches, horizon_steps, X_train, Y_train, X_test, Y_test, epochs):
    print(f"\n=== Horizon h={horizon_patches} ({horizon_steps} timesteps) ===")

    train_tokens, test_tokens = load_encoder_tokens(X_train, X_test)
    token_train_x, token_train_y = token_pairs(train_tokens, Y_train, horizon_patches)
    token_test_x, token_test_y = token_pairs(test_tokens, Y_test, horizon_patches)

    raw_train_x, raw_train_y = raw_pairs(X_train, Y_train, horizon_steps)
    raw_test_x, raw_test_y = raw_pairs(X_test, Y_test, horizon_steps)

    results = {}

    set_seed(43)
    print("  Training QuantFormer...")
    quantformer = QuantFormer(
        input_dim=EMBED_DIM,
        d_model=EMBED_DIM,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    )
    quantformer = train_model_with_validation(
        quantformer,
        token_train_x,
        token_train_y,
        epochs,
    )
    results["Proposed QuantFormer"] = evaluate_model(
        quantformer,
        DataLoader(TensorDataset(token_test_x, token_test_y), batch_size=BATCH_SIZE),
    )

    set_seed(43)
    print("  Training TCN...")
    tcn = train_model_with_validation(
        TCNPatchForecast(input_dim=X_train.size(-1)),
        raw_train_x,
        raw_train_y,
        epochs,
    )
    results["TCN"] = evaluate_model(
        tcn,
        DataLoader(TensorDataset(raw_test_x, raw_test_y), batch_size=BATCH_SIZE),
    )

    set_seed(43)
    print("  Training PatchTST...")
    patchtst = train_model_with_validation(
        PatchTST(
            patch_size=PATCH_SIZE,
            input_dim=X_train.size(-1),
            d_model=EMBED_DIM,
            num_heads=16,
            num_layers=4,
            d_ff=512,
            dropout=0.1,
        ),
        raw_train_x,
        raw_train_y,
        epochs,
    )
    results["PatchTST"] = evaluate_model(
        patchtst,
        DataLoader(TensorDataset(raw_test_x, raw_test_y), batch_size=BATCH_SIZE),
    )

    for name, (mse, mae, rmse) in results.items():
        print(f"  {name}: MSE={mse:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")
    return results


def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser(
        description="Train and evaluate all synthetic forecasting models at h=3 and h=5."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5], choices=[3, 5])
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    set_seed(44)
    X_train, Y_train = load_xy(DATA_TRAIN)
    X_test, Y_test = load_xy(DATA_TEST)
    print(f"Device: {DEVICE}")
    print(f"Train source: {DATA_TRAIN} | X={tuple(X_train.shape)}")
    print(f"Test source:  {DATA_TEST} | X={tuple(X_test.shape)}")
    print(f"Epochs per model and horizon: {args.epochs}")

    all_results = {}
    for horizon in args.horizons:
        all_results[horizon] = train_and_evaluate_horizon(
            horizon,
            HORIZONS[horizon],
            X_train,
            Y_train,
            X_test,
            Y_test,
            args.epochs,
        )

    print("\n=== Final synthetic long-horizon test metrics ===")
    for horizon, results in all_results.items():
        print(f"\nh={horizon} ({HORIZONS[horizon]} timesteps)")
        for name, (mse, mae, rmse) in results.items():
            print(f"{name:>22} | MSE={mse:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
