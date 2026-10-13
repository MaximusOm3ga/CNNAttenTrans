import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches
from Paper.proposed.modelCNN.model import AutoencoderTokenExtractor, TokenSelfAttention
from Paper.proposed.transformerModel.transformer import QuantFormer

DATA_PATH = REPO_ROOT / "Paper" / "dataset" / "paper_dataset.pt"
TEST_PATH = REPO_ROOT / "Paper" / "dataset" / "paper_new_test.pt"
PATCH_SIZE = 8
EMBED_DIM = 256
HORIZON_PATCHES = 1
BATCH_SIZE = 32
ENCODER_EPOCHS = 2000
TRANSFORMER_EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ComponentEncoder(nn.Module):
    def __init__(self, input_dim: int, use_dense: bool, use_pool: bool):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.use_pool = use_pool
        self.conv0 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(64)

        if use_dense:
            self.dense_blocks = nn.ModuleList()
            channels = 64
            for _ in range(3):
                block = nn.ModuleList()
                for _ in range(3):
                    block.append(nn.Sequential(
                        nn.BatchNorm1d(channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(channels, 32, kernel_size=3, padding=1),
                    ))
                    channels += 32
                self.dense_blocks.append(block)
            self.plain_layers = None
        else:
            channels = 64
            self.dense_blocks = None
            self.plain_layers = nn.ModuleList([
                nn.Sequential(
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                )
                for _ in range(9)
            ])

        self.final_channels = channels
        self.pool_attn = nn.Conv1d(channels, 1, kernel_size=1)
        self.token_proj = nn.Linear(channels, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, features = x.shape
        patches = steps // self.patch_size
        x = x[:, :patches * self.patch_size].reshape(
            batch * patches, self.patch_size, features
        ).transpose(1, 2)
        x = torch.relu(self.bn0(self.conv0(x)))

        if self.dense_blocks is not None:
            for block in self.dense_blocks:
                for layer in block:
                    x = torch.cat([x, layer(x)], dim=1)
        else:
            for layer in self.plain_layers:
                x = x + layer(x)

        if self.use_pool:
            weights = torch.softmax(self.pool_attn(x), dim=-1)
            x = (x * weights).sum(dim=-1)
        else:
            x = x.mean(dim=-1)
        return self.token_proj(x).reshape(batch, patches, EMBED_DIM)


class EncoderAutoencoder(nn.Module):
    def __init__(self, input_dim: int, use_dense: bool, use_pool: bool, use_token_attention: bool):
        super().__init__()
        self.encoder = ComponentEncoder(input_dim, use_dense, use_pool)
        self.use_token_attention = use_token_attention
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, EMBED_DIM))
        self.token_attention = TokenSelfAttention(EMBED_DIM) if use_token_attention else nn.Identity()
        self.decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, PATCH_SIZE * input_dim),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):
        tokens = self.encoder(x)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        tokens = self.token_attention(tokens)
        reconstruction = self.decoder(tokens).reshape(
            x.size(0), tokens.size(1), PATCH_SIZE, self.input_dim
        )
        return reconstruction


def load_data():
    if not DATA_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Run Paper/dataset/data.py to create both synthetic dataset files.")
    train_data = torch.load(DATA_PATH, map_location="cpu")
    test_data = torch.load(TEST_PATH, map_location="cpu")
    X, Y = train_data["X"].float(), train_data["Y"].float()
    X_test, Y_test = test_data["X"].float(), test_data["Y"].float()
    split = int(0.8 * X.size(0))
    return X[:split], Y[:split], X[split:], Y[split:], X_test, Y_test


def train_encoder(variant, X_train, epochs):
    settings = {
        "full": (True, True, True),
        "no_dense": (False, True, True),
        "no_pool": (True, False, True),
        "no_token_attention": (True, True, False),
    }
    use_dense, use_pool, use_token_attention = settings[variant]
    if variant == "full":
        model = AutoencoderTokenExtractor(
            input_size=X_train.size(-1),
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
        ).to(DEVICE)
    else:
        model = EncoderAutoencoder(
            X_train.size(-1), use_dense, use_pool, use_token_attention
        ).to(DEVICE)
    split = int(0.9 * X_train.size(0))
    train_loader = DataLoader(
        TensorDataset(X_train[:split]), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_train[split:]), batch_size=BATCH_SIZE, shuffle=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()
    best_val = float("inf")
    best_encoder = None
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            effective = xb[:, :(xb.size(1) // PATCH_SIZE) * PATCH_SIZE]
            target = effective.reshape(
                xb.size(0), effective.size(1) // PATCH_SIZE, PATCH_SIZE, xb.size(2)
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(effective)
            if isinstance(reconstruction, tuple):
                reconstruction = reconstruction[0]
            loss = criterion(reconstruction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                effective = xb[:, :(xb.size(1) // PATCH_SIZE) * PATCH_SIZE]
                target = effective.reshape(
                    xb.size(0), effective.size(1) // PATCH_SIZE, PATCH_SIZE, xb.size(2)
                )
                reconstruction = model(effective)
                if isinstance(reconstruction, tuple):
                    reconstruction = reconstruction[0]
                val_sum += criterion(reconstruction, target).item() * xb.size(0)
                val_count += xb.size(0)
        val_loss = val_sum / val_count
        scheduler.step()
        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            best_encoder = {
                key: value.detach().cpu().clone()
                for key, value in model.encoder.state_dict().items()
            }
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            elapsed = time.perf_counter() - started
            print(
                f"    encoder {variant}: epoch {epoch + 1}/{epochs} | "
                f"val={val_loss:.6f} | elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    model.encoder.load_state_dict(best_encoder)
    return model.encoder


@torch.no_grad()
def encode(encoder, X):
    encoder.eval()
    outputs = []
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    for (xb,) in loader:
        outputs.append(encoder(xb.to(DEVICE)).cpu())
    return torch.cat(outputs)


def forecast_pairs(tokens, Y):
    targets = reduce_to_patches(Y[:, :tokens.size(1) * PATCH_SIZE], PATCH_SIZE, "mean")
    return make_forecast_pairs(tokens, targets, HORIZON_PATCHES)


def train_forecaster(tokens_train, y_train, epochs):
    split = int(0.8 * tokens_train.size(0))
    X_tr, Y_tr = forecast_pairs(tokens_train[:split], y_train[:split])
    X_val, Y_val = forecast_pairs(tokens_train[split:], y_train[split:])
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE)
    model = QuantFormer(
        input_dim=tokens_train.size(-1), d_model=EMBED_DIM, num_heads=16,
        num_layers=4, d_ff=512, dropout=0.1, output_dim=1,
    ).to(DEVICE)
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
            loss = criterion(model(xb), yb)
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    transformer: epoch {epoch + 1}/{epochs} | val={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, tokens, Y):
    X_test, Y_test = forecast_pairs(tokens, Y)
    loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE)
    model.eval()
    predictions, targets = [], []
    for xb, yb in loader:
        predictions.append(model(xb.to(DEVICE)).cpu())
        targets.append(yb)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mse = nn.functional.mse_loss(prediction, target).item()
    mae = nn.functional.l1_loss(prediction, target).item()
    return mse, mae, math.sqrt(mse)


def main():
    parser = argparse.ArgumentParser(description="Faithful two-stage synthetic encoder ablation.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--encoder-epochs", type=int, default=ENCODER_EPOCHS)
    parser.add_argument("--transformer-epochs", type=int, default=TRANSFORMER_EPOCHS)
    args = parser.parse_args()
    if args.runs <= 0 or args.encoder_epochs <= 0 or args.transformer_epochs <= 0:
        raise ValueError("runs and epoch counts must be positive")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
    variants = ("full", "no_dense", "no_pool", "no_token_attention")
    results = {variant: [] for variant in variants}
    print(f"Faithful synthetic ablation | device={DEVICE}")
    print(f"Encoder epochs={args.encoder_epochs} | Transformer epochs={args.transformer_epochs}")

    for run in range(args.runs):
        seed = BASE_SEED + run
        print(f"\n=== Run {run + 1}/{args.runs} | internal seed={seed} ===")
        for variant in variants:
            set_seed(seed)
            print(f"  Training encoder variant: {variant}")
            encoder = train_encoder(variant, X_train, args.encoder_epochs)
            tokens_train = encode(encoder, X_train)
            tokens_val = encode(encoder, X_val)
            tokens_test = encode(encoder, X_test)
            model = train_forecaster(
                torch.cat([tokens_train, tokens_val]),
                torch.cat([Y_train, Y_val]),
                args.transformer_epochs,
            )
            metrics = evaluate(model, tokens_test, Y_test)
            results[variant].append(metrics)
            print(
                f"{variant:>20} | MSE={metrics[0]:.6f} | "
                f"MAE={metrics[1]:.6f} | RMSE={metrics[2]:.6f}"
            )

    print("\n=== Aggregate results across runs ===")
    print("Values are mean +/- standard deviation.")
    for variant in variants:
        values = torch.tensor(results[variant], dtype=torch.float64)
        mean = values.mean(0)
        std = values.std(0, unbiased=args.runs > 1)
        print(
            f"{variant:>20} | MSE={mean[0]:.6f} +/- {std[0]:.6f} | "
            f"MAE={mean[1]:.6f} +/- {std[1]:.6f} | "
            f"RMSE={mean[2]:.6f} +/- {std[2]:.6f}"
        )


if __name__ == "__main__":
    main()
