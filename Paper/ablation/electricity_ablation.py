import math
import runpy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.transformerModel.transformer import QuantFormer

PATCH_SIZE = 8
HORIZON = 1
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ELEC_DIR = Path(__file__).resolve().parents[1] / "RealDatasetsEval" / "ElecDS" / "proposed_rl_ds"
WINDOWS_PATH = ELEC_DIR / "electricity_windows.pt"
TOKENS_PATH = ELEC_DIR / "electricity_tokens.pt"


class RawPatchEmbedder(nn.Module):
    def __init__(self, input_dim: int = 6, patch_size: int = PATCH_SIZE, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Linear(input_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape
        P = T // self.patch_size
        x = x[:, :P * self.patch_size, :].reshape(B, P, self.patch_size * F)
        return self.net(x)


class LinearForecastHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return self.head(x).squeeze(-1)


def ensure_elec_data():
    if not WINDOWS_PATH.exists():
        dataset_prep = ELEC_DIR / "dataset_prep.py"
        if not dataset_prep.exists():
            raise FileNotFoundError(f"Missing dataset prep script: {dataset_prep}")
        runpy.run_path(str(dataset_prep), run_name="__main__")
    if not TOKENS_PATH.exists():
        token_ex = ELEC_DIR / "extract_tokens_electricity.py"
        if not token_ex.exists():
            raise FileNotFoundError(f"Missing token extraction script: {token_ex}")
        runpy.run_path(str(token_ex), run_name="__main__")


def load_windows():
    ensure_elec_data()
    data = torch.load(WINDOWS_PATH, map_location="cpu")
    X_train = data["X_train"].float()
    Y_train = data["Y_train"].float()
    X_test = data["X_test"].float()
    Y_test = data["Y_test"].float()
    return X_train, Y_train, X_test, Y_test


def load_tokens():
    ensure_elec_data()
    data = torch.load(TOKENS_PATH, map_location="cpu")
    return data["tokens_train"].float(), data["Y_train"].float(), data["tokens_test"].float(), data["Y_test"].float()


def reduce_to_patches(y: torch.Tensor, patch_size: int = PATCH_SIZE):
    B, T = y.shape
    P = T // patch_size
    y = y[:, :P * patch_size].reshape(B, P, patch_size)
    return y.mean(dim=-1)


def build_forecast_pairs(tokens: torch.Tensor, y_patched: torch.Tensor, horizon: int = HORIZON):
    if tokens.size(1) <= horizon:
        raise ValueError(f"Not enough patches: {tokens.size(1)} <= horizon {horizon}")
    return tokens[:, :tokens.size(1) - horizon, :], y_patched[:, horizon:]


def make_loader(X, Y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_variant(name: str, X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor, Y_test: torch.Tensor, variant: str):
    if variant in {"full", "no_pos"}:
        tokens_train, _, tokens_test, _ = load_tokens()
        y_train = reduce_to_patches(Y_train, PATCH_SIZE)
        y_test = reduce_to_patches(Y_test, PATCH_SIZE)
        X_tr, y_tr = build_forecast_pairs(tokens_train, y_train)
        X_te, y_te = build_forecast_pairs(tokens_test, y_test)

        model = QuantFormer(
            input_dim=X_tr.size(-1),
            d_model=256,
            num_heads=16,
            num_layers=4,
            d_ff=512,
            dropout=0.1,
            output_dim=1,
            use_positional_encoding=(variant == "full"),
        ).to(DEVICE)

        train_loader = make_loader(X_tr, y_tr)
        test_loader = make_loader(X_te, y_te, shuffle=False)

    elif variant == "no_transformer":
        tokens_train, _, tokens_test, _ = load_tokens()
        y_train = reduce_to_patches(Y_train, PATCH_SIZE)
        y_test = reduce_to_patches(Y_test, PATCH_SIZE)
        X_tr, y_tr = build_forecast_pairs(tokens_train, y_train)
        X_te, y_te = build_forecast_pairs(tokens_test, y_test)
        model = LinearForecastHead(d_model=X_tr.size(-1)).to(DEVICE)
        train_loader = make_loader(X_tr, y_tr)
        test_loader = make_loader(X_te, y_te, shuffle=False)

    elif variant == "no_cnn":
        X_train_dev = X_train.to(DEVICE)
        X_test_dev = X_test.to(DEVICE)
        patcher = RawPatchEmbedder(input_dim=X_train.size(-1), patch_size=PATCH_SIZE, d_model=256).to(DEVICE)
        y_train = reduce_to_patches(Y_train, PATCH_SIZE)
        y_test = reduce_to_patches(Y_test, PATCH_SIZE)
        X_tr = patcher(X_train_dev).detach()
        X_te = patcher(X_test_dev).detach()
        X_tr, y_tr = build_forecast_pairs(X_tr, y_train)
        X_te, y_te = build_forecast_pairs(X_te, y_test)
        model = QuantFormer(
            input_dim=X_tr.size(-1),
            d_model=256,
            num_heads=16,
            num_layers=4,
            d_ff=512,
            dropout=0.1,
            output_dim=1,
            use_positional_encoding=True,
        ).to(DEVICE)
        train_loader = make_loader(X_tr, y_tr)
        test_loader = make_loader(X_te, y_te, shuffle=False)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape != yb.shape:
                raise RuntimeError(f"{name} shape mismatch: pred {tuple(pred.shape)} vs y {tuple(yb.shape)}")
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            total_mse += nn.MSELoss(reduction="sum")(pred, yb).item()
            total_mae += nn.L1Loss(reduction="sum")(pred, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count
    rmse = math.sqrt(mse)
    print(f"{name:>18} | MSE={mse:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}")
    return mse, mae, rmse


def main():
    X_train, Y_train, X_test, Y_test = load_windows()
    variants = ["full", "no_pos", "no_transformer", "no_cnn"]
    print("Ablation study on Electricity Load Diagrams (ElecDS)")
    print("Variants: full, no_pos, no_transformer, no_cnn")
    for variant in variants:
        train_variant(f"variant:{variant}", X_train, Y_train, X_test, Y_test, variant)


if __name__ == "__main__":
    main()
