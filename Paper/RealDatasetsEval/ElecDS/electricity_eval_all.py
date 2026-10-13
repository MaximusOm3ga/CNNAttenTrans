import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.baseline_utils import make_forecast_pairs, reduce_to_patches
from Paper.baselinePatchTST.model import PatchTST
from Paper.baselineTCN.model import TCN
from Paper.proposed.transformerModel.transformer import QuantFormer

PATCH_SIZE = 8
FORECAST_HORIZON_PATCHES = 1
BATCH_SIZE = 32
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ELEC_DIR = Path(__file__).resolve().parent / "proposed_rl_ds"
WINDOWS_PATH = ELEC_DIR / "electricity_windows.pt"
TOKENS_PATH = ELEC_DIR / "electricity_tokens.pt"
PROPOSED_WEIGHTS = ELEC_DIR / "quantformer_electricity.pth"
TCN_WEIGHTS = ELEC_DIR / "baseline_tcn_electricity.pth"
PATCHTST_WEIGHTS = ELEC_DIR / "baseline_patchtst_electricity.pth"


def _load_windows():
    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Missing electricity windows dataset: {WINDOWS_PATH}")
    data = torch.load(WINDOWS_PATH, map_location="cpu")
    meta = data.get("meta", {})
    if meta.get("normalization") != "train_only" or "target_mean" not in meta:
        raise ValueError(
            "Dataset was not prepared with train-only input and target normalization. Rebuild the ElecDS dataset using "
            "Paper/RealDatasetsEval/ElecDS/proposed_rl_ds/dataset_prep.py"
        )

    X_train = data["X_train"].float()
    Y_train = data["Y_train"].float()
    X_test = data["X_test"].float()
    Y_test = data["Y_test"].float()
    return X_train, Y_train, X_test, Y_test


def _load_tokens():
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(f"Missing electricity token dataset: {TOKENS_PATH}")
    data = torch.load(TOKENS_PATH, map_location="cpu")
    return data["tokens_test"].float(), data["Y_test"].float()


def _metric_summary(model_name: str, mse: float, mae: float) -> str:
    rmse = math.sqrt(mse)
    return (
        f"{model_name}\n"
        f"  MSE : {mse:.6f}\n"
        f"  MAE : {mae:.6f}\n"
        f"  RMSE: {rmse:.6f}"
    )


def evaluate_proposed(weights_path: Path = PROPOSED_WEIGHTS, batch_size: int = BATCH_SIZE):
    tokens, Y_raw = _load_tokens()
    P = tokens.size(1)
    Y_patched = Y_raw[:, : P * PATCH_SIZE].view(Y_raw.size(0), P, PATCH_SIZE).mean(dim=-1)
    X = tokens[:, :-FORECAST_HORIZON_PATCHES]
    Y = Y_patched[:, FORECAST_HORIZON_PATCHES:]

    dl = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)
    model = QuantFormer(
        input_dim=tokens.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        output_dim=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            if preds.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: preds {tuple(preds.shape)} vs yb {tuple(yb.shape)}")
            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count
    return _metric_summary("Proposed QuantFormer (ElecDS)", mse, mae)


def _train_tcn_model(train_X, train_Y, weights_path: Path, epochs: int = EPOCHS):
    horizon_steps = PATCH_SIZE * FORECAST_HORIZON_PATCHES
    X_f, Y_f_full = make_forecast_pairs(train_X, train_Y, horizon=horizon_steps)
    Y_f_patch = reduce_to_patches(Y_f_full, PATCH_SIZE, reduction="mean")
    split = int(0.8 * X_f.size(0))
    train_dl = DataLoader(
        TensorDataset(X_f[:split], Y_f_patch[:split]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(X_f[split:], Y_f_patch[split:]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TCN(input_dim=train_X.size(-1)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            pred_patch = reduce_to_patches(pred, PATCH_SIZE, reduction="mean")
            loss = criterion(pred_patch, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred_patch = reduce_to_patches(model(xb), PATCH_SIZE, reduction="mean")
                val_loss += criterion(pred_patch, yb).item() * yb.size(0)
                val_count += yb.size(0)
        val_loss /= val_count
        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)

    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
    return model


def evaluate_tcn(weights_path: Path = TCN_WEIGHTS, batch_size: int = BATCH_SIZE):
    if not weights_path.exists():
        X_train, Y_train, _, _ = _load_windows()
        _train_tcn_model(X_train, Y_train, weights_path)

    _, _, X_test, Y_test = _load_windows()
    horizon_steps = PATCH_SIZE * FORECAST_HORIZON_PATCHES
    X_f, Y_f_full = make_forecast_pairs(X_test, Y_test, horizon=horizon_steps)
    ds = TensorDataset(X_f, Y_f_full)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = TCN(input_dim=X_test.size(-1)).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds_t = model(xb)
            preds_patch = reduce_to_patches(preds_t, PATCH_SIZE, reduction="mean")
            yb_patch = reduce_to_patches(yb, PATCH_SIZE, reduction="mean")
            total_mse += mse_loss(preds_patch, yb_patch).item()
            total_mae += mae_loss(preds_patch, yb_patch).item()
            total_count += yb_patch.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count
    return _metric_summary("TCN (ElecDS)", mse, mae)


def _train_patchtst_model(train_X, train_Y, weights_path: Path, epochs: int = EPOCHS):
    horizon_steps = PATCH_SIZE * FORECAST_HORIZON_PATCHES
    X_f, Y_f_full = make_forecast_pairs(train_X, train_Y, horizon=horizon_steps)
    T_in = int(X_f.size(1))
    P_in = T_in // PATCH_SIZE
    if P_in <= FORECAST_HORIZON_PATCHES:
        raise ValueError(f"Not enough input patches P_in={P_in} for forecast horizon {FORECAST_HORIZON_PATCHES}")
    P_target = P_in - FORECAST_HORIZON_PATCHES
    T_target = P_target * PATCH_SIZE
    Y_f_full = Y_f_full[:, :T_target]
    Y_f = reduce_to_patches(Y_f_full, PATCH_SIZE, reduction="mean")

    split = int(0.8 * X_f.size(0))
    train_loader = DataLoader(
        TensorDataset(X_f[:split], Y_f[:split]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_f[split:], Y_f[split:]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=train_X.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)[:, :-FORECAST_HORIZON_PATCHES]
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)[:, :-FORECAST_HORIZON_PATCHES]
                val_loss += criterion(preds, yb).item() * yb.size(0)
                val_count += yb.size(0)

        val_loss /= val_count
        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)

    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
    return model


def evaluate_patchtst(weights_path: Path = PATCHTST_WEIGHTS, batch_size: int = BATCH_SIZE):
    if not weights_path.exists():
        X_train, Y_train, _, _ = _load_windows()
        _train_patchtst_model(X_train, Y_train, weights_path)

    _, _, X_test, Y_test = _load_windows()
    horizon_steps = PATCH_SIZE * FORECAST_HORIZON_PATCHES
    X_f, Y_f_full = make_forecast_pairs(X_test, Y_test, horizon=horizon_steps)
    T_in = int(X_f.size(1))
    P_in = T_in // PATCH_SIZE
    if P_in <= FORECAST_HORIZON_PATCHES:
        raise ValueError(f"Not enough input patches P_in={P_in} for forecast horizon {FORECAST_HORIZON_PATCHES}")
    P_target = P_in - FORECAST_HORIZON_PATCHES
    T_target = P_target * PATCH_SIZE
    Y_f_full = Y_f_full[:, :T_target]
    Y_f = reduce_to_patches(Y_f_full, PATCH_SIZE, reduction="mean")

    ds = TensorDataset(X_f, Y_f)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = PatchTST(
        patch_size=PATCH_SIZE,
        input_dim=X_test.size(-1),
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)[:, :-FORECAST_HORIZON_PATCHES]
            total_mse += mse_loss(preds, yb).item()
            total_mae += mae_loss(preds, yb).item()
            total_count += yb.numel()

    mse = total_mse / total_count
    mae = total_mae / total_count
    return _metric_summary("PatchTST (ElecDS)", mse, mae)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the proposed QuantFormer, PatchTST, and TCN on the ElecDS dataset.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs used if baseline weights are not present.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing weights; do not train missing baseline models.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Dataset folder: {ELEC_DIR}")

    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {WINDOWS_PATH}. Run dataset_prep.py first.")

    data = torch.load(WINDOWS_PATH, map_location="cpu")
    if data.get("meta", {}).get("normalization") != "train_only":
        print("Detected train/test leakage in the ElecDS dataset; rebuilding with train-only normalization...")
        import runpy
        runpy.run_path(str(ELEC_DIR / "dataset_prep.py"), run_name="__main__")

    if not args.skip_train:
        X_train, Y_train, _, _ = _load_windows()
        if not PROPOSED_WEIGHTS.exists():
            print(f"Training QuantFormer weights to {PROPOSED_WEIGHTS}")
            import runpy
            original_argv = list(sys.argv)
            sys.argv = [
                str(ELEC_DIR / "train_quantformer_electricity.py"),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ]
            try:
                runpy.run_path(str(ELEC_DIR / "train_quantformer_electricity.py"), run_name="__main__")
            finally:
                sys.argv = original_argv
        if not TCN_WEIGHTS.exists():
            print(f"Training TCN weights to {TCN_WEIGHTS}")
            _train_tcn_model(X_train, Y_train, TCN_WEIGHTS, epochs=args.epochs)
        if not PATCHTST_WEIGHTS.exists():
            print(f"Training PatchTST weights to {PATCHTST_WEIGHTS}")
            _train_patchtst_model(X_train, Y_train, PATCHTST_WEIGHTS, epochs=args.epochs)

    print("\n" + evaluate_proposed(PROPOSED_WEIGHTS, batch_size=args.batch_size))
    print("\n" + evaluate_tcn(TCN_WEIGHTS, batch_size=args.batch_size))
    print("\n" + evaluate_patchtst(PATCHTST_WEIGHTS, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
