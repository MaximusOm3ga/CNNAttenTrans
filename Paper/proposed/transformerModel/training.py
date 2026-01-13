import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Transformer.proposed.transformerModel.transformer import QuantFormer

CONTEXT_LEN = 10
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
D_MODEL = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

FORECAST_HORIZON = 1

HERE = os.path.dirname(__file__)
DEFAULT_TOKENS_PATH = "./synthetic_tokens_new.pt"
DEFAULT_OUTPUT_PATH = "./quantformer_price_new.pt"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


def _log(msg: str):
    print(f"[trainer] {msg}")


def _is_bad(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()


def _reduce_y_to_patches(y: torch.Tensor, P: int, patch_size: int, reduction: str = "mean") -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f"Expected y to be 2D (B,T), got {tuple(y.shape)}")

    B, T = y.shape
    T_eff = P * patch_size
    if T_eff <= 0:
        raise ValueError(f"Invalid T_eff={T_eff} from P={P}, patch_size={patch_size}")
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
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be (N,P,E), got {tuple(tokens.shape)}")
    if y_patched.dim() != 2:
        raise ValueError(f"y_patched must be (N,P), got {tuple(y_patched.shape)}")
    if tokens.size(0) != y_patched.size(0) or tokens.size(1) != y_patched.size(1):
        raise ValueError(f"tokens/y_patched mismatch: {tuple(tokens.shape)} vs {tuple(y_patched.shape)}")

    P = int(tokens.size(1))
    if P <= horizon:
        raise ValueError(f"Not enough patches P={P} for horizon={horizon}")

    x_f = tokens[:, : P - horizon, :]
    y_f = y_patched[:, horizon:]
    return x_f, y_f


def create_dataloaders(tokens_path: str, batch_size: int, horizon: int):
    obj = torch.load(tokens_path, map_location="cpu")
    tokens = obj["tokens"].float()
    Y = obj["Y"].float()
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}

    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens to be 3D (N,P,E), got {tuple(tokens.shape)}")
    if Y.dim() != 2:
        raise ValueError(f"Expected Y to be 2D (N,T), got {tuple(Y.shape)}")
    if tokens.size(0) != Y.size(0):
        raise ValueError(f"N mismatch: tokens N={tokens.size(0)} vs Y N={Y.size(0)}")

    patch_size = int(meta.get("patch_size", 4))
    P = int(tokens.size(1))

    Y_patched = _reduce_y_to_patches(Y, P=P, patch_size=patch_size, reduction="mean")

    if _is_bad(tokens) or _is_bad(Y_patched):
        raise RuntimeError("Found NaNs/Infs in tokens or patched targets")

    tokens_f, Y_f = _make_forecast_pairs(tokens, Y_patched, horizon=horizon)

    split = int(0.8 * tokens_f.size(0))
    train_ds = TensorDataset(tokens_f[:split], Y_f[:split])
    val_ds = TensorDataset(tokens_f[split:], Y_f[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = int(tokens_f.size(-1))
    _log(
        f"Loaded tokens: {tuple(tokens.shape)} | Y: {tuple(Y.shape)} -> Y_patched: {tuple(Y_patched.shape)} | "
        f"forecast: tokens_f {tuple(tokens_f.shape)} -> Y_f {tuple(Y_f.shape)} | patch_size={patch_size} | horizon={horizon}"
    )
    return train_loader, val_loader, input_dim


def train_quantformer_price(
    tokens_path: str = DEFAULT_TOKENS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    context_len: int = CONTEXT_LEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    horizon: int = FORECAST_HORIZON,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")

    train_loader, val_loader, input_dim = create_dataloaders(
        tokens_path=tokens_path,
        batch_size=batch_size,
        horizon=horizon,
    )

    sample_tokens, _ = next(iter(train_loader))
    P_eff = int(sample_tokens.size(1))
    if context_len > P_eff:
        _log(f"context_len={context_len} > available_patches={P_eff}; clamping.")
        context_len = P_eff

    _log(f"Model input_dim={input_dim}, context_len={context_len}, forecast_horizon={horizon}")

    model = QuantFormer(
        input_dim=input_dim,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        output_dim=1,
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y_true in train_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X)

            if y_pred.shape != y_true.shape:
                raise RuntimeError(f"Shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}")

            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y_true in val_loader:
                X = X.to(device)
                y_true = y_true.to(device)
                y_pred = model(X)

                if y_pred.shape != y_true.shape:
                    raise RuntimeError(f"(val) Shape mismatch: y_pred {tuple(y_pred.shape)} vs y_true {tuple(y_true.shape)}")

                val_loss += criterion(y_pred, y_true).item() * X.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch + 1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val and math.isfinite(val_loss):
            best_val = val_loss
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), output_path)
            _log(f"Saved best model -> {os.path.abspath(output_path)}")

    _log("Training complete.")


if __name__ == "__main__":
    train_quantformer_price()
