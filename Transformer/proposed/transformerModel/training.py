import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Transformer.proposed.transformerModel.quantformer import QuantFormer

CONTEXT_LEN = 10
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
D_MODEL = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

HERE = os.path.dirname(__file__)
DEFAULT_TOKENS_PATH = "./synthetic_tokens.pt"
DEFAULT_OUTPUT_PATH = "./quantformer_price.pt"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


def _log(msg: str):
    print(f"[trainer] {msg}")


def _is_bad(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()


def _load_pt_with_meta(path: str):
    abspath = os.path.abspath(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
        meta = obj.get("meta", {})
    else:
        data = obj
        meta = {}
    if not torch.is_tensor(data):
        raise ValueError(f"Loaded object is not a tensor at {abspath}")
    _log(f"Loaded {abspath} | shape={tuple(data.shape)} | dtype={data.dtype} | meta.source={meta.get('source', 'unknown')}")
    if _is_bad(data):
        raise ValueError(f"Tensor contains NaN/Inf at {abspath}")
    return data, meta


def create_dataloaders(tokens_path, batch_size):
    data = torch.load(tokens_path)
    tokens = data["tokens"].float()
    Y      = data["Y"].float()
    split = int(0.8 * tokens.size(0))
    train_ds = TensorDataset(tokens[:split], Y[:split])
    val_ds   = TensorDataset(tokens[split:], Y[split:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        tokens.size(-1),
    )


def train_quantformer_price(
    tokens_path: str = DEFAULT_TOKENS_PATH,
    prices_path: str = "./prices.pt",
    output_path: str = DEFAULT_OUTPUT_PATH,
    context_len: int = CONTEXT_LEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")
    train_loader, val_loader, input_dim = create_dataloaders(
        tokens_path=tokens_path,
        batch_size=batch_size
    )
    _log(f"Model input_dim={input_dim}, context_len={context_len}")
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
