import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.modelCNN.model import DenseNetTokenEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_windows.pt"
ENCODER_PATH = BASE_DIR / "weather_encoder.pth"
OUTPUT_PATH = BASE_DIR / "weather_tokens.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 8
EMBED_DIM = 256


def encode_weather_tokens(batch_size: int = 64) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing weather windows dataset: {DATA_PATH}. Run dataset_prep.py first.")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Missing weather encoder weights: {ENCODER_PATH}. Run train_weather_encoder.py first."
        )

    data = torch.load(DATA_PATH, map_location="cpu")
    X_train = data["X_train"].float()
    X_test = data["X_test"].float()
    Y_train = data["Y_train"].float()
    Y_test = data["Y_test"].float()

    encoder = DenseNetTokenEncoder(
        input_size=X_train.size(-1),
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()

    def _encode(name, X):
        dl = DataLoader(
            TensorDataset(X),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
        )
        outs = []
        total = len(dl)
        print(f"Encoding {name}: {len(X)} windows in {total} batches on {DEVICE}", flush=True)
        with torch.no_grad():
            for batch_idx, (xb,) in enumerate(dl, start=1):
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    encoded = encoder(xb)
                outs.append(encoded.float().cpu())
                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total:
                    print(
                        f"{name}: batch {batch_idx}/{total} "
                        f"({batch_idx / total:.0%})",
                        flush=True,
                    )
        return torch.cat(outs, dim=0)

    tokens_train = _encode("train", X_train)
    tokens_test = _encode("test", X_test)

    torch.save(
        {
            "tokens_train": tokens_train,
            "tokens_test": tokens_test,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "meta": {
                "dataset": "WeatherDS",
                "patch_size": PATCH_SIZE,
                "embed_dim": EMBED_DIM,
            },
        },
        OUTPUT_PATH,
    )

    print(f"Saved weather tokens -> {OUTPUT_PATH}")
    print(f"tokens_train: {tokens_train.shape}, tokens_test: {tokens_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CNN tokens for WeatherDS.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    encode_weather_tokens(batch_size=args.batch_size)
