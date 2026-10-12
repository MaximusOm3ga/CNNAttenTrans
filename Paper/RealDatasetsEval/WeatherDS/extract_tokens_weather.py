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


def encode_weather_tokens() -> None:
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

    def _encode(X):
        dl = DataLoader(TensorDataset(X), batch_size=32, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in dl:
                outs.append(encoder(xb.to(DEVICE)).cpu())
        return torch.cat(outs, dim=0)

    tokens_train = _encode(X_train)
    tokens_test = _encode(X_test)

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
    encode_weather_tokens()
