import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Paper.proposed.modelCNN.model import DenseNetTokenEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_windows.pt"
ENCODER_PATH = BASE_DIR / "electricity_encoder.pth"
OUTPUT_PATH = BASE_DIR / "electricity_tokens.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(encoder, X):
    dl = DataLoader(TensorDataset(X), batch_size=32)
    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            outs.append(encoder(xb.to(DEVICE)).cpu())
    return torch.cat(outs)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Run dataset_prep.py first.")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing encoder weights: {ENCODER_PATH}")

    DATA = torch.load(DATA_PATH, map_location="cpu")
    X_train = DATA["X_train"].float()
    X_test = DATA["X_test"].float()

    encoder = DenseNetTokenEncoder(
        input_size=6,
        patch_size=8,
        embed_dim=256,
    ).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()

    tokens_train = encode(encoder, X_train)
    tokens_test = encode(encoder, X_test)

    torch.save(
        {
            "tokens_train": tokens_train,
            "tokens_test": tokens_test,
            "Y_train": DATA["Y_train"],
            "Y_test": DATA["Y_test"],
            "meta": DATA["meta"],
        },
        OUTPUT_PATH,
    )

    print(f"Tokens saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
