import os
import torch
from Transformer.proposed.modelCNN.model import DenseNetTokenEncoder

EMBED_DIM = 256

HERE = os.path.dirname(__file__)
DATASET_PATH = "../../dataset/paper_dataset.pt"
ENCODER_WEIGHTS = "./trained_encoder_new.pth"
OUTPUT_PATH = "../transformerModel/synthetic_tokens_new.pt"


data = torch.load(DATASET_PATH)
X = data["X"]

if X.dim() != 3:
    raise ValueError(f"Expected X to be 3D (N, T, D), got {X.shape}")

encoder = DenseNetTokenEncoder(input_size=X.shape[-1], patch_size=4, embed_dim=EMBED_DIM)
encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location="cpu"))
encoder.eval()

with torch.no_grad():
    tokens = encoder(X.float())

torch.save(
    {
        "tokens": tokens,
        "Y": data["Y"],
        "meta": {
            "source": "synthetic",
            "patch_size": 4,
            "aligned": "timestep",
        },
    },
    OUTPUT_PATH,
)

print(f"Saved tokens -> {OUTPUT_PATH}")
