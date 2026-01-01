import torch
from Transformer.proposed.modelCNN.model import DenseNetTokenEncoder
from Transformer.proposed.transformerModel.quantformer import QuantFormer

EMBED_DIM = 256
D_MODEL = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DATASET_PATH = "../../dataset/paper_new_test.pt"
ENCODER_WEIGHTS   = r"C:\Coding\REPOS\QUANT_ALGORITHMS\Transformer\proposed\modelCNN\trained_encoder.pth"
MODEL_WEIGHTS     = "quantformer_price.pt"

data = torch.load(TEST_DATASET_PATH)

X_test = data["X"].float()
Y_test = data["Y"].float()

assert X_test.dim() == 3
assert Y_test.dim() == 2
assert X_test.size(0) == Y_test.size(0)
assert X_test.size(1) == Y_test.size(1)

N_test, T, F = X_test.shape

encoder = DenseNetTokenEncoder(
    input_size=F,
    patch_size=1,
    embed_dim=EMBED_DIM,
)

encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=DEVICE))
encoder.to(DEVICE)
encoder.eval()

with torch.no_grad():
    tokens_test = encoder(X_test.to(DEVICE))

model = QuantFormer(
    input_dim=tokens_test.size(-1),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    output_dim=1,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

with torch.no_grad():
    preds = model(tokens_test)

mse = ((preds - Y_test.to(DEVICE)) ** 2).mean().item()
mae = (preds - Y_test.to(DEVICE)).abs().mean().item()
rmse = mse ** 0.5

print("Synthetic test evaluation (unseen dataset)")
print(f"MSE :  {mse:.6f}")
print(f"RMSE:  {rmse:.6f}")
print(f"MAE :  {mae:.6f}")
