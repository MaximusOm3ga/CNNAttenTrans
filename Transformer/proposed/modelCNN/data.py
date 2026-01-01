import torch
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 32


data = torch.load("../../dataset/paper_dataset.pt")
X = data["X"]

if X.dim() != 3:
    raise ValueError(f"Expected X to be 3D (N, T, D), got {X.shape}")

stock_data = X.float()
print(f"Synthetic data shape (X): {stock_data.shape}")

total = stock_data.size(0)
train_end = int(0.8 * total)

train_data = stock_data[:train_end]
val_data = stock_data[train_end:]

train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE)
