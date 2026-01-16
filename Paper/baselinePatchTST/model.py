import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_features, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_features, embed_dim)

    def forward(self, x):
        B, T, F = x.shape
        P = T // self.patch_size
        x = x[:, : P * self.patch_size, :]
        x = x.view(B, P, self.patch_size * F)
        return self.proj(x)


class PatchTST(nn.Module):
    def __init__(
        self,
        patch_size,
        input_dim,
        d_model=256,
        num_heads=16,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.patch_embed(x)
        K = z.size(1)
        z = z + self.pos_embed[:, :K]
        h = self.encoder(z)
        y = self.head(h).squeeze(-1)
        return y
