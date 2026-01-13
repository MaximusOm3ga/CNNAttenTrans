import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        B, T, _ = x.shape

        x = self.input_proj(x)

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos)

        x = self.encoder(x)

        x = self.head(x)
        return x.squeeze(-1)
