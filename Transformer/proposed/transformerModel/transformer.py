import torch
import torch.nn as nn


class QuantFormer(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 16,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        output_dim: int | None = None,
        max_context_len: int = 1024,
        pos_dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else None
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_embed = nn.Embedding(max_context_len, d_model)
            self.pos_dropout = nn.Dropout(pos_dropout)
        else:
            self.pos_embed = None
            self.pos_dropout = nn.Identity()
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(d_model, output_dim) if output_dim is not None else None

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        if self.input_projection is not None:
            x = self.input_projection(x)
        if self.use_positional_encoding and self.pos_embed is not None:
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            x = x + self.pos_embed(pos)
            x = self.pos_dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        if self.output_head is not None:
            x = self.output_head(x)
            return x.squeeze(-1)
        return x
