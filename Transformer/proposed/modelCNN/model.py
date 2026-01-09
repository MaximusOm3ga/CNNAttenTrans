import torch.nn as nn
import torch

CNN_INPUT_SIZE = 6
CNN_PATCH_SIZE = 4
CNN_EMBED_DIM = 256
CNN_GROWTH_RATE = 32
CNN_NUM_DENSE_BLOCKS = 3


class DenseNetTokenEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = CNN_INPUT_SIZE,
        patch_size: int = CNN_PATCH_SIZE,
        embed_dim: int = CNN_EMBED_DIM,
        growth_rate: int = CNN_GROWTH_RATE,
        num_dense_blocks: int = CNN_NUM_DENSE_BLOCKS,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.conv0 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(64)

        self.dense_blocks = nn.ModuleList()
        in_channels = 64

        for _ in range(num_dense_blocks):
            dense_block = nn.ModuleList()
            for _ in range(3):
                layer = nn.Sequential(
                    nn.BatchNorm1d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
                )
                dense_block.append(layer)
                in_channels += growth_rate
            self.dense_blocks.append(dense_block)

        self.final_channels = in_channels

        self.pool_attn = nn.Sequential(
            nn.Conv1d(self.final_channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        self.token_proj = nn.Linear(self.final_channels, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, D = x.shape

        patches = []
        for start in range(0, T - self.patch_size + 1, self.patch_size):
            patches.append(x[:, start:start + self.patch_size, :])

        patches = torch.stack(patches, dim=1)
        B, P, S, D = patches.shape
        x = patches.view(B * P, D, S)

        x = torch.relu(self.bn0(self.conv0(x)))

        for dense_block in self.dense_blocks:
            for layer in dense_block:
                out = layer(x)
                x = torch.cat([x, out], dim=1)

        weights = self.pool_attn(x)
        x = (x * weights).sum(dim=-1)

        tokens = self.token_proj(x)
        tokens = self.dropout(tokens)

        return tokens.view(B, P, self.embed_dim)


class TokenSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_head=16, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_head,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        attn_out, _ = self.mha(tokens, tokens, tokens)
        return self.norm(tokens + attn_out)


class TokenDecoder(nn.Module):
    def __init__(self, embed_dim=256, patch_size=4, output_size=6):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, patch_size * output_size)
        self.output_size = output_size
        self.patch_size = patch_size

    def forward(self, tokens):
        B, P, D = tokens.shape
        x = torch.relu(self.fc1(tokens))
        x = self.fc2(x)
        return x.view(B, P, self.patch_size, self.output_size)


class AutoencoderTokenExtractor(nn.Module):
    def __init__(self, input_size=6, patch_size=4, embed_dim=256):
        super().__init__()
        self.encoder = DenseNetTokenEncoder(input_size, patch_size, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim))

        self.attn = TokenSelfAttention(embed_dim)
        self.decoder = TokenDecoder(embed_dim, patch_size, input_size)

    def forward(self, x):
        tokens = self.encoder(x)

        B, P, D = tokens.shape
        tokens = tokens + self.pos_embed[:, :P]

        tokens = self.attn(tokens)
        reconstructed = self.decoder(tokens)

        return reconstructed, tokens
