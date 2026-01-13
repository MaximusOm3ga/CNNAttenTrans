# baselineTCN/model.py
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[..., :x.size(-1)]  # causal crop
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(
        self,
        input_dim,
        channels=(64, 64, 64, 64),
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_ch, ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )
            in_ch = ch

        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.network(x)
        y = self.head(h)       # (B, 1, T)
        return y.squeeze(1)    # (B, T)
