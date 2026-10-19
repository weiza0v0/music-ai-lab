"""Small CNN for Log-Mel genre classification (8 GB-VRAM / CPU friendly)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.net(x)


class GenreCNN(nn.Module):
    """4 conv blocks -> global average pool -> linear head.

    Input: (B, 1, 64, 216) log-Mel.
    """

    def __init__(self, n_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, width),
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
            ConvBlock(width * 4, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4 * 8, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CRNN(nn.Module):
    """CNN front-end + bidirectional GRU over time — tests temporal modelling."""

    def __init__(self, n_classes: int = 10, width: int = 32, hidden: int = 96,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, width, pool=(2, 2)),
            ConvBlock(width, width * 2, pool=(2, 2)),
            ConvBlock(width * 2, width * 4, pool=(2, 1)),
        )
        self.gru = nn.GRU(width * 4, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        f = self.features(x)                 # (B, C, F', T')
        f = f.mean(dim=2)                    # collapse frequency -> (B, C, T')
        f = f.transpose(1, 2)                # (B, T', C)
        out, _ = self.gru(f)
        return self.head(out.mean(dim=1))


class SmallTransformer(nn.Module):
    """Lightweight Transformer over Log-Mel time frames (patch projection)."""

    def __init__(self, n_classes: int = 10, d_model: int = 128, nhead: int = 4,
                 layers: int = 2, n_frames: int = 216, n_mels: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):                    # (B, 1, 64, 216)
        x = x.squeeze(1).transpose(1, 2)     # (B, T, 64)
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(self.norm(h.mean(dim=1)))


MODELS = {"cnn": GenreCNN, "crnn": CRNN, "transformer": SmallTransformer}


def build(name: str, n_classes: int = 10, **kw) -> nn.Module:
    return MODELS[name](n_classes=n_classes, **kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
