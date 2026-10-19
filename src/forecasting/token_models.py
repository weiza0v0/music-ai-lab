"""Models for phase-1 token forecasting.

Everything predicts the *next* EnCodec frame from the causal past, exactly like a
language model.  The primary metric is therefore reported in two modes:

* teacher-forced  : feed the ground-truth token at t-1, predict t  (the usual
                    "how good is the distribution" number);
* free-running    : feed your own prediction, predict t  (what the model would
                    actually do when generating).

Both are reported for every model so the gap is visible instead of hidden.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CODEBOOKS = 4
VOCAB = 2048
N_FRAMES = 300
PREFIX_FRAMES = 250
TARGET_FRAMES = N_FRAMES - PREFIX_FRAMES
TAIL_START = PREFIX_FRAMES - 1   # first position whose logits are used


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_windows(shards_dir: str | Path) -> dict:
    """Load every cached shard into one dict of numpy arrays."""
    shards = sorted(Path(shards_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shards_dir}")
    codes, meta = [], {k: [] for k in ("track_id", "genre", "artist", "split", "offset")}
    for p in shards:
        d = np.load(p, allow_pickle=False)
        codes.append(d["codes"].astype(np.int16))
        for k in meta:
            meta[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in meta.items()}
    out["codes"] = np.concatenate(codes)
    return out


def split_arrays(data: dict, split: str) -> dict:
    m = data["split"] == split
    return {"codes": data["codes"][m].astype(np.int64),
            "track_id": data["track_id"][m], "genre": data["genre"][m],
            "artist": data["artist"][m], "offset": data["offset"][m]}


# --------------------------------------------------------------------------- #
# baselines (no learning, but they share the evaluation loop)
# --------------------------------------------------------------------------- #
class Persistence:
    """Repeat the last prefix token for all 50 future frames."""
    name = "persistence"

    def predict(self, codes: np.ndarray) -> np.ndarray:
        # codes: [N, 4, 300] -> [N, 4, 50]
        last = codes[:, :, PREFIX_FRAMES - 1:PREFIX_FRAMES]
        return np.repeat(last, TARGET_FRAMES, axis=2)


class FrequencyPrior:
    """Most frequent token of each codebook in the training set."""
    name = "frequency"

    def __init__(self, train_codes: np.ndarray):
        self.top = np.array([np.bincount(train_codes[:, k, :].ravel(),
                                         minlength=VOCAB).argmax()
                             for k in range(N_CODEBOOKS)])

    def predict(self, codes: np.ndarray) -> np.ndarray:
        n = codes.shape[0]
        return np.broadcast_to(self.top[None, :, None], (n, N_CODEBOOKS, TARGET_FRAMES)).copy()


class Bigram:
    """P(t | t-1) per codebook, backing off to the unigram prior."""
    name = "bigram"

    def __init__(self, train_codes: np.ndarray, alpha: float = 0.5):
        self.logp = []
        counts1 = []
        for k in range(N_CODEBOOKS):
            c = train_codes[:, k, :].ravel()
            uni = np.bincount(c, minlength=VOCAB).astype(np.float64)
            self.logp.append(np.log(uni + 1.0) - np.log(uni.sum() + VOCAB))
            # bigram in a sparse dict: (prev * VOCAB + cur) -> count
            prev, cur = c[:-1].astype(np.int64), c[1:].astype(np.int64)
            key = prev * VOCAB + cur
            uniq, cnt = np.unique(key, return_counts=True)
            counts1.append((uniq, cnt))
        self.counts = counts1
        self.uni = [np.bincount(train_codes[:, k, :].ravel(), minlength=VOCAB)
                    for k in range(N_CODEBOOKS)]
        self.alpha = alpha

    def _next_dist(self, k: int, prev: int) -> np.ndarray:
        uni = self.uni[k]
        p = (uni + 1.0) / (uni.sum() + VOCAB)
        uniq, cnt = self.counts[k]
        lo, hi = np.searchsorted(uniq, prev * VOCAB), np.searchsorted(uniq, (prev + 1) * VOCAB)
        if hi > lo:
            idx = (uniq[lo:hi] - prev * VOCAB).astype(np.int64)
            row = np.zeros(VOCAB)
            row[idx] = cnt[lo:hi]
            tot = row.sum()
            p = (row + self.alpha * VOCAB * p) / (tot + self.alpha * VOCAB)
        return p

    def predict(self, codes: np.ndarray) -> np.ndarray:
        out = np.zeros((codes.shape[0], N_CODEBOOKS, TARGET_FRAMES), dtype=np.int64)
        for i in range(codes.shape[0]):
            for k in range(N_CODEBOOKS):
                prev = int(codes[i, k, PREFIX_FRAMES - 1])
                for t in range(TARGET_FRAMES):
                    p = self._next_dist(k, prev)
                    cur = int(p.argmax())
                    out[i, k, t] = cur
                    prev = cur
        return out


# --------------------------------------------------------------------------- #
# neural forecasters: one trunk, four codebook heads (MusicGen-style parallel)
# --------------------------------------------------------------------------- #
class _Trunk(nn.Module):
    def __init__(self, d_model: int, n_heads_out: int = N_CODEBOOKS, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(VOCAB, d_model) for _ in range(n_heads_out)])
        self.pos = nn.Embedding(N_FRAMES, d_model)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d_model, VOCAB) for _ in range(n_heads_out)])
        self.d_model = d_model
        self.n_heads_out = n_heads_out

    def tail_heads(self, h: torch.Tensor) -> torch.Tensor:
        """Only the last TARGET_FRAMES+1 positions carry a useful prediction.

        Computing 2048-way logits for all 300 frames costs ~6x more than needed and
        dominated the step time (measured 285 ms/step before this change).
        """
        h = h[:, TAIL_START:]
        return torch.stack([self.heads[k](h) for k in range(self.n_heads_out)], dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, T] -> [B, T, d]
        h = sum(self.emb[k](x[:, k]) for k in range(self.n_heads_out))
        h = h + self.pos(torch.arange(x.shape[-1], device=x.device))[None]
        return self.drop(h)


class LSTMForecaster(_Trunk):
    name = "lstm"

    def __init__(self, d_model: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__(d_model, dropout=dropout)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rnn(self.embed(x))[0]                  # [B, T, d]
        return self.tail_heads(h)


class TransformerForecaster(_Trunk):
    name = "transformer"

    def __init__(self, d_model: int = 192, layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        T = h.shape[1]
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.tail_heads(h)


def build_model(kind: str, **kw) -> nn.Module:
    if kind == "lstm":
        return LSTMForecaster(**kw)
    if kind == "transformer":
        return TransformerForecaster(**kw)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# losses / metrics
# --------------------------------------------------------------------------- #
def token_nll(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean CE over the 50 target frames and all codebooks."""
    logits = logits[:, :, :TARGET_FRAMES]   # predicts frames 250..299
    loss = 0.0
    for k in range(N_CODEBOOKS):
        loss = loss + F.cross_entropy(logits[:, k].reshape(-1, VOCAB),
                                      y[:, k, PREFIX_FRAMES:].reshape(-1))
    return loss / N_CODEBOOKS


@torch.no_grad()
def accuracies(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """Top-1 / Top-5 per codebook on the 50 target frames (teacher forced)."""
    logits = logits[:, :, :TARGET_FRAMES]
    out = {}
    t1, t5 = [], []
    for k in range(N_CODEBOOKS):
        lg = logits[:, k]
        tgt = y[:, k, PREFIX_FRAMES:]
        top5 = lg.topk(5, dim=-1).indices
        t1.append((top5[..., 0] == tgt).float().mean().item())
        t5.append((top5 == tgt.unsqueeze(-1)).any(-1).float().mean().item())
    out["top1"] = float(np.mean(t1))
    out["top5"] = float(np.mean(t5))
    out["top1_cb"] = t1
    return out


@torch.no_grad()
def free_running(model: nn.Module, codes: torch.Tensor, device: str,
                 batch: int = 128) -> torch.Tensor:
    """Autoregressive roll-out of the 50 target frames (model feeds itself)."""
    model.eval()
    outs = []
    for i in range(0, codes.shape[0], batch):
        x = codes[i:i + batch].to(device).clone()
        for t in range(PREFIX_FRAMES, N_FRAMES):
            logits = model(x[:, :, :t])
            nxt = logits[:, :, -1].argmax(-1)           # [B, K]
            x[:, :, t] = nxt
        outs.append(x[:, :, PREFIX_FRAMES:].cpu())
    return torch.cat(outs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def perplexity(nll: float) -> float:
    return float(math.exp(nll))
