"""Thin wrapper around the EnCodec 32 kHz tokenizer used by MusicGen.

The tokenizer is the *fixed* part of phase 1: nothing here is trained.  Keeping
it in one place also makes the compatibility claim testable — the checkpoint
id, sample rate, frame rate and codebook layout are printed and cached so the
report can quote them instead of guessing.
"""
from __future__ import annotations

import numpy as np
import torch

ENCODEC_ID = "facebook/encodec_32khz"
SAMPLE_RATE = 32_000
BANDWIDTH = 2.2  # kbps; the only bandwidth this checkpoint was trained for
N_CODEBOOKS = 4
FRAME_RATE = 50  # tokens per second per codebook


def load_encodec(device: str = "cuda", dtype=torch.float32):
    from transformers import EncodecModel

    model = EncodecModel.from_pretrained(ENCODEC_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_waveforms(model, waveforms: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Encode a batch of fixed-length mono 32 kHz clips.

    waveforms: float32 array [B, T] (all rows the same length).
    returns: int64 array [B, n_codebooks, frames] with frames = T / 640 at 50 Hz.
    """
    x = torch.from_numpy(np.ascontiguousarray(waveforms, dtype=np.float32))
    x = x[:, None, :].to(device)
    enc = model.encode(x)
    codes = enc.audio_codes
    # transformers >= 5 returns (nb_frames, batch, n_q, T); chunking is off here,
    # so nb_frames == 1 and the batch axis has to be taken, not the chunk axis.
    if codes.dim() == 4:
        assert codes.shape[0] == 1, f"unexpected chunked encode output {tuple(codes.shape)}"
        codes = codes[0]
    return codes.to("cpu").numpy()


@torch.no_grad()
def decode_tokens(model, codes: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Decode [B, n_codebooks, frames] token arrays back to [B, samples] audio."""
    c = torch.from_numpy(np.ascontiguousarray(codes.astype(np.int64))).to(device)
    c = c[None]  # (nb_frames=1, batch, n_q, T)
    # no volume normalisation is applied on the encode side, so one None scale
    # per chunk is exactly what decode expects
    out = model.decode(c, audio_scales=[None])
    return out.audio_values.squeeze(1).to("cpu").numpy()
