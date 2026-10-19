"""Shared audio loading / feature extraction for every model.

Both the classical baselines (SVM, MLP) and the deep models (CNN, CRNN,
Transformer) read audio through this module, so the only thing that differs
between the two families is the representation, not the data.
"""
from __future__ import annotations

import os

# Keep BLAS/FFT single-threaded inside worker processes; the parent process
# parallelises across clips instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import librosa

SAMPLE_RATE = 22050
CLIP_SECONDS = 5
N_MELS = 64
N_FFT = 1024
HOP = 512
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_FRAMES = 1 + CLIP_SAMPLES // HOP  # 216 for 5 s @ 22.05 kHz


def load_clip(path: str, offset: float | None = None, duration: float = CLIP_SECONDS) -> np.ndarray:
    """Load a mono float32 clip. `offset=None` takes the centre of the track."""
    info = librosa.get_duration(path=path)
    if offset is None:
        offset = max(0.0, (info - duration) / 2.0)
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=offset, duration=duration)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y[:CLIP_SAMPLES].astype(np.float32)


def log_mel(y: np.ndarray) -> np.ndarray:
    """64-bin log-Mel spectrogram, shape (64, N_FRAMES), power dB."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    db = librosa.power_to_db(mel, ref=np.max)
    # Pad/crop to a fixed frame count so batches stack.
    if db.shape[1] < N_FRAMES:
        db = np.pad(db, ((0, 0), (0, N_FRAMES - db.shape[1])), mode="edge")
    return db[:, :N_FRAMES].astype(np.float32)


def handcrafted(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Track-level hand-crafted feature vector (mean/std pooled over frames).

    MFCC + deltas, chroma, spectral shape, zero-crossing rate, RMS and tempo.
    This is the *same* vector for SVM and MLP — the fair-comparison rule.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    tempo = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP)
    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        contrast.mean(axis=1), contrast.std(axis=1),
        np.array([centroid.mean(), centroid.std(), bandwidth.mean(), bandwidth.std(),
                  rolloff.mean(), rolloff.std(), zcr.mean(), zcr.std(),
                  rms.mean(), rms.std(), float(np.ravel(tempo)[0]) / 100.0]),
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_both(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (handcrafted_vector, logmel_2d) for one file — used by workers."""
    y = load_clip(path)
    return handcrafted(y), log_mel(y)
