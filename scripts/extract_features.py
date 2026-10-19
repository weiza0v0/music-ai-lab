"""Cache both representations for every track in the split file.

Output: data/processed/features.npz
  track_ids  (N,)        global track id
  genres     (N,)        label string
  splits     (N,)        train / val / test
  X_feat     (N, D)      hand-crafted vector  -> SVM, MLP
  X_mel      (N, 64, 216) log-Mel spectrogram -> CNN (and later CRNN/TF)

Doing this once keeps every model on identical clips, which is the fairness
rule from the experiment protocol.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.audio_features import extract_both  # noqa: E402

# Force single-threaded numerical libs before workers import numpy/librosa.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"


def _one(path: str):
    try:
        return extract_both(path), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{path}: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--splits", default=str(ROOT / "data/splits/splits.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/features.npz"))
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    if "path" not in df.columns:
        df["path"] = df["track_id"].map(lambda t: f"data/raw/gtzan/genres/{t.split('.')[0]}/{t}.wav")
    t0 = time.time()
    print(f"[features] {len(df)} tracks with {args.workers} workers")
    with mp.Pool(args.workers) as pool:
        results = pool.map(_one, df["path"].tolist(), chunksize=8)

    feats, mels, keep, errors = [], [], [], []
    for (payload, err), row in zip(results, df.itertuples()):
        if err:
            errors.append(err)
            continue
        f, m = payload
        feats.append(f)
        mels.append(m)
        keep.append(row.Index)

    if errors:
        print(f"[features] {len(errors)} failures:")
        for e in errors[:10]:
            print("  ", e)
    X_feat = np.stack(feats)
    X_mel = np.stack(mels)
    sub = df.loc[keep]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        track_ids=sub["track_id"].to_numpy(),
        genres=sub["genre"].to_numpy(),
        splits=sub["split"].to_numpy(),
        X_feat=X_feat,
        X_mel=X_mel.astype(np.float16),  # half precision: 2x smaller, plenty for 8-bit dB range
    )
    print(f"[features] X_feat={X_feat.shape} X_mel={X_mel.shape} "
          f"in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
