"""Phase 1 data prep: cut 6 s windows (5 s prefix + 1 s target) and encode them
into EnCodec 32 kHz discrete tokens.

Design notes
------------
* Windows never cross the train/val/test boundary: the window index is built
  from the artist-disjoint split file, so every window of a track inherits that
  track's split.  This is the same protocol as the classification experiments.
* Encoding is GPU-bound and the whole job is resumable: work is cut into fixed
  shards, an existing shard is skipped, so a crash or a reboot only costs the
  current shard.
* Tokens are stored as uint16 (codebook size is 2048) to keep the cache small.

Output: data/features/token_prediction/shards/shard_XXXX.npz with keys
    codes      uint16 [N, 4, 300]   (50 Hz x 6 s, 4 codebooks)
    track_id   <U6  [N]
    genre      <U32 [N]
    artist     <U64 [N]
    split      <U5  [N]
    offset     float32 [N]          (seconds into the track)
    duration   float32 [N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits/fma/splits.csv"
OUT = ROOT / "data/features/token_prediction"
SR = 32_000
PREFIX_S = 5.0
TARGET_S = 1.0
WIN_S = PREFIX_S + TARGET_S
WIN_SAMPLES = int(WIN_S * SR)


def _load_one(args) -> tuple[int, np.ndarray | None]:
    """Worker: decode one window as mono 32 kHz float32."""
    idx, path, offset, win_s = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import librosa

    try:
        y, _ = librosa.load(path, sr=SR, mono=True, offset=float(offset), duration=win_s)
    except Exception:
        return idx, None
    n = int(win_s * SR)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return idx, y[:n].astype(np.float32)


def build_index(windows_per_track: int, limit: int | None, win_s: float) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(SPLITS, dtype={"track_id": str})
    df = df[df["ok"].astype(bool)]
    rows: list[dict] = []
    for r in df.itertuples(index=False):
        dur = float(r.duration)
        if dur < win_s + 0.5:
            continue
        # deterministic offsets: early and late part of the track
        cand = [2.0, max(2.0, dur - win_s - 1.0), max(2.0, dur / 2.0)]
        offsets = []
        for c in cand[:windows_per_track]:
            if all(abs(c - o) > win_s for o in offsets):
                offsets.append(c)
        for k, off in enumerate(offsets):
            rows.append(dict(track_id=str(r.track_id), genre=str(r.genre), artist=str(r.artist),
                             split=str(r.split), path=str(ROOT / r.path), offset=off,
                             duration=dur, window=k))
    if limit is not None:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(keep)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-per-track", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--encode-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefix-s", type=float, default=PREFIX_S)
    ap.add_argument("--target-s", type=float, default=TARGET_S)
    ap.add_argument("--subdir", default="", help="extra path component under data/features/")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    from src.forecasting.encodec_io import load_encodec, encode_waveforms

    win_s = args.prefix_s + args.target_s
    out_dir = OUT / args.subdir if args.subdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = out_dir / "shards"
    shards.mkdir(exist_ok=True)

    index = build_index(args.windows_per_track, args.limit, win_s)
    meta = dict(tokenizer="facebook/encodec_32khz", sample_rate=SR, frame_rate=50,
                n_codebooks=4, codebook_size=2048, prefix_s=args.prefix_s,
                target_s=args.target_s, frames=int(win_s * 50),
                windows_per_track=args.windows_per_track, n_windows=len(index),
                source_split=str(SPLITS.relative_to(ROOT)))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[tokens] {len(index)} windows from {len({r['track_id'] for r in index})} tracks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encodec(device)
    n_shards = (len(index) + args.shard_size - 1) // args.shard_size
    pool = ProcessPoolExecutor(max_workers=args.workers)
    n_fail = 0
    with torch.no_grad():
        for s in range(n_shards):
            path = shards / f"shard_{s:04d}.npz"
            if path.exists() and not args.overwrite:
                continue
            chunk = index[s * args.shard_size:(s + 1) * args.shard_size]
            jobs = [(i, r["path"], r["offset"], win_s) for i, r in enumerate(chunk)]
            waves: dict[int, np.ndarray] = {}
            for i, y in pool.map(_load_one, jobs):
                if y is None:
                    n_fail += 1
                else:
                    waves[i] = y
            keep = [i for i in range(len(chunk)) if i in waves]
            if not keep:
                continue
            batch = np.stack([waves[i] for i in keep])
            # EnCodec activations are large: 256 x 6 s in one forward needs > 10 GiB,
            # so encode in small mini-batches and free the cache between them.
            codes = np.concatenate(
                [encode_waveforms(model, batch[j:j + args.encode_batch], device)
                 for j in range(0, len(batch), args.encode_batch)], axis=0)
            torch.cuda.empty_cache()
            np.savez_compressed(
                path,
                codes=codes.astype(np.uint16),
                track_id=np.array([chunk[i]["track_id"] for i in keep], dtype="U8"),
                genre=np.array([chunk[i]["genre"] for i in keep], dtype="U32"),
                artist=np.array([chunk[i]["artist"] for i in keep], dtype="U64"),
                split=np.array([chunk[i]["split"] for i in keep], dtype="U5"),
                offset=np.array([chunk[i]["offset"] for i in keep], dtype=np.float32),
                duration=np.array([chunk[i]["duration"] for i in keep], dtype=np.float32),
            )
            print(f"[tokens] shard {s + 1}/{n_shards} n={len(keep)} "
                  f"({len(keep) * win_s:.0f} s audio) failed={n_fail}", flush=True)
    pool.shutdown()
    print("[tokens] done", out_dir)


if __name__ == "__main__":
    sys.exit(main())
