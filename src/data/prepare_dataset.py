"""Stage B — dataset preparation for the Music AI Lab MVP.

1. Download / unpack GTZAN.
2. Audit every track (decode, sample rate, duration, duplicate hashes).
3. Build a fixed, stratified *track-level* 70/15/15 split (seed 42).

The split is written to data/splits/splits.csv and is the single source of
truth for every model, so no model can see a test track during training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
GTZAN = RAW / "gtzan"
ARCHIVE = RAW / "genres.tar.gz"
URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def download() -> None:
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 1_000_000_000:
        print(f"[download] already present: {ARCHIVE} ({ARCHIVE.stat().st_size/1e9:.2f} GB)")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[download] {URL}")
    urllib.request.urlretrieve(URL, ARCHIVE)
    print(f"[download] saved {ARCHIVE.stat().st_size/1e9:.2f} GB")


def extract() -> None:
    if (GTZAN / "genres").is_dir() and len(list((GTZAN / "genres").glob("*/*.wav"))) >= 1000:
        print("[extract] already extracted")
        return
    if not ARCHIVE.exists():
        raise SystemExit("archive missing; run --download first")
    GTZAN.mkdir(parents=True, exist_ok=True)
    print("[extract] unpacking")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        # The archive contains top-level `genres/`.
        tf.extractall(GTZAN)
    print("[extract] done")


def audit() -> pd.DataFrame:
    import soundfile as sf

    rows, hashes = [], defaultdict(list)
    # The HuggingFace copy of GTZAN carries macOS AppleDouble stubs ("._*.wav");
    # they are metadata, not audio, so skip them.
    files = sorted(p for p in GTZAN.glob("genres/*/*.wav") if not p.name.startswith("._"))
    if not files:
        raise SystemExit(f"no wav files under {GTZAN}")
    for i, p in enumerate(files):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        hashes[h].append(p.stem)
        row = {"track_id": p.stem, "genre": p.parent.name, "path": str(p.relative_to(ROOT)),
               "split": None, "sr": None, "duration": None, "channels": None, "ok": False}
        try:
            info = sf.info(str(p))
            row.update(sr=info.samplerate, duration=round(info.duration, 3),
                       channels=info.channels, ok=info.samplerate == 22050 and info.duration > 2)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (i + 1) % 250 == 0:
            print(f"[audit] {i+1}/{len(files)}")
    df = pd.DataFrame(rows)
    df["md5"] = df["track_id"].map(
        {t: h for h, ts in hashes.items() for t in ts})
    dup_map = {t: h for h, ts in hashes.items() if len(ts) > 1 for t in ts}
    df["dup_hash"] = df["track_id"].map(dup_map).fillna("")
    df["duplicate"] = df["dup_hash"] != ""
    return df


def make_splits(df: pd.DataFrame, seed: int = 42,
                fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    for genre, idx in df.groupby("genre").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        train, val, test = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        df.loc[train, "split"] = "train"
        df.loc[val, "split"] = "val"
        df.loc[test, "split"] = "test"
    return df


def dedupe_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep decodable tracks only, and keep one copy of each identical file.

    GTZAN is known to contain 1 corrupt track and ~28 duplicated tracks. If a
    duplicate copy landed in train and another in test, the test score would be
    inflated; dropping the extra copies is the cheapest honest fix.
    """
    ok = df[df["ok"]].copy()
    ok = ok.sort_values("track_id").drop_duplicates(subset="md5", keep="first")
    excluded = df[~df["track_id"].isin(ok["track_id"])]
    return ok, excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.download:
        download()
    if args.extract:
        extract()
    if args.audit or args.split:
        df = audit()
        stats = {
            "dataset": "GTZAN",
            "n_tracks": int(len(df)),
            "genres": GENRES,
            "per_genre": dict(Counter(df["genre"])),
            "sample_rates": {str(k): int(v) for k, v in Counter(df["sr"]).items()},
            "channels": {str(k): int(v) for k, v in Counter(df["channels"]).items()},
            "duration_seconds": {"min": float(df["duration"].min()), "max": float(df["duration"].max()),
                                 "mean": float(df["duration"].mean())},
            "decode_failures": df.loc[~df["ok"], ["track_id", "path"]].to_dict("records"),
            "duplicate_tracks": sorted(df.loc[df["duplicate"], "track_id"].tolist()),
        }
        print(f"[audit] {stats['n_tracks']} tracks, failures={len(stats['decode_failures'])}, "
              f"duplicates={len(stats['duplicate_tracks'])}")
    if args.split:
        kept, excluded = dedupe_and_filter(df)
        stats["kept_tracks"] = int(len(kept))
        stats["excluded_tracks"] = excluded["track_id"].tolist()
        stats["split_protocol"] = {
            "seed": args.seed, "ratio": "70/15/15", "level": "track",
            "dedup": "md5 exact duplicates dropped (first kept)",
            "corrupt": "decode failures dropped",
        }
        kept = make_splits(kept, args.seed)
        out = ROOT / "data" / "splits" / "splits.csv"
        kept.to_csv(out, index=False)
        (ROOT / "data" / "splits" / "stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False))
        print("[split]", kept.groupby(["split", "genre"]).size().unstack(fill_value=0))
        print(f"[split] kept {len(kept)} / {len(df)} tracks (dropped {len(df)-len(kept)}: "
              f"corrupt or duplicate) -> {out}")
        # enforce the protocol's core invariants
        assert kept["split"].notna().all(), "every track must be assigned a split"
        assert set(kept["split"]) == {"train", "val", "test"}
        assert kept["md5"].is_unique, "duplicate audio must not survive the filter"


if __name__ == "__main__":
    main()
