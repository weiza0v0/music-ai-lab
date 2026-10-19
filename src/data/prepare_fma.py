"""FMA-small data preparation (main experiment).

Steps:
  python -m src.data.prepare_fma --extract --audit --split

Reads fma_metadata.zip / fma_small.zip from data/fma/raw, extracts to data/fma/,
builds a fixed track-level stratified 70/15/15 split (seed 42), and writes:
  data/splits/fma/splits.csv
  data/splits/fma/stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/fma/raw"
FMA = ROOT / "data/fma"
META = FMA / "metadata"
AUDIO = FMA / "fma_small"
OUT = ROOT / "data/splits/fma"
GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental",
          "International", "Pop", "Rock"]


def extract() -> None:
    META.mkdir(parents=True, exist_ok=True)
    FMA.mkdir(parents=True, exist_ok=True)
    meta_zip = RAW / "fma_metadata.zip"
    audio_zip = RAW / "fma_small.zip"
    if not (META / "tracks.csv").exists():
        print("[extract] metadata")
        with zipfile.ZipFile(meta_zip) as z:
            z.extractall(FMA)
    if not AUDIO.exists() or len(list(AUDIO.rglob("*.mp3"))) < 8000:
        print("[extract] audio (may take a few minutes)")
        with zipfile.ZipFile(audio_zip) as z:
            z.extractall(FMA)
    print("[extract] done")


def load_tracks() -> pd.DataFrame:
    path = META / "tracks.csv"
    df = pd.read_csv(path, index_col=0, header=[0, 1])
    df.index = [str(i).zfill(6) for i in df.index]
    return df


def audit() -> pd.DataFrame:
    import soundfile as sf

    df = load_tracks()
    # FMA metadata covers 106k tracks; only the 8,000-track "small" subset has audio.
    df = df[df[("set", "subset")] == "small"]
    df = df[df[("track", "genre_top")].notna()]
    # Flatten the two-level header into a simple frame before auditing.
    tracks = pd.DataFrame({
        "track_id": df.index.astype(str),
        "genre": df[("track", "genre_top")].astype(str).to_numpy(),
        "artist": df[("artist", "name")].astype(str).to_numpy(),
        "title": df[("track", "title")].astype(str).to_numpy(),
    })
    tracks["path"] = [f"data/fma/fma_small/{t[:3]}/{t}.mp3" for t in tracks["track_id"]]

    rows = []
    for i, rec in enumerate(tracks[["track_id", "genre", "artist", "title", "path"]]
                             .to_dict("records")):
        p = ROOT / rec["path"]
        row = {"track_id": rec["track_id"], "genre": rec["genre"],
               "artist": rec["artist"], "title": rec["title"], "path": rec["path"],
               "sr": None, "duration": None, "ok": False}
        if p.exists():
            try:
                info = sf.info(str(p))
                row.update(sr=info.samplerate, duration=round(info.duration, 3),
                           ok=info.samplerate > 8000 and info.duration > 10)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["error"] = "missing file"
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"[audit] {i+1}/{len(tracks)}")
    return pd.DataFrame(rows)


def make_artist_splits(df: pd.DataFrame, seed: int = 42,
                       fracs=(0.7, 0.15, 0.15)) -> pd.DataFrame:
    """Artist-disjoint split.

    GTZAN has no reliable artist metadata, but FMA does — and 81% of test
    tracks share an artist with train under a track-level split. Grouping by
    artist removes that shortcut. Greedy assignment by relative deficit keeps
    the overall 70/15/15 ratio; per-genre counts are reported, not forced.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["artist"] = df["artist"].fillna("").astype(str)
    # missing artist names become unique groups so we never merge strangers
    df["artist"] = [a if a and a != "nan" else f"_unknown_{t}"
                    for a, t in zip(df["artist"], df["track_id"])]
    sizes = df.groupby("artist").size()
    artists = list(sizes.index)
    rng.shuffle(artists)
    artists.sort(key=lambda a: -sizes[a])
    total = len(df)
    ratio = dict(zip(("train", "val", "test"), fracs))
    target = {k: total * v for k, v in ratio.items()}
    cur = {k: 0 for k in ratio}
    assign = {}
    for a in artists:
        k = max(ratio, key=lambda s: (target[s] - cur[s]) / target[s])
        assign[a] = k
        cur[k] += int(sizes[a])
    df["split"] = df["artist"].map(assign)
    return df


def make_splits(df: pd.DataFrame, seed: int = 42,
                fracs=(0.7, 0.15, 0.15)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    for genre, idx in df.groupby("genre").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * fracs[0]))
        n_va = int(round(n * fracs[1]))
        df.loc[idx[:n_tr], "split"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        df.loc[idx[n_tr + n_va:], "split"] = "test"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.extract:
        extract()
    if not args.audit and not args.split:
        return

    df = audit()
    ok = df[df["ok"]].copy()
    stats = {
        "dataset": "FMA-small",
        "n_metadata_tracks": int(len(df)),
        "n_decodable": int(len(ok)),
        "per_genre_all": dict(Counter(df["genre"])),
        "per_genre_ok": dict(Counter(ok["genre"])),
        "sample_rates": {str(k): int(v) for k, v in Counter(ok["sr"]).items()},
        "duration_seconds": {"min": float(ok["duration"].min()),
                             "max": float(ok["duration"].max()),
                             "mean": float(ok["duration"].mean())},
        "decode_failures": df.loc[~df["ok"], ["track_id", "path"]].to_dict("records"),
    }

    if args.split:
        # leakage comparison: track-level split first
        track_split = make_splits(ok, args.seed)
        tr_art = set(track_split.loc[track_split.split == "train", "artist"])
        track_leak = int(track_split.loc[track_split.split == "test", "artist"]
                         .isin(tr_art).sum())

        # primary split: artist-disjoint
        ok = make_artist_splits(ok, args.seed)
        tr_art2 = set(ok.loc[ok.split == "train", "artist"])
        stats["split"] = {
            "seed": args.seed, "ratio": "70/15/15", "level": "artist_disjoint",
            "note": "primary split groups every artist into exactly one split",
            "test_artists": int(ok.loc[ok.split == "test", "artist"].nunique()),
            "test_tracks_with_train_artist": int(
                ok.loc[ok.split == "test", "artist"].isin(tr_art2).sum()),
            "track_level_comparison": {
                "level": "track", "test_tracks_with_train_artist": track_leak,
                "test_tracks": int((track_split.split == "test").sum()),
            },
        }
        OUT.mkdir(parents=True, exist_ok=True)
        ok.to_csv(OUT / "splits.csv", index=False)
        track_split.to_csv(OUT / "splits_tracklevel.csv", index=False)
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print("[split]", ok.groupby(["split", "genre"]).size().unstack(fill_value=0))
        print(f"[split] kept {len(ok)} tracks -> {OUT/'splits.csv'} (artist-disjoint)")
        print(f"[split] track-level split saved -> splits_tracklevel.csv "
              f"(artist overlap {track_leak}/{int((track_split.split=='test').sum())})")


if __name__ == "__main__":
    main()
