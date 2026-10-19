"""Create a features.npz whose 'splits' field follows the track-level split.

Used only for the leakage-comparison ablation: same audio/features, different
split labels. Usage:
  python scripts/make_tracklevel_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/fma/processed/features.npz"
SPLITS = ROOT / "data/splits_fma/splits_tracklevel.csv"
OUT = ROOT / "data/fma/processed/features_tracklevel.npz"


def main() -> None:
    d = np.load(SRC, allow_pickle=True)
    df = pd.read_csv(SPLITS).set_index("track_id")
    m = df["split"]
    new = np.array([m[t] for t in d["track_ids"]], dtype=object)
    np.savez_compressed(OUT, track_ids=d["track_ids"], genres=d["genres"],
                        splits=new, X_feat=d["X_feat"], X_mel=d["X_mel"])
    print("wrote", OUT, "counts", pd.Series(new).value_counts().to_dict())


if __name__ == "__main__":
    main()
