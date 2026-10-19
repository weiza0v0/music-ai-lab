"""Recompute prediction flips (relative to clean_5s) for the robustness grid.

Uses the same decision rule as src/robustness.py (classical .predict(), deep argmax).
Writes results/gtzan/robustness_flips.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness import (CONDITIONS, MODEL_ORDER, evaluate_all,  # noqa: E402
                            extract_features, load_models, test_frame)

RES = ROOT / "results" / "gtzan"


def main() -> None:
    df = test_frame()
    feats, mels, _ = extract_features(df, workers=12)
    models, device = load_models(42)
    results, _ = evaluate_all(models, feats, mels, df, device)
    rows = []
    for m in MODEL_ORDER:
        base = results[(m, "clean_5s")][0]
        for c in CONDITIONS:
            pred = results[(m, c.name)][0]
            rows.append({"model": m, "condition": c.name,
                         "flips": int((pred != base).sum()), "n": len(pred)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "robustness_flips.csv", index=False)
    print(out.pivot(index="model", columns="condition", values="flips").to_string())


if __name__ == "__main__":
    main()
