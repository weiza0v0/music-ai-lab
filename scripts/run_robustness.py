"""Thin runner for the P0 robustness experiment.

The experiment itself lives in src/robustness.py so it is importable and can be
run as `python -m src.robustness`; this script only fixes sys.path for callers
that start from the repo root without `-m`.

    python scripts/run_robustness.py --workers 12
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness import main  # noqa: E402

if __name__ == "__main__":
    main()
