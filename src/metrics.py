"""Shared evaluation helpers: metrics, result rows, figure styling."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
FIG_DIR = RESULTS / "figures"
METRICS_CSV = RESULTS / "metrics.csv"

COLUMNS = ["model", "seed", "split", "n_samples", "accuracy", "macro_f1",
           "train_time_s", "infer_ms_per_clip", "n_params", "model_size_mb", "notes"]


def encode_labels(genres: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(genres)
    return y, le


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
             class_names: list[str]) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class_f1": {c: float(f) for c, f in zip(
            class_names, f1_score(y_true, y_pred, average=None))},
        "report": classification_report(y_true, y_pred, target_names=class_names, digits=3,
                                        zero_division=0),
    }
    return out


def append_metrics(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        key = ["model", "seed", "split"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["model", "seed"], inplace=True)
    df.to_csv(METRICS_CSV, index=False)
    print(f"[metrics] wrote {METRICS_CSV} ({len(df)} rows)")


def save_predictions(model_name: str, seed: int, split: str,
                     track_ids: np.ndarray, y_true: np.ndarray,
                     y_pred: np.ndarray, proba: np.ndarray,
                     class_names: list[str]) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"track_id": track_ids, "y_true": y_true, "y_pred": y_pred})
    for i, c in enumerate(class_names):
        df[f"p_{c}"] = proba[:, i]
    path = PRED_DIR / f"{model_name}_seed{seed}_{split}.csv"
    df.to_csv(path, index=False)
    return path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
