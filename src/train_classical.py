"""Stage C — classical baselines: SVM and MLP on the same hand-crafted features.

Protocol rules enforced here:
  * the StandardScaler is fit on TRAIN ONLY;
  * model selection uses the validation split, never test;
  * the test split is touched once, after the model is frozen.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (append_metrics, encode_labels, evaluate, save_json,  # noqa: E402
                         save_predictions)

DATA = ROOT / "data/processed/features.npz"
CKPT = ROOT / "checkpoints"


def load():
    d = np.load(DATA, allow_pickle=True)
    X = d["X_feat"]
    y, le = encode_labels(d["genres"])
    splits = d["splits"]
    return X, y, splits, le, d["track_ids"]


def make_model(kind: str, seed: int):
    if kind == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=seed),
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                          alpha=1e-4, batch_size=64, learning_rate_init=1e-3,
                          max_iter=300, early_stopping=True, n_iter_no_change=20,
                          validation_fraction=0.15, random_state=seed),
        )
    raise ValueError(kind)


def count_params(model) -> int:
    if hasattr(model, "named_steps"):
        from sklearn.svm import SVC as _SVC
        from sklearn.neural_network import MLPClassifier as _MLP
        est = model.named_steps[list(model.named_steps)[-1]]
        if isinstance(est, _SVC):
            return int(np.sum(est.n_support_) * model.named_steps["standardscaler"].n_features_in_)
        if isinstance(est, _MLP):
            return int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
    return 0


def run(kind: str, seed: int, X, y, splits, le, track_ids, latency_clips: int = 100):
    tr, va, te = (splits == s for s in ("train", "val", "test"))
    model = make_model(kind, seed)
    t0 = time.perf_counter()
    model.fit(X[tr], y[tr])
    train_time = time.perf_counter() - t0

    # Validation is for logging/diagnostics here; SVC/MLP hyper-params are fixed
    # a priori, so no test information leaks into the choice.
    val_pred = model.predict(X[va])
    val = evaluate(y[va], val_pred, model.predict_proba(X[va]), list(le.classes_))

    t0 = time.perf_counter()
    model.predict(X[te][:latency_clips])
    infer_ms = (time.perf_counter() - t0) / min(latency_clips, te.sum()) * 1000

    test_pred = model.predict(X[te])
    test_proba = model.predict_proba(X[te])
    test = evaluate(y[te], test_pred, test_proba, list(le.classes_))
    path = CKPT / f"{kind}_seed{seed}.joblib"
    CKPT.mkdir(exist_ok=True)
    bundle = {"kind": kind, "seed": seed, "model": model,
              "classes": list(le.classes_), "protocol": "gtzan-5s-center-v1"}
    joblib.dump(bundle, path)
    size_mb = path.stat().st_size / 1e6

    save_json({"val": {k: v for k, v in val.items() if k != "report"},
               "test": {k: v for k, v in test.items() if k != "report"},
               "test_report": test["report"],
               "class_names": list(le.classes_)},
              ROOT / "results" / f"{kind}_seed{seed}_eval.json")
    save_predictions(kind, seed, "val", track_ids[va], le.inverse_transform(y[va]),
                     le.inverse_transform(val_pred), model.predict_proba(X[va]), list(le.classes_))
    save_predictions(kind, seed, "test", track_ids[te], le.inverse_transform(y[te]),
                     le.inverse_transform(test_pred), test_proba, list(le.classes_))

    row = {"model": kind, "seed": seed, "split": "test", "n_samples": int(te.sum()),
           "accuracy": round(test["accuracy"], 4), "macro_f1": round(test["macro_f1"], 4),
           "train_time_s": round(train_time, 2), "infer_ms_per_clip": round(infer_ms, 3),
           "n_params": count_params(model), "model_size_mb": round(size_mb, 3),
           "notes": f"val_acc={val['accuracy']:.4f};val_f1={val['macro_f1']:.4f}"}
    print(f"[{kind}] seed={seed} test acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
          f"train={train_time:.1f}s infer={infer_ms:.3f}ms size={size_mb:.2f}MB")
    return row, test["macro_f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    X, y, splits, le, track_ids = load()
    print(f"[data] X={X.shape} train={(splits=='train').sum()} "
          f"val={(splits=='val').sum()} test={(splits=='test').sum()} classes={list(le.classes_)}")
    rows = []
    for kind in args.models:
        for seed in args.seeds:
            row, _ = run(kind, seed, X, y, splits, le, track_ids)
            rows.append(row)
    append_metrics(rows)


if __name__ == "__main__":
    main()
