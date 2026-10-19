"""Tests for the experiment protocol — these are the rules we must not break."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_split_is_track_level_disjoint():
    df = pd.read_csv(ROOT / "data/splits/gtzan/splits.csv")
    assert set(df["split"].unique()) == {"train", "val", "test"}
    # each track appears exactly once, so no track can straddle two splits
    assert df["track_id"].is_unique
    assert df.groupby("split")["track_id"].nunique().sum() == len(df)
    # stratified: every genre appears in every split
    assert (df.groupby(["split", "genre"]).size().unstack(fill_value=0) > 0).all().all()


def test_features_align_with_splits():
    d = np.load(ROOT / "data/features/gtzan/features.npz", allow_pickle=True)
    df = pd.read_csv(ROOT / "data/splits/gtzan/splits.csv")
    assert len(d["track_ids"]) == len(d["genres"]) == len(d["splits"]) == len(d["X_feat"])
    assert d["X_mel"].shape[1:] == (64, 216)
    assert set(d["track_ids"]).issubset(set(df["track_id"]))
    assert len(d["track_ids"]) == len(df), "some tracks failed feature extraction"
    # same clip for classical and deep models
    assert (d["splits"] == df.set_index("track_id").loc[d["track_ids"], "split"].to_numpy()).all()


def test_scaler_fitted_on_train_only():
    """StandardScaler must never see val/test statistics: reconstruct it."""
    import joblib
    from sklearn.preprocessing import StandardScaler
    d = np.load(ROOT / "data/features/gtzan/features.npz", allow_pickle=True)
    tr = d["splits"] == "train"
    ref = StandardScaler().fit(d["X_feat"][tr])
    bundle = joblib.load(ROOT / "checkpoints/gtzan/svm_seed42.joblib")
    got = bundle["model"].named_steps["standardscaler"]
    assert np.allclose(got.mean_, ref.mean_, atol=1e-5)
    assert np.allclose(got.scale_, ref.scale_, atol=1e-5)


def test_every_checkpoint_predicts_real_audio():
    from src.predict import available, predict_audio
    wav = sorted((ROOT / "data/raw/gtzan/genres/blues").glob("blues.0000*.wav"))[0]
    models = available()
    assert set(models) >= {"svm", "mlp", "cnn"}, models
    for m in models:
        out = predict_audio(wav, m)
        assert len(out["top3"]) == 3
        assert abs(sum(out["all"].values()) - 1.0) < 1e-4
        assert out["top3"][0][1] >= out["top3"][1][1] >= out["top3"][2][1]
