"""Run inference with a saved checkpoint — used by the CLI and the Streamlit demo.

The demo must call real weights; nothing here hard-codes a prediction.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("MUSIC_CKPT", ROOT / "checkpoints"))
CLASSICAL = {"svm", "mlp"}
DEEP = {"cnn", "crnn", "transformer"}


def available() -> list[str]:
    found = []
    for ck in CKPT.glob("*.joblib"):
        found.append(ck.stem.split("_seed")[0])
    for ck in CKPT.glob("*.pt"):
        found.append(ck.stem.split("_seed")[0])
    return sorted(set(found))


def _pick_ckpt(model: str, seed: int | None = None) -> Path:
    """Prefer an explicit seed, else seed 42 (the report default), else the first."""
    pattern = f"{model}_seed*.joblib" if model in CLASSICAL else f"{model}_seed*.pt"
    cands = sorted(CKPT.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"no checkpoint for {model!r} in {CKPT}")
    if seed is not None:
        hit = [c for c in cands if c.stem.endswith(f"_seed{seed}")]
        if not hit:
            raise FileNotFoundError(f"no checkpoint for {model!r} seed={seed}")
        return hit[0]
    for c in cands:
        if c.stem.endswith("_seed42"):
            return c
    return cands[0]


def predict_audio(audio_path: str | Path, model: str, topk: int = 3,
                  ckpt: str | Path | None = None) -> dict:
    """Return {'model', 'top3': [(genre, prob)...], 'all': {genre: prob}}."""
    from src.data.audio_features import handcrafted, load_clip, log_mel

    ckpt = Path(ckpt) if ckpt else _pick_ckpt(model)
    y = load_clip(str(audio_path))
    if model in CLASSICAL:
        bundle = joblib.load(ckpt)
        feat = handcrafted(y)[None, :]
        proba = bundle["model"].predict_proba(feat)[0]
        classes = bundle["classes"]
    elif model in DEEP:
        from src.models.cnn import build
        state = torch.load(ckpt, map_location="cpu")
        net = build(state["model"], n_classes=len(state["classes"]))
        net.load_state_dict(state["state_dict"])
        net.eval()
        mel = (log_mel(y)[None, None, :, :] - state["norm"]["mean"]) / state["norm"]["std"]
        with torch.no_grad():
            logits = net(torch.from_numpy(mel.astype(np.float32)))
        proba = torch.softmax(logits, 1)[0].numpy()
        classes = state["classes"]
    else:
        raise ValueError(f"unknown model {model!r}")

    order = np.argsort(proba)[::-1]
    return {
        "model": model,
        "top3": [(classes[i], float(proba[i])) for i in order[:topk]],
        "all": {classes[i]: float(proba[i]) for i in order},
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="cnn", choices=sorted(CLASSICAL | DEEP))
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    out = predict_audio(args.audio, args.model, ckpt=args.ckpt)
    print(f"model={out['model']}")
    for g, p in out["top3"]:
        print(f"  {g:10s} {p*100:5.1f}%")


if __name__ == "__main__":
    main()
