"""Decode predicted tokens back to audio and compare with the real continuation.

Two honest caveats are printed and written into the JSON:
  * the multi-codebook model predicts all 4 EnCodec streams, so the tokens are
    directly decodable — the single-codebook variant is NOT decodable and is
    reported as a limitation instead of being silently dropped;
  * the prediction is 1 s long, so the audio is a "next second" sample, not a
    full musical continuation.  Longer horizons need the generation stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.forecasting.encodec_io import SAMPLE_RATE, decode_tokens, load_encodec
from src.forecasting.token_models import (N_FRAMES, PREFIX_FRAMES, TARGET_FRAMES,
                                          build_model, load_windows, split_arrays)

ROOT = Path(__file__).resolve().parents[2]
TOKENS = Path(os.environ.get("MUSIC_TOKENS", ROOT / "data/features/token_prediction"))
CKPT = Path(os.environ.get("MUSIC_CKPT", ROOT / "checkpoints/token_prediction"))
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/token_prediction"))


def logmel(x: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 64) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256)
    return librosa.power_to_db(m + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-genre", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_dir = RES / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    data = load_windows(TOKENS / "shards")
    te = split_arrays(data, "test")

    ck = torch.load(CKPT / f"{args.model}_seed{args.seed}.pt", map_location=device)
    model = build_model(ck["kind"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # pick a few windows, one per genre
    genres = sorted(set(te["genre"].tolist()))
    picks = []
    for g in genres:
        idx = np.where(te["genre"] == g)[0]
        picks.extend(idx[:args.per_genre].tolist())
    print(f"[decode] {len(picks)} windows from genres {genres}")

    codes = torch.from_numpy(te["codes"][picks]).to(device)
    with torch.no_grad():
        x = codes.clone()
        for t in range(PREFIX_FRAMES, N_FRAMES):
            nxt = model(x[:, :, :t])[:, :, -1].argmax(-1)
            x[:, :, t] = nxt
        pred = x[:, :, PREFIX_FRAMES:].cpu().numpy()
    true = te["codes"][picks][:, :, PREFIX_FRAMES:]

    enc = load_encodec(device)
    pred_wav = decode_tokens(enc, pred, device)
    true_wav = decode_tokens(enc, true, device)

    import soundfile as sf
    rows = []
    for j, i in enumerate(picks):
        tid = str(te["track_id"][i])
        g = str(te["genre"][i])
        name = f"{tid}_{g}".replace(" ", "-")
        sf.write(audio_dir / f"{name}_pred.wav", pred_wav[j], SAMPLE_RATE)
        sf.write(audio_dir / f"{name}_true.wav", true_wav[j], SAMPLE_RATE)
        a, b = logmel(pred_wav[j]), logmel(true_wav[j])
        n = min(a.shape[1], b.shape[1])
        dist = float(np.sqrt(((a[:, :n] - b[:, :n]) ** 2).mean()))
        rows.append({"track_id": tid, "genre": g, "offset": float(te["offset"][i]),
                     "logmel_rmse": dist,
                     "pred_wav": f"audio/{name}_pred.wav",
                     "true_wav": f"audio/{name}_true.wav"})
        print(f"  {tid} {g:>15} logmel-RMSE {dist:.2f}")

    (RES / "decode.json").write_text(json.dumps(
        {"model": args.model, "seed": args.seed, "sample_rate": SAMPLE_RATE,
         "target_seconds": TARGET_FRAMES / 50, "items": rows,
         "note": "prediction is the next 1 s predicted autoregressively from the 5 s prefix; "
                 "single-codebook models are not decodable and are excluded here"},
        indent=2, ensure_ascii=False))
    print("[decode] wrote", RES / "decode.json")


if __name__ == "__main__":
    main()
