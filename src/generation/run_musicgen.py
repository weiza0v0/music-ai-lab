"""Phase 2: MusicGen-small prefix continuation (base model, and fine-tuned if present).

The official MusicGen repo supports audio continuation, but the plain
`transformers` path only exposes text conditioning.  What we do here is the
documented way of continuing a sequence: encode the prefix with the model's own
EnCodec, feed those codes as `input_ids` to the decoder (which builds MusicGen's
delay pattern internally) and let it roll forward.  Text conditioning (the genre
name) is kept the same for base and fine-tuned runs.

Everything is measured: wall time, peak GPU memory, output length, and a simple
objective distance to the real continuation.  Subjective quality is reported
separately and never inferred from a single number.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits/fma/splits.csv"
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/generation"))
MODEL_ID = "facebook/musicgen-small"
SR = 32_000
GENRE_TEXT = {
    "Blues": "blues music with guitar and slow tempo",
    "Classical": "classical orchestral music",
    "Country": "country music with acoustic guitar",
    "Disco": "disco music with four on the floor beat",
    "Electronic": "electronic dance music with synthesizers",
    "Folk": "folk music with acoustic instruments",
    "Hip-Hop": "hip hop music with drums and bass",
    "Jazz": "jazz music with saxophone and swing",
    "Pop": "pop music with catchy melody",
    "Reggae": "reggae music with offbeat rhythm",
    "Rock": "rock music with electric guitar and drums",
    "Metal": "heavy metal music with distorted guitar",
}


def load_model(model_id: str, device: str, dtype):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, processor, time.time() - t0


@torch.no_grad()
def encode_prompt(model, wav: np.ndarray, device: str) -> torch.Tensor:
    dt = next(model.parameters()).dtype
    x = torch.from_numpy(wav.astype(np.float32))[None, None].to(device=device, dtype=dt)
    codes = model.audio_encoder.encode(x).audio_codes
    if codes.dim() == 4:
        codes = codes[:, 0]
    return codes  # [1, 4, T]


@torch.no_grad()
def decode_codes(model, codes: torch.Tensor, device: str) -> np.ndarray:
    out = model.audio_encoder.decode(codes[:, None].to(device), audio_scales=[None])
    return out.audio_values.squeeze().float().cpu().numpy()


@torch.no_grad()
def continue_audio(model, processor, prompt_codes: torch.Tensor, text: str,
                   seconds: float, device: str, cfg: float, seed: int) -> np.ndarray:
    """Return the generated continuation only (prompt dropped)."""
    K = model.config.decoder.num_codebooks
    frames = int(seconds * 50)
    text_inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    hidden = model.get_encoder()(**text_inputs).last_hidden_state
    proj = model.enc_to_dec_proj(hidden) if hasattr(model, "enc_to_dec_proj") else hidden

    ids = prompt_codes.reshape(K, -1)                      # [K, T]
    enc_states = proj
    if cfg and cfg > 1:
        print("[musicgen] CFG disabled on this path (decoder-level CFG needs paired "
              "conditional/unconditional batches); falling back to guidance_scale=1.0")
    cfg = 1.0
    torch.manual_seed(seed)
    pad = model.config.decoder.pad_token_id
    out = model.decoder.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),   # otherwise the delay-pattern pads make
                                              # cumsum-1 hit -1 in the position embedding
        encoder_hidden_states=enc_states,
        decoder_start_token_id=pad,
        pad_token_id=pad,
        max_new_tokens=frames + K,     # delay pattern eats K frames at the end
        do_sample=True, top_k=250, temperature=1.0,
        guidance_scale=float(cfg or 1.0), num_return_sequences=1,
    )
    if cfg and cfg > 1:
        out = out[:1]
    gen = out[:, :, prompt_codes.shape[-1]:]
    if gen.shape[-1] > frames:
        gen = gen[:, :, :frames]
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="CFG 需要在 decoder 直连路径下手工对齐条件/无条件两路；默认关闭，保证 base 与 ft 可比")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-genre", default=None, help="只跑某个流派（便于单进程单曲调试）")
    ap.add_argument("--pick-index", type=int, default=0, help="该流派内取第几首测试曲")
    args = ap.parse_args()

    import librosa
    import pandas as pd
    import soundfile as sf

    (RES / "audio").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model, processor, load_s = load_model(args.model, device, dtype)
    print(f"[musicgen] loaded {args.model} in {load_s:.1f}s")

    df = pd.read_csv(SPLITS, dtype={"track_id": str})
    df = df[(df.split == "test") & df.ok.astype(bool)]
    # one track per genre for a readable comparison
    if args.only_genre:
        cand = df[df.genre == args.only_genre]
        picks = cand.iloc[args.pick_index:args.pick_index + 1]
    else:
        picks = df.groupby("genre", sort=True).head(1).head(args.n)

    rows = []
    for r in picks.itertuples(index=False):
        path = ROOT / r.path
        dur = float(r.duration)
        off = max(0.0, min(2.0, dur - 10.5))
        y, _ = librosa.load(path, sr=SR, mono=True, offset=off, duration=5.0 + args.seconds)
        prefix = y[: 5 * SR]
        true_cont = y[5 * SR: 5 * SR + int(args.seconds * SR)]
        if len(true_cont) < int(args.seconds * SR):
            true_cont = np.pad(true_cont, (0, int(args.seconds * SR) - len(true_cont)))

        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t0 = time.time()
        codes = encode_prompt(model, prefix, device)
        t_enc = time.time() - t0
        t0 = time.time()
        gen = continue_audio(model, processor, codes, GENRE_TEXT.get(r.genre, "music"),
                             args.seconds, device, args.cfg, args.seed)
        t_gen = time.time() - t0
        wav = decode_codes(model, gen, device)
        peak = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0

        name = f"{r.track_id}_{r.genre}".replace(" ", "-")
        sf.write(RES / "audio" / f"{name}_prompt.wav", prefix, SR)
        sf.write(RES / "audio" / f"{name}_{args.tag}.wav", wav[: int(args.seconds * SR)], SR)
        sf.write(RES / "audio" / f"{name}_true.wav", true_cont, SR)

        # objective distance on log-mel (64 bins), same length
        import librosa as lb
        m1 = lb.power_to_db(lb.feature.melspectrogram(y=wav, sr=SR, n_mels=64) + 1e-9)
        m2 = lb.power_to_db(lb.feature.melspectrogram(y=true_cont, sr=SR, n_mels=64) + 1e-9)
        n = min(m1.shape[1], m2.shape[1])
        rmse = float(np.sqrt(((m1[:, :n] - m2[:, :n]) ** 2).mean()))
        rms = float(np.sqrt((wav ** 2).mean()))
        rows.append(dict(track_id=str(r.track_id)[:6], genre=str(r.genre), tag=args.tag,
                         latency_s=round(t_gen, 2), enc_s=round(t_enc, 2),
                         peak_mem_mb=round(peak, 1), out_s=round(len(wav) / SR, 2),
                         logmel_rmse=round(rmse, 3), rms=round(rms, 4),
                         prompt_wav=f"audio/{name}_prompt.wav",
                         out_wav=f"audio/{name}_{args.tag}.wav",
                         true_wav=f"audio/{name}_true.wav"))
        print(f"  {r.track_id} {r.genre:>12} gen {t_gen:5.1f}s peak {peak:6.0f}MB "
              f"logmel {rmse:.2f}", flush=True)

    out = RES / f"metrics_{args.tag}.json"
    merged = []
    if out.exists():
        old = json.loads(out.read_text())
        keep = {x["track_id"]: x for x in old.get("items", [])}
        keep.update({x["track_id"]: x for x in rows})
        merged = list(keep.values())
    else:
        merged = rows
    out.write_text(json.dumps(dict(model=args.model, tag=args.tag, cfg=args.cfg,
                                   seconds=args.seconds, seed=args.seed,
                                   load_s=round(load_s, 1), items=merged), indent=2,
                              ensure_ascii=False))
    print("[musicgen] wrote", out)


if __name__ == "__main__":
    main()
