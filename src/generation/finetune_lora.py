"""Phase 2 (optional): parameter-efficient adaptation of MusicGen-small.

`peft` is not installed and installing it would add a dependency to a shared
conda env, so LoRA is implemented directly: every targeted projection w gets a
frozen base weight and a trainable low-rank update B(A(x)) * alpha/r.

The adaptation objective is the model's own: teacher-forced next-token CE over
the 4 interleaved codebooks, exactly as MusicGen was pretrained (delay pattern,
text conditioning, no tokenizer surgery).  If the memory budget is exceeded the
script stops and reports it — it does not silently fall back to a fake result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
TOKENS_FT = Path(os.environ.get("MUSIC_TOKENS_FT", ROOT / "data/features/token_prediction"))
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/generation"))
CKPT = Path(os.environ.get("MUSIC_CKPT", ROOT / "checkpoints/generation"))
MODEL_ID = "facebook/musicgen-small"
GENRE_TEXT = {
    "Blues": "blues music with guitar and slow tempo", "Classical": "classical orchestral music",
    "Country": "country music with acoustic guitar", "Disco": "disco music with four on the floor beat",
    "Electronic": "electronic dance music with synthesizers", "Folk": "folk music with acoustic instruments",
    "Hip-Hop": "hip hop music with drums and bass", "Jazz": "jazz music with saxophone and swing",
    "Pop": "pop music with catchy melody", "Reggae": "reggae music with offbeat rhythm",
    "Rock": "rock music with electric guitar and drums", "Metal": "heavy metal music with distorted guitar",
    # FMA-small 的另外三个流派：早期版本缺这三条，会被降级成泛化的 "music"，
    # 等于把条件信息丢掉。
    "Experimental": "experimental music with abstract electronic textures",
    "Instrumental": "instrumental music without vocals",
    "International": "international world music with traditional instruments",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.a = nn.Linear(base.in_features, r, bias=False).to(device=dev, dtype=dt)
        self.b = nn.Linear(r, base.out_features, bias=False).to(device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight.float()
        return w + (self.b.weight.float() @ self.a.weight.float()) * self.scale


TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
           "encoder_attn.q_proj", "encoder_attn.v_proj"]


def apply_lora(decoder, r: int, alpha: int) -> int:
    n = 0
    layers = decoder.model.decoder.layers
    for layer in layers:
        for name in TARGETS:
            parent_name, child = name.rsplit(".", 1)
            parent = getattr(layer, parent_name)
            mod = getattr(parent, child)
            if isinstance(mod, nn.Linear):
                setattr(parent, child, LoRALinear(mod, r, alpha))
                n += 1
    return n


def load_ft_windows() -> dict:
    shards = sorted((TOKENS_FT / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no fine-tune shards in {TOKENS_FT/'shards'}")
    out = {k: [] for k in ("codes", "split", "genre", "track_id")}
    for p in shards:
        d = np.load(p, allow_pickle=False)
        out["codes"].append(d["codes"].astype(np.int64))
        for k in ("split", "genre", "track_id"):
            out[k].append(d[k])
    return {k: (np.concatenate(v) if k == "codes" else np.concatenate(v)) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--prefix-frames", type=int, default=250)
    ap.add_argument("--total-frames", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--out", default="musicgen-small-lora")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="每次前向的窗口数；旧版写死 2，8 GB 卡只用了 16%% 显存")
    ap.add_argument("--mask-prefix", action="store_true",
                    help="只对后 --total-frames - --prefix-frames 帧算 loss")
    ap.add_argument("--result-name", default="finetune.json",
                    help="结果文件名，便于把修正版与旧的负结果并存")
    args = ap.parse_args()

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    # bfloat16 keeps the fp32 exponent range, so no loss scaler is needed on a
    # frozen fp16-style backbone with small trainable adapters.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = MusicgenForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = apply_lora(model.decoder, args.rank, args.alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[lora] {n_lora} adapted projections, {sum(p.numel() for p in trainable)/1e6:.2f}M params")

    data = load_ft_windows()
    tr = data["codes"][data["split"] == "train"]
    va = data["codes"][data["split"] == "val"]
    print(f"[lora] train {len(tr)} val {len(va)} windows, {args.total_frames} frames each")

    model.train()
    if hasattr(model.decoder, "gradient_checkpointing_enable"):
        model.decoder.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # 一次性编码全部流派的文本条件，再按索引取用。
    # 旧实现按 genres[0] 做缓存键：只要首个流派相同就命中，返回的是上一批的整块
    # 张量，导致同批里其余样本的文本条件被错配——静默错误，必须按流派拆开。
    genre_all = sorted(GENRE_TEXT)
    genre_idx = {g: i for i, g in enumerate(genre_all)}
    fallback_row = len(genre_all)
    with torch.no_grad():
        tin = processor(text=[GENRE_TEXT[g] for g in genre_all] + ["music"],
                        padding=True, return_tensors="pt").to(device)
        h_all = model.get_encoder()(**tin).last_hidden_state
        enc_all = model.enc_to_dec_proj(h_all) if hasattr(model, "enc_to_dec_proj") else h_all

    def encoder_states(names: list[str]):
        return enc_all[[genre_idx.get(n, fallback_row) for n in names]]

    rng = np.random.default_rng(0)
    hist, step, t_start = [], 0, time.time()
    vocab = model.config.decoder.vocab_size
    K = model.config.decoder.num_codebooks
    bs = args.batch_size
    order = rng.permutation(len(tr))
    ptr = 0
    try:
        while step < args.steps:
            if (time.time() - t_start) / 60 > args.max_minutes:
                print("[lora] time budget reached, stopping early")
                break
            if ptr + bs > len(order):
                order = rng.permutation(len(tr))
                ptr = 0
            idx = order[ptr:ptr + bs]
            ptr += bs
            batch = torch.from_numpy(tr[idx][:, :, :args.total_frames]).to(device)  # (B, K, T)
            enc = encoder_states([str(data["genre"][i]) for i in idx])
            # MusicgenForCausalLM 要求 input_ids 是 (B*K, T) 的扁平序列，
            # labels 是 (B, T, K)。旧实现把 (B, K, T) 直接同时当 input_ids 和
            # labels，库内部会按 labels.shape[1]==K 去截取，loss 只在最后 K 帧上
            # 对错位目标计算，实测 loss 11.08（高于均匀分布 7.62），等于没在学。
            flat = batch.reshape(-1, batch.shape[-1]).contiguous()
            labels = batch.transpose(1, 2).contiguous()          # (B, T, K)
            if args.mask_prefix:
                labels[:, :args.prefix_frames, :] = -100
            out = model.decoder(input_ids=flat, encoder_hidden_states=enc, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"[lora] step {step} non-finite loss, skipped", flush=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                cur = (time.time() - t_start) / 60
                hist.append({"step": step, "loss": float(loss)})
                print(f"[lora] step {step}/{args.steps} loss {float(loss):.4f} "
                      f"mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB {cur:.1f}min", flush=True)
    except torch.OutOfMemoryError:
        (RES / "finetune_failure.json").write_text(json.dumps(
            {"error": "CUDA OOM", "steps_done": step, "rank": args.rank,
             "total_frames": args.total_frames,
             "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6}, indent=2))
        print("[lora] OOM — recorded as failure, not as success")
        return

    # merge LoRA into the base weights so inference needs no adapter code
    n_merged = 0
    for layer in model.decoder.model.decoder.layers:
        for name in TARGETS:
            parent_name, child = name.rsplit(".", 1)
            mod = getattr(getattr(layer, parent_name), child)
            if isinstance(mod, LoRALinear):
                mod.base.weight.data = mod.merged_weight().to(mod.base.weight.dtype)
                setattr(getattr(layer, parent_name), child, mod.base)
                n_merged += 1
    model.eval()
    model.decoder.gradient_checkpointing_disable()
    out_dir = CKPT / args.out
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    meta = {"base_model": args.model, "rank": args.rank, "alpha": args.alpha,
            "steps": step, "lr": args.lr, "merged": n_merged,
            "batch_size": args.batch_size, "mask_prefix": bool(args.mask_prefix),
            "train_windows": int(len(tr)), "val_windows": int(len(va)),
            "total_frames": args.total_frames, "prefix_frames": args.prefix_frames,
            "train_minutes": round((time.time() - t_start) / 60, 1),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    (RES / args.result_name).write_text(json.dumps({"meta": meta, "history": hist}, indent=2))
    print("[lora] saved", out_dir, meta)


if __name__ == "__main__":
    main()
