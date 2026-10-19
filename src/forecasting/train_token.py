"""Phase 1: train / evaluate next-token forecasters on EnCodec tokens.

Protocol (identical to the classification experiments):
  train  -> parameter learning
  val    -> early stopping + hyper-parameter choice, never seen by test
  test   -> evaluated once, reported

Both the neural models and the count-based baselines see exactly the same test
windows and the same metric code, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.forecasting.token_models import (N_CODEBOOKS, N_FRAMES, PREFIX_FRAMES, TARGET_FRAMES,
                                          VOCAB, Bigram, FrequencyPrior, Persistence,
                                          accuracies, build_model, count_params, free_running,
                                          load_windows, split_arrays, token_nll)

ROOT = Path(__file__).resolve().parents[2]
TOKENS = Path(os.environ.get("MUSIC_TOKENS", ROOT / "data/features/token_prediction"))
CKPT = Path(os.environ.get("MUSIC_CKPT", ROOT / "checkpoints/token_prediction"))
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/token_prediction"))
SEEDS = (42, 123, 2024)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score(pred: np.ndarray, y: np.ndarray) -> dict:
    """Top-1 accuracy of a [N, 4, 50] prediction against the true frames."""
    tgt = y[:, :, PREFIX_FRAMES:]
    hit = (pred == tgt)
    return {"top1": float(hit.mean()),
            "top1_cb": [float(hit[:, k].mean()) for k in range(N_CODEBOOKS)],
            "top1_cb0": float(hit[:, 0].mean())}


# --------------------------------------------------------------------------- #
# neural training
# --------------------------------------------------------------------------- #
def train_one(kind: str, seed: int, tr: dict, va: dict, device: str,
              epochs: int, batch: int, lr: float, patience: int) -> tuple:
    set_seed(seed)
    model = build_model(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    xtr = torch.from_numpy(tr["codes"])
    ytr = xtr
    xva = torch.from_numpy(va["codes"]).to(device)
    n = len(xtr)
    best = {"val_nll": float("inf"), "epoch": -1, "state": None}
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = xtr[idx].to(device)
            loss = token_nll(model(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(float(token_nll(model(xva[j:j + 256]), xva[j:j + 256])) * len(xva[j:j + 256])
                     for j in range(0, len(xva), 256)) / len(xva)
        hist.append({"epoch": ep, "train_nll": tot / cnt, "val_nll": vl})
        print(f"  [{kind} seed{seed}] ep{ep:02d} train {tot / cnt:.4f} val {vl:.4f}", flush=True)
        if vl < best["val_nll"] - 1e-4:
            best = {"val_nll": vl, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif ep - best["epoch"] >= patience:
            break
    train_s = time.time() - t0
    model.load_state_dict(best["state"])
    model.eval()
    return model, best, hist, train_s


@torch.no_grad()
def evaluate_neural(model, te: dict, device: str, batch: int = 256) -> dict:
    x = torch.from_numpy(te["codes"]).to(device)
    nll, logits_all = 0.0, []
    for j in range(0, len(x), batch):
        lg = model(x[j:j + batch])
        nll += float(token_nll(lg, x[j:j + batch])) * len(x[j:j + batch])
        logits_all.append(lg.cpu())
    logits = torch.cat(logits_all)
    nll /= len(x)
    teacher = accuracies(logits, torch.from_numpy(te["codes"]))
    t0 = time.time()
    pred = free_running(model, torch.from_numpy(te["codes"]), device, batch=128)
    fr_time = time.time() - t0
    free = score(pred.numpy(), te["codes"])
    return {"nll": nll, "teacher_top1": teacher["top1"], "teacher_top5": teacher["top5"],
            "teacher_top1_cb0": teacher["top1_cb"][0],
            "free_top1": free["top1"], "free_top1_cb0": free["top1_cb0"],
            "free_top1_cb": free["top1_cb"],
            "rollout_s_per_1k": fr_time / max(1, len(x)) * 1000,
            "params": count_params(model),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["lstm", "transformer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    data = load_windows(TOKENS / "shards")
    tr, va, te = (split_arrays(data, s) for s in ("train", "val", "test"))
    print(f"[data] train {len(tr['codes'])} val {len(va['codes'])} test {len(te['codes'])} "
          f"tracks {len(set(te['track_id']))}")

    rows: list[dict] = []

    # ---- count-based baselines (same test set, same metric) ----
    for bl in (Persistence(), FrequencyPrior(tr["codes"]), Bigram(tr["codes"])):
        t0 = time.time()
        pred = bl.predict(te["codes"])
        dt = time.time() - t0
        s = score(pred, te["codes"])
        rows.append({"model": bl.name, "seed": 0, "nll": float("nan"),
                     "teacher_top1": s["top1"], "teacher_top5": float("nan"),
                     "teacher_top1_cb0": s["top1_cb0"],
                     "free_top1": s["top1"], "free_top1_cb0": s["top1_cb0"],
                     "rollout_s_per_1k": dt / len(te["codes"]) * 1000, "params": 0,
                     "peak_mem_mb": 0.0, "train_s": 0.0, "best_epoch": 0})
        print(f"[baseline] {bl.name}: top1 {s['top1']:.4f} cb0 {s['top1_cb0']:.4f}")

    # ---- neural models, multiple seeds ----
    epochs = 3 if args.quick else args.epochs
    for kind in args.kinds:
        for seed in args.seeds:
            model, best, hist, train_s = train_one(kind, seed, tr, va, device,
                                                   epochs, args.batch, args.lr, args.patience)
            metrics = evaluate_neural(model, te, device)
            torch.save({"state_dict": model.state_dict(), "kind": kind, "seed": seed,
                        "val_nll": best["val_nll"]},
                       CKPT / f"{kind}_seed{seed}.pt")
            (RES / f"{kind}_seed{seed}_history.json").write_text(
                json.dumps(hist, indent=1))
            row = {"model": kind, "seed": seed, **metrics,
                   "train_s": train_s, "best_epoch": best["epoch"]}
            rows.append(row)
            print(f"[{kind} seed{seed}] teacher top1 {metrics['teacher_top1']:.4f} "
                  f"free top1 {metrics['free_top1']:.4f} nll {metrics['nll']:.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics.csv", index=False)
    print(df[["model", "seed", "teacher_top1", "free_top1", "free_top1_cb0", "nll"]])
    (RES / "summary.json").write_text(json.dumps(
        dict(n_train=len(tr["codes"]), n_val=len(va["codes"]), n_test=len(te["codes"]),
             n_tracks_test=len(set(te["track_id"])),
             prefix_frames=PREFIX_FRAMES, target_frames=TARGET_FRAMES,
             vocab=VOCAB, n_codebooks=N_CODEBOOKS, rows=len(df)), indent=2))


if __name__ == "__main__":
    main()
