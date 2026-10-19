"""Stage D — train the deep baselines (CNN / CRNN / small Transformer) on Log-Mel.

Same split file, same cached clips as the classical baselines. Test is only
touched after early stopping has picked an epoch using the validation split.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (append_metrics, encode_labels, evaluate, save_json,  # noqa: E402
                         save_predictions)
from src.models.cnn import build, count_params  # noqa: E402

DATA = ROOT / "data/processed/features.npz"
CKPT = ROOT / "checkpoints"


def load_mel():
    d = np.load(DATA, allow_pickle=True)
    X = d["X_mel"].astype(np.float32)[:, None, :, :]  # (N,1,64,216)
    y, le = encode_labels(d["genres"])
    return X, y, d["splits"], le, d["track_ids"]


def make_loader(X, y, bs: int, shuffle: bool, seed: int = 0, device=None):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    g = torch.Generator().manual_seed(seed or 0)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, generator=g,
                      pin_memory=(device is not None and device.type == "cuda"))


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    logits, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits.append(model(xb).float().cpu())
        ys.append(yb)
    return torch.cat(logits), torch.cat(ys)


def macro_f1(y_true, y_pred, n_classes):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)),
                    zero_division=0)


def run(model_name: str, seed: int, epochs: int, bs: int, lr: float,
        patience: int, monitor: str = "macro_f1") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Deterministic kernels: without this, cuDNN autotuning moves test accuracy by
    # a few samples between identical runs (observed 76.2% -> 73.5% on seed 123).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    X, y, splits, le, track_ids = load_mel()
    tr, va, te = (splits == s for s in ("train", "val", "test"))

    # Normalisation statistics come from TRAIN ONLY.
    mu = float(X[tr].mean())
    sd = float(X[tr].std() + 1e-6)
    Xn = (X - mu) / sd

    train_loader = make_loader(Xn[tr], y[tr], bs, True, seed, device)
    val_loader = make_loader(Xn[va], y[va], bs, False, None, device)
    test_loader = make_loader(Xn[te], y[te], bs, False, None, device)

    model = build(model_name, n_classes=len(le.classes_)).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    hist, best, best_state, bad = [], -1.0, None, 0
    best_ep, best_val = 0, {}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        ep0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        train_loss = tot / n

        vlogits, vy = infer(model, val_loader, device)
        vpred = vlogits.argmax(1).numpy()
        v_acc = float((vpred == vy.numpy()).mean())
        v_f1 = float(macro_f1(vy.numpy(), vpred, len(le.classes_)))
        val_loss = float(nn.functional.cross_entropy(vlogits, vy).item())
        hist.append({"epoch": ep, "train_loss": round(train_loss, 4),
                     "val_loss": round(val_loss, 4), "val_acc": round(v_acc, 4),
                     "val_macro_f1": round(v_f1, 4),
                     "epoch_time_s": round(time.time() - ep0, 2)})
        score = v_f1 if monitor == "macro_f1" else v_acc
        print(f"[{model_name}] ep{ep:02d} loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"acc={v_acc:.3f} f1={v_f1:.3f} ({time.time()-ep0:.1f}s)", flush=True)
        if score > best:
            best, bad = score, 0
            best_ep = ep
            best_val = {"acc": v_acc, "macro_f1": v_f1, "loss": val_loss}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{model_name}] early stop at epoch {ep}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # ---- test, once ----
    t0 = time.time()
    logits, yt = infer(model, test_loader, device)
    infer_ms = (time.time() - t0) / max(1, len(yt)) * 1000
    proba = torch.softmax(logits, 1).numpy()
    pred = proba.argmax(1)
    res = evaluate(yt.numpy(), pred, proba, list(le.classes_))

    CKPT.mkdir(exist_ok=True)
    path = CKPT / f"{model_name}_seed{seed}.pt"
    torch.save({"model": model_name,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": list(le.classes_), "norm": {"mean": mu, "std": sd},
                "n_mels": 64, "n_frames": X.shape[-1], "seed": seed,
                "protocol": "gtzan-5s-center-v1"}, path)
    save_json({"history": hist,
               "val": best_val, "best_epoch": best_ep,
               "test": {k: v for k, v in res.items() if k != "report"},
               "test_report": res["report"], "classes": list(le.classes_)},
              ROOT / "results" / f"{model_name}_seed{seed}_eval.json")
    save_predictions(model_name, seed, "test", track_ids[te],
                     le.inverse_transform(yt.numpy()), le.inverse_transform(pred),
                     proba, list(le.classes_))

    row = {"model": model_name, "seed": seed, "split": "test", "n_samples": int(te.sum()),
           "accuracy": round(res["accuracy"], 4), "macro_f1": round(res["macro_f1"], 4),
           "train_time_s": round(train_time, 2), "infer_ms_per_clip": round(infer_ms, 3),
           "n_params": n_params, "model_size_mb": round(path.stat().st_size / 1e6, 3),
           "notes": f"best_ep={best_ep};best_val_f1={best:.4f};{device.type}"}
    print(f"[{model_name}] seed={seed} TEST acc={row['accuracy']:.4f} "
          f"macroF1={row['macro_f1']:.4f} params={n_params:,} "
          f"train={train_time:.1f}s size={row['model_size_mb']:.2f}MB")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "crnn", "transformer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for seed in args.seeds:
        rows.append(run(args.model, seed, args.epochs, args.batch_size, args.lr, args.patience))
    append_metrics(rows)


if __name__ == "__main__":
    main()
