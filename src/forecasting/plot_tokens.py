"""Figures for phase 1: learning curves, model comparison, per-codebook accuracy."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    str(Path.home() / ".fonts/wqy-microhei.ttc"),
]
for _f in _FONT_CANDIDATES:
    if Path(_f).exists():
        try:
            font_manager.fontManager.addfont(_f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
            break
        except Exception:  # noqa: BLE001
            continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/token_prediction"))
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LABEL = {"persistence": "Persistence", "frequency": "频率先验", "bigram": "Bigram",
         "lstm": "LSTM", "transformer": "Transformer"}
COLOR = {"persistence": "#9aa0a6", "frequency": "#b9a17a", "bigram": "#c9b79c",
         "lstm": "#6b8ea8", "transformer": "#c2703d"}
ORDER = ["persistence", "frequency", "bigram", "lstm", "transformer"]


def curves() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for kind, ax in zip(("lstm", "transformer"), axes):
        for i, seed in enumerate((42, 123, 2024)):
            p = RES / f"{kind}_seed{seed}_history.json"
            if not p.exists():
                continue
            h = pd.DataFrame(json.loads(p.read_text()))
            ax.plot(h.epoch, h.train_nll, color=COLOR[kind], alpha=0.45, lw=1.1,
                    ls="--" if i else "-")
            ax.plot(h.epoch, h.val_nll, color=COLOR[kind], alpha=0.9, lw=1.6,
                    label="val" if i == 0 else None)
        ax.set_title(LABEL[kind])
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy (nats/token)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("训练/验证 NLL（虚线 train，实线 val；三个种子）", fontsize=9, y=0.99)
    fig.tight_layout()
    out = FIG / "token_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def bars(df: pd.DataFrame) -> Path:
    g = df[df.seed > 0].groupby("model")[["teacher_top1", "free_top1"]].agg(["mean", "std"])
    base = df[df.seed == 0].set_index("model")
    models = [m for m in ORDER if m in set(df.model)]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.36
    for k, (col, lab) in enumerate((("teacher_top1", "teacher-forced"), ("free_top1", "free-running"))):
        if col == "teacher_top1":
            vals = [g.loc[m, (col, "mean")] if m in g.index else base.loc[m, col] for m in models]
            errs = [g.loc[m, (col, "std")] if m in g.index else 0 for m in models]
        else:
            vals = [g.loc[m, (col, "mean")] if m in g.index else base.loc[m, col] for m in models]
            errs = [g.loc[m, (col, "std")] if m in g.index else 0 for m in models]
        ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=["#9aa0a6", "#c2703d"][k], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models])
    ax.set_ylabel("Top-1 token accuracy")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(frameon=False)
    ax.set_title("测试集 Top-1（4 码本平均；误差棒为 3 个种子）", fontsize=9)
    fig.tight_layout()
    out = FIG / "token_top1.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def per_codebook(df: pd.DataFrame) -> Path:
    import ast
    rows = df[df.seed == 42]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(4)
    for _, r in rows.iterrows():
        vals = ast.literal_eval(r["free_top1_cb"]) if isinstance(r["free_top1_cb"], str) else r["free_top1_cb"]
        ax.bar(x + (list(rows.model).index(r.model) - len(rows) / 2) * 0.12, vals, 0.12,
               label=LABEL.get(r.model, r.model), color=COLOR.get(r.model, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"码本 {k}" for k in range(4)])
    ax.set_ylabel("free-running Top-1")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("逐码本 free-running Top-1（seed 42）", fontsize=9)
    fig.tight_layout()
    out = FIG / "token_per_codebook.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def decode_example() -> Path | None:
    p = RES / "decode.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    items = d["items"][:4]
    if not items:
        return None
    import librosa
    import soundfile as sf
    fig, axes = plt.subplots(2, len(items), figsize=(3.0 * len(items), 3.2), squeeze=False)
    for j, it in enumerate(items):
        yp, _ = sf.read(RES / it["pred_wav"])
        yt, _ = sf.read(RES / it["true_wav"])
        mp = librosa.power_to_db(librosa.feature.melspectrogram(y=yp, sr=32000, n_mels=64) + 1e-9)
        mt = librosa.power_to_db(librosa.feature.melspectrogram(y=yt, sr=32000, n_mels=64) + 1e-9)
        axes[0, j].imshow(mt, origin="lower", aspect="auto", cmap="magma")
        axes[1, j].imshow(mp, origin="lower", aspect="auto", cmap="magma")
        axes[0, j].set_title(f"{it['genre']}\n真实 1 s", fontsize=8)
        axes[1, j].set_title("预测 1 s", fontsize=8)
        for a in (axes[0, j], axes[1, j]):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("预测 Token 解码后的 Log-Mel 与真实后 1 秒对照", fontsize=9)
    fig.tight_layout()
    out = FIG / "token_decode.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(RES / "metrics.csv")
    outs = [curves(), bars(df), per_codebook(df)]
    outs.append(decode_example())
    print("[figures]", [str(o) for o in outs if o])


if __name__ == "__main__":
    main()
