"""功能 C：一首歌的「性格变化」——把音频切成滑动窗口逐段预测。

这不是预测未来音频，而是对不同时间片段分别做流派分类，用来展示
「同一首歌在不同段落里，模型听到的东西会变」。

CLI:
  python -m src.timeline data/raw/gtzan/genres/pop/pop.00000.wav \
      --model cnn --window 5 --hop 2.5
输出：
  results/gtzan/timeline.csv              每个窗口的 top-1 与全类别概率
  results/gtzan/figures/timeline_<id>.png 时间轴热力图
也可被 Streamlit Demo 直接 import。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for _f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"):
    if Path(_f).exists():
        try:
            font_manager.fontManager.addfont(_f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
            break
        except Exception:  # noqa: BLE001
            pass
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "gtzan"
CLASSICAL = {"svm", "mlp"}

FRAME_SR = 22050
N_FRAMES = 216


def _load_full(path: str, sr: int = FRAME_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _window_features(y: np.ndarray, window: float, sr: int = FRAME_SR):
    from src.data.audio_features import handcrafted, log_mel

    n = int(round(window * sr))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return handcrafted(y), log_mel(y)


def _load_model(model: str):
    if model in CLASSICAL:
        cands = sorted(CKPT.glob(f"{model}_seed*.joblib"))
        if not cands:
            raise FileNotFoundError(model)
        ck = next((c for c in cands if c.stem.endswith("_seed42")), cands[0])
        bundle = joblib.load(ck)
        return ("classical", bundle, ck)
    from src.models.cnn import build
    cands = sorted(CKPT.glob(f"{model}_seed*.pt"))
    if not cands:
        raise FileNotFoundError(model)
    ck = next((c for c in cands if c.stem.endswith("_seed42")), cands[0])
    state = torch.load(ck, map_location="cpu")
    net = build(state["model"], n_classes=len(state["classes"]))
    net.load_state_dict(state["state_dict"])
    net.eval()
    return ("deep", state, ck, net)


def predict_windows(y: np.ndarray, model: str = "cnn", window: float = 5.0,
                    hop: float = 2.5, sr: int = FRAME_SR, min_window: float = 2.0) -> pd.DataFrame:
    """Return one row per window: start, end, top1, prob, and per-class probabilities."""
    kind = _load_model(model)
    dur = len(y) / sr
    end_limit = max(0.0, dur - window)
    starts = list(np.arange(0.0, end_limit + 1e-9, hop))
    if not starts:
        starts = [0.0]
    # A track may be a few ms longer than 30 s; don't create a near-duplicate tail window.
    if end_limit > 0 and end_limit - starts[-1] > 0.1:
        starts.append(end_limit)
    starts = sorted({round(s, 2) for s in starts})

    rows = []
    for s in starts:
        seg = y[int(round(s * sr)): int(round(s * sr + window * sr))]
        if len(seg) < int(min_window * sr):
            continue
        feat, mel = _window_features(seg, window, sr)
        if kind[0] == "classical":
            bundle = kind[1]
            proba = bundle["model"].predict_proba(feat[None, :])[0]
            classes = bundle["classes"]
        else:
            state, net = kind[1], kind[3]
            m = (mel[None, None, :, :] - state["norm"]["mean"]) / state["norm"]["std"]
            with torch.no_grad():
                logits = net(torch.from_numpy(m.astype(np.float32)))
            proba = torch.softmax(logits, 1)[0].numpy()
            classes = state["classes"]
        order = np.argsort(proba)[::-1]
        row = {"start_s": round(float(s), 2),
               "end_s": round(float(min(s + window, dur)), 2),
               "top1": classes[order[0]], "prob": round(float(proba[order[0]]), 4)}
        for c, p in zip(classes, proba):
            row[f"p_{c}"] = round(float(p), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_timeline(df: pd.DataFrame, title: str = "", path: Path | None = None):
    """Heatmap: x = window start time, y = genre, colour = probability."""
    pcols = [c for c in df.columns if c.startswith("p_")]
    classes = [c[2:] for c in pcols]
    mat = df[pcols].to_numpy().T
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.7), 3.6))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xticks(range(len(df)), [f"{v:.1f}" for v in df["start_s"]], fontsize=8)
    ax.set_xlabel("窗口起始时间 (s)", fontsize=9)
    ax.set_ylabel("流派", fontsize=9)
    for j, (_, r) in enumerate(df.iterrows()):
        i = classes.index(r["top1"])
        ax.scatter(j, i, s=70, facecolors="none", edgecolors="#7CFC00", linewidths=1.6)
    ax.set_title(title or "同一首歌在不同时间窗口下的流派判断", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="概率")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="cnn", choices=["svm", "mlp", "cnn", "crnn", "transformer"])
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--hop", type=float, default=2.5)
    ap.add_argument("--csv", default=str(ROOT / "results/gtzan/timeline.csv"))
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    y = _load_full(args.audio)
    df = predict_windows(y, args.model, args.window, args.hop)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    stem = Path(args.audio).stem
    fig_path = Path(args.figure) if args.figure else ROOT / f"results/gtzan/figures/timeline_{stem}.png"
    fig = plot_timeline(df, f"{stem} · {args.model.upper()} · 每 {args.window:g}s 一窗，步长 {args.hop:g}s",
                        fig_path)
    plt.close(fig)

    print(df[["start_s", "end_s", "top1", "prob"]].to_string(index=False))
    changes = (df["top1"] != df["top1"].shift()).sum() - 1
    print(f"[timeline] {len(df)} windows, {changes} 次判断变化 -> {args.csv}, {fig_path}")


if __name__ == "__main__":
    main()
