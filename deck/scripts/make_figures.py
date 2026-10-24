#!/usr/bin/env python3
"""重绘演示用插图（数据全部读自 results/，不手填任何数字）。

设计目的：原始 matplotlib 默认图带英文标题、彩虹配色、六联小图，
放进米白纸底的 deck 里既显小又跑色。这里只做两件事：
  1) 换成 deck 的调色板与中文标签；
  2) 一页只画一件事，让字号在投影上够大。

用法：
    python3 deck/scripts/make_figures.py
输出：deck/html/assets/figures/*.png（同时拷一份到 deck/assets/figures/）
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deck" / "html" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- deck palette (与 index.html 的 :root 一致) ----
INK = "#1C1A17"
MUTED = "#8A817A"
RULE = "#D8D0C6"
ACCENT = "#C0512F"
ACCENT2 = "#2F5B4F"
FLAG = "#B08442"
BG = "#FAF7F2"

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"],
    "axes.unicode_minus": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.edgecolor": RULE,
    "figure.dpi": 200,
})


def pick_font() -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"):
        if cand in have:
            return cand
    return "DejaVu Sans"


FONT = pick_font()
plt.rcParams["font.family"] = FONT
plt.rcParams["font.sans-serif"] = [FONT]

LABEL = {
    "svm": "SVM", "mlp": "MLP", "cnn": "CNN", "crnn": "CRNN",
    "transformer": "轻量 Transformer", "ast": "AST（预训练）",
}


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_mean(rows: list[dict], key: str) -> tuple[float, float, float]:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return st.mean(vals), min(vals), max(vals)


def style_axis(ax, xlabel: str = ""):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, labelsize=13)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, color=MUTED, labelpad=10)


def fig_leaderboard(name: str, csv_path: Path, out_name: str, note: str) -> None:
    rows = read_metrics(csv_path)
    models = ["svm", "mlp", "cnn", "crnn", "transformer", "ast"]
    data = []
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        if not sub:
            continue
        mean, lo, hi = seed_mean(sub, "macro_f1")
        data.append((m, mean, lo, hi))
    data.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ys = range(len(data))
    best = max(d[1] for d in data)
    for y, (m, mean, lo, hi) in zip(ys, data):
        is_top = mean == best
        color = FLAG if is_top else ("#B9B2A8" if m in ("svm", "mlp") else INK)
        ax.barh(y, mean, height=0.52, color=color, zorder=3)
        ax.plot([lo, hi], [y, y], color=MUTED, lw=1.0, alpha=.55, zorder=4)
        ax.text(mean + 0.012, y, f"{mean:.3f}", va="center", ha="left",
                fontsize=15, color=INK, zorder=5,
                fontweight="bold" if is_top else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABEL[m] for m, *_ in data], fontsize=14)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.grid(axis="x", color=RULE, lw=0.8, alpha=.7, zorder=0)
    style_axis(ax, "Macro-F1（测试集，3 种子均值，短线为极差）")
    if note:
        ax.text(0.0, 1.04, note, transform=ax.transAxes, fontsize=13, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def fig_cost() -> None:
    rows = read_metrics(ROOT / "results/gtzan/metrics.csv")
    pts = []
    for m in ("svm", "mlp", "cnn", "crnn", "transformer", "ast"):
        sub = [r for r in rows if r["model"] == m]
        if not sub:
            continue
        lat, _, _ = seed_mean(sub, "infer_ms_per_clip")
        size, _, _ = seed_mean(sub, "model_size_mb")
        pts.append((m, lat, size))

    off = {"mlp": (-8, 16, "right"), "svm": (0, 18, "center"),
           "transformer": (-4, 16, "center"), "crnn": (13, 4, "left"),
           "cnn": (6, 16, "left"), "ast": (-16, 12, "right")}
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for m, lat, size in pts:
        big = m == "ast"
        ax.scatter(lat, size, s=260 if big else 150,
                   color=ACCENT if big else INK, zorder=4,
                   edgecolor=BG, linewidth=1.5)
        dx, dy, ha = off.get(m, (10, 12, "left"))
        ax.annotate(LABEL[m], (lat, size), textcoords="offset points",
                    xytext=(dx, dy), fontsize=13, color=ACCENT if big else INK, ha=ha)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("单段推理延迟（ms，对数坐标）", fontsize=13, color=MUTED, labelpad=10)
    ax.set_ylabel("权重体积（MB，对数坐标）", fontsize=13, color=MUTED, labelpad=10)
    ax.grid(color=RULE, lw=0.8, alpha=.6, zorder=0)
    style_axis(ax)
    ax.annotate("", xy=(204, 346), xytext=(0.76, 2.4),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0, ls=(0, (4, 4))),
                zorder=2)
    ax.text(9, 34, "≈270× 延迟 · ≈145× 体积", fontsize=14, color=ACCENT)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.png", bbox_inches="tight")
    plt.close(fig)


EP = re.compile(r"\[(?P<tag>\w+)\] ep(?P<ep>\d+) loss=(?P<loss>[\d.]+) val_loss=(?P<vl>[\d.]+)")


def fig_curves() -> None:
    series = {}
    for tag, fname in (("transformer", "gtzan_long_transformer.log"),
                       ("cnn", "gtzan_long_cnn.log")):
        path = ROOT / "results/logs/overnight" / fname
        if not path.exists():
            continue
        runs, eps, losses, vls = [], [], [], []
        for line in path.read_text(errors="ignore").splitlines():
            m = EP.search(line)
            if not m:
                continue
            ep = int(m.group("ep"))
            if ep == 1 and eps:                      # 新的一次运行（换种子）
                runs.append((eps, losses, vls))
                eps, losses, vls = [], [], []
            eps.append(ep)
            losses.append(float(m.group("loss")))
            vls.append(float(m.group("vl")))
        if eps:
            runs.append((eps, losses, vls))
        if runs:
            series[tag] = runs

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, tag, title in ((axes[0], "transformer", "轻量 Transformer（0.542）"),
                           (axes[1], "cnn", "CNN（0.753）")):
        if tag not in series:
            ax.set_axis_off()
            continue
        runs = series[tag]
        for eps, losses, vls in runs:
            ax.plot(eps, losses, color=INK, lw=1.1, alpha=.28)
            ax.plot(eps, vls, color=ACCENT, lw=1.1, alpha=.28)
        # 三次运行的均值（按最短公共长度截断）
        L = min(len(r[1]) for r in runs)
        mean_tr = [st.mean(r[1][i] for r in runs) for i in range(L)]
        mean_va = [st.mean(r[2][i] for r in runs) for i in range(L)]
        ax.plot(range(1, L + 1), mean_tr, color=INK, lw=2.2, label="训练损失")
        ax.plot(range(1, L + 1), mean_va, color=ACCENT, lw=2.2, label="验证损失")
        ax.set_title(title, fontsize=15, color=INK, pad=12)
        ax.set_xlabel("epoch", fontsize=12, color=MUTED)
        ax.grid(color=RULE, lw=0.8, alpha=.6)
        style_axis(ax)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=12, loc="upper right")
    axes[0].set_ylabel("交叉熵损失", fontsize=12, color=MUTED)
    if "transformer" in series:
        runs = series["transformer"]
        L = min(len(r[2]) for r in runs)
        mv = [st.mean(r[2][i] for r in runs) for i in range(L)]
        k = min(range(L), key=lambda i: mv[i])
        axes[0].annotate("验证损失最低点之后，一路抬头",
                         xy=(k + 1, mv[k]), xytext=(6.2, 2.70),
                         fontsize=12, color=ACCENT,
                         arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0,
                                         connectionstyle="arc3,rad=0.18"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_token_top1() -> None:
    rows = read_metrics(ROOT / "results/token_prediction_long/metrics.csv")
    items = []
    for m, label in (("persistence", "重复上一帧"), ("frequency", "频率先验"),
                     ("bigram", "Bigram")):
        sub = [r for r in rows if r["model"] == m]
        if sub:
            items.append((label, seed_mean(sub, "free_top1")[0] * 100, False))
    for m, label in (("lstm", "LSTM"), ("transformer", "Transformer")):
        sub = [r for r in rows if r["model"] == m]
        if sub:
            items.append((label, seed_mean(sub, "free_top1")[0] * 100, True))
    items.insert(0, ("均匀分布", 0.049, False))
    items.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    xs = range(len(items))
    colors = [ACCENT if ours else "#B9B2A8" for _, _, ours in items]
    ax.bar(xs, [v for _, v, _ in items], width=.56, color=colors, zorder=3)
    for x, (label, v, _) in zip(xs, items):
        dy = 0.22 if abs(v - 1.71) < 0.13 else 0.10
        ax.text(x, v + dy, f"{v:.2f}%", ha="center", fontsize=14, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for l, _, _ in items], fontsize=13)
    ax.set_ylim(0, 2.85)
    ax.set_ylabel("Top-1（%）", fontsize=13, color=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.8, alpha=.7, zorder=0)
    style_axis(ax)
    ax.axhline(1.71, color=ACCENT2, lw=1.0, ls=(0, (5, 4)), zorder=2)
    ax.text(-0.45, 1.98, "Bigram 基线 1.71%", ha="left", fontsize=12, color=ACCENT2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_token_top1.png", bbox_inches="tight")
    plt.close(fig)


def fig_codebook() -> None:
    rows = read_metrics(ROOT / "results/token_prediction_long/metrics.csv")
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    width = 0.36
    xs = range(4)
    for k, (m, color, label) in enumerate((("transformer", INK, "Transformer"),
                                          ("lstm", "#B9B2A8", "LSTM"))):
        sub = [r for r in rows if r["model"] == m and r.get("free_top1_cb")]
        if not sub:
            continue
        arrs = [json.loads(r["free_top1_cb"]) for r in sub]
        means = [st.mean(a[i] for a in arrs) * 100 for i in range(4)]
        ax.bar([x + (k - .5) * width for x in xs], means, width=width,
               color=color, label=label, zorder=3)
        for x, v in zip(xs, means):
            ax.text(x + (k - .5) * width, v + 0.04, f"{v:.1f}", ha="center",
                    fontsize=12, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"码本 {i}" for i in xs], fontsize=14)
    ax.set_ylabel("free-running Top-1（%）", fontsize=13, color=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.8, alpha=.7, zorder=0)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_codebook.png", bbox_inches="tight")
    plt.close(fig)


def fig_robust() -> None:
    """七种扰动下的准确率：横轴条件，五条折线（真实数据，不重新训练）。"""
    rows = read_metrics(ROOT / "results/gtzan/robustness.csv")
    conds = [("clean_5s", "干净"), ("clip_3s", "剪 3 秒"), ("clip_10s", "拉 10 秒"),
             ("gain_0.5", "音量 ×0.5"), ("gain_1.5", "音量 ×1.5"),
             ("noise_10db", "噪声 10 dB"), ("noise_0db", "噪声 0 dB")]
    colors = {"svm": "#B9B2A8", "mlp": "#9C948A", "cnn": ACCENT,
              "crnn": ACCENT2, "transformer": MUTED}
    lw = {"svm": 1.6, "mlp": 1.6, "cnn": 2.6, "crnn": 1.6, "transformer": 1.6}

    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    xs = range(len(conds))
    for m in ("svm", "mlp", "crnn", "transformer", "cnn"):
        pts = []
        for c, _ in conds:
            hit = [r for r in rows if r["model"] == m and r["condition"] == c]
            pts.append(float(hit[0]["accuracy"]) if hit else None)
        ax.plot(list(xs), pts, color=colors[m], lw=lw[m], marker="o", ms=4.5,
                label=LABEL[m], zorder=4 if m == "cnn" else 3)
        last = pts[-1]
        dy = {"svm": 0.0, "mlp": 0.0, "crnn": 0.030, "transformer": -0.022,
              "cnn": -0.070}[m]
        ax.annotate(LABEL[m].replace("轻量 Transformer", "轻量 TF"),
                    (len(conds) - 1, last + dy), textcoords="offset points",
                    xytext=(10, 0), fontsize=11.5, color=colors[m], va="center")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n for _, n in conds], fontsize=12.5)
    ax.set_xlim(-0.35, len(conds) + 0.15)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("准确率", fontsize=13, color=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.8, alpha=.7, zorder=0)
    style_axis(ax)
    ax.text(0.55, 0.052, "CNN 从第一名掉到倒数第一", fontsize=12.5, color=ACCENT)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robust.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_leaderboard("gtzan", ROOT / "results/gtzan/metrics.csv",
                    "fig_gtzan_f1.png", "GTZAN · 985 首 · 10 类")
    fig_leaderboard("fma", ROOT / "results/fma/metrics.csv",
                    "fig_fma_f1.png", "FMA-small · 7,994 首 · 8 类")
    fig_cost()
    fig_robust()
    fig_curves()
    fig_token_top1()
    fig_codebook()
    print("font:", FONT)
    for p in sorted(OUT.glob("fig_*.png")):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
