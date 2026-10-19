"""Stage E — turn saved evaluation artefacts into the figures the deck needs.

Reads results/metrics.csv + results/*_eval.json (never hard-codes numbers) and
writes results/figures/*.png plus results/summary.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CJK font: matplotlib's DejaVu has no Chinese glyphs, so register one we have.
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

ROOT = Path(__file__).resolve().parents[1]
import os
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results"))
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

MODEL_LABEL = {"svm": "SVM", "mlp": "MLP", "cnn": "CNN", "crnn": "CRNN",
               "transformer": "Transformer", "ast": "AST(预训练)",
               "cnn_aug": "CNN+SpecAug"}
COLORS = {"svm": "#8a8f98", "mlp": "#6b8ea8", "cnn": "#c2703d", "crnn": "#9c6b8f",
          "transformer": "#4f7a5f", "ast": "#b08d57", "cnn_aug": "#8d5a3b"}


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(RES / "metrics.csv")
    df = df[df["split"] == "test"].copy()
    # one row per (model, seed); for the leaderboard use the mean over seeds
    agg = df.groupby("model").agg(accuracy=("accuracy", "mean"),
                                  macro_f1=("macro_f1", "mean"),
                                  train_time_s=("train_time_s", "mean"),
                                  infer_ms_per_clip=("infer_ms_per_clip", "mean"),
                                  n_params=("n_params", "mean"),
                                  model_size_mb=("model_size_mb", "mean"),
                                  n_seeds=("seed", "nunique")).reset_index()
    agg["label"] = agg["model"].map(MODEL_LABEL)
    return agg


def leaderboard(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    order = agg.sort_values("macro_f1")
    y = np.arange(len(order))
    axes[0].barh(y, order["macro_f1"], color=[COLORS[m] for m in order["model"]])
    axes[0].set_yticks(y, order["label"])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Macro-F1 (test)")
    axes[0].set_title("谁更懂音乐？", loc="left", fontsize=12, fontweight="bold")
    for yi, v in zip(y, order["macro_f1"]):
        axes[0].text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

    s = order.sort_values("train_time_s")
    y2 = np.arange(len(s))
    axes[1].barh(y2, s["train_time_s"], color=[COLORS[m] for m in s["model"]])
    axes[1].set_yticks(y2, s["label"])
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.05, 1000)
    axes[1].set_xticks([0.1, 1, 10, 100])
    axes[1].set_xticklabels(["0.1", "1", "10", "100"])
    axes[1].set_xlabel("训练耗时 (s，对数坐标)")
    axes[1].set_title("计算成本", loc="left", fontsize=12, fontweight="bold")
    for yi, v in zip(y2, s["train_time_s"]):
        axes[1].text(v * 1.25, yi, f"{v:.1f}s", va="center", fontsize=9)
    fig.suptitle("Music AI Lab — GTZAN 10 流派分类擂台", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "leaderboard.png", bbox_inches="tight")
    plt.close(fig)


def eval_files() -> dict[str, Path]:
    """One eval file per model, preferring seed 42 when several runs exist."""
    files: dict[str, Path] = {}
    for p in sorted(RES.glob("*_seed*_eval.json")):
        model = p.name.split("_seed")[0]
        if model not in MODEL_LABEL:
            continue
        if model not in files or p.name.endswith("_seed42_eval.json"):
            files[model] = p
    return dict(sorted(files.items()))


def confusion_panel() -> None:
    rows = []
    for model, p in eval_files().items():
        d = json.loads(p.read_text())
        if "confusion_matrix" not in d.get("test", {}):
            continue
        rows.append((model, d))
    if not rows:
        return
    n = len(rows)
    ncols = 3 if n > 4 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.2 * nrows), squeeze=False)
    classes = rows[0][1]["classes"]
    for ax, (model, d) in zip(axes.ravel(), rows):
        cm = np.array(d["test"]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(f"{MODEL_LABEL[model]}  acc={d['test']['accuracy']:.3f}", fontsize=11)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cmn[i, j] > 0.5:
                    ax.text(j, i, f"{cmn[i,j]:.1f}", ha="center", va="center",
                            fontsize=6, color="white" if cmn[i, j] > 0.75 else "black")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, label="row-normalised")
    fig.savefig(FIG / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def training_curves() -> None:
    rows = []
    for model, p in eval_files().items():
        d = json.loads(p.read_text())
        if d.get("history"):
            rows.append((model, d["history"]))
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.9), squeeze=False)
    for ax, (model, hist) in zip(axes[0], rows):
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_loss"] for h in hist], color=COLORS[model], label="train loss")
        ax.plot(ep, [h["val_loss"] for h in hist], color=COLORS[model], ls="--", label="val loss")
        ax2 = ax.twinx()
        ax2.plot(ep, [h["val_macro_f1"] for h in hist], color="#333", lw=1, label="val F1")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("val Macro-F1（黑线）", fontsize=8)
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def summary(agg: pd.DataFrame) -> None:
    lines = ["# Music AI Lab — MVP 结果汇总", "",
             f"数据：{os.environ.get('MUSIC_SUMMARY_DATA', 'GTZAN 原始 1000 首，清洗后 985 首（剔除 1 首损坏 + 14 首重复）')}，"
             "按曲目分层划分 70/15/15，seed=42。",
             "每个模型使用同一批 5 s 中心片段；表中为测试集结果，多 seed 时取均值。", "",
             "| 模型 | 测试 Accuracy | 测试 Macro-F1 | 训练耗时(s) | 单片段推理(ms) | 参数量 | 模型大小(MB) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in agg.sort_values("macro_f1", ascending=False).iterrows():
        lines.append(f"| {r['label']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['train_time_s']:.1f} | {r['infer_ms_per_clip']:.2f} | "
                     f"{int(r['n_params']):,} | {r['model_size_mb']:.2f} |")
    # hardest genres from the best model
    best = agg.sort_values("macro_f1", ascending=False).iloc[0]
    p = eval_files().get(best["model"])
    if p:
        d = json.loads(p.read_text())
        f1s = sorted(d["test"]["per_class_f1"].items(), key=lambda kv: kv[1])
        lines += ["", f"## 最容易翻车的流派（{best['label']}）", ""]
        for g, f in f1s[:3]:
            lines.append(f"- {g}: F1={f:.3f}")
        lines += ["", f"最强流派：{f1s[-1][0]} (F1={f1s[-1][1]:.3f})"]
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(( RES / "summary.md").read_text())


def main() -> None:
    agg = load_metrics()
    leaderboard(agg)
    confusion_panel()
    training_curves()
    summary(agg)
    print(f"[figures] written to {FIG}")


if __name__ == "__main__":
    main()
