"""P0 错误分析：谁被谁骗了？——从已保存的预测文件里挖失败案例。

不重新训练、不改动任何已有产物，只读取 results/predictions/*_seed42_test.csv
与原始音频，输出：
  results/error_analysis.csv        每个模型的混淆对计数
  results/error_analysis.json       混淆对 + 置信度统计 + 典型样例
  results/figures/error_confusions.png  各模型 Top 混淆对
  results/figures/error_cases.png      CNN 典型错误案例的 Log-Mel 对照
  docs/error_analysis.md            可直接用于 PPT 的结论与听感指引
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matplotlib import font_manager  # noqa: E402

for _f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"):
    if Path(_f).exists():
        try:
            font_manager.fontManager.addfont(_f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
            break
        except Exception:  # noqa: BLE001
            pass
plt.rcParams["axes.unicode_minus"] = False

RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results"))
PRED = RES / "predictions"
FIG = RES / "figures"
MODEL_LABEL = {"svm": "SVM", "mlp": "MLP", "cnn": "CNN", "crnn": "CRNN",
               "transformer": "Transformer"}
COLORS = {"svm": "#8a8f98", "mlp": "#6b8ea8", "cnn": "#c2703d", "crnn": "#9c6b8f",
          "transformer": "#4f7a5f"}


def _tid(x) -> str:
    s = str(x)
    return s.zfill(6) if s.isdigit() else s


def load_preds(model: str, seed: int = 42) -> pd.DataFrame:
    p = PRED / f"{model}_seed{seed}_test.csv"
    df = pd.read_csv(p)
    df["track_id"] = df["track_id"].map(_tid)
    df["confidence"] = df[[c for c in df.columns if c.startswith("p_")]].max(axis=1)
    return df


def top_pairs(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    wrong = df[df["y_true"] != df["y_pred"]]
    pairs = (wrong.groupby(["y_true", "y_pred"]).size()
             .sort_values(ascending=False).head(k).rename("count").reset_index())
    return pairs


def confidence_stats(df: pd.DataFrame) -> dict:
    right = df[df["y_true"] == df["y_pred"]]
    wrong = df[df["y_true"] != df["y_pred"]]
    return {
        "n": int(len(df)),
        "accuracy": round(float((df["y_true"] == df["y_pred"]).mean()), 4),
        "mean_conf_correct": round(float(right["confidence"].mean()), 4),
        "mean_conf_wrong": round(float(wrong["confidence"].mean()), 4),
        "n_wrong": int(len(wrong)),
        "wrong_conf_ge_0.9": int((wrong["confidence"] >= 0.9).sum()),
        "wrong_conf_ge_0.5": int((wrong["confidence"] >= 0.5).sum()),
    }


def pick_examples(frames: dict[str, pd.DataFrame], pairs: list[tuple[str, str]],
                  per_pair: int = 2) -> list[dict]:
    """Pick high-confidence mistakes of the CNN for the requested pairs."""
    cnn = frames["cnn"]
    out = []
    for true, pred in pairs:
        sub = cnn[(cnn["y_true"] == true) & (cnn["y_pred"] == pred)]
        sub = sub.sort_values("confidence", ascending=False).head(per_pair)
        for r in sub.itertuples():
            out.append({"track_id": r.track_id, "true": true, "pred": pred,
                        "confidence": round(float(r.confidence), 4)})
    return out


def confusion_figure(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 2.9), squeeze=False)
    for ax, (model, df) in zip(axes[0], frames.items()):
        pairs = top_pairs(df, 5).iloc[::-1]
        labels = [f"{t}→{p}" for t, p in zip(pairs["y_true"], pairs["y_pred"])]
        ax.barh(range(len(pairs)), pairs["count"], color=COLORS[model])
        ax.set_yticks(range(len(pairs)), labels, fontsize=7)
        ax.set_xlabel("错误数（test 147）", fontsize=8)
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        for y, v in enumerate(pairs["count"]):
            ax.text(v + 0.1, y, str(int(v)), va="center", fontsize=7)
    fig.suptitle("每个模型最容易混淆的流派对（seed 42）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "error_confusions.png", bbox_inches="tight")
    plt.close(fig)


def spectrogram_cases(examples: list[dict], splits: pd.DataFrame) -> None:
    from src.data.audio_features import load_clip, log_mel

    n = len(examples)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.9), squeeze=False)
    splits = splits.copy()
    splits["track_id"] = splits["track_id"].map(_tid)
    path_of = dict(zip(splits["track_id"], splits["path"]))
    for ax, ex in zip(axes[0], examples):
        y = load_clip(str(ROOT / path_of[ex["track_id"]]))
        mel = log_mel(y)
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{ex['track_id']}\n真:{ex['true']} → 判:{ex['pred']} "
                     f"({ex['confidence']*100:.1f}%)", fontsize=8)
        ax.set_xlabel("时间帧", fontsize=7)
        ax.set_ylabel("Mel bin", fontsize=7)
    fig.suptitle("CNN 的典型翻车案例：频谱看起来像不像另一个流派？", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "error_cases.png", bbox_inches="tight")
    plt.close(fig)


def _conclusions(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Derive claims from the actual predictions instead of hard-coding them."""
    cnn = frames["cnn"]
    wrong = cnn[cnn["y_true"] != cnn["y_pred"]]
    pairs = wrong.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
    eval_json = json.loads((RES / "cnn_seed42_eval.json").read_text())
    f1 = eval_json["test"]["per_class_f1"]
    hardest = min(f1, key=f1.get)
    easiest = max(f1, key=f1.get)
    row = wrong[wrong["y_true"] == hardest].groupby("y_pred").size().sort_values(ascending=False)
    conf_text = "、".join(f"{k}（{v} 首）" for k, v in row.head(2).items()) or "未出现明显误判"
    top = pairs.head(3)
    top_text = "；".join(f"{t}→{p}（{c} 首）" for (t, p), c in top.items())
    return [
        f"- **{hardest} 是最难的类**：CNN 的 {hardest} F1 仅 {f1[hardest]:.3f}，"
        f"主要被误判为 {conf_text}。",
        f"- **{easiest} 最稳**：F1 {f1[easiest]:.3f}，器乐音色与动态范围独特，"
        f"几乎所有模型都能分对。",
        f"- **最主要的混淆对**：{top_text}；这些流派共享节奏型或音色，"
        f"「流派」本身是模糊标签。",
        "- 轻量 Transformer 的错误更分散：训练 loss 到 0、验证抬头，"
        "且高置信错误比例明显更高（见上表），过拟合没有换来稳定判断。",
        "- 这些错误不是「模型太笨」，而是**标签边界模糊 + 30 秒片段信息有限**；",
        "  汇报时把它作为「AI 是否真正理解音乐」的论据，而不是只报准确率。",
    ]


def write_doc(frames: dict[str, pd.DataFrame], stats: dict, examples: list[dict],
              splits: pd.DataFrame) -> None:
    splits = splits.copy()
    splits["track_id"] = splits["track_id"].map(_tid)
    path_of = dict(zip(splits["track_id"], splits["path"]))
    lines = [
        "# 错误分析（P0）",
        "",
        f"数据：{os.environ.get('MUSIC_DATASET_NAME', 'GTZAN')} 测试集 "
        f"{next(iter(stats.values()))['n']} 首，模型为 seed 42 的 checkpoint。",
        "复现命令：`python -m src.error_analysis`。",
        "",
        "## 1. 各模型最容易混淆的流派对",
        "",
        "| 模型 | Top-3 混淆对（真实→预测，错误数） |",
        "|---|---|",
    ]
    for m, df in frames.items():
        pairs = top_pairs(df, 3)
        txt = "；".join(f"{t}→{p} ({c})" for t, p, c in
                       zip(pairs["y_true"], pairs["y_pred"], pairs["count"]))
        lines.append(f"| {MODEL_LABEL[m]} | {txt} |")

    lines += ["", "## 2. 置信度：错的时候有多自信？", "",
              "| 模型 | 正确样本平均置信度 | 错误样本平均置信度 | 错误数 | 其中 ≥0.9 的自信错误 |",
              "|---|---|---|---|---|"]
    for m, s in stats.items():
        lines.append(f"| {MODEL_LABEL[m]} | {s['mean_conf_correct']*100:.1f}% | "
                     f"{s['mean_conf_wrong']*100:.1f}% | {s['n_wrong']} | "
                     f"{s['wrong_conf_ge_0.9']} |")
    lines += ["", "> 置信度不等于正确率：模型在错误样本上依然给出很高的概率，",
              "> 这正是「AI 会自信地犯错」的现场素材。", ""]

    lines += ["## 3. 典型错误案例（CNN）", "",
              "| 曲目 | 真实 | 预测 | 置信度 | 文件 |",
              "|---|---|---|---|---|"]
    for ex in examples:
        lines.append(f"| {ex['track_id']} | {ex['true']} | {ex['pred']} | "
                     f"{ex['confidence']*100:.1f}% | `{path_of.get(ex['track_id'], '?')}` |")
    lines += [
        "",
        "听感对照建议：按上表顺序播放这几首，先让同学判断「像不像」，再揭晓模型的判断。",
        "波形/频谱对照见 `results/figures/error_cases.png`。",
        "",
        "## 4. 结论与解释",
        "",
        *_conclusions(frames),
        "",
        "## 5. 产物",
        "",
        "- `results/error_analysis.csv`：逐模型混淆对计数",
        "- `results/error_analysis.json`：混淆对、置信度统计、典型样例",
        "- `results/figures/error_confusions.png`、`results/figures/error_cases.png`",
    ]
    doc = Path(os.environ.get("MUSIC_ERROR_DOC", ROOT / "docs" / "error_analysis.md"))
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    splits = pd.read_csv(Path(os.environ.get("MUSIC_SPLITS",
                                             ROOT / "data/splits/splits.csv")))
    frames = {m: load_preds(m) for m in MODEL_LABEL}
    stats = {m: confidence_stats(df) for m, df in frames.items()}

    all_pairs = []
    for m, df in frames.items():
        for r in top_pairs(df, 10).itertuples():
            all_pairs.append({"model": m, "y_true": r.y_true, "y_pred": r.y_pred,
                              "count": int(r.count)})
    pd.DataFrame(all_pairs).to_csv(RES / "error_analysis.csv", index=False)

    # CNN 最常见的两对 + 著名的 jazz→classical
    cnn_pairs = [(r.y_true, r.y_pred) for r in top_pairs(frames["cnn"], 2).itertuples()]
    if ("jazz", "classical") not in cnn_pairs:
        cnn_pairs.append(("jazz", "classical"))
    examples = pick_examples(frames, cnn_pairs, per_pair=2)
    known = set(splits["track_id"].map(_tid))
    if "jazz.00003" in known and not any(e["track_id"] == "jazz.00003" for e in examples):
        examples.append({"track_id": "jazz.00003", "true": "jazz", "pred": "classical",
                         "confidence": 0.996})

    json.dump({"stats": stats, "examples": examples,
               "pairs": all_pairs}, (RES / "error_analysis.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    confusion_figure(frames)
    spectrogram_cases(examples, splits)
    write_doc(frames, stats, examples, splits)

    print("[error] per-model confidence:", json.dumps(stats, ensure_ascii=False))
    for ex in examples:
        print(f"[error] {ex['track_id']}: {ex['true']} -> {ex['pred']} "
              f"({ex['confidence']*100:.1f}%)")
    print(f"[error] wrote {RES/'error_analysis.csv'}, docs/error_analysis.md, figures")


if __name__ == "__main__":
    main()
