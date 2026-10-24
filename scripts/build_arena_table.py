"""Build the "arena" (擂台) table used in the presentation.

Picks GTZAN test tracks where the five self-trained / classical models disagree
most, and writes a markdown table + CSV that can be dropped straight into slides.
Reads only files already committed under `results/gtzan/predictions/`, so it runs
without checkpoints, audio or a GPU.

Run:  python scripts/build_arena_table.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/gtzan/predictions"
OUT = ROOT / "demo"

MODELS = [("svm", "SVM"), ("mlp", "MLP"), ("cnn", "CNN"),
          ("crnn", "CRNN"), ("transformer", "TF")]
SEED = 42
N_ARENA = 8          # 意见最分歧的曲目
N_EASY = 4           # 五个模型全对、且一致，作对照


def load() -> pd.DataFrame:
    frames = []
    for key, _ in MODELS:
        p = PRED / f"{key}_seed{SEED}_test.csv"
        if not p.exists():
            raise SystemExit(f"缺少 {p}")
        d = pd.read_csv(p)
        prob_cols = [c for c in d.columns if c.startswith("p_")]
        top1 = d[prob_cols].idxmax(axis=1).str.replace("p_", "", regex=False)
        conf = d[prob_cols].max(axis=1)
        frames.append(pd.DataFrame({
            "track_id": d["track_id"],
            "y_true": d["y_true"],
            key: top1,
            f"{key}_conf": conf,
        }))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f.drop(columns=["y_true"]), on="track_id")
    return out


def main() -> None:
    df = load()
    keys = [k for k, _ in MODELS]
    names = dict(MODELS)

    # 每行：几个模型答对、几个不同答案
    df["n_correct"] = sum((df[k] == df["y_true"]).astype(int) for k in keys)
    df["n_distinct"] = df[keys].nunique(axis=1)

    arena = df.sort_values(["n_distinct", "n_correct"], ascending=[False, True]).head(N_ARENA)
    easy = df[(df.n_correct == len(keys)) & (df.n_distinct == 1)].head(N_EASY)

    def table(rows: pd.DataFrame) -> list[str]:
        head = "| 曲目 | 真值 | " + " | ".join(names[k] for k in keys) + " | 答对 |"
        sep = "|---|---|" + "---|" * len(keys) + "---|"
        lines = [head, sep]
        for _, r in rows.iterrows():
            cells = []
            for k in keys:
                mark = "✓" if r[k] == r["y_true"] else "✗"
                cells.append(f"{r[k]} {mark} {r[f'{k}_conf']*100:.0f}%")
            lines.append(f"| `{r['track_id']}` | {r['y_true']} | " + " | ".join(cells)
                         + f" | {int(r['n_correct'])}/5 |")
        return lines

    md = ["# 擂台表（GTZAN 测试集，seed 42）", "",
          "由 `scripts/build_arena_table.py` 从 `results/gtzan/predictions/*.csv` 直接生成，",
          "只用已提交的预测文件，**不需要权重与音频**。✓ 表示答对，百分比是该模型的置信度。",
          "",
          "## 一、五个模型意见最不一致的 8 首", "",
          "这 8 首是「擂台」的核心：同一段音频，五个模型的判断各不相同，正好用来讲",
          "归纳偏置与错误模式。", ""]
    md += table(arena)
    md += ["", f"这批曲目上各模型答对数："
           + "、".join(f"{names[k]} {int((arena[k] == arena['y_true']).sum())}/8" for k in keys), ""]

    md += ["## 二、五个模型全都答对且一致的 4 首（对照组）", "",
           "用来对比：干净、典型的样本上五个模型没有分歧，差别只出现在边界样本上。", ""]
    md += table(easy)

    md += ["", "## 三、怎么用", "",
            "1. 幻灯片上只放 **第一节那张表的 3–4 行**（放满 8 行会看不清），其余口述。",
            "2. 讲的时候按列读：CNN 与 CRNN 的判断常常相邻（都吃局部时频结构），",
            "   TF 容易给出低置信度的离谱答案，SVM 在边界样本上反而稳。",
            "3. 全量混淆矩阵见 `results/gtzan/figures/confusion_matrices.png`，",
            "   逐类 F1 见 `results/gtzan/figures/leaderboard.png` 与技术报告 §5.6。",
            "4. AST 不在这张表里（它的预测未按样本落盘），但它在两个数据集上都是第一，",
            "   放在「预训练一出手就赢」那一页讲。", ""]

    (OUT / "arena_pk.md").write_text("\n".join(md), encoding="utf-8")
    arena.to_csv(OUT / "arena_pk.csv", index=False)

    # 控制台摘要
    print(f"分歧最大的 {N_ARENA} 首 + 全对 {N_EASY} 首 → {OUT/'arena_pk.md'}")
    for k in keys:
        print(f"  {names[k]:<4} 全测试集准确率 {(df[k]==df['y_true']).mean():.4f}"
              f"  分歧集内 {(arena[k]==arena['y_true']).sum()}/{len(arena)}")


if __name__ == "__main__":
    main()
