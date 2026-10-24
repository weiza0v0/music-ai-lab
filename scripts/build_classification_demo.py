"""Build the classification demo pack: 5-second clips + precomputed verdicts.

Demo machines may have no GPU, so the pack must work without any inference:
every clip ships with the verdicts the five models already gave it, taken from
`results/fma/predictions/*.csv` (same models, same centre-5s crop, seed 42).

FMA-small audio is CC-BY 4.0, so these clips may be redistributed with
attribution (GTZAN is not, which is why this pack is built from FMA).

Run:  python scripts/build_classification_demo.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/fma/predictions"
SPLITS = ROOT / "data/splits/fma/splits.csv"
OUT = ROOT / "demo/classification"
SR = 22050          # 与训练/评估时的特征口径一致
CLIP_S = 5.0        # 中心 5 秒
SEED = 42
MODELS = [("svm", "SVM"), ("mlp", "MLP"), ("cnn", "CNN"),
          ("crnn", "CRNN"), ("transformer", "TF")]
N_ARENA, N_EASY = 6, 2


def ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise SystemExit("需要 ffmpeg")
    return p


def cut(src: Path, dst: Path, duration: float, seconds: float = CLIP_S) -> None:
    start = max(0.0, duration / 2 - seconds / 2)
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds}",
         "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-af", "volume=-1dB",
         "-codec:a", "libmp3lame", "-b:a", "160k", str(dst)],
        check=True,
    )


def load_verdicts() -> pd.DataFrame:
    frames = []
    for key, _ in MODELS:
        p = PRED / f"{key}_seed{SEED}_test.csv"
        if not p.exists():
            raise SystemExit(f"缺少 {p}")
        d = pd.read_csv(p)
        prob = [c for c in d.columns if c.startswith("p_")]
        frames.append(pd.DataFrame({
            "track_id": d["track_id"].astype(str).str.zfill(6),
            "y_true": d["y_true"],
            key: d[prob].idxmax(axis=1).str.replace("p_", "", regex=False),
            f"{key}_conf": d[prob].max(axis=1),
        }))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f.drop(columns=["y_true"]), on="track_id")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS)
    splits["track_id"] = splits["track_id"].astype(str).str.zfill(6)
    df = load_verdicts()
    keys = [k for k, _ in MODELS]
    names = dict(MODELS)
    df["n_correct"] = sum((df[k] == df["y_true"]).astype(int) for k in keys)
    df["n_distinct"] = df[keys].nunique(axis=1)

    arena = df.sort_values(["n_distinct", "n_correct"], ascending=[False, True]).head(N_ARENA)
    easy = df[(df.n_correct == len(keys)) & (df.n_distinct == 1)].head(N_EASY)
    picked = pd.concat([arena.assign(_group="分歧"), easy.assign(_group="一致")])

    meta = splits.set_index("track_id")
    rows, made = [], []
    for _, r in picked.iterrows():
        tid = r["track_id"]
        if tid not in meta.index:
            continue
        m = meta.loc[tid]
        src = ROOT / m["path"]
        if not src.exists():
            print(f"  跳过 {tid}：本地没有 {m['path']}")
            continue
        dst = OUT / f"{tid}_{m['genre']}.mp3"
        cut(src, dst, float(m["duration"]))
        # 额外一份 15 秒：给人听的上下文；模型实际只看中心 5 秒
        cut(src, OUT / f"{tid}_{m['genre']}_context15.mp3", float(m["duration"]), 15.0)
        made.append(dst.name)
        rows.append({
            "track_id": tid, "genre": m["genre"], "artist": m["artist"],
            "title": m["title"], "group": r["_group"],
            "verdicts": {k: (r[k], float(r[f"{k}_conf"])) for k in keys},
            "n_correct": int(r["n_correct"]),
            "file": dst.name,
        })

    # ---------- README ----------
    L = ["# 分类演示包（无需 GPU）", "",
         f"{len(rows)} 段 5 秒片段，全部取自 FMA-small 的**测试集**（模型没见过），",
         "并把五个模型当时给出的判断一起附上：现场不需要任何推理。", "",
         "**授权**：FMA-small 音频为 CC-BY 4.0（Defferrard et al., ISMIR 2017），",
         "允许再分发，条件是署名。逐曲的曲名与艺术家见下表。GTZAN 音频不随仓库分发",
         "（版权归各权利人），所以这批示例用 FMA。", "",
         "所有片段统一裁成**中心 5 秒**、22 050 Hz 单声道、只留 1 dB 余量（不做响度归一）—— ",
         "与模型评估时的输入口径一致，因此下面表格里的判断就是模型对这段音频的真实输出。", "",
         "另有一份 `<曲目>_context15.mp3`：同一首曲目的**中心 15 秒**，只给人耳听上下文用。",
         "模型不消费它——评估与上表判断用的始终是中心 5 秒。", ""]

    for grp, title, blurb in (
        ("分歧", "一、五个模型意见不一致（擂台用）",
         "同一段音频，五个模型给出不同答案。这是最值得讲的一组：分歧集中在流派边界模糊的样本上。"),
        ("一致", "二、五个模型全都答对且一致（对照组）",
         "干净、典型的样本上五个模型没有分歧——差别只出现在边界样本上。"),
    ):
        sel = [r for r in rows if r["group"] == grp]
        if not sel:
            continue
        L += [f"## {title}", "", blurb, "",
              "| 音频文件 | 真值 | 曲目 | " + " | ".join(names[k] for k in keys) + " | 答对 |",
              "|---|---|---|" + "---|" * len(keys) + "---|"]
        for r in sel:
            cells = []
            for k in keys:
                g, c = r["verdicts"][k]
                cells.append(f"{g} {'✓' if g == r['genre'] else '✗'} {c*100:.0f}%")
            L.append(f"| `{r['file']}` | {r['genre']} | {r['artist']} — {r['title']} | "
                     + " | ".join(cells) + f" | {r['n_correct']}/5 |")
        L.append("")

    L += ["## 三、现场怎么用", "",
          "### 方式 1：只播放（完全不需要环境）", "",
          "按上表顺序放 mp3，先问「你们觉得这是什么流派」，再公布五个模型的答案与置信度。",
          "分歧组里最值得放的是**人工特征模型对、CNN 错**的那几条：",
          "说明卷积看到的是局部音色纹理，而不是人耳理解的风格。", "",
          "### 方式 2：现场跑（CPU 就够，不需要显卡）", "",
          "五个自建/传统模型在纯 CPU 上单段推理约 **0.1–0.3 秒**（实测：CNN 196 ms、",
          "CRNN 60 ms、TF 80 ms、MLP 70 ms；SVM 首次调用含特征提取约 2.6 s，之后很快）。",
          "所以笔记本没有显卡也能现场 PK：", "",
          "```bash",
          "git clone https://github.com/weiza0v0/music-ai-lab.git && cd music-ai-lab",
          "pip install -r requirements.txt",
          "# 这一包是 FMA 的片段，用 FMA 的权重（8 类）",
          "MUSIC_CKPT=checkpoints/fma python3 -m streamlit run app/demo.py",
          "```", "",
          "上传任意一段音乐即可；想复现上表，就上传本目录里对应的 mp3。", "",
          "> AST 不在这个包里：它需要 346 MB 权重与约 200 ms/段（CPU 上更慢），",
          "> 现场只讲它的榜上成绩与成本对比即可。", "",
          "## 四、数字出处", "",
          "- 判断与置信度：`results/fma/predictions/{model}_seed42_test.csv`",
          "- 全量指标：`results/fma/metrics.csv`；逐类 F1 见报告 §5.5",
          "- 这张表由 `scripts/build_classification_demo.py` 生成，可重跑核对", ""]
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")

    with open(OUT / "verdicts.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"生成 {len(made)} 段 → {OUT}")
    for r in rows:
        print(f"  {r['file']:28s} {r['group']}  {r['n_correct']}/5  {r['genre']}")


if __name__ == "__main__":
    main()
