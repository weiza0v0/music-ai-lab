"""生成「原曲 / 生成曲」对照包：demo/pairs/

命名规则让文件名自己说清楚是哪一首、哪一段，方便直接拖进幻灯片：

    1_000995_Experimental_原曲30秒.mp3
    2_000995_Experimental_前缀5秒+真实3秒.mp3
    3_000995_Experimental_前缀5秒+原始模型续写3秒.mp3
    4_000995_Experimental_前缀5秒+LoRA续写3秒.mp3

Run:  python scripts/build_pair_demo.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "results/generation"
OUT = ROOT / "demo/pairs"
SPLITS = ROOT / "data/splits/fma/splits.csv"

# 有长样例的三条：前两条是「改善最大」与「反而变差」，第三条是典型样本
SHOWCASE = ("000995", "001482", "000010")
LONG_TAILS = {  # 000995 上额外跑过的更长续写
    "000995": (("原始模型", "probe12", 12), ("LoRA", "ft8", 8)),
}


def ff() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise SystemExit("需要 ffmpeg")
    return p


def cut(src: Path, dst: Path, seconds: float | None = None) -> None:
    cmd = [ff(), "-y", "-v", "error"]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", str(src), "-ac", "1", "-ar", "32000", "-af", "volume=-1dB",
            "-codec:a", "libmp3lame", "-b:a", "160k", str(dst)]
    subprocess.run(cmd, check=True)


def join(parts: list[Path], dst: Path) -> None:
    cmd = [ff(), "-y", "-v", "error"]
    for p in parts:
        cmd += ["-i", str(p)]
    filt = "".join(f"[{i}:a]volume=-1dB[a{i}];" for i in range(len(parts)))
    filt += "".join(f"[a{i}]" for i in range(len(parts)))
    filt += f"concat=n={len(parts)}:v=0:a=1[out]"
    cmd += ["-filter_complex", filt, "-map", "[out]",
            "-codec:a", "libmp3lame", "-b:a", "160k", "-ac", "1", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS)
    splits["track_id"] = splits["track_id"].astype(str).str.zfill(6)
    meta = splits.set_index("track_id")

    # 每条曲目的 base / LoRA 客观距离（越低越好）
    def rmse(tag: str, tid: str) -> float | None:
        p = GEN / f"metrics_{tag}.json"
        if not p.exists():
            return None
        for it in json.loads(p.read_text())["items"]:
            if str(it["track_id"]).zfill(6) == tid:
                return float(it["logmel_rmse"])
        return None

    rows = []
    stems = sorted({w.name.replace("_prompt.wav", "")
                    for w in (GEN / "audio").glob("*_prompt.wav")})
    for stem in stems:
        tid, genre = stem.split("_", 1)
        if tid not in meta.index:
            continue
        m = meta.loc[tid]
        src = ROOT / m["path"]
        base_lbl = f"{tid}_{genre}"

        # 1) 整曲 30 秒（原曲）
        if src.exists():
            cut(src, OUT / f"1_{base_lbl}_原曲30秒.mp3", None)

        # 2) 前缀 + 真实后 3 秒
        parts2 = [GEN / "audio" / f"{stem}_prompt.wav", GEN / "audio" / f"{stem}_true.wav"]
        if all(p.exists() for p in parts2):
            join(parts2, OUT / f"2_{base_lbl}_前缀5秒+真实3秒.mp3")

        # 3) 前缀 + 原始模型续写 3 秒
        parts3 = [GEN / "audio" / f"{stem}_prompt.wav", GEN / "audio" / f"{stem}_base.wav"]
        if all(p.exists() for p in parts3):
            join(parts3, OUT / f"3_{base_lbl}_前缀5秒+原始模型续写3秒.mp3")

        # 4) 前缀 + LoRA 续写 3 秒
        parts4 = [GEN / "audio" / f"{stem}_prompt.wav", GEN / "audio" / f"{stem}_ft.wav"]
        if all(p.exists() for p in parts4):
            join(parts4, OUT / f"4_{base_lbl}_前缀5秒+LoRA续写3秒.mp3")

        # 5) 更长续写（仅 000995）
        n = 5
        for label, tag, secs in LONG_TAILS.get(tid, ()):
            tail = GEN / "audio" / f"{stem}_{tag}.wav"
            if tail.exists():
                join([GEN / "audio" / f"{stem}_prompt.wav", tail],
                     OUT / f"{n}_{base_lbl}_前缀5秒+{label}续写{secs}秒.mp3")
                n += 1

        # 6) 四段串联（前缀 → 真实 → 原始 → LoRA）
        ab = GEN / "audio"
        seq = [ab / f"{stem}_{t}.wav" for t in ("prompt", "true", "base", "ft")]
        if all(p.exists() for p in seq):
            join(seq, OUT / f"{n}_{base_lbl}_四段串联_前缀-真实-原始-LoRA.mp3")

        b, f = rmse("base", tid), rmse("ft", tid)
        rows.append({
            "tid": tid, "genre": genre, "artist": m["artist"], "title": m["title"],
            "base": b, "ft": f, "delta": (f - b) if (b is not None and f is not None) else None,
        })

    # ---------- 索引 ----------
    L = ["# 原曲 / 生成曲 对照包", "",
         "每个文件的名字就是说明：开头数字表示播放顺序，后面是「曲目号_流派」与这一段是什么。", "",
         "**播放顺序**：1 听原曲 → 2 听真实后续 → 3 听原始模型续写 → 4 听 LoRA 续写。",
         "2/3/4 三条的**前 5 秒完全相同**（同一段前缀），只有后面的 3 秒不同。", "",
         "| 曲目 | 流派 | 艺术家 — 曲名 | 原始模型 RMSE | LoRA RMSE | 差值 |",
         "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["delta"] is None, x["delta"])):
        d = f"{r['delta']:+.2f}" if r["delta"] is not None else "—"
        b = f"{r['base']:.2f}" if r["base"] is not None else "—"
        f_ = f"{r['ft']:.2f}" if r["ft"] is not None else "—"
        note = ""
        if r["tid"] == "000995":
            note = "（改善最大，另有 8/12 秒长样例）"
        elif r["tid"] == "001482":
            note = "（**反而变差**，主动放它更可信）"
        L.append(f"| `{r['tid']}` | {r['genre']} | {r['artist']} — {r['title']}{note} | {b} | {f_} | {d} |")

    L += ["", "RMSE 是「生成频谱与真实后 3 秒的距离」，越低越好；8 首均值 19.23 → 17.91。", "",
          "## 文件清单", ""]
    for p in sorted(OUT.glob("*.mp3")):
        L.append(f"- `{p.name}`")
    L += ["", "## 说明", "",
          "- 每条 2/3/4 都以同一段 5 秒前缀开头，所以可以直接连着放，听众只需注意后 3 秒。",
          "- 只有 `000995` 有多于 3 秒的续写：那是唯一跑通过更长续写的曲目（见 `demo/README.md` §3.5）。",
          "- 音频未做响度归一，只留 1 dB 余量；真实录音的前缀本来就比续写段安静一些。",
          "- 素材与 `demo/audio/` 里的同名文件是同一份音频，这里只是换成了自解释的中文名。", ""]
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"生成 {len(list(OUT.glob('*.mp3')))} 个文件 → {OUT}")


if __name__ == "__main__":
    main()
