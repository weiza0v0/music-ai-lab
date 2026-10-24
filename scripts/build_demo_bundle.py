"""Build the presentation asset bundle under `demo/`.

Turns the wav files produced by the generation stage into mp3 clips that can be
dropped straight into slides, and writes `demo/audio/README.md` describing every
clip together with the measured Log-Mel RMSE of the track it came from.

Run:  python scripts/build_demo_bundle.py
Requires: ffmpeg (libmp3lame) on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "results/generation"
TOK = ROOT / "results/token_prediction"
OUT = ROOT / "demo/audio"

# 挑选三条测试曲：典型、改善最大、反而变差 —— 三个都要讲，最后一条正是诚实性的证据
SHOWCASE = ("000010", "000995", "001482")

GUARD_SECONDS = 0.6  # A/B 串联时每段之间的静音，便于听众分清段落


def need(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        raise SystemExit(f"缺少 {cmd}，请先安装 ffmpeg")
    return p


def to_mp3(src: Path, dst: Path, loudnorm: bool = True) -> Path:
    ff = need("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = "loudnorm=I=-18:TP=-2:LRA=11" if loudnorm else "anull"
    subprocess.run(
        [ff, "-y", "-v", "error", "-i", str(src), "-af", af,
         "-codec:a", "libmp3lame", "-b:a", "160k", "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def load_metrics() -> dict[str, dict]:
    """返回 {track_id: {'genre':.., 'base':rmse, 'ft':rmse, ...}}"""
    out: dict[str, dict] = {}
    for tag in ("base", "ft", "ft_v1_negative", "v3", "full", "r16"):
        p = GEN / f"metrics_{tag}.json"
        if not p.exists():
            continue
        for it in json.loads(p.read_text())["items"]:
            row = out.setdefault(it["track_id"], {"genre": it["genre"]})
            row[tag] = float(it["logmel_rmse"])
    return out


def main() -> None:
    need("ffmpeg")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    made: list[tuple[str, str]] = []

    # ---------- 一、前缀续写：prompt / true / base / ft ----------
    for wav in sorted((GEN / "audio").glob("*_prompt.wav")):
        stem = wav.name.replace("_prompt.wav", "")
        tid = stem.split("_")[0]
        if tid not in metrics:
            continue
        for part, desc in (("prompt", "5 秒前缀（模型输入）"),
                           ("true", "真实后 3 秒（参考答案）"),
                           ("base", "MusicGen-small 原始续写"),
                           ("ft", "LoRA 适配后续写")):
            src = GEN / "audio" / f"{stem}_{part}.wav"
            if not src.exists():
                continue
            dst = OUT / f"{stem}__{part}.mp3"
            to_mp3(src, dst)
            made.append((dst.name, f"{stem} — {desc}"))

    # ---------- 二、A/B 串联：一条音频听完四段 ----------
    for tid in SHOWCASE:
        stem = next((w.name.replace("_prompt.wav", "")
                     for w in (GEN / "audio").glob(f"{tid}_*_prompt.wav")), None)
        if stem is None:
            continue
        parts = [GEN / "audio" / f"{stem}_{p}.wav"
                 for p in ("prompt", "true", "base", "ft")]
        if not all(p.exists() for p in parts):
            continue
        ff = need("ffmpeg")
        dst = OUT / f"AB__{stem}.mp3"
        cmd = [ff, "-y", "-v", "error"]
        for p in parts:
            cmd += ["-i", str(p)]
        # 每段后接 GUARD_SECONDS 静音，四段顺序：前缀 → 真实 → 原始 → 适配
        filt = "".join(
            f"[{i}:a]loudnorm=I=-18:TP=-2:LRA=11,apad=pad_dur={GUARD_SECONDS}[a{i}];"
            for i in range(len(parts))
        ) + "".join(f"[a{i}]" for i in range(len(parts))) + \
            f"concat=n={len(parts)}:v=0:a=1[out]"
        cmd += ["-filter_complex", filt, "-map", "[out]",
                "-codec:a", "libmp3lame", "-b:a", "160k", "-ac", "1", str(dst)]
        subprocess.run(cmd, check=True)
        made.append((dst.name,
                     f"{stem} — 【串联】前缀 → 真实后 3 秒 → 原始续写 → LoRA 续写"
                     f"（每段间隔 {GUARD_SECONDS}s）"))

    # ---------- 三、Token 预测：1 秒预测 vs 真实 ----------
    dec = TOK / "decode.json"
    if dec.exists():
        for it in json.loads(dec.read_text())["items"]:
            tid, genre = it["track_id"], it["genre"]
            if tid not in SHOWCASE:
                continue
            for key, desc in (("pred_wav", "模型预测的 1 秒"),
                              ("true_wav", "真实的 1 秒")):
                src = TOK / it[key]
                if not src.exists():
                    continue
                dst = OUT / f"TOK__{tid}_{genre}__{key.replace('_wav', '')}.mp3"
                to_mp3(src, dst)
                made.append((dst.name, f"{tid} {genre} — {desc}（1 秒）"))

    # ---------- 四、清单 ----------
    lines = [
        "# 演示音频包",
        "",
        "由 `scripts/build_demo_bundle.py` 从 `results/generation/audio/` 与",
        "`results/token_prediction/audio/` 生成，**不是重新合成的**——每个文件都对应报告里的一次实测。",
        "",
        "所有片段已做响度对齐（`loudnorm I=-18 LUFS`），目的是让听众比较**音色与结构**，",
        "而不是被音量差异带偏；这也会削掉原始文件之间本来就有的电平差。",
        "",
        "## 一、前缀续写（3 秒）",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
    ]
    for name, desc in made:
        if name.startswith(("AB__", "TOK__")):
            continue
        lines.append(f"| `{name}` | {desc} |")

    lines += ["", "## 二、A/B 串联（一条听完四段，适合直接插幻灯片）", "",
              "| 文件 | 内容 |", "|---|---|"]
    for name, desc in made:
        if name.startswith("AB__"):
            lines.append(f"| `{name}` | {desc} |")

    lines += ["", "## 三、Token 预测（1 秒）", "",
              "| 文件 | 内容 |", "|---|---|"]
    for name, desc in made:
        if name.startswith("TOK__"):
            lines.append(f"| `{name}` | {desc} |")

    lines += ["", "## 四、每条曲目的客观距离（Log-Mel RMSE，越低越好）", "",
              "| 曲目 | 流派 | 原始 | +LoRA | 差值 | 讲什么 |", "|---|---|---|---|---|---|"]
    note = {"000010": "典型样本：小幅稳定改善",
            "000995": "改善最大：−3.65 dB，最适合现场听出差别",
            "001482": "**反而变差** +0.33 dB：8 首里唯一退步的一首，用来讲“指标不是万能的”"}
    for tid in SHOWCASE:
        m = metrics.get(tid)
        if not m:
            continue
        b, f = m.get("base"), m.get("ft")
        if b is None or f is None:
            continue
        lines.append(f"| {tid} | {m['genre']} | {b:.2f} | {f:.2f} | {f - b:+.2f} | {note.get(tid, '')} |")

    lines += [
        "",
        "全 8 首的逐曲表见技术报告附录 A.5（表 29）；均值 19.23 → 17.91，8 首中 7 首改善，",
        "去掉改善最大的一首后仍有 −0.99 dB。",
        "",
        "## 五、播放建议",
        "",
        "1. **先放 `AB__000995_Experimental.mp3`**：听众能在 12 秒内听出「前缀 → 真实 → 原始 → 适配」",
        "   四段的差别，这是全场最有说服力的一条。",
        "2. 再放 `AB__001482_Electronic.mp3`：**这一条适配后更差**。主动放它，然后解释为什么",
        "   （8 首样本、配对检验、指标只奖励贴近原曲），比只放好消息可信得多。",
        "3. 若时间紧，只放这两条就够。",
        "",
        "## 六、注意",
        "",
        "- 这些都是**真实后 3 秒**可对照的样本，不是自由创作。别把它说成“AI 作曲”。",
        "- 32 kHz 单声道原始采样，转 mp3 只为便于插入幻灯片；报告里的测量全部在 wav 上完成。",
        "- 生成音频的版权与许可问题见技术报告 §7.10：**不要公开发布这批音频**，课堂演示可用。",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"生成 {len(made)} 个文件 → {OUT}")
    for name, _ in made:
        print("  ", name)


if __name__ == "__main__":
    main()
