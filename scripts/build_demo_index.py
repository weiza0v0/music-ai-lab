"""生成 demo/index.csv：整个演示素材包的中文清单（给人在 Excel 里找文件用）。

文件名全部保持纯 ASCII（Windows 解 zip 时中文名会变乱码），
中文说明集中写在这张表里。用 utf-8-sig 写出，Excel 直接双击打开不乱码。

Run:  python scripts/build_demo_index.py
"""
from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def dur(p: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True)
        return f"{float(out.stdout.strip()):.1f}"
    except Exception:
        return ""


PAIR_TAG = {
    "ORIGINAL_30s": "原曲整段 30 秒（人听上下文）",
    "PREFIX5_TRUE3": "5 秒前缀 + 真实后 3 秒（参考答案）",
    "PREFIX5_BASE3": "5 秒前缀 + MusicGen 原始续写 3 秒",
    "PREFIX5_LORA3": "5 秒前缀 + LoRA 适配后续写 3 秒",
    "PREFIX5_BASE12": "5 秒前缀 + 原始模型续写 12 秒（长样例，仅 000995）",
    "PREFIX5_LORA8": "5 秒前缀 + LoRA 续写 8 秒（长样例，仅 000995）",
    "AB_PREFIX-TRUE-BASE-LORA": "四段串联：前缀 → 真实 → 原始 → LoRA",
}


def describe(p: Path) -> tuple[str, str]:
    """返回 (类别, 中文说明)"""
    name = p.name
    rel = p.relative_to(DEMO).as_posix()

    # 非音频：README / JSON / CSV
    if p.suffix.lower() != ".mp3":
        doc = {
            "README.md": "说明文档（先读这个）",
            "index.csv": "本清单",
            "verdicts.json": "五模型对 8 段示例的判断（机器可读）",
            "arena_pk.md": "GTZAN 擂台表（8 首分歧最大 + 4 首全对）",
            "arena_pk.csv": "擂台表的 CSV 版",
        }
        return "其他", doc.get(name, "")

    if rel.startswith("pairs/"):
        for key, desc in PAIR_TAG.items():
            if name.endswith(key + ".mp3"):
                return "原曲/生成曲对照", desc
        return "原曲/生成曲对照", ""

    if rel.startswith("classification/"):
        if name.endswith("_context15.mp3"):
            return "判别式示例", "同一曲目中心 15 秒（只给人听，模型不看）"
        return "判别式示例", "中心 5 秒（模型实际输入），配 README 里的五模型判断表"

    if rel.startswith("audio/"):
        heads = (
            ("AB__", "四段串联：前缀 → 真实 → 原始 → LoRA（16 秒，含间隔）"),
            ("REAL__", "无缝：5 秒前缀 + 真实后 3 秒"),
            ("GENbase__", "无缝：5 秒前缀 + 原始模型续写"),
            ("GENft__", "无缝：5 秒前缀 + LoRA 续写"),
            ("FULL30__", "整曲 30 秒原始素材"),
            ("LONG__", "长样例：5 秒前缀 + 8/12 秒续写"),
            ("TOK__", "Token 预测的 1 秒（pred=模型、true=真实）"),
        )
        for head, desc in heads:
            if name.startswith(head):
                return "生成式音频", desc
        tails = (
            ("__prompt.mp3", "5 秒前缀（模型输入）"),
            ("__true.mp3", "真实后 3 秒"),
            ("__base.mp3", "MusicGen 原始续写 3 秒"),
            ("__ft.mp3", "LoRA 续写 3 秒"),
            ("__ft_v1.mp3", "初版 LoRA（标签维度错误，负结果存档）"),
            ("__v3.mp3", "3600 步消融"),
            ("__full.mp3", "全窗口消融"),
            ("__r16.mp3", "rank16 消融"),
            ("__base8.mp3", "原始模型续写 8 秒（长样例）"),
            ("__ft8.mp3", "LoRA 续写 8 秒（长样例）"),
            ("__probe4.mp3", "原始模型续写 4 秒（长度实验）"),
            ("__probe6.mp3", "原始模型续写 6 秒（长度实验）"),
            ("__probe8.mp3", "原始模型续写 8 秒（长度实验）"),
            ("__probe12.mp3", "原始模型续写 12 秒（长度实验）"),
            ("__probeLoRA8.mp3", "LoRA 续写 8 秒（长度实验）"),
        )
        for tail, desc in tails:
            if name.endswith(tail):
                return "生成式音频", desc
        return "生成式音频", ""

    return "其他", ""


def main() -> None:
    rows = []
    for p in sorted(DEMO.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".mp3", ".md", ".json", ".csv"}:
            continue
        if p.name == "index.csv":
            continue
        kind, desc = describe(p)
        rows.append({
            "类别": kind,
            "文件": p.relative_to(DEMO).as_posix(),
            "中文说明": desc,
            "时长(秒)": dur(p) if p.suffix.lower() == ".mp3" else "",
        })

    order = {"原曲/生成曲对照": 0, "生成式音频": 1, "判别式示例": 2, "其他": 3}
    rows.sort(key=lambda r: (order.get(r["类别"], 9), r["文件"]))

    out = DEMO / "index.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["类别", "文件", "中文说明", "时长(秒)"])
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {out}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
