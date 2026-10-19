"""Figures for phase 2 generation: spectrogram comparison + cost bars."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib import font_manager

_FONT_CANDIDATES = ["/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
                    str(Path.home() / ".fonts/wqy-microhei.ttc")]
for _f in _FONT_CANDIDATES:
    if Path(_f).exists():
        try:
            font_manager.fontManager.addfont(_f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
            break
        except Exception:  # noqa: BLE001
            continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/generation"))
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _spec(ax, path: Path, title: str):
    y, sr = sf.read(path)
    import librosa
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64) + 1e-9)
    ax.imshow(m, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def spectrograms() -> Path | None:
    d = json.loads((RES / "metrics_base.json").read_text())
    items = d["items"][:3]
    if not items:
        return None
    ft = None
    p_ft = RES / "metrics_ft.json"
    if p_ft.exists():
        ft = {x["track_id"]: x for x in json.loads(p_ft.read_text())["items"]}
    sec = d.get("seconds", 5)
    ncol = 4 if ft else 3
    fig, axes = plt.subplots(len(items), ncol, figsize=(3.0 * ncol, 2.2 * len(items)),
                             squeeze=False)
    for i, it in enumerate(items):
        _spec(axes[i, 0], RES / "audio" / Path(it["prompt_wav"]).name, "前缀 5 s")
        _spec(axes[i, 1], RES / "audio" / Path(it["true_wav"]).name, f"真实后 {sec:g} s")
        _spec(axes[i, 2], RES / "audio" / Path(it["out_wav"]).name, f"MusicGen 续写 {sec:g} s")
        if ft and it["track_id"] in ft:
            _spec(axes[i, 3], RES / "audio" / Path(ft[it["track_id"]]["out_wav"]).name,
                  f"LoRA 续写 {sec:g} s")
        axes[i, 0].set_ylabel(it["genre"], fontsize=8)
    fig.suptitle("前缀、真实后半段与生成续写的 Log-Mel 对照", fontsize=9)
    fig.tight_layout()
    out = FIG / "generation_spec.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def cost_bars() -> Path | None:
    rows = []
    for tag, label in (("base", "MusicGen-small"), ("ft", "MusicGen + LoRA")):
        p = RES / f"metrics_{tag}.json"
        if not p.exists():
            continue
        items = json.loads(p.read_text())["items"]
        rows.append((label, np.mean([x["latency_s"] for x in items]),
                     np.mean([x["peak_mem_mb"] for x in items]),
                     np.mean([x["logmel_rmse"] for x in items])))
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, k, title in zip(axes, (1, 2, 3), ("生成耗时 (s)", "峰值显存 (MB)", "Log-Mel RMSE")):
        ax.bar([r[0] for r in rows], [r[k] for r in rows],
               color=["#c2703d", "#6b8ea8"][: len(rows)])
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("续写成本与客观距离", fontsize=9)
    fig.tight_layout()
    out = FIG / "generation_cost.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outs = [spectrograms(), cost_bars()]
    print("[figures]", [str(o) for o in outs if o])


if __name__ == "__main__":
    main()
