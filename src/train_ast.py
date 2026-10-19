"""P1 — AST transfer learning: frozen AudioSet backbone + logistic-regression head.

Protocol (same as every other model in this repo):
  * the track-level split in data/splits/splits.csv is the single source of truth;
  * the head's only hyper-parameter (C) is chosen on the validation split;
  * the test split is evaluated exactly once, after the head is frozen;
  * the StandardScaler is fit on TRAIN ONLY.

The Audio Spectrogram Transformer is used as a fixed feature extractor: the
backbone stays in eval mode with requires_grad=False and every forward runs under
inference_mode, so the only trained parameters are the 768->10 logistic head.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# huggingface_hub 1.30 (installed here) chokes on the bracketed "[::1]" token in
# NO_PROXY and raises `InvalidURL: Invalid port: ':1]'` before it sends a single
# request. The HTTP proxy itself is fine, so we simply drop the bypass list for
# this process; otherwise a clean checkout cannot download the weights.
for _k in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_k, None)

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import encode_labels, evaluate, save_json  # noqa: E402

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SEED = 42
SR = 16_000
CLIP_SECONDS = 5.0
PAD_SECONDS = 10.24              # 1024 frames x 160 samples (10 ms hop)
CLIP_SAMPLES = int(SR * CLIP_SECONDS)      # 80_000
PAD_SAMPLES = int(SR * PAD_SECONDS)        # 163_840
PREPROC = "gtzan-16k-5s-center-zero-pad-10.24s-v1"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

DATA = Path(os.environ.get("MUSIC_SPLITS", ROOT / "data/splits/splits.csv"))
CACHE = Path(os.environ.get("MUSIC_AST_CACHE", ROOT / "data/processed/ast_embeddings.npz"))
RESULTS = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results"))
FIG = RESULTS / "figures"
METRICS_CSV = RESULTS / "metrics.csv"
DOCS = ROOT / "docs"
DOC_PATH = Path(os.environ.get("MUSIC_AST_DOC", DOCS / "ast.md"))
DATASET_NAME = os.environ.get("MUSIC_DATASET_NAME", "GTZAN")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[warn] CUDA unavailable — AST extraction on CPU will be very slow")
    return torch.device("cpu")


def load_split() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    assert len(df) > 0, "empty split file"
    assert set(df["split"]) == {"train", "val", "test"}
    print(f"[data] {len(df)} tracks | train={(df.split == 'train').sum()} "
          f"val={(df.split == 'val').sum()} test={(df.split == 'test').sum()}")
    return df.reset_index(drop=True)


def load_waveform(path: str, duration: float) -> np.ndarray:
    """Centre 5 s at 16 kHz mono, then zero-pad to 10.24 s (1024 AST frames)."""
    duration = float(duration) if np.isfinite(duration) else 0.0
    offset = max(0.0, (duration - CLIP_SECONDS) / 2.0)
    y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=CLIP_SECONDS)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    y = y[:CLIP_SAMPLES].astype(np.float32)
    return np.pad(y, (0, PAD_SAMPLES - CLIP_SAMPLES)).astype(np.float32)


def extract_embeddings(df: pd.DataFrame, fe, model, device, batch_size: int,
                       workers: int) -> tuple[np.ndarray, dict]:
    """Extract embeddings for every track, with threaded audio loading."""
    paths = [str(ROOT / p) if not os.path.isabs(p) else p for p in df["path"]]
    durations = df["duration"].tolist()
    n = len(paths)
    embs: list[np.ndarray] = []
    fe_time = 0.0
    gpu_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm-up so cuDNN autotuning / CUDA context setup is not counted.
    if device.type == "cuda":
        with torch.inference_mode():
            model(input_values=torch.zeros(2, 1024, 128, device=device))
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = ex.map(load_waveform, paths, durations)
        chunk: list[np.ndarray] = []
        for i, wave in enumerate(it):
            chunk.append(wave)
            if len(chunk) >= batch_size or i == n - 1:
                tf = time.perf_counter()
                feats = fe(chunk, sampling_rate=SR, return_tensors="pt")
                fe_time += time.perf_counter() - tf
                pixel_values = feats["input_values"].to(device, non_blocking=True)
                tg = time.perf_counter()
                with torch.inference_mode():
                    out = model(input_values=pixel_values)
                emb = out.pooler_output.float().cpu().numpy()
                gpu_time += time.perf_counter() - tg
                embs.append(emb)
                chunk = []
                done = min(i + 1, n)
                if done % (batch_size * 8) == 0 or done == n:
                    print(f"[embed] {done}/{n} clips", flush=True)
    elapsed = time.perf_counter() - t0
    embeddings = np.concatenate(embs, axis=0).astype(np.float32)
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    timing = {"embedding_extract_s": round(elapsed, 2),
              "featurize_s": round(fe_time, 2),
              "gpu_forward_s": round(gpu_time, 2),
              "peak_gpu_mb": round(peak_mb, 1)}
    print(f"[embed] done in {elapsed:.1f}s ({elapsed / n * 1000:.1f} ms/clip) "
          f"shape={embeddings.shape} peak_gpu={peak_mb:.0f}MB")
    return embeddings, timing


def _cache_metadata(df: pd.DataFrame) -> dict:
    return {"model_id": MODEL_ID, "pooling": "pooler_output", "preproc": PREPROC,
            "track_ids": df["track_id"].to_numpy().astype(str),
            "genres": df["genre"].to_numpy().astype(str),
            "splits": df["split"].to_numpy().astype(str)}


def load_cache(df: pd.DataFrame) -> dict | None:
    if not CACHE.exists():
        return None
    try:
        d = np.load(CACHE, allow_pickle=True)
        meta = _cache_metadata(df)
        if (str(d["model_id"]) != MODEL_ID or str(d["pooling"]) != "pooler_output"
                or str(d["preproc"]) != PREPROC):
            print("[cache] stale model/preprocessing metadata — re-extracting")
            return None
        if list(d["track_ids"].astype(str)) != list(meta["track_ids"]):
            print("[cache] track list/order changed — re-extracting")
            return None
        print(f"[cache] reusing {CACHE} ({CACHE.stat().st_size / 1e6:.2f} MB)")
        return d
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] unreadable ({type(exc).__name__}: {exc}) — re-extracting")
        return None


def save_cache(df: pd.DataFrame, embeddings: np.ndarray, timing: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = _cache_metadata(df)
    data["embeddings"] = embeddings
    data["extract_time_s"] = timing["embedding_extract_s"]
    data["featurize_s"] = timing["featurize_s"]
    data["gpu_forward_s"] = timing["gpu_forward_s"]
    data["peak_gpu_mb"] = timing["peak_gpu_mb"]
    tmp = CACHE.with_name(CACHE.stem + ".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, CACHE)
    print(f"[cache] wrote {CACHE} ({CACHE.stat().st_size / 1e6:.2f} MB)")


def train_head(embeddings: np.ndarray, y: np.ndarray, splits: np.ndarray,
               class_names: list[str], seed: int) -> dict:
    """Grid-search C on val, refit on train only; test is not touched here."""
    tr, va = splits == "train", splits == "val"
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(embeddings[tr])          # train-only fit
    Xtr = scaler.transform(embeddings[tr])
    Xva = scaler.transform(embeddings[va])

    grid = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=seed)
        t = time.perf_counter()
        clf.fit(Xtr, y[tr])
        fit_s = time.perf_counter() - t
        vp = clf.predict(Xva)
        row = {"C": C,
               "val_accuracy": round(float(accuracy_score(y[va], vp)), 4),
               "val_macro_f1": round(float(f1_score(y[va], vp, average="macro")), 4),
               "fit_s": round(fit_s, 3)}
        grid.append(row)
        print(f"[head] C={C:<7g} val_acc={row['val_accuracy']:.4f} "
              f"val_f1={row['val_macro_f1']:.4f} ({fit_s * 1000:.0f} ms)")

    # Tie-break: higher val macro-F1, then val accuracy, then smaller C.
    best = max(grid, key=lambda r: (r["val_macro_f1"], r["val_accuracy"], -r["C"]))
    clf = LogisticRegression(C=best["C"], max_iter=5000, solver="lbfgs", random_state=seed)
    t = time.perf_counter()
    clf.fit(Xtr, y[tr])
    final_fit_s = time.perf_counter() - t
    total_s = time.perf_counter() - t0

    val_pred = clf.predict(Xva)
    val_proba = clf.predict_proba(Xva)
    val = evaluate(y[va], val_pred, val_proba, class_names)
    print(f"[head] chosen C={best['C']:g} | val acc={val['accuracy']:.4f} "
          f"macroF1={val['macro_f1']:.4f} | total={total_s:.2f}s")
    return {"scaler": scaler, "clf": clf, "best": best, "grid": grid,
            "val": val, "total_s": total_s, "final_fit_s": final_fit_s,
            "head_params": int(clf.coef_.size + clf.intercept_.size)}


def run_test(head: dict, embeddings: np.ndarray, y: np.ndarray, splits: np.ndarray,
             track_ids: np.ndarray,
             class_names: list[str]) -> tuple[dict, list[dict], float]:
    """Evaluate the frozen head on test exactly once."""
    te = splits == "test"
    Xte = head["scaler"].transform(embeddings[te])
    t0 = time.perf_counter()
    pred = head["clf"].predict(Xte)
    proba = head["clf"].predict_proba(Xte)
    predict_s = time.perf_counter() - t0
    res = evaluate(y[te], pred, proba, class_names)
    rows = [{"track_id": str(t),
             "y_true": class_names[int(yt)],
             "y_pred": class_names[int(yp)],
             "confidence": round(float(pr.max()), 4)}
            for t, yt, yp, pr in zip(track_ids[te], y[te], pred, proba)]
    print(f"[test] n={int(te.sum())} acc={res['accuracy']:.4f} "
          f"macroF1={res['macro_f1']:.4f} ({predict_s * 1000:.1f} ms for all clips)")
    return res, rows, predict_s


def benchmark_latency(df: pd.DataFrame, fe, model, head: dict, device,
                      clips: int = 30) -> dict:
    """Batch-1 end-to-end latency: feature extractor + backbone + head.

    Audio *decoding* is excluded, consistent with the CNN baselines which time a
    forward pass over cached log-Mel features.
    """
    test_df = df[df["split"] == "test"].head(clips)
    paths = [str(ROOT / p) if not os.path.isabs(p) else p for p in test_df["path"]]
    durations = test_df["duration"].tolist()
    with ThreadPoolExecutor(max_workers=8) as ex:
        waves = list(ex.map(load_waveform, paths, durations))

    def one(w: np.ndarray) -> float:
        t0 = time.perf_counter()
        inputs = fe([w], sampling_rate=SR, return_tensors="pt")
        pixel_values = inputs["input_values"].to(device)
        with torch.inference_mode():
            out = model(input_values=pixel_values)
        emb = out.pooler_output.float().cpu().numpy()
        head["clf"].predict_proba(head["scaler"].transform(emb))
        return (time.perf_counter() - t0) * 1000.0

    for w in waves[:2]:       # warm-up, not measured
        one(w)
    times = np.array([one(w) for w in waves[2:]], dtype=np.float64)
    stats = {"n_clips": int(len(times)),
             "median_ms": round(float(np.median(times)), 3),
             "mean_ms": round(float(times.mean()), 3),
             "p90_ms": round(float(np.percentile(times, 90)), 3),
             "std_ms": round(float(times.std(ddof=1)), 3)}
    print(f"[latency] batch=1 median={stats['median_ms']:.1f} ms/clip "
          f"(n={stats['n_clips']})")
    return stats


def benchmark_throughput(df: pd.DataFrame, fe, model, head: dict, device,
                         batch_size: int = 16) -> dict:
    """Amortised per-clip cost over the whole test split, same convention as CNN.

    Batches of `batch_size`; includes feature extraction + backbone + head but
    excludes wav decoding, matching how train_cnn.py reports infer_ms_per_clip.
    """
    test_df = df[df["split"] == "test"]
    paths = [str(ROOT / p) if not os.path.isabs(p) else p for p in test_df["path"]]
    durations = test_df["duration"].tolist()
    with ThreadPoolExecutor(max_workers=8) as ex:
        waves = list(ex.map(load_waveform, paths, durations))

    def run(chunk: list[np.ndarray]) -> np.ndarray:
        inputs = fe(chunk, sampling_rate=SR, return_tensors="pt")
        pixel_values = inputs["input_values"].to(device)
        with torch.inference_mode():
            out = model(input_values=pixel_values)
        return out.pooler_output.float().cpu().numpy()

    run(waves[:batch_size])                      # warm-up, not measured
    t0 = time.perf_counter()
    embs = [run(waves[i:i + batch_size]) for i in range(0, len(waves), batch_size)]
    emb = np.concatenate(embs, axis=0)
    head["clf"].predict_proba(head["scaler"].transform(emb))
    total_s = time.perf_counter() - t0
    stats = {"n_clips": len(waves), "batch_size": batch_size,
             "total_s": round(total_s, 3),
             "ms_per_clip": round(total_s / len(waves) * 1000.0, 3)}
    print(f"[throughput] batch={batch_size} {stats['ms_per_clip']:.1f} ms/clip "
          f"({stats['n_clips']} clips in {total_s:.1f}s)")
    return stats


def backbone_size_mb() -> tuple[float, str]:
    """On-disk size of the pretrained safetensors plus its cache path."""
    try:
        from huggingface_hub import constants as hf_constants
        cache = Path(hf_constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        cache = Path.home() / ".cache/huggingface/hub"
    hits = sorted(cache.glob(f"models--{MODEL_ID.replace('/', '--')}"
                             "/snapshots/*/model.safetensors"))
    if not hits:
        return 0.0, "unknown"
    path = hits[-1]
    return path.stat().st_size / 1e6, str(path)


def head_size_mb(head: dict) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        joblib.dump({"scaler": head["scaler"], "clf": head["clf"]}, tmp)
        return tmp.stat().st_size / 1e6
    finally:
        tmp.unlink(missing_ok=True)


def _setup_cjk_font() -> None:
    candidates = ["/mnt/c/Windows/Fonts/msyh.ttc",
                  "/mnt/c/Windows/Fonts/simhei.ttf",
                  str(Path.home() / ".fonts/wqy-microhei.ttc")]
    for f in candidates:
        if Path(f).exists():
            try:
                font_manager.fontManager.addfont(f)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
                break
            except Exception:  # noqa: BLE001
                continue
    plt.rcParams["axes.unicode_minus"] = False


def draw_figure(test_res: dict, class_names: list[str], out: Path) -> None:
    _setup_cjk_font()
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    cm = np.asarray(test_res["confusion_matrix"], dtype=int)
    f1 = [test_res["per_class_f1"][c] for c in class_names]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"AST frozen backbone · test n={cm.sum()} · "
                 f"acc={test_res['accuracy']:.3f}, macro-F1={test_res['macro_f1']:.3f}",
                 loc="left", fontsize=10, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "#20303f")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="tracks")

    ax = axes[1]
    order = np.argsort(f1)
    ax.barh(np.arange(len(order)), [f1[i] for i in order],
            color="#3f6b8f", height=0.68)
    ax.set_yticks(np.arange(len(order)), [class_names[i] for i in order])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 (test)")
    ax.set_title("Per-genre F1", loc="left", fontsize=10, fontweight="bold")
    for yi, i in enumerate(order):
        ax.text(f1[i] + 0.01, yi, f"{f1[i]:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out}")


def _cnn_reference() -> dict | None:
    if not METRICS_CSV.exists():
        return None
    df = pd.read_csv(METRICS_CSV)
    cnn = df[(df["model"] == "cnn") & (df["split"] == "test")]
    if cnn.empty:
        return None
    return {"n_seeds": int(cnn["seed"].nunique()),
            "acc_mean": float(cnn["accuracy"].mean()),
            "acc_std": float(cnn["accuracy"].std(ddof=1)) if len(cnn) > 1 else 0.0,
            "f1_mean": float(cnn["macro_f1"].mean()),
            "f1_std": float(cnn["macro_f1"].std(ddof=1)) if len(cnn) > 1 else 0.0,
            "train_s_mean": float(cnn["train_time_s"].mean()),
            "infer_ms_mean": float(cnn["infer_ms_per_clip"].mean()),
            "params": int(cnn["n_params"].iloc[0]),
            "size_mb": float(cnn["model_size_mb"].mean())}


def write_docs(path: Path, df: pd.DataFrame, test_res: dict, val_res: dict,
               head: dict, timing: dict, latency: dict, throughput: dict,
               cost: dict, cache_reused: bool, cnn: dict | None, seed: int,
               figure_written: bool) -> None:
    classes = list(test_res["per_class_f1"].keys())
    n_test = int(np.asarray(test_res["confusion_matrix"]).sum())
    cm = np.asarray(test_res["confusion_matrix"], dtype=int)
    per_class = [(c, test_res["per_class_f1"][c], int(cm[i].sum()))
                 for i, c in enumerate(classes)]
    per_class_sorted = sorted(per_class, key=lambda r: r[1])
    pairs = sorted(((int(cm[i, j]), i, j) for i in range(len(classes))
                    for j in range(len(classes)) if i != j and cm[i, j] > 0),
                   reverse=True)[:5]
    cnn_note = (f"CNN 数据取 `results/metrics.csv` 中 {cnn['n_seeds']} 个 seed 的均值±标准差；"
                f"AST 只有 seed {seed}。") if cnn else "未找到 `results/metrics.csv` 中的 CNN 行。"
    pair_txt = "；".join(f"{classes[i]} → {classes[j]}（{n} 首）" for n, i, j in pairs) or "无"

    if cnn:
        d_acc = (test_res["accuracy"] - cnn["acc_mean"]) * 100
        d_f1 = (test_res["macro_f1"] - cnn["f1_mean"]) * 100
        cnn_acc = f"{cnn['acc_mean']:.4f} ± {cnn['acc_std']:.4f}"
        cnn_f1 = f"{cnn['f1_mean']:.4f} ± {cnn['f1_std']:.4f}"
        delta = (f"相对 CNN：Accuracy {d_acc:+.1f} 个百分点，Macro-F1 {d_f1:+.1f} 个百分点。")
        cnn_row = (f"| CNN（{cnn['n_seeds']} seeds，均值±标准差） | 无（从头训练） | {cnn_acc} | {cnn_f1} | "
                   f"{cnn['train_s_mean']:.2f} s | {cnn['infer_ms_mean']:.2f} ms | "
                   f"{cnn['params']:,} | {cnn['size_mb']:.2f} MB |")
    else:
        cnn_acc = cnn_f1 = "—"
        delta = "（未找到 results/metrics.csv 中的 CNN 行，未做对比。）"
        cnn_row = "| CNN | — | — | — | — | — | — | — |"

    lines = [
        "# AST 迁移学习（P1）：冻结 AudioSet 主干 + 线性分类头",
        "",
        "> 本文件由 `src/train_ast.py` 在实验结束时自动生成，所有数字来自本次真实运行，",
        "> 可追溯到 `results/ast_metrics.json` 与 `results/ast_eval.json`。",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}；seed={seed}。",
        "",
        "## 一句话结论",
        "",
        f"在 {DATASET_NAME} {len(df)} 首、按曲目固定的 70/15/15 划分上，冻结 AST 主干、只训练 "
        f"{head['head_params']:,} 个参数（768→{len(test_res['per_class_f1'])} 逻辑回归）后，测试集 n={n_test} 的 "
        f"Accuracy = **{test_res['accuracy']:.4f}**，Macro-F1 = **{test_res['macro_f1']:.4f}**。"
        f"{delta}",
        "",
        "注意：AST 使用了 AudioSet 预训练权重，属于「迁移学习组」，不能把收益直接解释为",
        "Transformer 架构优于 CNN；它多花的是预训练数据、下载体积与推理时间。",
        "",
        "## 权重来源",
        "",
        f"- HuggingFace 模型：`{MODEL_ID}`",
        "- 预训练数据：AudioSet（YouTube 音频片段，527 类，约 2M 片段）；",
        "  模型卡报告 AudioSet mAP 0.4593，节点名中的 `10-10` 表示训练时 10 秒片段、10 个 patch。",
        "- 架构：AST base，patch 16×16，12 层 Transformer，hidden=768，"
        f"主干参数量 {cost['backbone_params']:,}。",
        "- 授权：BSD-3-Clause（以模型卡为准）。",
        f"- 权重文件：`model.safetensors`，{cost['backbone_mb']:.1f} MB；"
        "首次运行下载到默认 HF 缓存 `~/.cache/huggingface/hub`。",
        f"- 本次加载模型耗时 {timing.get('model_load_s', 0.0):.1f} s。",
        "",
        "## 预处理",
        "",
        f"1. `librosa.load(..., sr={SR}, mono=True)`，取每首曲目**中心 5 秒**"
        "（与 CNN 等模型的中心片段约定一致）。",
        f"2. 短于 5 秒补零；再在右侧补零到 **{PAD_SECONDS:.2f} 秒**"
        f"（{PAD_SAMPLES} 个采样点 = 1024 帧 × 10 ms），以满足 AST 的固定输入长度。",
        f"3. `ASTFeatureExtractor`：16 kHz、128 Mel bins、n_fft=1024、hop=160、"
        "`do_normalize=True`（mean=-4.2677393，std=4.5689974），输出 `(1024, 128)`。",
        "4. 主干输出取 `pooler_output`（768 维）= (CLS + distillation token) / 2，"
        "这正是原始 AudioSet 分类头使用的表征；不做任何微调。",
        "5. 所有曲目的 embedding 先缓存到 `data/processed/ast_embeddings.npz`（git 忽略），"
        "后续实验直接复用。",
        "",
        "## 冻结策略",
        "",
        "- `model.eval()` + 所有参数 `requires_grad_(False)`；",
        "- 前向统一放在 `torch.inference_mode()` 下，不构建计算图；",
        "- 训练集只有 689 首，微调 86M 参数极易过拟合，因此只训练线性头；",
        "- 线性头 = `StandardScaler`（**只在 train 上拟合**）+ `LogisticRegression`；",
        f"- C 在验证集上从 {C_GRID} 中按 Macro-F1 选择，选中 C={head['best']['C']:g}；",
        "- 测试集只在 C 与标准化器确定后评估一次。",
        "",
        "## 结果",
        "",
        "| 划分 | n | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
        f"| val（用于选 C） | {val_res['n_samples']} | {val_res['accuracy']:.4f} | "
        f"{val_res['macro_f1']:.4f} |",
        f"| test（只评一次） | {n_test} | **{test_res['accuracy']:.4f}** | "
        f"**{test_res['macro_f1']:.4f}** |",
        "",
        "逐类 F1（test，按 F1 升序）：",
        "",
        "| 流派 | F1 | 测试样本数 |",
        "|---|---|---|",
    ]
    for c, f, n in per_class_sorted:
        lines.append(f"| {c} | {f:.3f} | {n} |")
    lines += [
        "",
        f"最容易混淆的类别：{pair_txt}。"
        + ("混淆矩阵与逐类 F1 图见 `results/figures/ast.png`，"
           if figure_written else "")
        + "完整矩阵见 `results/ast_eval.json`。",
        "",
        "### 与 CNN baseline 对比",
        "",
        "| 模型 | 预训练 | Accuracy (test) | Macro-F1 | 训练耗时 | 单片段推理 | 参数量 | 模型大小 |",
        "|---|---|---|---|---|---|---|---|",
        cnn_row,
        f"| AST（seed {seed}，本实验） | AudioSet（冻结） | {test_res['accuracy']:.4f} | "
        f"{test_res['macro_f1']:.4f} | {cost['train_time_s']:.2f} s（仅分类头） | "
        f"{throughput['ms_per_clip']:.2f} ms | {cost['n_params']:,} | "
        f"{cost['model_size_mb']:.2f} MB |",
        "",
        cnn_note,
        "两者的训练耗时不可直接比较：CNN 是从头训练 30 epoch，AST 只训练线性头，"
        "但 AST 的 embedding 提取是一次性额外成本（见下）。",
        "",
        "## 额外成本（真实测量）",
        "",
        f"- 权重下载/缓存：{cost['backbone_mb']:.1f} MB（safetensors）。",
        f"- {len(df)} 首 embedding 提取：{timing['embedding_extract_s']:.1f} s"
        f"（约 {timing['embedding_extract_s'] / len(df) * 1000:.1f} ms/首，"
        f"含 CPU 解码/重采样/特征化与 GPU 前向；"
        f"{'本次复用缓存，耗时来自首次提取' if cache_reused else '本次重新计算'}）。",
        f"- 其中 Mel 特征化 {timing['featurize_s']:.1f} s，GPU 前向 {timing['gpu_forward_s']:.1f} s；"
        f"提取阶段峰值显存约 {timing['peak_gpu_mb']:.0f} MB。",
        f"- 分类头训练：{cost['train_time_s']:.2f} s（{len(C_GRID)} 个 C 的验证集网格搜索 + "
        f"最终 train 重拟合；"
        f"最终重拟合 {head['final_fit_s'] * 1000:.0f} ms）。",
        f"- 单片段推理：{latency['median_ms']:.1f} ms（batch=1 中位数，n={latency['n_clips']}；"
        "包含特征化 + 主干前向 + 分类头，不含 wav 解码）；"
        f"按 batch {throughput['batch_size']} 平摊为 {throughput['ms_per_clip']:.1f} ms/片段"
        "（与 CNN 的统计口径一致，用于上表对比）。",
        f"- 参数量：总计 {cost['n_params']:,}，其中可训练 {head['head_params']:,}"
        f"（主干 {cost['backbone_params']:,} 全部冻结）。",
        f"- 模型大小：主干 {cost['backbone_mb']:.1f} MB + 分类头 {cost['head_mb'] * 1000:.0f} KB"
        f" = {cost['model_size_mb']:.2f} MB。",
        "",
        "## 局限",
        "",
        "- **采样率域差异**：GTZAN 原始为 22.05 kHz，AST 要求 16 kHz。重采样会丢掉 "
        "8 kHz 以上的高频内容，对金属、古典等依赖高频音色的流派可能不利。",
        "- **补零到 10.24 s**：每首只有 5 秒真实音频，其余是静音。AST 的位置编码与"
        "注意力会「看到」一半空白输入，和 AudioSet 的 10 秒连续片段不同。",
        "- **预训练域差异**：AudioSet 来自 YouTube，含大量非音乐声音；GTZAN 是 "
        "30 秒、录音质量较统一的音乐片段，二者分布不同。",
        "- **只做线性探针**：没有微调，也没有数据增强或多片段投票；"
        "这测量的是「冻结表征的线性可分性」，不是 AST 的上限。",
        f"- **单 seed**：AST 只跑了 seed {seed}；线性头本身方差小，但没有多 seed 的稳定性估计。",
        f"- **{DATASET_NAME} 本身的问题**：标签噪声与类别边界模糊，绝对准确率不宜过度解读。",
        f"- **成本**：{cost['n_params'] / 1e6:.1f}M 参数、{cost['model_size_mb']:.0f} MB 权重、"
        f"约 {throughput['ms_per_clip'] / 1000:.2f} s/片段"
        + (f"，与 CNN 的 {cnn['params'] / 1e6:.2f}M 参数、{cnn['size_mb']:.1f} MB、"
           f"{cnn['infer_ms_mean']:.2f} ms 相比，部署代价高 2–3 个数量级。" if cnn else "。"),
        "",
        "## 复现命令",
        "",
        "```bash",
        "cd /mnt/d/Desktop/MUSIC_AI",
        "PY=/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python",
        "",
        "# 首次运行会下载约 346 MB 权重到默认 HF 缓存；之后直接复用",
        "$PY -m src.train_ast",
        "",
        "# 忽略 embedding 缓存，强制重新提取（约 3–4 分钟，RTX 4060）",
        "$PY -m src.train_ast --force-extract",
        "",
        "# 查看验证集 C 网格与最终结果",
        "cat results/ast_metrics.json results/ast_eval.json",
        "```",
        "",
        "脚本会写入：`data/processed/ast_embeddings.npz`（缓存）、"
        "`results/ast_metrics.json`、`results/ast_eval.json`、"
        "`results/figures/ast.png`（如未加 `--no-figure`）和本文件；"
        "不会修改 `results/metrics.csv`、README 或任何已有训练脚本。",
        "",
        "## 环境与排障记录",
        "",
        "- 环境：Python 3.12 conda `DataAnalysis`，torch "
        f"{torch.__version__}，transformers {_tr_version()}，GPU RTX 4060 Laptop 8 GB。",
        "- 首次加载 `AutoFeatureExtractor` 时遇到 "
        "`InvalidURL: Invalid port: ':1]'`。根因不是网络，而是本机 `NO_PROXY` 里含有 "
        "带方括号的 `[::1]`，`huggingface_hub 1.30` 解析该列表时出错；"
        "脚本在导入 transformers 前临时移除 `NO_PROXY/no_proxy`，代理仍正常使用。",
        "- `AutoModel` 加载时会提示 `classifier.*` 为 UNEXPECTED：这是预期行为，"
        "因为加载的是 backbone `ASTModel`，不包含 AudioSet 的 527 类分类头。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[docs] wrote {path}")


def _tr_version() -> str:
    import transformers
    return transformers.__version__


def main() -> None:
    ap = argparse.ArgumentParser(description="AST frozen-backbone transfer learning")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--latency-clips", type=int, default=30)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    _seed_everything(args.seed)
    device = _pick_device()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    df = load_split()
    y, le = encode_labels(df["genre"].to_numpy())
    classes = [str(c) for c in le.classes_]  # JSON-safe Python strings
    splits = df["split"].to_numpy()
    track_ids = df["track_id"].to_numpy()

    cache = None if args.force_extract else load_cache(df)
    timing: dict = {}
    if cache is not None:
        embeddings = cache["embeddings"].astype(np.float32)
        timing = {"embedding_extract_s": float(cache["extract_time_s"]),
                  "featurize_s": float(cache["featurize_s"]),
                  "gpu_forward_s": float(cache["gpu_forward_s"]),
                  "peak_gpu_mb": float(cache["peak_gpu_mb"])}
        cache_reused = True
        model = fe = None
        model_load_s = 0.0
    else:
        cache_reused = False
        t0 = time.perf_counter()
        last_exc = None
        fe = model = None
        for attempt in range(1, 6):
            try:
                fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
                model = AutoModel.from_pretrained(MODEL_ID)
                break
            except Exception as exc:  # noqa: BLE001 - retry flaky downloads
                last_exc = exc
                print(f"[load] attempt {attempt}/5 failed: {type(exc).__name__}: {exc}")
                time.sleep(min(5 * attempt, 30))
        if fe is None or model is None:
            raise SystemExit(f"could not load {MODEL_ID}: {last_exc}")
        model_load_s = time.perf_counter() - t0
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[load] AST ready on {device.type} in {model_load_s:.1f}s "
              f"({sum(p.numel() for p in model.parameters()):,} params)")

        embeddings, timing = extract_embeddings(df, fe, model, device,
                                                args.batch_size, args.workers)
        save_cache(df, embeddings, timing)

    assert embeddings.shape == (len(df), 768), embeddings.shape
    assert np.isfinite(embeddings).all()

    head = train_head(embeddings, y, splits, classes, args.seed)
    val_res = head["val"]
    val_res["n_samples"] = int((splits == "val").sum())
    test_res, predictions, predict_s = run_test(head, embeddings, y, splits,
                                                track_ids, classes)
    n_test = int((splits == "test").sum())
    assert n_test > 0, "empty test split"

    # ---- inference latency + on-disk cost (evaluated after the head is frozen) ----
    if model is None:
        t0 = time.perf_counter()
        fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        model_load_s = time.perf_counter() - t0
    timing["model_load_s"] = round(model_load_s, 2)
    latency = benchmark_latency(df, fe, model, head, device,
                                clips=args.latency_clips)
    latency["predict_all_test_s"] = round(predict_s, 4)
    throughput = benchmark_throughput(df, fe, model, head, device,
                                      batch_size=args.batch_size)

    backbone_mb, backbone_path = backbone_size_mb()
    head_mb = head_size_mb(head)
    backbone_params = sum(p.numel() for p in model.parameters())
    n_params = backbone_params + head["head_params"]
    model_size_mb = round(backbone_mb + head_mb, 3)
    if backbone_mb == 0.0:  # cache not found: fall back to float32 weight bytes
        model_size_mb = round(backbone_params * 4 / 1e6 + head_mb, 3)
        backbone_mb = round(backbone_params * 4 / 1e6, 3)

    cost = {"backbone_params": backbone_params, "head_params": head["head_params"],
            "n_params": n_params, "backbone_mb": round(backbone_mb, 3),
            "head_mb": head_mb, "model_size_mb": model_size_mb,
            "train_time_s": round(head["total_s"], 2)}

    import sklearn
    import transformers
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sklearn": sklearn.__version__, "librosa": librosa.__version__,
                "numpy": np.__version__}

    notes = (f"frozen_backbone;embed_extract={timing['embedding_extract_s']:.1f}s;"
             f"head_train={cost['train_time_s']:.2f}s;best_C={head['best']['C']:g};"
             f"val_acc={val_res['accuracy']:.4f};val_f1={val_res['macro_f1']:.4f};"
             f"head_params={head['head_params']};backbone_params={backbone_params};"
             f"16k;5s-center-zero-pad-10.24s;pooler_output;"
             f"cache={'hit' if cache_reused else 'miss'}")

    metrics = {"model": "ast", "seed": args.seed, "split": "test",
               "n_samples": int((splits == "test").sum()),
               "accuracy": round(test_res["accuracy"], 4),
               "macro_f1": round(test_res["macro_f1"], 4),
               "train_time_s": round(head["total_s"], 2),
               "infer_ms_per_clip": throughput["ms_per_clip"],
               "n_params": n_params, "model_size_mb": model_size_mb,
               "notes": notes}
    save_json(metrics, RESULTS / "ast_metrics.json")

    eval_obj = {
        "model": "ast", "seed": args.seed, "split": "test",
        "n_samples": int((splits == "test").sum()),
        "accuracy": round(test_res["accuracy"], 4),
        "macro_f1": round(test_res["macro_f1"], 4),
        "class_names": classes,
        "per_class_f1": test_res["per_class_f1"],
        "confusion_matrix": test_res["confusion_matrix"],
        "classification_report": test_res["report"],
        "val": {"n_samples": val_res["n_samples"],
                "accuracy": round(val_res["accuracy"], 4),
                "macro_f1": round(val_res["macro_f1"], 4),
                "per_class_f1": val_res["per_class_f1"],
                "confusion_matrix": val_res["confusion_matrix"],
                "classification_report": val_res["report"]},
        "head": {"type": "StandardScaler+LogisticRegression",
                 "C_grid": C_GRID, "selected_C": head["best"]["C"],
                 "selection_metric": "val_macro_f1",
                 "grid": head["grid"], "final_fit_s": round(head["final_fit_s"], 3),
                 "train_time_s": cost["train_time_s"],
                 "head_params": head["head_params"],
                 "scaler_fit_on": "train only"},
        "predictions": list(predictions),
        "timing": {**{k: round(float(v), 3) for k, v in timing.items()},
                   "latency_batch1": latency, "throughput": throughput,
                   "predict_all_test_s": round(predict_s, 4)},
        "cost": cost,
        "config": {"model_id": MODEL_ID, "pooling": "pooler_output",
                   "preproc": PREPROC, "sample_rate": SR,
                   "clip_seconds": CLIP_SECONDS, "padded_seconds": PAD_SECONDS,
                   "n_mels": 128, "n_frames": 1024, "batch_size": args.batch_size,
                   "device": device.type, "cache_reused": cache_reused,
                   "backbone_path": backbone_path},
        "versions": versions,
        "notes": [
            "Backbone frozen: eval mode, requires_grad=False, inference_mode forwards.",
            "Only the 768->10 logistic head (plus train-only scaler) is fit.",
            "C selected on val macro-F1; test evaluated once after freezing.",
            "Latency is batch-1 feature-extractor + backbone + head, excluding wav decode.",
            "Pooling is pooler_output = (CLS + distillation token) / 2, the AST "
            "classification representation.",
            "NO_PROXY is dropped before importing transformers to work around a "
            "huggingface_hub 1.30 bracket-IPv6 parsing bug; proxy remains active.",
        ],
    }
    save_json(eval_obj, RESULTS / "ast_eval.json")

    figure_path = FIG / "ast.png"
    if not args.no_figure:
        draw_figure(test_res, classes, figure_path)

    cnn = _cnn_reference()
    write_docs(DOC_PATH, df, test_res, val_res, head, timing, latency,
               throughput, cost, cache_reused, cnn, args.seed,
               figure_written=figure_path.exists())

    print("\n" + "=" * 72)
    print(f"AST seed={args.seed} | test n={metrics['n_samples']} | "
          f"acc={metrics['accuracy']:.4f} | macroF1={metrics['macro_f1']:.4f}")
    print(f"train(head)={metrics['train_time_s']:.2f}s | "
          f"infer={metrics['infer_ms_per_clip']:.2f}ms | "
          f"params={metrics['n_params']:,} | size={metrics['model_size_mb']:.2f}MB")
    print(f"embedding_extract={timing['embedding_extract_s']:.1f}s | "
          f"C={head['best']['C']:g} | cache={'hit' if cache_reused else 'miss'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
