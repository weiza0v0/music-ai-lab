"""Stage G — P0 robustness experiment: how stable are the five baselines?

All 7 conditions are evaluated on the same 147 held-out test tracks, with the
same rule as training: the split file decides the test set, classical models
read the 209-d hand-crafted vector, deep models read (1, 64, 216) log-Mel and
normalise with the mean/std stored *in the checkpoint* (never re-fitted here).

Conditions
----------
clean_5s    centre 5 s                  -> the protocol used everywhere else
clip_3s     centre 3 s                  -> log-Mel stretched 130 -> 216 frames
clip_10s    centre 10 s                 -> log-Mel compressed 431 -> 216 frames
gain_0.5    waveform x 0.5, clipped     -> quieter playback
gain_1.5    waveform x 1.5, clipped     -> louder playback (clip ratio recorded)
noise_10db  + white Gaussian noise, SNR 10 dB
noise_0db   + white Gaussian noise, SNR 0 dB

`delta_acc` / `delta_f1` are `condition - clean` for the same model, so a
negative number is a drop. Results are written to results/gtzan/robustness.csv and
results/gtzan/figures/robustness.png.

Run:
    python -m src.robustness                 # full experiment
    python -m src.robustness --workers 12
    python -m src.robustness --plot-only     # redraw the figure from the CSV
"""
from __future__ import annotations

import os

# Must precede numpy/librosa so fork-ed workers inherit single-threaded BLAS;
# the parent parallelises across tracks instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.audio_features import (HOP, N_FFT, N_FRAMES, N_MELS,  # noqa: E402
                                     SAMPLE_RATE, handcrafted)

SPLITS = ROOT / "data/splits/gtzan/splits.csv"
CKPT = ROOT / "checkpoints" / "gtzan"
RES = ROOT / "results" / "gtzan"
FIG = RES / "figures"
OUT_CSV = RES / "robustness.csv"
OUT_PNG = FIG / "robustness.png"
CACHE = ROOT / "data/features/gtzan/features.npz"

MODEL_ORDER = ("svm", "mlp", "cnn", "crnn", "transformer")
CLASSICAL = frozenset({"svm", "mlp"})
MODEL_LABEL = {"svm": "SVM", "mlp": "MLP", "cnn": "CNN", "crnn": "CRNN",
               "transformer": "Transformer"}
COLORS = {"svm": "#8a8f98", "mlp": "#6b8ea8", "cnn": "#c2703d", "crnn": "#9c6b8f",
          "transformer": "#4f7a5f"}


@dataclass(frozen=True)
class Condition:
    """One perturbation. `duration` is the window length in seconds."""

    name: str
    duration: float
    gain: float | None = None
    snr_db: float | None = None


CONDITIONS = (
    Condition("clean_5s", 5.0),
    Condition("clip_3s", 3.0),
    Condition("clip_10s", 10.0),
    Condition("gain_0.5", 5.0, gain=0.5),
    Condition("gain_1.5", 5.0, gain=1.5),
    Condition("noise_10db", 5.0, snr_db=10.0),
    Condition("noise_0db", 5.0, snr_db=0.0),
)
COND_LABEL = {
    "clean_5s": "clean 5 s",
    "clip_3s": "3 s 片段",
    "clip_10s": "10 s 片段",
    "gain_0.5": "增益 ×0.5",
    "gain_1.5": "增益 ×1.5",
    "noise_10db": "噪声 10 dB",
    "noise_0db": "噪声 0 dB",
}


# --------------------------------------------------------------------------- #
# audio -> representation                                                     #
# --------------------------------------------------------------------------- #
def load_window(path: str | Path, duration: float) -> np.ndarray:
    """Centre `duration` seconds of a track, padded/truncated to exact samples.

    Mirrors src.data.audio_features.load_clip, which is hard-wired to 5 s.
    """
    total = librosa.get_duration(path=str(path))
    offset = max(0.0, (total - duration) / 2.0)
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                        offset=offset, duration=duration)
    n = int(round(SAMPLE_RATE * duration))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype(np.float32)


def _stable_seed(track_id: str, cond_name: str) -> int:
    """Reproducible per-(track, condition) RNG seed — Python hash is salted."""
    return zlib.crc32(f"{track_id}|{cond_name}".encode("utf-8")) & 0xFFFFFFFF


def apply_condition(y: np.ndarray, cond: Condition, track_id: str
                    ) -> tuple[np.ndarray, dict]:
    """Apply gain / additive noise; clip to [-1, 1] and record how much clipped."""
    stats: dict = {"clip_ratio": None, "gain": cond.gain, "snr_db": cond.snr_db}
    if cond.gain is not None:
        y = y * np.float32(cond.gain)
    elif cond.snr_db is not None:
        rng = np.random.default_rng(_stable_seed(track_id, cond.name))
        p_sig = float(np.mean(np.square(y.astype(np.float64))))
        p_noise = p_sig / (10.0 ** (cond.snr_db / 10.0))
        noise = rng.normal(0.0, float(np.sqrt(p_noise)), size=y.shape)
        y = y + noise.astype(np.float32)
    else:
        return y, stats                      # clean / clip_*: waveform untouched

    stats["clip_ratio"] = float(np.mean(np.abs(y) > 1.0))
    return np.clip(y, -1.0, 1.0).astype(np.float32), stats


def log_mel_fixed(y: np.ndarray) -> tuple[np.ndarray, int]:
    """64-bin log-Mel, time axis linearly interpolated to exactly N_FRAMES.

    Same mel parameters as src.data.audio_features.log_mel; the only addition is
    the np.interp stretch/compress so 3 s and 10 s clips also feed (64, 216).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    db = librosa.power_to_db(mel, ref=np.max)
    native = int(db.shape[1])
    if native == N_FRAMES:
        return db.astype(np.float32), native
    old = np.arange(native, dtype=np.float64)
    new = np.linspace(0.0, native - 1.0, N_FRAMES)
    out = np.empty((db.shape[0], N_FRAMES), dtype=np.float32)
    for i in range(db.shape[0]):
        out[i] = np.interp(new, old, db[i])
    return out, native


def _extract_track(job: tuple[int, str, str]):
    """Worker: one track -> (feat[cond], mel[cond], meta[cond]) for all conditions."""
    idx, track_id, path = job
    try:
        # The 5 s window is reused by clean + gain + noise; 3 s / 10 s load once.
        y5 = load_window(path, 5.0)
        feats, mels, meta = {}, {}, {}
        for cond in CONDITIONS:
            y = y5 if cond.duration == 5.0 else load_window(path, cond.duration)
            y_pert, stats = apply_condition(y, cond, track_id)
            mel, native = log_mel_fixed(y_pert)
            stats["frames"] = native
            stats["n_samples"] = int(len(y_pert))
            feats[cond.name] = handcrafted(y_pert)
            mels[cond.name] = mel
            meta[cond.name] = stats
    except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
        return idx, None, f"{track_id}: {type(exc).__name__}: {exc}"
    return idx, (feats, mels, meta), None


# --------------------------------------------------------------------------- #
# feature extraction                                                          #
# --------------------------------------------------------------------------- #
def test_frame() -> pd.DataFrame:
    df = pd.read_csv(SPLITS)
    df = df[df["split"] == "test"].copy().reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"no test rows in {SPLITS}")
    if "path" not in df.columns:
        df["path"] = df["track_id"].map(
            lambda t: f"data/raw/gtzan/genres/{t.split('.')[0]}/{t}.wav")
    return df


def extract_features(df: pd.DataFrame, workers: int = 12):
    n = len(df)
    feats = {c.name: np.zeros((n, 209), np.float32) for c in CONDITIONS}
    mels = {c.name: np.zeros((n, N_MELS, N_FRAMES), np.float32) for c in CONDITIONS}
    meta = {c.name: [None] * n for c in CONDITIONS}
    jobs = [(i, str(r.track_id), str(ROOT / str(r.path)))
            for i, r in enumerate(df.itertuples(index=False))]

    t0 = time.time()
    done, errors = 0, []
    print(f"[robustness] extracting {n} tracks x {len(CONDITIONS)} conditions "
          f"with {workers} workers", flush=True)
    with mp.Pool(processes=workers) as pool:
        for idx, payload, err in pool.imap_unordered(_extract_track, jobs, chunksize=2):
            done += 1
            if err:
                errors.append(err)
            else:
                f, m, st = payload
                for name in feats:
                    feats[name][idx] = f[name]
                    mels[name][idx] = m[name]
                    meta[name][idx] = st[name]
            if done % 20 == 0 or done == n:
                print(f"[robustness] {done}/{n} tracks ({time.time() - t0:.1f}s)",
                      flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} track(s) failed:\n" + "\n".join(errors[:10]))
    print(f"[robustness] features ready in {time.time() - t0:.1f}s", flush=True)
    return feats, mels, meta


# --------------------------------------------------------------------------- #
# models                                                                      #
# --------------------------------------------------------------------------- #
def _ckpt(model: str, seed: int) -> Path:
    pat = "*.joblib" if model in CLASSICAL else "*.pt"
    path = CKPT / f"{model}_seed{seed}{Path(pat).suffix}"
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def load_models(seed: int):
    """Load the 5 checkpoints; deep nets are rebuilt from their saved config."""
    import torch

    from src.models.cnn import build

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict = {}
    for name in MODEL_ORDER:
        path = _ckpt(name, seed)
        if name in CLASSICAL:
            bundle = joblib.load(path)
            models[name] = {"classes": [str(c) for c in bundle["classes"]],
                            "model": bundle["model"]}
        else:
            state = torch.load(path, map_location="cpu")
            net = build(state["model"], n_classes=len(state["classes"]))
            net.load_state_dict(state["state_dict"])
            net.eval().to(device)
            models[name] = {"classes": [str(c) for c in state["classes"]],
                            "net": net, "norm": state["norm"]}
    return models, device


def _proba(name: str, models: dict, feat: np.ndarray, mel: np.ndarray,
           device, batch: int = 64) -> np.ndarray:
    if name in CLASSICAL:
        return models[name]["model"].predict_proba(feat)
    import torch

    norm = models[name]["norm"]
    x = ((mel - np.float32(norm["mean"])) / np.float32(norm["std"]))[:, None, :, :]
    xt = torch.from_numpy(x.astype(np.float32))
    net = models[name]["net"]
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            out.append(torch.softmax(net(xt[i:i + batch].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_all(models: dict, feats: dict, mels: dict, df: pd.DataFrame, device):
    """Return {(model, condition): (pred_idx, y_true_idx, accuracy, macro_f1)}."""
    from sklearn.metrics import f1_score

    classes = models[MODEL_ORDER[0]]["classes"]
    for name in MODEL_ORDER:
        if models[name]["classes"] != classes:
            raise RuntimeError(f"class order mismatch for {name}")
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([cls_idx[g] for g in df["genre"]], dtype=np.int64)

    results: dict = {}
    for name in MODEL_ORDER:
        for cond in CONDITIONS:
            # Use each model's own decision rule, exactly as in training/eval:
            # SVM/MLP -> .predict() (SVM's Platt probabilities can disagree with
            # the OVO decision on ~7% of samples), deep models -> softmax argmax.
            if name in ("svm", "mlp"):
                pred = models[name]["model"].predict(feats[cond.name]).astype(np.int64)
            else:
                proba = _proba(name, models, feats[cond.name], mels[cond.name], device)
                pred = proba.argmax(axis=1).astype(np.int64)
            acc = float((pred == y_true).mean())
            f1 = float(f1_score(y_true, pred, average="macro",
                                labels=list(range(len(classes))), zero_division=0))
            results[(name, cond.name)] = (pred, y_true, acc, f1)
            print(f"[robustness] {MODEL_LABEL[name]:>11s} {cond.name:<10s} "
                  f"acc={acc:.4f} macroF1={f1:.4f}", flush=True)
    return results, y_true


# --------------------------------------------------------------------------- #
# reporting                                                                   #
# --------------------------------------------------------------------------- #
def write_csv(results: dict) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        base_acc = results[(name, "clean_5s")][2]
        base_f1 = results[(name, "clean_5s")][3]
        for cond in CONDITIONS:
            _, _, acc, f1 = results[(name, cond.name)]
            rows.append({
                "model": name,
                "condition": cond.name,
                "n": int(len(results[(name, cond.name)][1])),
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "delta_acc": round(acc - base_acc, 4),
                "delta_f1": round(f1 - base_f1, 4),
            })
    out = pd.DataFrame(rows, columns=["model", "condition", "n", "accuracy",
                                      "macro_f1", "delta_acc", "delta_f1"])
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[robustness] wrote {OUT_CSV} ({len(out)} rows)")
    return out


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # CJK-safe: matplotlib's bundled DejaVu has no Chinese glyphs. Register every
    # available face (regular + bold) and then pick the first family name.
    chosen = None
    for f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyhbd.ttc",
              "/mnt/c/Windows/Fonts/simhei.ttf",
              str(Path.home() / ".fonts/wqy-microhei.ttc")):
        if not Path(f).exists():
            continue
        try:
            font_manager.fontManager.addfont(f)
            if chosen is None:
                chosen = font_manager.FontProperties(fname=f).get_name()
        except Exception:  # noqa: BLE001
            continue
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def plot_results(csv_path: Path = OUT_CSV, out_png: Path = OUT_PNG) -> Path:
    plt = _setup_matplotlib()
    df = pd.read_csv(csv_path)
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    conds = [c.name for c in CONDITIONS if c.name in set(df["condition"])]
    acc = df.pivot(index="model", columns="condition", values="accuracy").loc[models, conds]
    delta = (df.pivot(index="model", columns="condition", values="delta_acc")
             .loc[models, conds] * 100.0)          # percentage points

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.75, 1.0]})
    x = np.arange(len(conds))
    w = 0.82 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, acc.loc[m].values, w,
               label=MODEL_LABEL[m], color=COLORS[m], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, [COND_LABEL[c] for c in conds], fontsize=9)
    ax.set_ylim(0.0, 0.92)
    ax.set_ylabel("测试 Accuracy（n=147）")
    ax.set_title("七种扰动下的准确率", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.7)
    ax.set_axisbelow(True)

    data = delta.values
    vmax = max(float(np.abs(data).max()), 0.5)
    im = axh.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axh.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds],
                   rotation=40, ha="right", fontsize=8)
    axh.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(conds)):
            v = data[i, j]
            axh.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(v) > 0.6 * vmax else "#333333")
    axh.set_title("相对 clean 的变化（百分点）", loc="left", fontsize=12,
                  fontweight="bold")
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
    cb.set_label("Δ Accuracy（百分点）", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Music AI Lab — 五个模型的鲁棒性（GTZAN 测试集，seed 42）",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(models), frameon=False, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[robustness] wrote {out_png}")
    return out_png


def check_clean_alignment(feats: dict, mels: dict, df: pd.DataFrame) -> dict | None:
    """Recomputed clean_5s must match the cached features every model trained on."""
    if not CACHE.exists():
        print("[robustness] cached features.npz missing — skip clean alignment check")
        return None
    d = np.load(CACHE, allow_pickle=True)
    pos = {str(t): i for i, t in enumerate(d["track_ids"])}
    if not all(str(t) in pos for t in df["track_id"]):
        print("[robustness] cached features do not cover the test split — skip check")
        return None
    rows = np.array([pos[str(t)] for t in df["track_id"]])
    feat_mae = float(np.abs(feats["clean_5s"] - d["X_feat"][rows]).mean())
    mel_mae = float(np.abs(mels["clean_5s"] -
                           d["X_mel"][rows].astype(np.float32)).mean())
    print(f"[robustness] clean_5s vs cached features: feat MAE={feat_mae:.2e}, "
          f"mel MAE={mel_mae:.2e} dB")
    return {"feat_mae": feat_mae, "mel_mae": mel_mae}


def compare_with_metrics(acc_by_model: dict, seed: int) -> dict[str, float]:
    """Clean accuracy should reproduce the seed-42 rows of results/gtzan/metrics.csv."""
    path = RES / "metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    ref = df[(df["split"] == "test") & (df["seed"] == seed)]
    diffs = {}
    print(f"[robustness] clean_5s vs results/gtzan/metrics.csv (seed {seed}):")
    for name in MODEL_ORDER:
        hit = ref[ref["model"] == name]
        if hit.empty:
            continue
        r = float(hit.iloc[0]["accuracy"])
        d = acc_by_model[name] - r
        diffs[name] = d
        print(f"  {MODEL_LABEL[name]:>11s} ours={acc_by_model[name]:.4f} "
              f"reference={r:.4f} diff={d:+.4f}")
    return diffs


def report(results: dict, meta: dict, df: pd.DataFrame) -> None:
    print("\n=== per-model robustness summary (test n=%d) ===" % len(df))
    print(f"{'model':<12}{'clean':>7}{'worst cond':>14}{'min acc':>9}"
          f"{'delta':>8}{'mean |Δ|':>9}")
    for name in MODEL_ORDER:
        base = results[(name, "clean_5s")][2]
        worst = min((c.name for c in CONDITIONS),
                    key=lambda c: results[(name, c)][2])
        mn = results[(name, worst)][2]
        mean_abs = float(np.mean([abs(results[(name, c.name)][2] - base)
                                  for c in CONDITIONS if c.name != "clean_5s"]))
        print(f"{MODEL_LABEL[name]:<12}{base:>7.4f}{worst:>14}{mn:>9.4f}"
              f"{mn - base:>+8.4f}{mean_abs:>9.4f}")

    print("\n=== prediction flips vs clean (count / 147) ===")
    header = "model".ljust(12) + "".join(f"{COND_LABEL[c.name]:>12}" for c in CONDITIONS[1:])
    print(header)
    for name in MODEL_ORDER:
        base = results[(name, "clean_5s")][0]
        row = [str(int((results[(name, c.name)][0] != base).sum()))
               for c in CONDITIONS[1:]]
        print(f"{MODEL_LABEL[name]:<12}" + "".join(f"{v:>12}" for v in row))

    print("\n=== condition diagnostics ===")
    for cond in CONDITIONS:
        frames = {m.get("frames") for m in meta[cond.name] if m}
        clips = [m["clip_ratio"] for m in meta[cond.name]
                 if m and m["clip_ratio"] is not None]
        line = f"{cond.name:<11} duration={cond.duration:.0f}s frames={sorted(frames)}"
        if clips:
            line += f" clipped_samples={np.mean(clips) * 100:.3f}% (mean), " \
                    f"{np.max(clips) * 100:.2f}% (max)"
        print(line)

    print("\n=== full table ===")
    print(f"{'model':<12}{'condition':<12}{'n':>4}{'acc':>8}{'f1':>8}"
          f"{'Δacc':>8}{'Δf1':>8}")
    for name in MODEL_ORDER:
        base_acc = results[(name, "clean_5s")][2]
        base_f1 = results[(name, "clean_5s")][3]
        for cond in CONDITIONS:
            _, _, acc, f1 = results[(name, cond.name)]
            print(f"{MODEL_LABEL[name]:<12}{cond.name:<12}{len(df):>4}"
                  f"{acc:>8.4f}{f1:>8.4f}{acc - base_acc:>+8.4f}{f1 - base_f1:>+8.4f}")


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="P0 robustness experiment")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=42,
                    help="which checkpoint seed to evaluate (default 42)")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N test tracks, results are NOT written")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw results/gtzan/figures/robustness.png from results/gtzan/robustness.csv")
    args = ap.parse_args(argv)

    if args.plot_only:
        plot_results()
        return

    df = test_frame()
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[debug] limited to {len(df)} tracks — results will NOT be written")
    print(f"[robustness] test tracks={len(df)} seed={args.seed} "
          f"conditions={len(CONDITIONS)} models={len(MODEL_ORDER)}")

    feats, mels, meta = extract_features(df, workers=args.workers)
    check_clean_alignment(feats, mels, df)
    models, device = load_models(args.seed)
    print(f"[robustness] models loaded on {device.type}")
    results, _ = evaluate_all(models, feats, mels, df, device)

    if args.limit:
        print("[debug] done — nothing written (drop --limit for the real run)")
        return

    out = write_csv(results)
    plot_results()
    report(results, meta, df)
    compare_with_metrics({m: results[(m, "clean_5s")][2] for m in MODEL_ORDER}, args.seed)
    print(f"\n[robustness] rows={len(out)} models={out['model'].nunique()} "
          f"conditions={out['condition'].nunique()} n_per_row={sorted(out['n'].unique())}")


if __name__ == "__main__":
    main()
