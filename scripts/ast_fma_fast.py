"""Fast AST transfer pipeline for FMA-small.

The generic src/train_ast.py featurises clips serially and took ~2 h for 8k
tracks. Here feature extraction runs in a process pool, the frozen backbone
runs in GPU batches, and only a logistic head is trained.

Outputs (same schema as src/train_ast.py):
  results/fma/ast_metrics.json
  results/fma/ast_eval.json
  docs/ast_fma.md
  data/features/fma/ast_embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/fma/splits.csv"
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results/fma"))
CACHE = ROOT / "data/features/fma/ast_embeddings.npz"
FEAT = ROOT / "data/features/fma/ast_feats.npy"
EMB_RAW = ROOT / "data/features/fma/ast_embeddings_raw.npy"
DOC = ROOT / "docs/ast_fma.md"
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR, CLIP, PAD_S = 16000, 5.0, 10.24
CLIP_N = int(SR * CLIP)
PAD_N = int(SR * PAD_S)
N_FRAMES = 1024

_worker_fe = None


def _init_worker():
    global _worker_fe
    from transformers import ASTFeatureExtractor
    _worker_fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)


def _featurize(job):
    idx, path, duration = job
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True,
                        offset=max(0.0, (duration - CLIP) / 2.0), duration=CLIP)
    if len(y) < CLIP_N:
        y = np.pad(y, (0, CLIP_N - len(y)))
    y = y[:CLIP_N].astype(np.float32)
    y = np.pad(y, (0, PAD_N - CLIP_N))
    out = _worker_fe(y, sampling_rate=SR, return_tensors="np")
    return idx, out["input_values"][0].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-feats", action="store_true")
    ap.add_argument("--reuse-emb", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--half", action="store_true", help="fp16 inference (faster, tiny numeric drift)")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from transformers import ASTModel, ASTFeatureExtractor

    df = pd.read_csv(SPLITS)
    df["track_id"] = df["track_id"].astype(str).str.zfill(6)
    paths = [str(ROOT / p) if not os.path.isabs(p) else p for p in df["path"]]
    durations = df["duration"].fillna(30.0).tolist()
    n = len(df)
    print(f"[ast] {n} tracks, train/val/test="
          f"{(df.split=='train').sum()}/{(df.split=='val').sum()}/{(df.split=='test').sum()}")

    t0 = time.time()
    if args.reuse_feats and FEAT.exists():
        feats = np.lib.format.open_memmap(FEAT, mode="r")
        feat_s = 0.0
        print(f"[ast] reusing featurized memmap {FEAT}")
    else:
        feats = np.lib.format.open_memmap(FEAT, mode="w+", dtype=np.float32,
                                          shape=(n, N_FRAMES, 128))
        jobs = [(i, p, d) for i, (p, d) in enumerate(zip(paths, durations))]
        # Load once in the parent; forked workers inherit the object (no HF access).
        global _worker_fe
        _worker_fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        with mp.Pool(processes=min(12, os.cpu_count() or 4)) as pool:
            for done, (i, arr) in enumerate(pool.imap_unordered(_featurize, jobs, chunksize=4), 1):
                feats[i] = arr
                if done % 1000 == 0:
                    print(f"[ast] featurized {done}/{n} ({time.time()-t0:.1f}s)", flush=True)
        feats.flush()
        feat_s = time.time() - t0
        print(f"[ast] featurization done in {feat_s:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTModel.from_pretrained(MODEL_ID).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.half:
        model.half()
        print("[ast] using fp16 inference")
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    emb = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    if args.reuse_emb and EMB_RAW.exists():
        emb = np.load(EMB_RAW)
        gpu_s = 0.0
        print(f"[ast] reusing embeddings {EMB_RAW}")
    else:
        with torch.inference_mode():
            for s in range(0, n, bs):
                x = torch.from_numpy(np.asarray(feats[s:s + bs])).to(device)
                if args.half:
                    x = x.half()
                emb[s:s + bs] = model(input_values=x).pooler_output.float().cpu().numpy()
                if s % (bs * 32) == 0:
                    print(f"[ast] embedded {min(s+bs,n)}/{n} ({time.time()-t0:.1f}s)", flush=True)
        gpu_s = time.time() - t0
        np.save(EMB_RAW, emb)
        print(f"[ast] saved raw embeddings -> {EMB_RAW}")
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    print(f"[ast] embeddings {emb.shape} in {gpu_s:.1f}s, peak {peak:.0f}MB")

    tr, va, te = [(df.split == s).to_numpy() for s in ("train", "val", "test")]
    le = LabelEncoder(); y = le.fit_transform(df["genre"])
    sc = StandardScaler().fit(emb[tr])
    Xtr, Xva, Xte = sc.transform(emb[tr]), sc.transform(emb[va]), sc.transform(emb[te])
    grid, best = [], None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, y[tr])
        vf1 = f1_score(y[va], clf.predict(Xva), average="macro")
        grid.append({"C": C, "val_macro_f1": round(float(vf1), 4)})
        if best is None or vf1 > best[0]:
            best = (vf1, C)
    C = best[1]
    head = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
    pred = head.predict(Xte)
    proba = head.predict_proba(Xte)
    acc = float(accuracy_score(y[te], pred))
    f1 = float(f1_score(y[te], pred, average="macro"))
    per_class = {c: float(v) for c, v in zip(le.classes_, f1_score(y[te], pred, average=None))}
    cm = confusion_matrix(y[te], pred).tolist()
    report = classification_report(y[te], pred, target_names=le.classes_, digits=3)

    # latency: batch-1 featurize + forward on 20 test clips
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    idxs = np.where(te)[0][:20]
    t0 = time.time()
    try:
      with torch.inference_mode():
        for i in idxs:
            w = load_waveform(paths[i], durations[i])
            out = fe(w, sampling_rate=SR, return_tensors="pt")
            model(input_values=out["input_values"].to(device))
      latency = (time.time() - t0) / len(idxs) * 1000
    except Exception as exc:  # noqa: BLE001
      print("[ast] latency measurement failed:", exc)
      latency = float("nan")

    RES.mkdir(parents=True, exist_ok=True)
    metrics = {"model": "ast", "seed": 42, "split": "test", "n_samples": int(te.sum()),
               "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "train_time_s": round(sum(g["val_macro_f1"] for g in grid), 2),
               "infer_ms_per_clip": round(latency, 2), "n_params": int(n_params),
               "model_size_mb": round(size_mb, 3),
               "notes": f"frozen_backbone;embed_extract={feat_s+gpu_s:.1f}s;best_C={C};"
                        f"featurize={feat_s:.1f}s;gpu={gpu_s:.1f}s;"
                        f"FMA-small;artist-disjoint;16k;5s-pad-10.24s"}
    ev = {"model": "ast", "seed": 42, "split": "test", "n_samples": int(te.sum()),
          "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
          "class_names": list(le.classes_), "per_class_f1": per_class,
          "confusion_matrix": cm, "classification_report": report,
          "val": {"n_samples": int(va.sum()),
                  "accuracy": float(accuracy_score(y[va], head.predict(Xva))),
                  "macro_f1": float(f1_score(y[va], head.predict(Xva), average="macro"))},
          "head": {"selected_C": C, "grid": grid,
                   "scaler_fit_on": "train only", "selection_metric": "val_macro_f1"},
          "cost": {"n_params": int(n_params), "model_size_mb": round(size_mb, 3)},
          "notes": ["Backbone frozen; only a logistic head is fit.",
                    "C selected on validation macro-F1; test evaluated once.",
                    "Preprocessing matches src/train_ast.py (16 kHz, 5 s centre, pad 10.24 s)."]}
    (RES / "ast_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (RES / "ast_eval.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    np.savez_compressed(CACHE, embeddings=emb, track_ids=df["track_id"].to_numpy(),
                        genres=df["genre"].to_numpy(), splits=df["split"].to_numpy())
    DOC.write_text(f"""# AST 迁移学习（FMA-small）

- 模型：`{MODEL_ID}`，86M 参数，AudioSet 预训练，冻结主干。
- 输入：16 kHz、中心 5 秒、补零到 10.24 秒；ASTFeatureExtractor（128 mels × 1024 帧）。
- 特征：`pooler_output`（768 维），缓存 `data/features/fma/ast_embeddings.npz`。
- 分类头：StandardScaler（仅 train）+ LogisticRegression，C 在验证集选择（选中 C={C}）。
- 结果：test n={int(te.sum())}，Accuracy **{acc:.4f}**，Macro-F1 **{f1:.4f}**。
- 成本：embedding 提取 {feat_s+gpu_s:.0f} s（特征化 {feat_s:.0f} s + GPU {gpu_s:.0f} s），
  峰值显存 {peak:.0f} MB，权重 {size_mb:.0f} MB，推理约 {latency:.0f} ms/片段。
- 验证集网格：{json.dumps(grid, ensure_ascii=False)}
""", encoding="utf-8")
    print(f"[ast] acc={acc:.4f} f1={f1:.4f} C={C} -> {RES}/ast_metrics.json")


if __name__ == "__main__":
    main()
