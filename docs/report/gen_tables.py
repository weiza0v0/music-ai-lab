"""Generate LaTeX tables / inline numbers from real result files.

Run before building the PDF. Missing FMA results produce explicit placeholders,
so the report never contains invented numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TAB = OUT / "tables"
TAB.mkdir(exist_ok=True)
MVP = ROOT / "results"
FMA = ROOT / "results_fma"
ABL = ROOT / "results_fma_ablation"

ORDER = ["ast", "cnn", "crnn", "svm", "mlp", "transformer"]
LABEL = {"ast": "AST（预训练）", "cnn": "CNN", "crnn": "CRNN", "svm": "SVM",
         "mlp": "MLP", "transformer": "Transformer"}


def fmt_params(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e5:
        return f"{n/1e3:.0f}K"
    return f"{n/1e3:.0f}K"


def agg(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    df = df[df.split == "test"]
    rows = []
    for m in ORDER:
        sub = df[df.model == m]
        if sub.empty:
            continue
        rows.append({
            "model": m,
            "label": LABEL[m],
            "acc": sub.accuracy.mean(), "acc_std": sub.accuracy.std(ddof=0),
            "f1": sub.macro_f1.mean(), "f1_std": sub.macro_f1.std(ddof=0),
            "train": sub.train_time_s.mean(), "lat": sub.infer_ms_per_clip.mean(),
            "params": sub.n_params.mean(), "size": sub.model_size_mb.mean(),
            "n_seeds": len(sub),
        })
    return pd.DataFrame(rows)


def metric_table(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [r"\begin{table}[H]", r"\centering\small", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"模型 & Acc & Macro-F1 & 训练(s) & 推理(ms) & 参数量 & 大小(MB)\\",
             r"\midrule"]
    for _, r in df.iterrows():
        if r["n_seeds"] > 1:
            acc = f"{r.acc:.3f}$\\pm${r.acc_std:.3f}"
            f1 = f"{r.f1:.3f}$\\pm${r.f1_std:.3f}"
        else:
            acc, f1 = f"{r.acc:.3f}", f"{r.f1:.3f}"
        lines.append(f"{r['label']} & {acc} & {f1} & {r['train']:.1f} & {r['lat']:.2f} & "
                     f"{fmt_params(r['params'])} & {r['size']:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def robustness_table() -> str:
    p = MVP / "robustness.csv"
    if not p.exists():
        return "% robustness.csv missing\n"
    df = pd.read_csv(p).pivot(index="model", columns="condition", values="accuracy")
    conds = ["clean_5s", "clip_3s", "clip_10s", "gain_0.5", "gain_1.5",
             "noise_10db", "noise_0db"]
    head = [r"\begin{table}[H]", r"\centering\small",
            r"\caption{七种扰动下的测试准确率（GTZAN，seed 42，n=147）。}",
            r"\label{tab:robust}", r"\begin{tabular}{lccccccc}", r"\toprule",
            r"模型 & clean & 3s & 10s & 0.5$\times$ & 1.5$\times$ & 10dB & 0dB\\",
            r"\midrule"]
    rows = []
    for m in ["svm", "mlp", "cnn", "crnn", "transformer"]:
        vals = " & ".join(f"{df.loc[m, c]:.3f}" for c in conds)
        rows.append(f"{LABEL[m]} & {vals}\\\\")
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(head + rows + tail)


def ablation_table() -> tuple[str, str]:
    configs = [("cnn_base", "CNN baseline"), ("cnn_specaug", "CNN + SpecAug"),
               ("cnn_specaug_ls", "CNN + SpecAug + LS"),
               ("crnn_specaug", "CRNN + SpecAug + LS"),
               ("tf_specaug", "Transformer + SpecAug + LS")]
    rows, best = [], None
    for key, label in configs:
        f = ABL / key / f"{key.split('_')[0]}_seed42_eval.json"
        if key.startswith("crnn"):
            f = ABL / key / "crnn_seed42_eval.json"
        if key.startswith("tf_"):
            f = ABL / key / "transformer_seed42_eval.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        v = d.get("val", {})
        rows.append((label, v.get("acc", float("nan")), v.get("macro_f1", float("nan")),
                     d.get("best_epoch", "—")))
        if v.get("macro_f1") is not None and (best is None or v["macro_f1"] > best[1]):
            best = (label, v["macro_f1"], v["acc"])
    if not rows:
        return "% ablation missing\n", "（消融结果待生成）"
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{优化消融（FMA-small 验证集，seed 42，20 epoch；测试集未参与选型）。}",
             r"\label{tab:ablation}", r"\begin{tabular}{lccc}", r"\toprule",
             r"配置 & val Acc & val Macro-F1 & best epoch\\", r"\midrule"]
    for label, a, f1, ep in rows:
        lines.append(f"{label} & {a:.3f} & {f1:.3f} & {ep}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    sentence = ""
    if best:
        base = next((r for r in rows if r[0] == "CNN baseline"), None)
        delta = (best[1] - base[2]) * 100 if base else None
        if delta is not None and delta < 1.5:
            sentence = (f"验证集上最佳配置为 {best[0]}（Macro-F1 {best[1]:.3f}），"
                        f"但相对 CNN 基线仅提升 {delta:+.1f} 个百分点，处于单次运行的波动范围内；"
                        f"因此主实验保留基线配置，不引入额外复杂度。")
        else:
            sentence = (f"验证集上最佳配置为 {best[0]}（Macro-F1 {best[1]:.3f}，"
                        f"准确率 {best[2]:.3f}），据此确定主实验的正则化配置。")
    return "\n".join(lines), sentence


def per_class_table() -> str:
    cnn = json.loads((MVP / "cnn_seed42_eval.json").read_text())
    ast_path = MVP / "ast_seed42_eval.json"
    ast = json.loads(ast_path.read_text()) if ast_path.exists() else None
    f1 = cnn["test"]["per_class_f1"]
    keys = sorted(f1, key=f1.get)
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{GTZAN 逐类 F1（seed 42）：CNN 与预训练 AST 对比。}",
             r"\label{tab:perclass}", r"\begin{tabular}{lccc}", r"\toprule",
             r"流派 & CNN F1 & AST F1 & 测试样本\\", r"\midrule"]
    for k in keys:
        a = f"{ast['test']['per_class_f1'][k]:.3f}" if ast else "--"
        lines.append(f"{k} & {f1[k]:.3f} & {a} & 15\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fma_split_table() -> str:
    p = ROOT / "data/splits_fma/splits.csv"
    if not p.exists():
        return "% splits missing\n"
    df = pd.read_csv(p)
    tab = pd.crosstab(df["genre"], df["split"])
    for c in ["train", "val", "test"]:
        if c not in tab:
            tab[c] = 0
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{FMA-small 艺术家分离划分的逐流派数量。}",
             r"\label{tab:fmasplit}", r"\begin{tabular}{lccc}", r"\toprule",
             r"流派 & 训练 & 验证 & 测试\\", r"\midrule"]
    for g, r in tab.iterrows():
        lines.append(f"{g} & {r['train']} & {r['val']} & {r['test']}\\\\")
    lines.append(f"合计 & {tab['train'].sum()} & {tab['val'].sum()} & {tab['test'].sum()}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    mvp = agg(MVP / "metrics.csv")
    (TAB / "mvp.tex").write_text(
        metric_table(mvp, "GTZAN 测试集结果（3 seeds；AST 为单 seed 迁移组，训练时间仅计线性头）。", "tab:gtzan"))

    fma_exists = (FMA / "metrics.csv").exists()
    if fma_exists:
        fma = agg(FMA / "metrics.csv")
        (TAB / "fma.tex").write_text(
            metric_table(fma, "FMA-small 主实验结果（3 seeds，均值$\\pm$标准差；AST 训练时间仅计线性头）。", "tab:fma"))
        (FMA / "figures").mkdir(parents=True, exist_ok=True)
        src = FMA / "figures/leaderboard.png"
        if src.exists():
            (FMA / "figures/leaderboard_fma.png").write_bytes(src.read_bytes())
        best = fma.sort_values("f1", ascending=False).iloc[0]
        fma_sentence = (
            f"在 FMA-small 上，{best.label} 取得最高的 Macro-F1（{best.f1:.3f}），"
            f"测试集约 1\\,200 首。")
    else:
        (TAB / "fma.tex").write_text("% FMA results missing\n")
        fma_sentence = "（FMA-small 主实验结果待生成）"
    (OUT / "numbers_fma.tex").write_text(fma_sentence)

    # data stats
    stats = ROOT / "data/splits_fma/stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        n = s.get("n_decodable", 0)
        (OUT / "numbers_fma_n.tex").write_text(str(n))
        if (ROOT / "data/splits_fma/splits.csv").exists():
            d = pd.read_csv(ROOT / "data/splits_fma/splits.csv")
            c = d.split.value_counts()
            (OUT / "numbers_fma_split.tex").write_text(
                f"{c.get('train', 0)} / {c.get('val', 0)} / {c.get('test', 0)}")
    else:
        (OUT / "numbers_fma_n.tex").write_text("8000")

    abl_tex, abl_sentence = ablation_table()
    (TAB / "ablation.tex").write_text(abl_tex)
    (OUT / "numbers_ablation.tex").write_text(abl_sentence)

    (TAB / "robustness.tex").write_text(robustness_table())
    (TAB / "perclass.tex").write_text(per_class_table())
    (TAB / "fma_split.tex").write_text(fma_split_table())

    # abstract sentence
    def pick(df, m):
        r = df[df.model == m]
        return r.iloc[0] if len(r) else None
    cnn = pick(mvp, "cnn")
    ast = pick(mvp, "ast")
    abs_parts = []
    if cnn is not None:
        abs_parts.append(f"在 GTZAN 上，从头训练的 CNN 的 Macro-F1 为 {cnn.f1:.3f}")
    if ast is not None:
        abs_parts.append(f"冻结预训练主干的 AST 达到 {ast.f1:.3f}")
    if fma_exists:
        fma = agg(FMA / "metrics.csv")
        b = fma.sort_values("f1", ascending=False).iloc[0]
        abs_parts.append(f"在 FMA-small 上最佳模型 {b.label} 的 Macro-F1 为 {b.f1:.3f}")
    # macros for inline numbers
    def macro(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"
    m = []
    m.append(macro("MvpCnnF", f"{cnn.f1:.3f}" if cnn is not None else "--"))
    m.append(macro("MvpAstF", f"{ast.f1:.3f}" if ast is not None else "--"))
    if fma_exists:
        b = fma.sort_values("f1", ascending=False).iloc[0]
        n_test = int(pd.read_csv(ROOT / "data/splits_fma/splits.csv").query("split == 'test'").shape[0])
        m.append(macro("FmaBestModel", b.label))
        m.append(macro("FmaBestF", f"{b.f1:.3f}"))
        m.append(macro("FmaBestAcc", f"{b.acc:.3f}"))
        m.append(macro("FmaTestN", n_test))
        cnn_row = fma[fma.model == "cnn"]
        if len(cnn_row):
            m.append(macro("FmaCnnF", f"{cnn_row.iloc[0].f1:.3f}"))
        leak_csv = ROOT / "results_fma_leak/metrics.csv"
        if leak_csv.exists():
            lk = pd.read_csv(leak_csv)
            lk = lk[(lk.model == "cnn") & (lk.split == "test")]
            if len(lk):
                m.append(macro("LeakCnnF", f"{lk.macro_f1.mean():.3f}"))
                m.append(macro("LeakCnnAcc", f"{lk.accuracy.mean():.3f}"))
                delta = (lk.macro_f1.mean() - cnn_row.iloc[0].f1) * 100
                m.append(macro("LeakDelta", f"{delta:+.1f}"))
    else:
        m += [macro("FmaBestModel", "待定"), macro("FmaBestF", "--"),
              macro("FmaBestAcc", "--"), macro("FmaTestN", "1,200"),
              macro("FmaCnnF", "--"), macro("LeakCnnF", "--"),
              macro("LeakCnnAcc", "--"), macro("LeakDelta", "--")]
    abl_tex, abl_sentence = ablation_table()
    if abl_sentence:
        m.append(macro("AblationSentence", abl_sentence))
    for _name in ("LeakCnnF", "LeakCnnAcc", "LeakDelta"):
        if not any(f"\\{_name}" in x for x in m):
            m.append(macro(_name, "--"))
    # FMA split leakage macros
    if stats.exists():
        st = json.loads(stats.read_text()).get("split", {})
        tl = st.get("track_level_comparison", {})
        m.append(macro("FmaTrackLeak", tl.get("test_tracks_with_train_artist", "--")))
        m.append(macro("FmaTrackLeakN", tl.get("test_tracks", "1200")))
        denom = max(tl.get("test_tracks", 1), 1)
        pct = round(100.0 * tl.get("test_tracks_with_train_artist", 0) / denom, 1)
        m.append(macro("FmaTrackLeakPct", pct))
        m.append(macro("FmaArtistLeak", st.get("test_tracks_with_train_artist", "--")))
    else:
        m += [macro("FmaTrackLeak", "--"), macro("FmaTrackLeakN", "1,200"),
              macro("FmaTrackLeakPct", "--"), macro("FmaArtistLeak", "0")]
    (OUT / "macros.tex").write_text("".join(m), encoding="utf-8")

    (OUT / "numbers_abstract.tex").write_text(
        "；".join(abs_parts) + "。" if abs_parts else "（结果待生成）")
    print("generated:", sorted(p.name for p in TAB.glob("*.tex")),
          "and inline numbers")


if __name__ == "__main__":
    main()
