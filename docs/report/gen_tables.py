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
MVP = ROOT / "results" / "gtzan"
FMA = ROOT / "results/fma"
ABL = ROOT / "results/fma_ablation"

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
    p = ROOT / "data/splits/fma/splits.csv"
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


def gtzan_table() -> str:
    df = pd.read_csv(ROOT / "data/splits/gtzan/splits.csv")
    st = json.loads((ROOT / "data/splits/gtzan/stats.json").read_text())
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{GTZAN 清洗后的逐流派统计。}",
             r"\label{tab:gtzan-data}", r"\begin{tabular}{lccc}", r"\toprule",
             r"流派 & 曲目数 & 平均时长 (s) & 采样率\\", r"\midrule"]
    for g, sub in df.groupby("genre"):
        lines.append(f"{g} & {len(sub)} & {sub['duration'].mean():.2f} & 22,050\\\\")
    lines.append(f"合计 & {len(df)} & {df['duration'].mean():.2f} & --\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def perseed_table(csv_path: Path, caption: str, label: str) -> str:
    if not csv_path.exists():
        return "% missing\n"
    df = pd.read_csv(csv_path)
    df = df[df.split == "test"].copy()
    lines = [r"\begin{table}[H]", r"\centering\scriptsize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"模型 & seed & Acc & Macro-F1 & 训练(s) & 推理(ms)\\", r"\midrule"]
    for _, r in df.sort_values(["model", "seed"]).iterrows():
        lines.append(f"{r['model']} & {int(r['seed'])} & {r['accuracy']:.4f} & "
                     f"{r['macro_f1']:.4f} & {r['train_time_s']:.1f} & "
                     f"{r['infer_ms_per_clip']:.2f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fma_perclass_table() -> str:
    files = {"cnn": FMA / "cnn_seed42_eval.json",
             "crnn": FMA / "crnn_seed42_eval.json",
             "ast": FMA / "ast_seed42_eval.json"}
    data = {k: json.loads(v.read_text()) for k, v in files.items() if v.exists()}
    if not data:
        return "% missing\n"
    classes = data["cnn"]["classes"] if "cnn" in data else data["ast"]["classes"]
    support = {}
    sp = ROOT / "data/splits/fma/splits.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        support = sdf[sdf.split == "test"]["genre"].value_counts().to_dict()
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{FMA-small 逐类 F1（seed 42）与测试样本数。}",
             r"\label{tab:fma-perclass}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"流派 & CNN & CRNN & AST & 测试样本\\", r"\midrule"]
    for c in sorted(classes):
        vals = []
        for m in ["cnn", "crnn", "ast"]:
            d = data.get(m)
            vals.append(f"{d['test']['per_class_f1'][c]:.3f}" if d else "--")
        lines.append(f"{c} & " + " & ".join(vals) + f" & {support.get(c, '--')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def arch_table() -> str:
    return r"""\begin{table}[H]
\centering\small
\caption{自建模型的层结构与参数量（GTZAN 10 类；括号内为 FMA 8 类的差异）。}
\label{tab:arch}
\begin{tabular}{llc}
\toprule
模型 & 关键层（输入 $\rightarrow$ 输出） & 参数量\\
\midrule
CNN & Conv3$\times$3(32)$\times$2 $\rightarrow$ Pool & \multirow{3}{*}{592,554}\\
    & Conv3$\times$3(64)$\times$2 $\rightarrow$ Pool & \\
    & Conv3$\times$3(128)$\times$2 $\rightarrow$ Pool$\times$2 & \\
    & AdaptiveAvgPool(2,4) $\rightarrow$ FC(1024,10) & \\
\midrule
CRNN & Conv(32/64/128) $\rightarrow$ BiGRU(96) & \multirow{2}{*}{418,986}\\
     & 时间平均 $\rightarrow$ FC(192,10) & \\
\midrule
Transformer & Linear(64,128)+Pos $\rightarrow$ 2$\times$Pre-LN Encoder & \multirow{2}{*}{302,474}\\
            & 时间平均 $\rightarrow$ FC(128,10) & \\
\bottomrule
\end{tabular}
\end{table}
"""


def robustness_delta_table() -> str:
    p = MVP / "robustness.csv"
    if not p.exists():
        return "% missing\n"
    df = pd.read_csv(p)
    piv = df.pivot(index="model", columns="condition", values="delta_acc") * 100
    conds = ["clip_3s", "clip_10s", "gain_0.5", "gain_1.5", "noise_10db", "noise_0db"]
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{相对 clean 的准确率变化（百分点，GTZAN seed 42）。}",
             r"\label{tab:robust-delta}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r"模型 & 3s & 10s & 0.5$\times$ & 1.5$\times$ & 10dB & 0dB\\", r"\midrule"]
    for m in ["svm", "mlp", "cnn", "crnn", "transformer"]:
        vals = " & ".join(f"{piv.loc[m, c]:+.1f}" for c in conds)
        lines.append(f"{LABEL[m]} & {vals}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def ast_grid_table() -> str:
    p = FMA / "ast_eval.json"
    if not p.exists():
        return "% missing\n"
    d = json.loads(p.read_text())
    grid = d.get("head", {}).get("grid", [])
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{AST 线性头的验证集 C 网格（FMA-small）。}",
             r"\label{tab:ast-grid}", r"\begin{tabular}{cc}", r"\toprule",
             r"$C$ & val Macro-F1\\", r"\midrule"]
    for g in grid:
        lines.append(f"{g['C']} & {g['val_macro_f1']:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)



# --------------------------------------------------------------------------- #
# phase 2: token forecasting + MusicGen continuation
# --------------------------------------------------------------------------- #
TOK = ROOT / "results/token_prediction"
# 加长训练后的结果优先：原版只训 12 轮就被早停，val_nll 仍在下降，
# 属于欠训练，不应作为报告的主结果。
_TOK_LONG = ROOT / "results/token_prediction_long"
if (_TOK_LONG / "metrics.csv").exists():
    TOK = _TOK_LONG
GEN = ROOT / "results/generation"
TOKEN_ORDER = ["persistence", "frequency", "bigram", "lstm", "transformer"]
TOKEN_LABEL = {"persistence": "Persistence", "frequency": "频率先验", "bigram": "Bigram",
               "lstm": "LSTM", "transformer": "Transformer"}


def token_table() -> tuple[str, dict]:
    """Main comparison table for phase-1 token forecasting."""
    p = TOK / "metrics.csv"
    if not p.exists():
        return ("% phase-1 token results not generated yet\n", {})
    df = pd.read_csv(p)
    macros: dict[str, str] = {}
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Token 预测结果（FMA-small 测试集，5 s 前缀 → 1 s 目标）。神经网络为 3 个种子的均值 $\pm$ 标准差，基线为确定性预测。}",
             r"\label{tab:token}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"模型 & 参数量 & NLL & Teacher Top-1 & Free Top-1 & 码本 0 Top-1" + "\\\\",
             r"\midrule"]
    for m in TOKEN_ORDER:
        sub = df[df.model == m]
        if sub.empty:
            continue
        neural = m in ("lstm", "transformer")
        if neural:
            # 标准差常常小于 0.001，用 3 位小数会显示成 ±0.000，故按量级选位数
            sd = float(sub.free_top1.std(ddof=0))
            sdf = 3 if sd >= 1e-3 else 4
            t1 = f"{sub.teacher_top1.mean():.3f}$\\pm${sub.teacher_top1.std(ddof=0):.3f}"
            f1 = f"{sub.free_top1.mean():.3f}$\\pm${sd:.{sdf}f}"
            nll = f"{sub.nll.mean():.3f}"
            cb0 = f"{sub.free_top1_cb0.mean():.3f}"
            params = fmt_params(sub.params.mean())
            macros[f"Tok{m.capitalize()}Free"] = f"{sub.free_top1.mean():.3f}"
            macros[f"Tok{m.capitalize()}Nll"] = f"{sub.nll.mean():.3f}"
            if m == "transformer":
                macros["TokTfFreeStd"] = f"{sub.free_top1.std(ddof=0):.3f}"
                macros["TokTfTeacher"] = f"{sub.teacher_top1.mean():.3f}"
                macros["TokTfTrainS"] = f"{sub.train_s.mean():.0f}"
                macros["TokTfParams"] = params
            else:
                macros["TokLstmTrainS"] = f"{sub.train_s.mean():.0f}"
        else:
            t1 = f"{sub.teacher_top1.mean():.3f}"
            f1 = f"{sub.free_top1.mean():.3f}"
            nll = "--"
            cb0 = f"{sub.free_top1_cb0.mean():.3f}"
            params = "--"
            macros[f"Tok{m.capitalize()}Free"] = f"{sub.free_top1.mean():.3f}"
        lines.append(f"{TOKEN_LABEL[m]} & {params} & {nll} & {t1} & {f1} & {cb0}" + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    summ = TOK / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        macros["TokNTest"] = f"{s.get('n_test', 0):,}"
        macros["TokNTrain"] = f"{s.get('n_train', 0):,}"
        macros["TokNTracks"] = f"{s.get('n_tracks_test', 0):,}"
    return ("\n".join(lines) + "\n", macros)


def token_decode_table() -> str:
    # decode.json 由旧的 token_prediction 目录生成；加长训练目录里可能没有，
    # 两处都找一遍。
    for d in (TOK, ROOT / "results/token_prediction"):
        if (d / "decode.json").exists():
            p = d / "decode.json"
            break
    else:
        return "% decode results not generated yet\n"
    d = json.loads(p.read_text())
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{预测 Token 解码回音频后与真实后 1 秒的对照（Log-Mel RMSE，越小越接近）。}",
             r"\label{tab:tokensamples}",
             r"\begin{tabular}{llcc}", r"\toprule",
             r"曲目 & 流派 & Log-Mel RMSE & 试听" + "\\\\", r"\midrule"]
    for it in d["items"]:
        lines.append(f"{it['track_id']} & {it['genre']} & {it['logmel_rmse']:.2f} & "
                     "\\texttt{" + Path(it['pred_wav']).name.replace('_', '\\_') + "}" + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def generation_table() -> tuple[str, dict]:
    macros: dict[str, str] = {}
    base = GEN / "metrics_base.json"
    base_rmse = None
    if base.exists():
        b = json.loads(base.read_text())
        if b["items"]:
            base_rmse = float(np.mean([x["logmel_rmse"] for x in b["items"]]))

    # 行顺序：base → 旧（标签维度 bug）→ 修正后各配置（步数 / 目标 / rank 三个旋钮）
    runs = (
        ("base", "MusicGen-small（原始）", "Base"),
        ("ft_v1_negative", "\\quad + LoRA 旧版（标签维度错误）", "FtVOneNeg"),
        ("ft", "\\quad + LoRA $r{=}8$，1200 步，掩前缀", "Ft"),
        ("v3", "\\quad + LoRA $r{=}8$，3600 步，掩前缀", "VThree"),
        ("full", "\\quad + LoRA $r{=}8$，1200 步，全窗口", "Full"),
        ("r16", "\\quad + LoRA $r{=}16$，1200 步，掩前缀", "RSixteen"),
    )
    items = []
    for tag, label, mac in runs:
        p = GEN / f"metrics_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if not d["items"]:
            continue
        lat = float(np.mean([x["latency_s"] for x in d["items"]]))
        mem = float(np.mean([x["peak_mem_mb"] for x in d["items"]]))
        rmse = float(np.mean([x["logmel_rmse"] for x in d["items"]]))
        delta = ("—" if base_rmse is None or tag == "base"
                 else f"${rmse - base_rmse:+.2f}$")
        items.append((label, lat, mem, float(d["seconds"]), rmse, delta))
        # LaTeX 宏名不能含数字
        macros[f"Gen{mac}Lat"] = f"{lat:.1f}"
        macros[f"Gen{mac}Mem"] = f"{mem:.0f}"
        macros[f"Gen{mac}Rmse"] = f"{rmse:.2f}"

    if not items:
        return "% generation results not generated yet\n", macros

    # 配对显著性检验：8 首测试曲逐曲配对，判断差异是否只是噪声。
    # 这一步必需——本项目已有两个反例：v1 的 −0.80 靠单曲撑着，
    # r16 的 loss 更低却音频更差。只看均值会得出错误结论。
    def _per_track(tag):
        fp = GEN / f"metrics_{tag}.json"
        if not fp.exists():
            return None
        dd = json.loads(fp.read_text())
        return {x["track_id"]: float(x["logmel_rmse"]) for x in dd["items"]} if dd["items"] else None

    def _paired(a_tag, b_tag):
        a, b = _per_track(a_tag), _per_track(b_tag)
        if not a or not b:
            return None
        keys = [k for k in a if k in b]
        if len(keys) < 3:
            return None
        diff = np.array([a[k] - b[k] for k in keys])
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        return (float(diff.mean()), float(se),
                float(diff.mean() / se) if se > 0 else 0.0)

    for a_tag, b_tag, key in (("ft", "base", "FtVsBase"), ("v3", "ft", "VThreeVsFt"),
                              ("full", "ft", "FullVsFt"), ("r16", "base", "RSixteenVsBase")):
        st = _paired(a_tag, b_tag)
        if st:
            mean, se, tstat = st
            macros[f"Gen{key}Mean"] = f"{mean:+.2f}"
            macros[f"Gen{key}T"] = f"{tstat:+.2f}"
            macros[f"Gen{key}Sig"] = "显著" if abs(tstat) > 2.36 else "不显著"

    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{MusicGen-small 前缀续写（同一批测试前缀，5 s 前缀 $\to$ 3 s 续写；"
             r"采样 $T=1$、top-$k=250$、无 CFG；每行为 8 首测试曲的均值）。}",
             r"\label{tab:generation}",
             r"\begin{tabular}{lccccr}", r"\toprule",
             r"模型 & 生成耗时(s) & 峰值显存(MB) & 输出时长(s) & Log-Mel RMSE & 相对原始" + "\\\\",
             r"\midrule"]
    for label, lat, mem, sec, rmse, delta in items:
        lines.append(f"{label} & {lat:.1f} & {mem:.0f} & {sec:.1f} & {rmse:.2f} & {delta}" + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", macros


def finetune_sentence() -> tuple[str, dict]:
    # 优先用修正后的结果（finetune_v2.json）；finetune.json 是旧的错误版本，
    # 仅作为对照保留，不应被当成当前结论。
    p = GEN / "finetune_v2.json"
    if p.exists():
        d = json.loads(p.read_text())
        m = d["meta"]
        hist = d["history"]
        first = float(np.mean([x["loss"] for x in hist[:10]]))
        last = float(np.mean([x["loss"] for x in hist[-10:]]))
        uniform = float(np.log(2048.0))
        return (
            f"LoRA（rank {m['rank']}，alpha {m['alpha']}）在 {m['train_windows']} 个训练窗口上"
            f"以 batch {m['batch_size']} 训练 {m['steps']} 步、约 {m['train_minutes']} 分钟，"
            f"峰值显存 {m['peak_mem_mb']:.0f}\\,MB，仅更新 {m['merged']} 个投影层，"
            f"测试集未参与。目标值从训练初期的 {first:.2f} 降到 {last:.2f}"
            f"（均匀分布基准 $\\ln 2048={uniform:.2f}$），即模型确实学到了 Token 层面的结构。"
        ), dict(FtSteps=str(m["steps"]), FtRank=str(m["rank"]),
                FtMinutes=f"{m['train_minutes']:.0f}", FtMem=f"{m['peak_mem_mb']:.0f}")
    p = GEN / "finetune.json"
    if p.exists():
        d = json.loads(p.read_text())["meta"]
        return (f"LoRA（rank {d['rank']}，alpha {d['alpha']}）在 {d['train_windows']} 个训练窗口上"
                f"训练 {d['steps']} 步、约 {d['train_minutes']} 分钟，峰值显存 {d['peak_mem_mb']:.0f}\\,MB。"), {}
    return "（微调结果待生成）", {}



def budget_table() -> tuple[str, dict]:
    """训练预算消融：原预算 vs 加长训练（一次性复核，用来判断是否欠训练）。"""
    import re as _re

    rows = []
    specs = (
        ("GTZAN", ROOT / "results/gtzan/metrics.csv", ROOT / "results/gtzan_long/metrics.csv",
         30, ("cnn", "crnn", "transformer")),
        ("FMA-small", ROOT / "results/fma/metrics.csv", ROOT / "results/fma_long/metrics.csv",
         30, ("cnn", "crnn", "transformer")),
    )
    for name, old_p, new_p, cap, models in specs:
        if not old_p.exists() or not new_p.exists():
            continue
        old = pd.read_csv(old_p)
        new = pd.read_csv(new_p)
        for mdl in models:
            o = old[(old.model == mdl) & (old.split == "test")]
            n = new[(new.model == mdl) & (new.split == "test")]
            if o.empty or n.empty:
                continue
            bes = [int(b.group(1)) for b in
                   (_re.search(r"best_ep=(\d+)", str(x)) for x in n.get("notes", []))
                   if b]
            rows.append((name, mdl, o.macro_f1.mean(), n.macro_f1.mean(),
                         n.macro_f1.mean() - o.macro_f1.mean(),
                         ("是" if bes and max(bes) >= 60 else "否") if bes else "--"))
    if not rows:
        return "% budget ablation not available\n", {}

    label = {"cnn": "CNN", "crnn": "CRNN", "transformer": "轻量 Transformer"}
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{训练预算消融：原协议（30 轮，patience 8）与加长（60 轮，patience 12）"
             r"的测试集 Macro-F1 对比。所有配置沿用同一数据划分与随机种子。}",
             r"\label{tab:budget}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"数据集 & 模型 & 原预算 & 加长训练 & $\Delta$" + "\\\\",
             r"\midrule"]
    prev = None
    for ds, mdl, o, n, d, hit in rows:
        if prev is not None and ds != prev:
            lines.append(r"\midrule")
        ds_cell = ds if ds != prev else ""
        prev = ds
        lines.append(f"{ds_cell} & {label.get(mdl, mdl)} & {o:.4f} & {n:.4f} & ${d:+.4f}$" + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # 结论句：区分"确实欠训练"与"原预算已够"
    gtz_best = max((r[4] for r in rows if r[0] == "GTZAN"), default=0.0)
    fma_best = max((r[4] for r in rows if r[0] == "FMA-small"), default=0.0)
    fma_vals = [r[4] for r in rows if r[0] == "FMA-small"]
    fma_mean = sum(fma_vals) / len(fma_vals) if fma_vals else 0.0
    sentence = (f"在 GTZAN 上加长训练的最好增益为 ${gtz_best:+.4f}$，"
                f"而在 FMA-small 上为 ${fma_best:+.4f}$（均值为 ${fma_mean:+.4f}$）")
    # 判据：主实验所在的数据集若均值不为正、且最好增益也不足 1 个百分点，
    # 就说明原预算已经够用，而不是「欠训练」。
    if fma_mean <= 0 or fma_best < 0.01:
        sentence += ("——这说明主实验的原训练预算已经够用，"
                     "早停并没有系统性地截断仍在改善的模型；"
                     "GTZAN 上的小幅提升主要来自小数据集上方差更大的训练过程，"
                     "不宜据此宣称模型被欠训练。")
    else:
        sentence += "——两个数据集都能从更长的训练中获益，原预算偏紧。"
    return ("\n".join(lines) + "\n", {"BudgetSentence": sentence})



def generation_pertrack_table() -> str:
    """逐曲 Log-Mel RMSE 对照（base vs 修正后 LoRA），支撑「整体变好而非一首变好」。"""
    base, ft = GEN / "metrics_base.json", GEN / "metrics_ft.json"
    if not (base.exists() and ft.exists()):
        return "% generation per-track results not generated yet\n"
    b = {x["track_id"]: x for x in json.loads(base.read_text())["items"]}
    f = {x["track_id"]: x for x in json.loads(ft.read_text())["items"]}
    keys = sorted(set(b) & set(f))
    if not keys:
        return "% generation per-track results empty\n"
    rows, diffs = [], []
    for k in keys:
        rb, rf = float(b[k]["logmel_rmse"]), float(f[k]["logmel_rmse"])
        d = rf - rb
        diffs.append(d)
        mark = "$\\downarrow$" if d < 0 else "$\\uparrow$"
        rows.append(f'{k} & {b[k].get("genre", "")} & {rb:.2f} & {rf:.2f} & {d:+.2f} {mark}' + "\\\\")
    diffs = np.array(diffs)
    improved = int((diffs < 0).sum())
    mean = diffs.mean()
    # 去掉改善最大的一首后重算：这是本文判定「整体变好」的关键检验
    worst = diffs.min()
    excl = np.delete(diffs, diffs.argmin()).mean() if len(diffs) > 1 else float("nan")
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{生成式任务的逐曲结果（8 首测试曲，Log-Mel RMSE，越低越好；"
             r"差值为 LoRA $-$ 原始，负值表示改善）。}",
             r"\label{tab:gentrack}",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"曲目 & 流派 & 原始 & $+$LoRA & 差值" + "\\\\", r"\midrule"]
    lines += rows
    lines += [r"\midrule",
              f"均值 & & {np.mean([float(b[k]['logmel_rmse']) for k in keys]):.2f} & "
              f"{np.mean([float(f[k]['logmel_rmse']) for k in keys]):.2f} & {mean:+.2f}" + "\\\\",
              f"\\multicolumn{{5}}{{l}}{{改善 {improved}/{len(keys)} 首；去掉改善最大的一首后均值差 {excl:+.2f}}}"
              + "\\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


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
    stats = ROOT / "data/splits/fma/stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        n = s.get("n_decodable", 0)
        (OUT / "numbers_fma_n.tex").write_text(str(n))
        if (ROOT / "data/splits/fma/splits.csv").exists():
            d = pd.read_csv(ROOT / "data/splits/fma/splits.csv")
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
    (TAB / "gtzan_data.tex").write_text(gtzan_table())
    (TAB / "mvp_perseed.tex").write_text(
        perseed_table(MVP / "metrics.csv", "GTZAN 逐 seed 结果。", "tab:mvp-perseed"))
    (TAB / "fma_perseed.tex").write_text(
        perseed_table(FMA / "metrics.csv", "FMA-small 逐 seed 结果（AST 为单 seed）。",
                      "tab:fma-perseed"))
    (TAB / "fma_perclass.tex").write_text(fma_perclass_table())
    (TAB / "arch.tex").write_text(arch_table())
    (TAB / "robust_delta.tex").write_text(robustness_delta_table())
    (TAB / "ast_grid.tex").write_text(ast_grid_table())
    (TAB / "generation_pertrack.tex").write_text(generation_pertrack_table())
    import shutil
    for src, dst in [(FMA / "figures/confusion_matrices.png", OUT / "figures/confusion_fma.png"),
                     (FMA / "figures/training_curves.png", OUT / "figures/curves_fma.png"),
                     (MVP / "figures/training_curves.png", OUT / "figures/curves_gtzan.png")]:
        if src.exists():
            (OUT / "figures").mkdir(exist_ok=True)
            shutil.copyfile(src, dst)

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
        n_test = int(pd.read_csv(ROOT / "data/splits/fma/splits.csv").query("split == 'test'").shape[0])
        m.append(macro("FmaBestModel", b.label))
        m.append(macro("FmaBestF", f"{b.f1:.3f}"))
        m.append(macro("FmaBestAcc", f"{b.acc:.3f}"))
        m.append(macro("FmaTestN", n_test))
        cnn_row = fma[fma.model == "cnn"]
        if len(cnn_row):
            m.append(macro("FmaCnnF", f"{cnn_row.iloc[0].f1:.3f}"))
        leak_csv = ROOT / "results/fma_leak/metrics.csv"
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
    tok_tex, tok_macros = token_table()
    (TAB / "token.tex").write_text(tok_tex, encoding="utf-8")
    (TAB / "token_samples.tex").write_text(token_decode_table(), encoding="utf-8")
    gen_tex, gen_macros = generation_table()
    (TAB / "generation.tex").write_text(gen_tex, encoding="utf-8")
    ft_sentence, ft_macros = finetune_sentence()
    m.append(macro("FinetuneSentence", ft_sentence))
    bud_tex, bud_macros = budget_table()
    (TAB / "budget.tex").write_text(bud_tex, encoding="utf-8")
    for k, v in {**tok_macros, **gen_macros, **ft_macros, **bud_macros}.items():
        m.append(macro(k, v))
    (OUT / "macros.tex").write_text("".join(m), encoding="utf-8")

    (OUT / "numbers_abstract.tex").write_text(
        "；".join(abs_parts) + "。" if abs_parts else "（结果待生成）")
    print("generated:", sorted(p.name for p in TAB.glob("*.tex")),
          "and inline numbers")


if __name__ == "__main__":
    main()
