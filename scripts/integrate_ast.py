"""把 AST 迁移学习结果并入统一结果体系。

读取 results/gtzan/ast_metrics.json / results/gtzan/ast_eval.json（由 src/train_ast.py 产生），
1) 生成与其它模型同 schema 的 results/gtzan/ast_seed42_eval.json；
2) 把 AST 行追加进 results/gtzan/metrics.csv（按 model/seed/split 去重）。

之后运行 `python -m src.evaluate` 即可让 AST 出现在排行榜与汇总里。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("MUSIC_RESULTS", ROOT / "results" / "gtzan"))
COLUMNS = ["model", "seed", "split", "n_samples", "accuracy", "macro_f1",
           "train_time_s", "infer_ms_per_clip", "n_params", "model_size_mb", "notes"]


def main() -> None:
    metrics = json.loads((RES / "ast_metrics.json").read_text())
    ev = json.loads((RES / "ast_eval.json").read_text())

    normalized = {
        "history": [],
        "val": {"acc": ev["val"]["accuracy"], "macro_f1": ev["val"]["macro_f1"],
                "n_samples": ev["val"].get("n_samples")},
        "test": {
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "confusion_matrix": ev["confusion_matrix"],
            "per_class_f1": ev["per_class_f1"],
        },
        "test_report": ev["classification_report"],
        "classes": ev["class_names"],
        "source": "results/gtzan/ast_eval.json",
        "note": "frozen AST backbone + logistic head (transfer-learning group)",
    }
    (RES / "ast_seed42_eval.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    row = {k: metrics.get(k) for k in COLUMNS}
    row["model"] = "ast"
    row["seed"] = 42
    row["split"] = "test"
    row["n_samples"] = int(ev["n_samples"])
    row["notes"] = metrics.get("notes", "")
    df = pd.DataFrame([row], columns=COLUMNS)
    if (RES / "metrics.csv").exists():
        old = pd.read_csv(RES / "metrics.csv")
        old = old[~((old["model"] == "ast") & (old["seed"] == 42) & (old["split"] == "test"))]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["model", "seed"], inplace=True)
    df.to_csv(RES / "metrics.csv", index=False)
    print(f"[ast] merged: acc={row['accuracy']} f1={row['macro_f1']} "
          f"rows={len(df)} -> {RES/'metrics.csv'}")
    print("[ast] normalized eval ->", RES / "ast_seed42_eval.json")


if __name__ == "__main__":
    main()
