# 验收清单（对照 `README.md`「验收清单」的 12 项 Definition of Done）

> ✅ = 已达成且有产物；◐ = 部分达成；⬜ = 未做。
> 每个 ✅ 都能在仓库里找到对应文件，PPT 数字可从 `results/` 追溯。

## 实验基础

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 1 | 环境审计报告与可复现依赖 | ✅ | `docs/environment_report.md`、`requirements.txt`；含 GPU 沙箱排查全过程 |
| 2 | 数据集说明、授权检查与固定划分 | ✅ | `docs/data_card.md`、`data/splits/splits.csv`、`stats.json` |
| 3 | SVM、MLP、CNN、CRNN、Transformer + AST 均可运行 | ✅ | `checkpoints/*`（15 个权重）+ AST 主干；`results/metrics.csv` 16 行 |
| 4 | 训练日志、模型权重与配置文件完整 | ◐ | 权重 + 逐 epoch history（`*_eval.json`）+ `configs/`；缺独立 stdout 日志文件 |

## 实验结果

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 5 | Accuracy、Macro-F1 与混淆矩阵 | ✅ | `results/metrics.csv`、`figures/confusion_matrices.png`、逐类 F1 |
| 6 | 至少三个随机种子的核心结果 | ✅ | 5 个自建模型 × 3 seed（42/123/2024）；AST 为单 seed 的迁移组，已在文档注明 |
| 7 | 训练时间、推理延迟、模型大小和显存对比 | ◐ | 前三项全有；AST 记录了峰值 1139.7 MB，自建模型未单独记录 |
| 8 | 至少两个错误案例及一次鲁棒性实验 | ✅ | `docs/error_analysis.md`（5 个案例）+ `docs/robustness.md`（5 模型 × 7 扰动） |

## 展示材料

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 9 | 可上传音频的真实推理 Demo | ✅ | `app/demo.py` + `src/predict.py`，HTTP 200 冒烟通过 |
| 10 | 模型排行榜与可视化图表 | ✅ | `figures/leaderboard.png`、`training_curves.png`、`summary.md` |
| 11 | PPT 中所有数字均能追溯到结果文件 | ◐ | 数字均来自 `metrics.csv` / `*_eval.json`；PPT 尚未制作 |
| 12 | 备用录屏、演示音频与彩排完成 | ⬜ | 未做 |

## MVP 判定

**“最小 MVP” 已达成并扩展**：数据 → 特征 → 5 个自建模型 → 3 seed 复测 →
P0 错误分析 / 鲁棒性 → P1 AST 迁移 → P2 时间轴 → 真实推理 Demo 全链路可运行、
可复现。协议约束由 `tests/test_protocol.py` 自动校验，并由独立校验员只读复核
（`docs/audit_report.md`，6/6 PASS）。

距离完整课程项目还差：FMA-small 主实验、PPT 与现场彩排。
