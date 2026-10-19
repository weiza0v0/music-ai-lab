# 验收清单（对照 `README.md`「验收清单」的 12 项 Definition of Done）

> ✅ = 已达成且有产物；◐ = 部分达成；⬜ = 未做。
> 每个 ✅ 都能在仓库里找到对应文件，PPT 数字可从 `results/` 追溯。

## 实验基础

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 1 | 环境审计报告与可复现依赖 | ✅ | `docs/environment_report.md`、`requirements.txt`；含 GPU 沙箱排查全过程 |
| 2 | 数据集说明、授权检查与固定划分 | ✅ | `docs/data_card.md`、`data/splits/splits.csv`、`stats.json` |
| 3 | SVM、MLP、CNN、CRNN、Transformer 均可运行 | ✅ | `checkpoints/*`（15 个权重）、`results/metrics.csv` 15 行 |
| 4 | 训练日志、模型权重与配置文件完整 | ◐ | 权重 + 逐 epoch history（`*_eval.json`）+ `configs/`；缺独立 stdout 日志文件 |

## 实验结果

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 5 | Accuracy、Macro-F1 与混淆矩阵 | ✅ | `results/metrics.csv`、`figures/confusion_matrices.png`、逐类 F1 |
| 6 | 至少三个随机种子的核心结果 | ✅ | seed 42/123/2024 全部跑完（5 模型 × 3 seed）；均值±std 见 `results/summary.md` |
| 7 | 训练时间、推理延迟、模型大小和显存对比 | ◐ | 前三项有；峰值显存未记录（8 GB 卡上 batch=32 远未打满） |
| 8 | 至少两个错误案例及一次鲁棒性实验 | ◐ | 混淆矩阵已定位 rock→country/blues；鲁棒性扰动未做 |

## 展示材料

| # | 条目 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 9 | 可上传音频的真实推理 Demo | ✅ | `app/demo.py` + `src/predict.py`，HTTP 200 冒烟通过 |
| 10 | 模型排行榜与可视化图表 | ✅ | `figures/leaderboard.png`、`training_curves.png`、`summary.md` |
| 11 | PPT 中所有数字均能追溯到结果文件 | ◐ | 数字均来自 `metrics.csv` / `*_eval.json`；PPT 尚未制作 |
| 12 | 备用录屏、演示音频与彩排完成 | ⬜ | 未做 |

## MVP 判定

**“最小 MVP” 已达成**：数据 → 特征 → 5 个模型 → 3 seed 复测 → 评估 → 图表 →
真实推理 Demo 全链路可运行、可复现。协议约束（按曲目划分、去重、训练集拟合
标准化、测试集只用一次）由 `tests/test_protocol.py` 自动校验，并由独立校验员
只读复核（`docs/audit_report.md`，6/6 PASS）。

距离完整课程项目还差：鲁棒性实验、AST、FMA-small、PPT 与彩排。
