# 独立审计报告（第三方只读复核）

审计对象：本仓库的数据划分、特征缓存、训练代码、checkpoint 与结果文件。
审计方式：不信任已有结论，重新读取原始文件并复算；训练复现放在 `/tmp` 隔离目录，
不修改仓库。共进行两轮：第一轮（结果 15 行初版）与第二轮（修复 + 确定性重跑后的
最终状态）。两轮结论都是 **6/6 PASS**。

## 最终结论

**没有发现数据泄漏、测试集污染或伪造结果的证据。**
第一轮发现的 6 个报告层面的小问题中，4 个已修复、2 个属于固有行为并已写入文档。

| # | 审计项 | 结果 | 最终证据 |
|---|---|---|---|
| 1 | 划分完整性 | PASS | 985 首 / 985 唯一 track / 985 唯一 md5；689 + 149 + 147 = 985；跨集合 md5 与 track 交集均为 0；10 流派 × 3 集合齐全；seed=42 可逐行复现划分；1000 = 985 + 14 重复 + 1 损坏 |
| 2a | 传统标准化器只用训练集 | PASS | 6/6 `.joblib`：`mean_/scale_` 与仅用 train 拟合的 Scaler 最大差 **0.0**，且不等于全量拟合 |
| 2b | 深度归一化只用训练集 | PASS | 9/9 `.pt`：mean **−38.72503662**、std **15.42299271**，与 train-mel 复算完全一致；全量拟合会是 −38.7882 / 15.3898 |
| 2c | 早停不看测试集 | PASS | `train_cnn.py`：归一化统计取 `X[train]`；每轮指标来自 val；best 状态按 val F1 保存；test loader 在早停循环结束后才首次迭代 |
| 3 | 结果一致性 | PASS | 15/15 预测 CSV 复算 acc/F1 与 `metrics.csv` 最大差 **4e-5**、与 `*_eval.json` 差 **0**；所有行 `split=test`、`n_samples=147`；`y_true` 与划分文件完全一致 |
| 4 | 可复现性 | PASS | `pytest -q tests` → **4 passed**；CLI 推理 jazz 样本 top-1 = jazz（96.2%）；seed 42 重训得到**逐字节相同**的 checkpoint（cnn/crnn/transformer 三个均验证） |
| 5 | Demo 真实性 | PASS | `app/demo.py` / `src/predict.py` 无硬编码、无随机兜底；5 个模型默认解析到 `*_seed42`；同音频不同 seed 输出不同；白噪声 → classical 0.759 |
| 6 | 产物自洽 | PASS | 9 个深度 `.pt`（1.22–2.39 MB）；参数量 592,554 / 418,986 / 302,474 与 `metrics.csv` 一致；全部 `strict=True` 加载并输出 (1,10)；15 个 checkpoint 类别顺序一致 |

## 最终结果（test，n=147，5 模型 × 3 seed）

| 模型 | Accuracy | Macro-F1 |
|---|---|---|
| CNN | 0.755 ± 0.007 | 0.753 ± 0.006 |
| CRNN | 0.726 ± 0.034 | 0.723 ± 0.034 |
| SVM | 0.708 ± 0.000 | 0.699 ± 0.000 |
| MLP | 0.633 ± 0.051 | 0.630 ± 0.055 |
| Transformer | 0.544 ± 0.025 | 0.541 ± 0.019 |

## 第一轮问题的处理状态

| # | 问题 | 状态 | 说明 |
|---|---|---|---|
| 1 | `best_ep` 记的是训练轮数 | 已修复 | 9/9 的 `best_ep` == JSON `best_epoch` == val-F1 最优轮 |
| 2 | eval JSON 的 `val` 用最后一轮 | 已修复 | 改为最佳轮指标（如 cnn_seed42 val F1 0.7276，末轮 0.7160） |
| 3 | CLI 默认 seed 123 | 已修复 | 5 个模型均默认 seed 42，也支持显式 seed |
| 4 | 示例曲目属于训练集 | 已修复 | Demo 与 README 改用测试集曲目（blues.00008 / classical.00001） |
| 5 | SVM 概率与硬预测不一致 | 保留（固有） | OVO 决策 vs Platt 概率，8–10/147 行不同；准确率统一用 `predict()`，README 已注明 |
| 6 | 文档未更新 seed 数 | 已修复 | README、验收清单、实验日志已同步 3-seed 结果 |

## 确定性说明

代码已设 `cudnn.benchmark=False`、`cudnn.deterministic=True`、
`use_deterministic_algorithms(True, warn_only=True)`，并对 DataLoader 使用带种子的
generator。校验员在 `/tmp` 隔离副本重训 cnn/crnn/transformer 的 seed 42，得到与仓库
**sha256 完全相同**的权重文件；同 seed 的 30 轮结果也逐位一致。

局限：PyTorch 会警告 `adaptive_avg_pool2d_backward_cuda` 无确定性实现（`warn_only`
下继续训练）。因此可复现性在本机/本 torch 版本下是**实测成立**，不是跨硬件硬保证；
如需硬保证，设 `warn_only=False` 并配置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 其他

- 仓库未初始化 `.git`，无法审计代码来源；但所有产物内部自洽、可复算。
- 已知真实错误案例：测试集 `jazz.00003` 被 CNN 以 99.6% 置信度判成 `classical`，
  适合作为「AI 自信犯错」的演示素材。
- 审计过程未修改仓库任何源文件/数据/产物。
