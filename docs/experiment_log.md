# 实验日志（Music AI Lab MVP）

> 对应 `README.md`「分阶段计划与进度」中的阶段 01–06（A–H）。每条记录都来自真实执行，未运行的项目
> 明确标为未做。

## 阶段 A：环境审计

| 项 | 结果 |
|---|---|
| OS | WSL2 Ubuntu 22.04（内核 6.18.54.1-microsoft-standard-WSL2） |
| Python | 3.12.13（conda env `DataAnalysis`） |
| torch | 2.12.1+cu130，CUDA 可用 True |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB，算力 (8,9) |
| CPU / RAM | 16 逻辑核 / 7.8 GiB |
| 磁盘 | D: 1.4 TB 可用 |
| 音频栈 | ffmpeg 4.4.2，soundfile 0.14.0，librosa 1.0.0 |

**GPU 问题定位过程**（本次最有价值的排查）：

1. `nvidia-smi` → `Failed to initialize NVML: GPU access blocked by the operating system`
2. `ctypes.CDLL('libcuda.so.1').cuInit(0)` → `100`（CUDA_ERROR_NO_DEVICE），设备数 0
3. Windows 侧 `/mnt/c/Windows/System32/nvidia-smi.exe` 正常（驱动 595.79 / CUDA 13.2）
   → 排除 Windows 驱动问题
4. `open('/dev/dxg', O_RDWR)` → `EACCES`（节点权限是 666，说明不是文件权限）
5. DSH 文件策略从 `workspace-write` 切到 `danger-full-access` 后，同一命令：
   `open ok`、`cuInit=0`、`device_count=1`、`nvidia-smi` 正常

结论：GPU 本身健康，是**进程沙箱拦了 `/dev/dxg` 设备访问**；CUDA/NVML 把它翻译成
「操作系统阻止了 GPU 访问」。此后全部训练走 GPU。报告见
`docs/environment_report.md`。

## 阶段 B：数据准备

- 来源：HuggingFace `marsyas/gtzan`，`data/genres.tar.gz`（1.17 GiB，下载约 4 分钟）。
- 解压后发现 1000 个真实 wav + 1000 个 macOS AppleDouble `._*.wav` 垃圾文件，
  已在审计中过滤并在清理时删除。
- 审计（`src/data/prepare_dataset.py --audit`）：

| 指标 | 值 |
|---|---|
| 曲目数 | 1000 |
| 每类 | 100 |
| 采样率 | 22050 Hz（999 首合法，1 首损坏） |
| 声道 | 单声道 |
| 时长 | 29.93–30.65 s，均值 30.02 s |
| 解码失败 | `jazz.00054`（GTZAN 经典损坏文件） |
| 重复曲目 | 28 首（MD5 相同） |

- 清洗策略：剔除损坏文件；按 MD5 只保留重复组的第一首。
- 划分：按曲目分层 70/15/15，seed=42 → **985 首**（train 689 / val 149 / test 147），
  每类 65–70 训练样本。
- 产物：`data/splits/gtzan/splits.csv`（含 md5、样本率、时长）、`stats.json`。

## 阶段 C：特征与传统 Baseline

- `scripts/extract_features.py`：985 首 × 8 进程，49.5 s，产出
  `X_feat (985, 209)` 与 `X_mel (985, 64, 216)`（float16 存储）。
- 人工特征：40 MFCC 均值/标准差、40 ΔMFCC、12 chroma、7 谱对比度、谱质心/带宽/
  rolloff、ZCR、RMS、tempo —— SVM 与 MLP 完全共用。
- StandardScaler 只在 train 拟合（测试 `test_scaler_fitted_on_train_only` 校验）。

| 模型 | test Acc | Macro-F1 | 训练 |
|---|---|---|---|
| SVM (RBF, C=10) | 0.708 | 0.699 | 0.2 s |
| MLP (256-128) | 0.639 | 0.633 | 0.6 s |

注：sklearn 1.9 对 `SVC(probability=True)` 给出 deprecation warning，功能正常；
后续可换 `CalibratedClassifierCV` 消除。

## 阶段 D：深度 Baseline

- 统一训练器 `src/train_cnn.py`：AdamW + CosineAnnealing、按 val Macro-F1 早停、
  保存最佳权重、测试集只评一次。
- 输入统一为 64×216 Log-Mel，归一化统计量取训练集。
- **踩坑**：第一版忘记 `.to(device)`，模型实际在 CPU 上训练（30 epoch 296 s）。
  修正后 1.8 s/epoch，总 56 s，准确率还从 71.4% 提到 73.5%。

单 seed=42 结果：

| 模型 | 结构要点 | test Acc | Macro-F1 |
|---|---|---|---|
| CNN | 4×ConvBlock(32/64/128/128)+GAP | 0.735 | 0.731 |
| CRNN | CNN + BiGRU(96) | 0.721 | 0.715 |
| Transformer | 2 层、d=128、4 头，帧级自注意力 | 0.565 | 0.557 |

随后按计划补跑 seed 123 / 2024，最终 15 次训练（5 模型 × 3 seed）：

| 模型 | Accuracy（均值±std） | Macro-F1（均值±std） |
|---|---|---|
| CNN | 0.755 ± 0.007 | 0.753 ± 0.006 |
| CRNN | 0.726 ± 0.034 | 0.723 ± 0.034 |
| SVM | 0.708 ± 0.000 | 0.699 ± 0.000 |
| MLP | 0.633 ± 0.051 | 0.630 ± 0.055 |
| Transformer | 0.544 ± 0.025 | 0.541 ± 0.019 |

结论：CNN 领先 SVM 约 4.7 个点且三个 seed 极稳（±0.7）；CRNN 次之但波动较大
（±3.4）；MLP 对初始化最敏感（±5.1）；Transformer 三个 seed 都只有 52–56%。

**确定性插曲**：第一轮 3-seed 跑完后再重跑，CNN seed 123 从 76.2% 变成 73.5%，
原因是 cuDNN 自动调优 + 非确定 kernel。开启 `cudnn.deterministic=True`、
`benchmark=False`、`use_deterministic_algorithms(True)` 后重跑，并用 seed 42
二次复跑验证逐位一致（0.7483 / 0.7493），独立校验员在 `/tmp` 隔离目录重训也得到
**逐字节相同**的 checkpoint。此后所有结果以此为准。
注意：`torch.use_deterministic_algorithms(True, warn_only=True)` 下
`adaptive_avg_pool2d_backward_cuda` 会警告「无确定性实现」；实测当前环境可复现，
但属于经验性保证而非跨硬件硬保证（如需硬保证可设 `warn_only=False` 并配置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`）。

Transformer 训练 loss 迅速到 0、验证 loss 反升 → 985 首小数据上的过拟合典型案例；
早停在第 22 epoch。

## 阶段 E：评估与图表

- `results/gtzan/metrics.csv`：15 行（5 模型 × 3 seed；acc / macro-F1 / 训练时间 / 推理延迟 /
  参数量 / 大小）。`evaluate.py` 自动按模型聚合均值。
- `results/gtzan/*_seed42_eval.json`：混淆矩阵、逐类 F1、classification report、训练曲线。
- `results/gtzan/predictions/*.csv`：val/test 逐样本预测与 10 类概率。
- 图：`leaderboard.png`、`confusion_matrices.png`、`training_curves.png`（中文字体用
  Windows 微软雅黑，已避免方框乱码）。
- `results/gtzan/summary.md`：汇总表 + 最易混淆流派。

## 阶段 G：Demo

- `app/demo.py`：上传音频 → 波形/Log-Mel → 各模型 Top-3 + 全类别概率 → 多模型对比
  （意见不一致时高亮）。所有预测调用 `src/predict.py` 加载真实 checkpoint。
- 冒烟验证：`streamlit run --server.headless true`，HTTP 200；CLI 对 jazz 样本三模型
  均给出 96%+ 的 jazz（真实权重，无硬编码）。

## 阶段 H：交付

- README（快速开始 / 协议 / 结果 / 排障）、数据卡、实验日志、结果汇总、测试。

## 独立审计（第三方校验员）

只读复核结论：**6/6 PASS**，无数据泄漏、无伪造结果。详见 `docs/audit_report.md`。
审计同时发现 6 个报告层面的小问题（best_ep 记错、eval 的 val 用最后一轮、
CLI 默认 seed 123、示例用了训练集曲目、SVM 概率与硬预测不一致、文档未更新），
已逐条修复并重跑深度模型。

## 尚未完成（下一轮候选）

1. 鲁棒性：片段长度 3 s / 10 s、加噪、音量缩放。
2. 错误分析：rock↔country、disco↔pop 混淆案例听感对照；jazz.00003 的
   「99% 自信误判为 classical」可作为典型案例。
3. AST 迁移学习（`MIT/ast-finetuned-audioset-10-10-0.4593`）。
4. FMA-small 主实验（若磁盘与时间允许）。
5. PPT 故事线与现场盲测素材。

## 阶段后扩展（P0 / P1 / P2）

MVP 完成后按 `README.md`「下一步计划」继续做了三组实验：

### P0 错误分析

- `python -m src.error_analysis`；读取 seed 42 的预测文件复算，不重新训练。
- CNN 最大混淆对：`reggae→pop`(3)、`blues→reggae`(2)、`hiphop→pop`(2)；
  `rock` F1 0.429 最低，`classical` 0.933 最高。
- 置信度：Transformer 有 16/64 个错误样本置信度 ≥ 0.9；CNN 3/37。
- 产物：`docs/error_analysis.md`、`results/gtzan/error_analysis.csv/.json`、
  `figures/error_confusions.png`、`figures/error_cases.png`。

### P0 鲁棒性

- `python -m src.robustness --workers 12`；147 首测试曲目 × 7 条件 × 5 模型。
- clean 行与 `metrics.csv` 零误差；连跑两次 CSV 逐字节相同。
- 噪声破坏最大（10 dB 平均约 −42 pp，0 dB 约 −50 pp）；深度模型对增益完全不敏感；
  3 s 片段最伤 CRNN（−53.1 pp）；除增益外每个扰动下第一名都是 SVM。
- 产物：`docs/robustness.md`、`results/gtzan/robustness.csv`、`figures/robustness.png`。

### P1 AST 迁移

- `python -m src.train_ast`；冻结 86M 主干，只训练 7,690 参数线性头。
- test n=147：Acc 0.8231 / Macro-F1 0.8135；比 CNN 高 6.8 / 6.0 个百分点。
- 成本：346 MB、embedding 提取 214 s、推理 204 ms/片段。
- 产物：`docs/ast.md`、`results/gtzan/ast_metrics.json`、`results/gtzan/ast_eval.json`。

### P2 分段预测时间轴（功能 C）

- `python -m src.timeline <audio> --model cnn`；5 秒窗、2.5 秒步长。
- 147 首测试曲目中 88 首至少出现一次窗口级判断变化；`rock.00072` 改判 9 次。
- 已集成进 Streamlit Demo 的「🎞 进阶」区。
- 产物：`docs/timeline.md`、`results/gtzan/timeline_scan.csv`、`figures/timeline_*.png`。

## 主实验（FMA-small）

### 数据与划分

- 8,000 首、8 流派（每类 1,000），审计后 7,994 首可解码；采样率 44.1k/48k/22.05k。
- 曲目级划分下 971/1,199（81.0%）测试曲目与训练集共享艺术家 → 主实验改为
  **艺术家分离**划分：5,596/1,199/1,199，测试集与训练集艺术家重叠 0。
- 曲目级划分保留（`splits_tracklevel.csv`）作为泄漏对照。
- 数据卡：`docs/fma_data.md`；特征：7,994 ×（209 维人工特征 + 64×216 Log-Mel）。

### 基线结果（3 seeds，test n=1,199）

| 模型 | Accuracy | Macro-F1 | 训练(s)/seed |
|---|---|---|---|
| CRNN | 0.502 ± 0.009 | 0.492 ± 0.006 | 371 |
| CNN | 0.480 ± 0.005 | 0.470 ± 0.002 | 334 |
| SVM | 0.446 ± 0.000 | 0.440 ± 0.000 | 16 |
| MLP | 0.428 ± 0.011 | 0.422 ± 0.012 | 4 |
| Transformer | 0.387 ± 0.010 | 0.378 ± 0.019 | 80 |

### 泄漏对照

- 同一批特征、仅换成曲目级划分，CNN seed 42：Acc 0.5663 / Macro-F1 0.5682。
- 对比艺术家分离（0.480 / 0.470），Macro-F1 虚高 **+9.9 pp**。

### 优化消融（仅验证集，seed 42，20 epoch）

| 配置 | val Acc | val Macro-F1 |
|---|---|---|
| CNN baseline | 0.542 | 0.522 |
| CNN + SpecAug | 0.540 | 0.520 |
| CNN + SpecAug + LS | 0.554 | 0.529 |
| CRNN + SpecAug + LS | 0.560 | 0.524 |
| Transformer + SpecAug + LS | 0.422 | 0.394 |

- 最大提升 +0.7 pp，处于单次运行波动范围内；主实验保留基线配置。
- 结论：瓶颈是标签边界与音乐多样性，不是单纯过拟合。

### AST 迁移

- 冻结 AudioSet AST 主干（86M 参数）+ 线性头，16 kHz、中心 5 秒补零到 10.24 秒。
- test n=1,199：Accuracy **0.5963** / Macro-F1 **0.5903**，比最好的自建模型 CRNN
  （0.492）高 9.8 个百分点；权重 345 MB，推理约 231 ms/片段。
- 实现中改为多进程特征化 + GPU 批量前向（fp16），把 8,000 首的提取时间从预估
  2 小时降到约 7 分钟；结果见 `docs/ast_fma.md`、`results/fma/ast_metrics.json`。

---

## 二期扩展（Token 预测与音乐生成）

### 阶段 0：审计

| 项目 | 结论 |
|---|---|
| 环境 | torch 2.12.1+cu130 / RTX 4060 8 GB / 内存 7.9 GB / 磁盘余 1.4 TB |
| 依赖 | `transformers` 5.16.1 已含 `EncodecModel` 与 `MusicgenForConditionalGeneration`，无需 `audiocraft` |
| 数据 | FMA-small artist-disjoint 划分可直接复用（5,596/1,199/1,199） |
| 产物 | `docs/phase0_audit_phase2.md` |

### 阶段一：EnCodec Token 预测

- Tokenizer：`facebook/encodec_32khz`，32 kHz、4 码本、2048 词表、50 Hz；6 s 窗口 = 4×300 Token。
- 数据：每首 2 个 6 s 窗口（5 s 前缀 + 1 s 目标），共 15,988 窗口，
  train/val/test = 11,192 / 2,398 / 2,398，窗口不跨集合。
- 模型：Persistence、频率先验、Bigram、LSTM(2×256)、Transformer(4 层, d=192)；
  AdamW + 余弦退火 + 早停，batch 64，12 epoch，3 seed。
- 结果（test，free-running Top-1 / teacher-forced Top-1 / NLL）：

| 模型 | Free Top-1 | Teacher Top-1 | NLL |
|---|---|---|---|
| Persistence | 1.62% | 1.62% | -- |
| 频率先验 | 1.16% | 1.16% | -- |
| Bigram | 1.71% | 1.71% | -- |
| LSTM | 2.08% | 2.78% | 6.56 |
| Transformer | **2.19%** | **3.36%** | **6.52** |

- 训练轮数：上表为加长到 45 轮后的结果（`results/token_prediction_long/`）。原
  `--epochs 15 --patience 3` 在第 12 轮早停，而验证 NLL 当时仍在下降，属欠训练；
  12 轮版本 LSTM free Top-1 只有 1.61%，低于 Bigram 的 1.71%，那是假负结果。
- 是否仍欠训练：**已收敛**。最优轮次虽在 41–45，但验证 NLL 每 5 轮的下降量
  呈几何衰减（LSTM 0.130→0.030→0.0048→0.0021，Transformer 0.155→0.021→0.0032→0.0013），
  外推剩余可降仅 0.001–0.002，小于末 5 轮抖动（0.0035 / 0.0021）。
  即 best_epoch 落在上限是平台期抖动，不是仍在改善；无需继续加长。

- 解码：Transformer 自回归 50 帧 × 4 码本 → EnCodec 解码 1 秒音频，
  与真实后 1 秒的 Log-Mel RMSE 为 7.7–21.1 dB（8 个流派样例已保存）。
- 单码本不可单独解码，故本文不做单码本主结果。

### 阶段二：MusicGen-small 续写与 LoRA

- 续写：前缀 5 s → 4×250 Token → decoder.generate（delay pattern），文本条件为流派描述。
  8 个流派各 1 首：生成 3 秒耗时 **3.7 s（均值）**、峰值显存 **1,393 MB**、
  Log-Mel RMSE **19.2 dB**。
- 长度限制：>150 帧触发位置索引越界断言，统一取 3 秒，base 与 LoRA 同参数。
- LoRA 初版：144 个投影层、rank 8、alpha 16、2.36 M 可训练参数，300 步 1.7 分钟、
  峰值 1,295 MB；loss 8.81 → 7.81（均匀 7.62），**未收敛**，当时记为负结果。
- **根因排查（2026-10-20）**：`MusicgenForCausalLM` 要求 `input_ids=(B·K,T)`、
  `labels=(B,T,K)`，初版把 `(B,K,T)` 同时当两者传入，库据 `labels.shape[1]`
  截取 logits，loss 只在末尾 K 帧上与错位目标计算。未训练原始模型对照：
  错误布局 11.08、正确布局 5.46、均匀分布 7.62 —— 错误布局下"比乱猜还差"。
- 修正后：batch 2→16、300→1200 步、只对续写帧计 loss，loss 7.60 → **5.98**；
  音频 Log-Mel RMSE 19.23 → **17.91**，8 首中 **7 首变好**；去掉改善最大的一首后
  仍有 **−0.99**（初版同样处理后只剩 −0.16，属噪声）。
- 消融：rank 16 的 loss 更低（5.68）但音频更差（18.59）——**目标值低 ≠ 音频好**；
  不掩前缀（全窗口）17.64，与掩码版差 0.27，n=8 上不显著。
- 产物：`results/generation/`（音频、metrics_{base,ft,r16,full}.json、
  finetune_v2.json、finetune_v1_negative.json）、`checkpoints/generation/`。

### 训练预算消融（2026-10-20 复核）

问题：部分模型的最优轮次正好落在 30 轮上限（GTZAN CNN seed123、GTZAN CRNN seed42、
FMA CRNN seed42），需检验"继续训练是否更好"。统一加长到 60 轮、patience 12，
其余协议不变：

| 数据集 | CNN | CRNN | 轻量 Transformer |
|---|---|---|---|
| GTZAN | 0.7530 → 0.7606 (+0.0075) | 0.7234 → 0.7370 (+0.0136) | 0.5415 → 0.5348 (−0.0067) |
| FMA-small | 0.4696 → 0.4611 (−0.0085) | 0.4921 → 0.4844 (−0.0077) | 0.3784 → 0.3808 (+0.0024) |

结论：**FMA 上无系统性收益（均值 −0.46 pp），说明主实验原训练预算够用**，
早停没有截断仍在改善的模型。GTZAN 的提升出现在仅 147 首的测试集上、单一种子，
不能与随机波动区分；轻量 Transformer 更差与其早过拟合（最优轮 7–13）一致。
因此**主表保持 30 轮协议不变**，加长结果作为稳健性检查。

产物：`results/gtzan_long/`、`results/fma_long/`、报告 §6.5 与表 14。
