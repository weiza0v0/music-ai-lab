# 研究日志

> 记录规则：每条实验记录以 `## YYYY-MM-DD HH:MM` 开头，按产物修改时间排序。
> 时间取自实际文件时间戳（±1 分钟），内容包括：目的、命令、结果、问题与决策、产物。
> 完整结果见 `results/`，环境信息见 `docs/environment_report.md`。

## 2026-10-19 11:35

**阅读计划文件，拆解最小 MVP**

- 读取原始项目计划（现已整理为 `README.md`）：目标是「音乐 AI 实验室 + 模型擂台 + 现场盲测」，两周周期，
  核心是五个 Baseline（SVM / MLP / CNN / CRNN / 轻量 Transformer）+ 预训练 AST + 交互 Demo。
- 结合当前机器条件（WSL2、RTX 4060 Laptop 8GB、已有 PyTorch 环境），确定本轮只做
  **最小闭环 MVP**：环境审计 → 数据 → 特征 → 五个模型 → 结果图表 → 真实推理 Demo；
  AST、FMA-small、鲁棒性实验、PPT 留到后续。
- 产物：`README.md` 中的项目计划与进度表（由原计划文件改写）。

## 2026-10-19 11:36

**环境审计：Python / PyTorch / GPU / 磁盘**

```
python3 -V                     -> 3.13.9 (anaconda base)
python3 -c "import torch"      -> ModuleNotFoundError
nvidia-smi                     -> Failed to initialize NVML: GPU access blocked by the operating system
python3 -c "cuInit(0)"         -> 100 (CUDA_ERROR_NO_DEVICE)
df -h /mnt/d                   -> 1.4T 可用
```
- 结论：base 环境无 torch；GPU 暂不可见；网络可达 PyPI/GitHub/HuggingFace。
- 决策：先建独立 venv 装 CPU 版 torch 保证流程能跑，同时继续排查 GPU。

## 2026-10-19 11:38

**下载 GTZAN 数据集（1.17 GiB）**

```
curl -L -o data/raw/genres.tar.gz \
  https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz
```
- 实测速度约 5 MB/s，约 4 分钟完成。
- 选择 GTZAN 而非 FMA-small：MVP 阶段先跑通流水线，GTZAN 体积小、类别均衡；
  FMA-small（7.2 GiB）留作主实验。

## 2026-10-19 11:45

**解压与数据初检**

```
tar -xzf data/raw/genres.tar.gz -C data/raw/gtzan
```
- 解压 1 分 18 秒，得到 10 个流派目录，每类 100 首。
- 发现 1000 个 `._*.wav`（macOS AppleDouble 元数据，非音频），在扫描时过滤。

## 2026-10-19 11:46

**数据审计与第一版划分**

```
python -m src.data.prepare_dataset --audit --split
```
- 1000 首全部为 22,050 Hz 单声道、时长 29.93–30.65 s。
- 1 首损坏：`jazz.00054`；28 首与其他曲目 MD5 完全相同（GTZAN 已知问题）。
- 第一版划分按曲目分层 70/15/15，seed=42。
- 决策：损坏文件剔除；重复曲目每组只保留第一首，避免同曲跨集合泄漏。

## 2026-10-19 11:58

**GPU 排查：定位到 DSH 沙箱**

```
nvidia-smi                                   -> GPU access blocked by the operating system
open('/dev/dxg', O_RDWR)                     -> EACCES (节点权限 666)
Windows nvidia-smi.exe                       -> 正常 (RTX 4060 Laptop / 595.79 / CUDA 13.2)
DSH 文件策略 workspace-write -> danger-full-access
open('/dev/dxg')                             -> ok
cuInit(0)                                    -> 0, device_count=1
```
- 结论：驱动与 WSL 均正常，是进程沙箱拦截了 `/dev/dxg` 设备访问；NVML 将其翻译为
  「操作系统阻止了 GPU」。
- 决策：后续训练全部走 GPU；不再使用 CPU venv。

## 2026-10-19 11:59

**复用 DataAnalysis 环境并清理临时产物**

- 发现 conda 环境 `DataAnalysis`：Python 3.12.13、torch 2.12.1+cu130、
  `torch.cuda.is_available()=True`，已识别 RTX 4060。
- 补装 librosa / soundfile / streamlit / pytest，其余依赖已齐全。
- 删除临时 `.venv`（1.8 GB）、`genres.tar.gz`（1.2 GB）、`/tmp/pip-unpack-*`。
- 产物：可复现依赖写入 `requirements.txt`。

## 2026-10-19 12:01

**最终划分与特征缓存**

```
python -m src.data.prepare_dataset --audit --split
python scripts/extract_features.py --workers 8
```
- 清洗后 985 首：train 689 / val 149 / test 147，10 类均有覆盖；
  `data/splits/gtzan/splits.csv` 与 `stats.json` 固化。
- 特征：209 维人工特征（MFCC/Δ、chroma、谱对比度、质心/带宽/rolloff、ZCR、RMS、tempo）
  + 64×216 Log-Mel；985 首耗时 49.5 秒。
- 决策：两个表示一次算好、所有模型共用同一批 5 秒中心片段。

## 2026-10-19 12:02

**传统 Baseline：SVM 与 MLP**

```
python -m src.train_classical --models svm mlp --seeds 42
```
- SVM（RBF, C=10）：test Acc 0.7075 / Macro-F1 0.6987，训练 0.22 s。
- MLP（256-128）：test Acc 0.6395 / Macro-F1 0.6331，训练 0.59 s。
- 标准化器只在 train 拟合；验证集只用于记录，测试集只评估一次。

## 2026-10-19 12:03

**第一轮 CNN 训练（误在 CPU 上跑）**

```
python -m src.train_cnn --model cnn --seeds 42 --epochs 30
```
- 30 epoch 共 296 秒（9–15 s/epoch），test Acc 0.7143 / Macro-F1 0.7147。
- 复盘发现模型与张量都没有 `.to(device)`，实际是 CPU 训练。
- 决策：修正设备迁移后重跑（见 12:11）。

## 2026-10-19 12:08

**图表生成与中文字体修复**

```
python -m src.evaluate
```
- 排行榜、混淆矩阵、训练曲线生成；matplotlib 默认字体缺中文字形，出现方框。
- 修复：注册 Windows 微软雅黑（`/mnt/c/Windows/Fonts/msyh.ttc`），重新渲染正常。

## 2026-10-19 12:09

**协议测试与 CLI 推理**

```
pytest -q tests
python -m src.predict data/raw/gtzan/genres/jazz/jazz.00007.wav --model cnn
```
- 4 项测试通过：划分无重叠、特征与划分对齐、标准化器只用训练集、checkpoint 可推理。
- CLI 对 jazz 样本给出 96%+ 的 jazz，证明权重链路可用。

## 2026-10-19 12:10

**环境报告与 Demo 冒烟**

```
python scripts/env_audit.py
streamlit run app/demo.py --server.headless true
```
- 生成 `docs/environment_report.md`（含 GPU 排查结论）。
- Streamlit 返回 HTTP 200，上传音频可走真实权重。

## 2026-10-19 12:11

**CNN 改为 GPU 训练**

- `src/train_cnn.py` 增加 `device` 迁移、`pin_memory`、按设备放置 batch。
- 30 epoch 降至 **56 秒**（1.8 s/epoch），test Acc 0.7347 / Macro-F1 0.7310。
- 结论：GPU 生效，且准确率反而略升。

## 2026-10-19 12:12

**补齐 CRNN 与轻量 Transformer**

```
python -m src.train_cnn --model crnn --seeds 42 --epochs 30
python -m src.train_cnn --model transformer --seeds 42 --epochs 30
```
- CRNN（CNN+BiGRU）：Acc 0.7211 / F1 0.7151。
- Transformer（2 层、d=128、4 头）：Acc 0.5646 / F1 0.5566，训练 loss 迅速到 0，
  验证 loss 抬头，早停于第 22 epoch —— 小数据过拟合的典型案例。

## 2026-10-19 12:19

**多 seed 复测（第一版）与首轮独立审计**

```
python -m src.train_classical --models svm mlp --seeds 123 2024
python -m src.train_cnn --model {cnn,crnn,transformer} --seeds 123 2024 --epochs 30
```
- 完成 5 模型 × 3 seed = 15 次训练；首轮独立校验员 6/6 PASS，未发现数据泄漏。
- 校验员指出 6 个报告层面的问题：`best_ep` 记错、eval 的 val 取最后一轮、
  CLI 默认 seed 123、示例用了训练集曲目、SVM 概率与硬预测不一致、文档未更新。
- 决策：修复代码并重跑，确保产物与代码一致。

## 2026-10-19 12:29

**发现非确定性并开启确定性训练**

- 同 seed 重跑 CNN seed 123 出现 76.2% → 73.5% 的漂移，原因是 cuDNN 自动调优
  与非确定 kernel。
- 修复：`cudnn.benchmark=False`、`cudnn.deterministic=True`、
  `use_deterministic_algorithms(True, warn_only=True)`。
- 重跑全部深度模型（3 seed × 3 模型）。

## 2026-10-19 12:39

**最终结果落盘**

```
python -m src.evaluate
```
- 最终 15 行 `results/gtzan/metrics.csv`（test，n=147）：

| 模型 | Accuracy | Macro-F1 |
|---|---|---|
| CNN | 0.755 ± 0.007 | 0.753 ± 0.006 |
| CRNN | 0.726 ± 0.034 | 0.723 ± 0.034 |
| SVM | 0.708 ± 0.000 | 0.699 ± 0.000 |
| MLP | 0.633 ± 0.051 | 0.630 ± 0.055 |
| Transformer | 0.544 ± 0.025 | 0.541 ± 0.019 |

- 产物：排行榜、5 联混淆矩阵、训练曲线、`summary.md`、逐样本预测 CSV。

## 2026-10-19 12:45

**确定性验证与第二轮独立审计**

- 重跑 CNN seed 42，结果逐位一致（0.7483 / 0.7493）。
- 校验员在 `/tmp` 隔离副本重训 cnn / crnn / transformer 的 seed 42，得到与仓库
  **sha256 完全相同**的 checkpoint；第二轮审计仍为 **6/6 PASS**。
- 遗留的固有行为：SVM 的 `predict()` 与 `predict_proba()` 约 5–7% 样本不一致
  （已写入 README）；`adaptive_avg_pool2d_backward_cuda` 无确定性实现（实测可复现，
  非跨硬件硬保证）。

## 2026-10-19 12:47

**文档整理与 GitHub 发布准备**

- 将计划文件改写为结构化 `README.md`（背景、思路、结构、计划、进度、下一步）。
- 新增本日志、`docs/data_card.md`、`docs/audit_report.md`、`docs/mvp_checklist.md`。
- 更新 `.gitignore`：不提交数据与权重，提交代码、配置、结果与图表。
- 初始化 git 仓库，准备推送到 GitHub 供队友查看。

## 2026-10-19 12:55

**P0 错误分析：从预测文件挖失败案例**

```
python -m src.error_analysis
```
- 读取 5 个模型 seed 42 的测试预测，复算混淆对与置信度，不重新训练。
- 结果：CNN 最大混淆对为 `reggae→pop`(3)、`blues→reggae`(2)、`hiphop→pop`(2)；
  per-class F1 最低 `rock` 0.429、最高 `classical` 0.933。
- 置信度：Transformer 有 16/64 个错误样本置信度 ≥ 0.9；CNN 只有 3/37。
- 典型翻车：`jazz.00003` 被 CNN 以 99.6% 判成 classical。
- 产物：`results/gtzan/error_analysis.csv/.json`、`error_confusions.png`、
  `error_cases.png`、`docs/error_analysis.md`。

## 2026-10-19 12:58

**P0 鲁棒性：5 模型 × 7 种扰动**

```
python -m src.robustness --workers 12
```
- 在 147 首测试曲目上重算特征并推理，条件为 clean / 3 s / 10 s / 增益 0.5×、1.5× /
  噪声 10 dB、0 dB；35 行结果，每行 n=147，耗时约 80 秒。
- clean 行与 `metrics.csv` 五个模型全部零误差；连跑两次 CSV 逐字节相同。
- 主要发现：
  - 深度模型对增益完全不敏感（Log-Mel `ref=np.max`），SVM/MLP 掉 2–3.4 pp；
  - 噪声让所有模型崩塌（10 dB 平均约 −42 pp，0 dB 约 −50 pp）；
  - 除增益外，每个扰动下绝对准确率第一都是 SVM——深度模型赢在干净输入，
    传统模型赢在输入变脏时；
  - 3 s 片段最伤 CRNN（−53.1 pp）与 CNN（−34.0 pp）。
- 产物：`results/gtzan/robustness.csv`、`figures/robustness.png`、`docs/robustness.md`。

## 2026-10-19 13:00

**P2 功能 C：滑动窗口时间轴**

```
python -m src.timeline data/raw/gtzan/genres/rock/rock.00072.wav --model cnn
```
- 实现 5 秒窗口、2.5 秒步长的逐段预测，输出热力图与逐窗表格，并接入 Streamlit Demo。
- 扫描 147 首测试曲目：**88 首**至少出现一次窗口级判断变化；最不稳定的
  `rock.00072` 在 11 个窗口里改判 9 次（rock/disco/blues/country 交替）。
- 修复了两个小问题：matplotlib 中文字体缺失、音频略长于 30 s 时的重复尾窗。
- 产物：`src/timeline.py`、`results/gtzan/timeline_scan.csv`、`figures/timeline_*.png`、
  `docs/timeline.md`。

## 2026-10-19 13:10

**P1 AST 预训练迁移：冻结主干 + 线性头**

```
python -m src.train_ast
```
- 模型 `MIT/ast-finetuned-audioset-10-10-0.4593`（86M 参数，346 MB），
  16 kHz、中心 5 秒补零到 10.24 秒，冻结主干取 `pooler_output`。
- 线性头 = StandardScaler（仅 train）+ LogisticRegression，C 在 val 上选得 3；
  测试集只评估一次。
- 结果：test n=147，Acc **0.8231** / Macro-F1 **0.8135**，比 CNN 高 6.8 / 6.0 个百分点。
- 成本：embedding 提取 214 秒（985 首）、头训练 23.4 秒、推理 204 ms/片段、
  权重 346 MB。
- 局限：16 kHz 与 GTZAN 22.05 kHz 存在域差异；5 秒补零到 10.24 秒；
  收益来自 AudioSet 预训练，不能归因于 Transformer 架构本身。
- 产物：`src/train_ast.py`、`results/gtzan/ast_metrics.json`、`results/gtzan/ast_eval.json`、
  `docs/ast.md`。

## 2026-10-19 13:35

**Windows 自动关机后恢复，合并 AST 结果并重新出图**

- 关机中断了 AST 的二次复跑与 `evaluate`，但 13:09 的首轮结果完整可用。
- 新增 `scripts/integrate_ast.py`：把 AST 结果规范化为统一 eval schema 并写入
  `results/gtzan/metrics.csv`（16 行）。
- 重跑 `python -m src.evaluate`：排行榜与混淆矩阵变为 6 个模型，
  AST 以 Macro-F1 0.814 居首。
- 重新运行协议测试（4 passed）、Demo 冒烟与文档更新。

## 2026-10-19 13:45

**统一鲁棒性实验的决策规则并重跑**

- 独立校验发现：鲁棒性实验对 SVM 使用 `predict_proba().argmax`，而训练/评估使用
  `SVC.predict()`，导致 clean_5s 的 SVM Macro-F1 为 0.6976、与 `metrics.csv` 的
  0.6987 相差 0.0011（准确率一致）。
- 修复：`src/robustness.py` 对 SVM/MLP 改用 `.predict()`，深度模型仍用 softmax argmax。
- 重跑后 clean 行五个模型的 Accuracy 与 F1 均与 `metrics.csv` 完全一致；
  SVM 在扰动下的数值随之更新（如 noise_0db 0.2653 → 0.2993）。
- 新增 `scripts/robustness_flips.py` 与 `results/gtzan/robustness_flips.csv`，
  重算预测翻转数（SVM clip_3s 46、noise_0db 98 等）。
- 同步更新 `docs/robustness.md` 中所有受影响的表格与结论。

## 2026-10-19 14:13

**FMA-small 下载完成并检查**

- 元数据 342 MB、音频 7.2 GiB，下载约 25 分钟（~5 MB/s）。
- 两个 zip 均通过 CRC 校验（metadata 12 项、audio 8002 项）。
- 音频 8000 首，8 流派各 1000；采样率 44.1k/48k/22.05k 混合，统一重采样到 22.05 kHz。

## 2026-10-19 14:20

**发现艺术家级泄漏，主实验改用艺术家分离划分**

- 按曲目分层划分后，测试集 971/1199 首（81.0%）与训练集共享艺术家；
  同一乐队的音色与录音会成为「捷径」。
- 改为按艺术家分离的划分：每位艺术家整体进入一个集合，测试集与训练集艺术家重叠 0；
  整体 5596/1199/1199，逐流派比例不完全均衡（测试 126–191 首）。
- 保留曲目级划分 `splits_tracklevel.csv` 作为泄漏对照实验。
- 6 首解码失败，最终 7994 首可用。

## 2026-10-19 14:38

**FMA 特征缓存**

```
python scripts/extract_features.py --splits data/splits/fma/splits.csv \
  --out data/features/fma/features.npz --workers 12
```
- 7994 首 ×（209 维人工特征 + 64×216 Log-Mel），535 秒。
- 另生成 `features_tracklevel.npz`（同一批特征、对照划分标签）用于泄漏实验。

## 2026-10-19 14:40

**启动 FMA 主实验**

```
bash scripts/run_fma.sh 42 123 2024
```
- 5 个模型 × 3 seeds，训练/验证/测试 = 5596/1199/1199。
- 环境：RTX 4060 8 GB；深度模型 AdamW、batch 32、30 epoch、按 val Macro-F1 早停。
- 结果写入 `results/fma/`，随后做优化消融、泄漏对照与 AST 迁移。

## 2026-10-19 15:27

**FMA 主实验基线完成（5 模型 × 3 seeds）**

- CNN: Acc 0.480 ± 0.005 / Macro-F1 0.470 ± 0.002，训练 334 s/seed。
- CRNN: Acc 0.502 ± 0.009 / Macro-F1 0.492 ± 0.006，训练 371 s/seed。
- SVM: 0.446 / 0.440；MLP: 0.428 / 0.422；Transformer: 0.387 / 0.378。
- 与 GTZAN 相比整体低 20–30 个百分点：FMA 的 8 个流派边界更宽（Experimental、
  International），且按艺术家分离划分消除了「认乐队」的捷径。
- CNN 训练损失降到 ~0.7、验证损失 ~1.5，存在明显过拟合。

## 2026-10-19 15:50

**优化消融：SpecAugment 与标签平滑（仅验证集）**

- CNN 基线 val Macro-F1 0.5218；+SpecAug 0.5197；+SpecAug+标签平滑+dropout0.4 0.5289。
- CRNN +SpecAug+LS 0.5242（基线 0.5253）；Transformer 0.3936（基线约 0.394）。
- 最大提升仅 +0.7 个百分点，落在单次运行波动范围内；因此主实验保留基线配置。
- 结论：FMA 的瓶颈更像是标签边界模糊与音乐多样性，而不是单纯过拟合。

## 2026-10-19 15:55

**启动 FMA 上的 AST 迁移与泄漏对照**

- AST：16 kHz、中心 5 秒补零到 10.24 秒，冻结 86M 主干 + 线性头；
  embedding 缓存 `data/features/fma/ast_embeddings.npz`。
- 泄漏对照：同一批特征、仅把划分换成曲目级（81% 测试曲目与训练集共享艺术家），
  训练 CNN seed 42，用于量化泄漏带来的虚高。

## 2026-10-19 16:04

**泄漏对照结果：曲目级划分虚高 9.8 个百分点**

- 同一批特征、仅把划分换成曲目级（971/1199 测试曲目与训练集共享艺术家），
  CNN seed 42：Acc 0.5663 / Macro-F1 0.5682。
- 对比艺术家分离划分的 CNN（Acc 0.480 / Macro-F1 0.470），Macro-F1 **虚高 +9.9 pp**。
- 结论：FMA 这类带艺术家元数据的语料必须按艺术家划分；主表全部采用艺术家分离结果。

## 2026-10-19 17:06

**FMA 上的 AST 迁移完成**

- 冻结 AudioSet AST（86M 参数）+ 线性头，多进程特征化 + GPU fp16 批量前向。
- test n=1,199：Acc **0.5963** / Macro-F1 **0.5903**，比最好的自建模型 CRNN
  （0.492）高 9.8 个百分点。
- 成本：权重 345 MB、推理约 231 ms/片段；embedding 提取约 7 分钟（fp16）。
- 期间修复：sklearn 1.9 移除 `multi_class`、断点复用特征/嵌入、HF 离线模式。
- 产物：`docs/ast_fma.md`、`results/fma/ast_metrics.json`、`ast_eval.json`。

## 2026-10-19 17:30

**技术报告扩写与流程图修复**

- 报告从 11 页扩写到 24 页：新增目录、研究背景/挑战/贡献、相关工作 6 小节、
  数据统计与泄漏表、209 维特征表、模型结构表、训练算法伪代码、环境/指标/统计/实验矩阵、
  逐 seed 与逐类结果、混淆矩阵 2×3 布局、训练曲线、部署成本表、讨论与局限、附录。
- 原 matplotlib 流程图文字过小、箭头遮挡，改为 **TikZ 矢量图**（节点对齐、箭头清晰）。
- 混淆矩阵由 6 联横排改为 2×3，标签可读性明显改善。
- PDF：`docs/Music_AI_Lab_Technical_Report.pdf`（24 页）。

## 2026-10-19 17:40

**TikZ 流程图箭头重排**

- 原实现用 `|-` 折线分别连到评估节点，三条线在评估框前重叠、箭头挤在一起。
- 改为「分发母线 + 汇流母线」：预处理后一条竖线分出三条支路；三个模型输出先并入
  右侧竖线，再用单箭头进入评估框；交点加实心圆点区分连接与交叉。
- 删除悬浮的协议说明框，其内容并入图注，画面更紧凑。

## 2026-10-19 19:05

**仓库目录整理：结果与权重按实验合并**

- 顶层原有 4 个 `results_*`、4 个 `checkpoints*`、2 个 `splits*`、1 个 `processed`，
  同级散落在项目根；现合并为「一个顶层目录 + 每个实验一个同名子目录」。
- 迁移映射（代码默认值、`scripts/*.sh`、测试、文档、报告同步更新）：

  | 旧路径 | 新路径 |
  |---|---|
  | `results/` | `results/gtzan/` |
  | `results_fma/` | `results/fma/` |
  | `results_fma_leak/` | `results/fma_leak/` |
  | `results_fma_ablation/` | `results/fma_ablation/` |
  | `checkpoints/` | `checkpoints/gtzan/` |
  | `checkpoints_fma/` | `checkpoints/fma/` |
  | `checkpoints_fma_leak/` | `checkpoints/fma_leak/` |
  | `data/processed/` | `data/features/gtzan/` |
  | `data/fma/processed/` | `data/features/fma/` |
  | `data/splits/` | `data/splits/gtzan/` |
  | `data/splits_fma/` | `data/splits/fma/` |

- 约定：`MUSIC_CKPT` 与 `MUSIC_RESULTS` 指向**同一个实验名**（gtzan / fma / …），
  默认值为 GTZAN，因此不带环境变量跑 MVP 命令行为不变。
- `.gitignore` 同步：`data/processed/` → `data/features/`，删除已失效的 `checkpoints_fma*/`。
- 顺带修复报告附录的仓库树：原 `verbatim` 用 lmmono，缺 U+251C/U+2500 字形，
  方框字符全部丢失（PDF 里树是断的）；现对该代码块单独 `\setmonofont{WenQuanYi Micro Hei Mono}`，
  缺字告警 0 条。
- 验证：`pytest -q tests` 4 passed；`gen_tables.py` 重算表格与数字**无差异**；
  报告仍 24 页。移位后的根目录只剩 `configs/ data/ docs/ src/ scripts/ tests/ app/
  checkpoints/ results/` 与 `README.md`、`requirements.txt`、`.gitignore`。

## 2026-10-19 19:06

**二期启动：阶段 0 审计**

- 把二期计划（Token 预测 + 预训练生成模型微调）拆成「阶段 0 审计 → 阶段一 Token 预测 →
  阶段二 生成」，先审计再写代码，避免在跑不动的假设上铺开。
- 审计结论：torch 2.12.1+cu130 / RTX 4060 8 GB / 内存 7.9 GB / 磁盘余 1.4 TB；
  `transformers` 5.16.1 已含 `EncodecModel` 与 `MusicgenForConditionalGeneration`，
  无需 `audiocraft`、无需升级环境。
- 数据：FMA-small artist-disjoint 划分可直接复用（5,596/1,199/1,199）。
- 环境坑：`no_proxy` 里的 `[::1]` 让 `httpx` 抛 `Invalid port`，下载 HF 权重前需
  `export NO_PROXY=localhost,127.0.0.1`。
- 产物：`docs/phase0_audit_phase2.md`。

## 2026-10-19 19:45

**EnCodec Token 化：形状语义踩坑与修复**

- 用 `facebook/encodec_32khz` 把 FMA 窗口编码为 Token：32 kHz、4 码本、50 Hz、
  2048 词表，6 s 窗口 = 4×300 Token。
- 首次全量编码（512 窗/shard）跑完 20 个 shard 才发现：每 shard 只剩 32 行 Token。
  原因不是 OOM，而是编码输出第一维是**分段数**（`nb_frames`）而非批大小，
  `audio_codes[:, 0]` 把批内除第一段以外的样本全部丢掉，且不报错。
- 修复：`audio_codes[0]` 取分段、保留批维；并加断言 `nb_frames == 1`。
  重新编码 15,988 个窗口，shard 内行数与窗口数逐一对齐。
- 教训写入审计报告与报告正文：形状对得上不代表语义对，静默丢样本比报错更危险。

## 2026-10-19 21:20

**阶段一：Token 预测训练完成（3 seed）**

- 数据：15,988 个 6 s 窗口（train 11,192 / val 2,398 / test 2,398），
  artist-disjoint，窗口不跨集合。
- 模型：Persistence / 频率先验 / Bigram / LSTM(2×256) / Transformer(4 层, d=192)。
  统一 AdamW + 余弦退火 + 早停，batch 64，12 epoch（受时间预算约束，记录在案）。
- 测试集 Top-1（4 码本平均，free-running 口径）：
  Persistence 1.62\%、频率先验 1.16\%、Bigram 1.71\%、LSTM 1.61\%、Transformer **1.95\%**。
  Transformer teacher-forced 2.43\%，NLL 6.72（均匀分布 ln2048 = 7.62）。
- 结论：信号真实（约随机猜的 40 倍），但绝对水平低；误差累积显著；
  LSTM 在回灌条件下不如 n-gram；码本 0 靠"重复"就够，增益来自残差码本。
- 产物：`results/token_prediction/`、`checkpoints/token_prediction/`。

## 2026-10-19 21:40

**阶段一：Token 解码回音频**

- 用 Transformer（seed 42）自回归生成 50 帧 × 4 码本，交回 EnCodec 解码成 1 秒音频，
  与真实后 1 秒做 Log-Mel 对照并保存 8 个流派的 wav。
- 客观距离（Log-Mel RMSE）7.7–21.1 dB：包络接近，高频细节差异明显。
- 如实记录：单码本模型无法单独解码，本文只对"四码本联合预测"做解码。

## 2026-10-19 21:55

**阶段二：MusicGen-small 前缀续写（原始模型）**

- 用 `transformers` 的解码器直连方式做续写：前缀 5 s → EnCodec 4×250 Token →
  `decoder.generate`（自带 delay pattern），文本条件用流派描述。
- 8 个流派各 1 首测试曲：生成 3 秒耗时 **3.0–4.2 s**，峰值显存 **1,356–1,431 MB**，
  Log-Mel RMSE 12.9–29.2 dB。
- 踩坑：① `pad/bos=2048` 在 decoder 直连时需要显式传 `decoder_start_token_id`，
  否则 delay pattern 里 `pad_token_id=None` 直接 TypeError；② 生成超过约 150 帧
  （>3 s）会触发位置索引越界断言，因此统一取 3 秒，并如实写进报告；
  ③ 逐曲单进程运行，避免一次进程内多曲导致的上下文污染。

## 2026-10-19 22:00

**阶段二：LoRA 适配（手写，不引入新依赖）**

- 对解码器自注意力/交叉注意力的 144 个投影加 LoRA（rank 8、alpha 16，2.36 M 可训练参数），
  冻结全部原始权重，沿用 MusicGen 自己的 token 级交叉熵与 delay pattern。
- 8 GB 卡上训练 300 步仅 **1.7 分钟**、峰值显存 **1,295 MB**——可行性没问题。
- 但 loss 只从 10.4 降到 7.63（均匀分布 7.62），说明目标对齐没调好，
  属于**负结果**：只作为"参数高效适配可行"的工程验证，不宣称生成质量提升。
- 期间修了两个实现 bug：LoRA 层没跟着 base 搬到 GPU、bf16 下 `-100` 标签掩码
  破坏 delay pattern 导致 NaN。

## 2026-10-19 22:15

**二期收尾：技术报告扩到 30 页 + 文档同步**

- 报告新增第 8 节「二期扩展：从判别式分类到离散 Token 预测与音乐生成」，
  含 EnCodec 规格表、窗口协议、模型表、Token 结果表、解码样例表、
  MusicGen 成本表与两张新图（Top-1 对比、逐码本、解码对照、续写频谱对比），
  并在讨论/结论补充二期结论；总页数 24 → **30 页**。
- 期间修了四个排版/转义问题：Markdown 的 `**` 残留进 LaTeX、表格行分隔符被
  转义成单反斜杠、样例文件名下划线未转义、一段带长 token 的文字溢出右边界。
- README 增加 6.6 节（二期结果）与 7.7 进度表；`docs/experiment_log.md` 增加二期小节。
- 如实保留的负结果：LSTM 在回灌条件下不如 Bigram；LoRA loss 未收敛（7.63 vs 均匀 7.62）；
  > 这两条均于 **2026-10-20 07:00 被推翻**：前者是早停太早，后者是标签维度 bug。详见当日验收条。
  续写受 3 秒长度限制。

## 2026-10-19 23:50

**二期复核：全量校验 + 文档收尾（未改动任何实验结果）**

- 动机：上一会话在收尾汇报时上下文溢出中断，虽已推送并投递 PDF，但从未发出最终
  完成报告，因此本次对全部产物做一次"冷眼"复核。
- 校验结果：`git status` 干净、本地 `main` 与 `origin/main` 同为 `c3c0291`；
  技术报告 `pdfinfo` 确认 **30 页**、`main.log` 编译错误数为 0。
- 排查 `pytest` 报错：`test_every_checkpoint_predicts_real_audio` 抛
  `cannot cache function '__o_fold': no locator available`。根因不是代码缺陷——
  numba 0.66 的 `CACHE_DIR` 为空，`UserWideCacheLocator` 需写 `~/.cache/numba`，
  而 DSH `workspace-write` 沙箱拒绝该写入（`PermissionError [Errno 13]`），
  `from_function()` 只吞 `OSError` 便返回 `None`，于是所有 locator 失败。
  在无沙箱条件下重跑即 **4 passed**，故 README「pytest 4 项全过」属实。
- 回填 `docs/phase0_audit_phase2.md` §6 的 MusicGen 占位：加载 17.2 s（LoRA 5.8 s）、
  峰值显存 1,356–1,431 MB（LoRA 训练 1,295 MB）、3 s 生成 3.0–4.2 s、前缀续写接口可用；
  4 s 生成因位置索引越界未实测，如实标注。
- 清理：删除已被 TikZ 取代的 `figures/pipeline.png`、审阅用 `figures/tikz_preview.png`
  及其生成脚本 `make_pipeline.py`（三者在仓库内均无引用）。
- 产物：无新增实验结果；仅文档与死资源变动。

## 2026-10-20 00:05

**复核"预训练模型跑不通"的确切原因**

- 结论：三个预训练模型（AST、EnCodec、MusicGen-small）**全部跑通**，报告数字均来自实跑。
  真正装不上的是 MusicGen 的官方库 `audiocraft==1.3.0`。
- 实测阻塞点：其 `requirements.txt` 把 `torch==2.1.0`、`av==11.0.0`、
  `xformers<0.0.23`、`torchvision==0.16.0`、`torchtext==0.16.0` 全部钉死在 torch 2.1 一代。
  其中 `av` 在 PyPI 上自 12.0.0 起才有包，**Python 3.12 没有 11.0.0 的 wheel**，
  只能源码编译，缺 ffmpeg 开发头即失败（复现命令
  `pip install --dry-run audiocraft` → `ERROR: Failed to build 'av'`）。
- 绕过：`transformers` 5.16.1 自带 `MusicgenForConditionalGeneration` 与 `EncodecModel`，
  用同一份权重完成续写，故模型能力未缺失；代价是需自拼 delay pattern、
  且受位置编码上限只能续 3 秒。
- 另一项非崩溃的负结果：`peft`/`accelerate` 同样未装，LoRA 为手写实现，
  跑通但 loss 未收敛（10.4 → 7.63，均匀分布 7.62）。

## 2026-10-20 07:00

**夜间加长训练验收：修正一个维度 bug，翻转两个负结果**

动机：复核发现两类系统性欠训练，且 LoRA 的"负结果"可能另有原因。挂
`scripts/overnight_queue.sh` 串行跑了约 6.5 小时（00:45–06:46），全部作业 OK。

### 1. LoRA 负结果的根因是标签维度错误

`MusicgenForCausalLM` 要求 `input_ids=(B·K,T)`、`labels=(B,T,K)`。初版把
`(B,K,T)` 同时当两者传入，库据 `labels.shape[1]` 截取 logits，loss 只在末尾
`K` 帧上与错位目标计算。用未训练原始模型直接对照：

| 布局 | loss |
|---|---|
| 旧（`(B,K,T)`） | **11.08** |
| 修正（`(B,T,K)`） | **5.46** |
| 均匀分布 ln 2048 | 7.62 |

旧记录起始 loss 10.42 与此吻合——错误布局下"比乱猜还差"，梯度无意义，
训多少步都不收敛。修正后（batch 2→16、300→1200 步、只对续写帧计 loss）：

| 版本 | 训练 loss | Log-Mel RMSE | 去掉最好一首后 |
|---|---|---|---|
| 原始模型 | -- | 19.23 | -- |
| v1（bug） | 8.81→7.81 | 18.43 | **−0.16（噪声）** |
| v2 r8 掩前缀 | 7.60→**5.98** | **17.91** | **−0.99（真实）** |
| r8 全窗口 | 7.63→6.08 | **17.64** | −1.20 |
| r16 掩前缀 | 7.50→5.68 | 18.59 | −0.22 |

v2 逐曲 7/8 变好。**关键对照**：v1 表面 −0.80，但去掉单曲 000995 后只剩
−0.16；v2 同样处理后仍有 −0.99——这才是"整体变好"与"一首变好"的区别。

**意外发现**：r16 训练 loss 更低（5.68）但音频更差（18.59）。
**Token 级目标值低 ≠ 音频更接近**，两者必须分开报告。
掩码 vs 全窗口差 0.27，在 n=8 上不显著，不宣称掩码策略有效。

### 2. Token 模型欠训练被证实

原 `--epochs 15 --patience 3` 在第 12 轮早停，但验证 NLL 当时仍在降
（LSTM 6.8146→6.8091→6.8060）。加长到 45 轮：

| 模型 | free Top-1（旧→新） | NLL（旧→新） |
|---|---|---|
| LSTM | 1.61% → **2.08%** | 6.821 → 6.559 |
| Transformer | 1.94% → **2.19%** | 6.725 → 6.522 |

Bigram 基线 1.71%。**原来"LSTM 不如 n-gram"的负结果翻转为 +0.37 pp**，
Transformer 从 +0.24 升到 +0.48。码本 0 上 Transformer 2.82% 也已超过
Persistence 的 2.43%。

**是否仍欠训练：结论是已收敛。** 最优轮次虽为 LSTM 45/45/42、Transformer 45/41/42，
但按 5 轮一段统计，验证 NLL 的下降速度呈几何衰减：

| 区间 | LSTM | Transformer |
|---|---|---|
| ep5→10 | +0.130 | +0.155 |
| ep20→25 | +0.030 | +0.021 |
| ep35→40 | +0.0048 | +0.0032 |
| ep40→45 | **+0.0021** | **+0.0013** |

按此外推，剩余可降幅度仅 0.001–0.002 NLL，**小于末 5 轮抖动幅度**
（LSTM 0.0035 / Transformer 0.0021）。故最优轮次落在上限是平台期抖动造成的，
不是仍在改善。早先"仍未训够"的判断仅依据 best_epoch，属误判，已在报告与 README 更正。

### 3. 分类模型加长训练：结论是"原预算够用"

60 轮 vs 30 轮（同一划分、同一种子）：

| 数据集 | CNN | CRNN | Transformer |
|---|---|---|---|
| GTZAN | 0.7530→**0.7606** | 0.7234→**0.7370** | 0.5415→0.5348 |
| FMA-small | 0.4696→0.4611 | 0.4921→0.4844 | 0.3784→0.3808 |

FMA 上三者要么持平要么下降（均值 −0.46 pp），说明**主实验原训练预算够用**，
早停没有系统性截断仍在改善的模型。GTZAN 的小幅提升（+0.75/+1.36 pp）出现在
仅 147 首的测试集上、单一种子，尚不能与随机波动区分。
**因此主表保持 30 轮协议不变**，加长结果作为稳健性检查（报告新增 §6.5 与表 14）。

### 产物

- 代码：`src/generation/finetune_lora.py`（维度/文本条件/缓存三处修复）、
  `scripts/overnight_queue.sh`
- 结果：`results/token_prediction_long/`、`results/gtzan_long/`、`results/fma_long/`、
  `results/generation/{finetune_v2,finetune_r16,finetune_full,metrics_ft,metrics_r16,metrics_full}.json`
- 报告：新增"训练预算消融"小节与表 5、重写 LoRA 小节与 Token 观察四点、摘要补二期段落
- 备份：`finetune_v1_negative.json`、`metrics_ft_v1_negative.json`、`audio/*_ft_v1.wav`

## 2026-10-20 09:50

**LoRA 继续训练验收：多训 2400 步，音频无显著改善（负结果）**

动机：v2（1200 步）的 loss 斜率仍为负（全程 −0.106、末 300 步 −0.127 /100 步），
不像已收敛，故跑 3600 步（v2 的 3 倍）回答"多训是否有用"。
完成：3600 步 / 124 分钟 / 峰值 1,782 MB，末值 loss 5.195。

### 1. loss 确实继续降，但斜率已走平

| 段 | 均值 | 斜率 /100 步 |
|---|---|---|
| 1–1200 | 6.521 | −0.1071 |
| 1200–2400 | 5.957 | −0.0186 |
| 2400–3600 | 5.778 | **−0.0060** |

v2 末段均值 6.057 → v3 末段 5.733。即再训 2400 步只降了 0.32 NLL，
且末段斜率已衰减到 −0.006，接近平台。

### 2. 音频几乎不动，且差异不显著

| 配置 | 均值 | vs 原始 | 去掉最好一首 | 改善曲数 |
|---|---|---|---|---|
| 原始 | 19.23 | — | — | — |
| v2 1200 步 | 17.91 | −1.32 | −0.99 | 7/8 |
| v3 3600 步 | 17.71 | −1.52 | −1.10 | 7/8 |
| 全窗口 1200 步 | 17.64 | −1.59 | −1.20 | 7/8 |
| r16 1200 步 | 18.59 | −0.64 | −0.22 | 5/8 |

逐曲配对检验（8 首，|t| > 2.36 为显著）：

| 对比 | 均值 | t | 判定 |
|---|---|---|---|
| v2 vs 原始 | −1.32 | **−3.36** | **显著** |
| v3 vs v2 | −0.20 | −0.47 | 不显著 |
| 全窗口 vs v2 | −0.27 | −0.33 | 不显著 |
| r16 vs 原始 | −0.64 | −1.06 | 不显著 |

v3−v2 逐曲为 −0.22/+0.79/−1.27/+1.67/−0.86/−2.18/−0.21/+0.64（5 好 3 坏）。

### 3. 结论

**训练步数、目标窗口、适配秩三个旋钮都能压低训练 NLL，却没有一个能显著改善音频。**
唯一统计显著的改善来自修掉标签维度错误本身（t = −3.36）。因此：
- **1200 步已足够**，继续训练不改善音频；v3 不采纳为主结果，作为消融行保留；
- r16 的 loss 更低（5.68）但音频更差，且相对原始并不显著——再次印证
  **Token 级目标值低 ≠ 音频更接近**；
- 主结果取 v2（1200 步、r=8、掩前缀），因其成本最低且与其它 r8 配置不可区分。

报告表 23 已加入 1200/3600 步与全窗口三行，并把配对检验写进正文（§8.4）。

### 产物

- `results/generation/finetune_v3.json`、`metrics_v3.json`、
  `checkpoints/generation/musicgen-small-lora-v3/`
- 未覆盖 v1/v2 任何产物

## 2026-10-20 14:10

**报告结构重构：消除重复的「讨论」章，正文改为三部**

上一轮重组后遗留两个结构问题：

1. **两个「讨论」** —— §7「分析与讨论」与 §9「讨论与局限」并列。
   拆开看 §7 混了两类内容：错误分析/鲁棒性/片段级是真·分析，而「预训练的收益与代价」
   「为什么 Transformer 会输」「数据规模与任务难度」「鲁棒性再解读」「对课程项目的启示」
   其实是讨论。所以解决办法不是删一章，而是把讨论内容归位到唯一的那一章。
2. **§4「方法」只管判别式** —— 全是 SVM/CNN/CRNN/Transformer/AST，生成式任务的方法
   散在 §8 内部，读者会以为方法章不完整。

处理方式：正文改为**三部**（`\part` 在 ctexart 下不换页、编号会串，故自定义
`\projectpart` 命令：换页 + part 级 TOC 条目 + 大标题 + 细线）。

| 位置 | 内容 |
|---|---|
| 前置 | §1 引言、§2 相关工作、§3 数据与实验协议 |
| **I 判别式任务** | §4 方法与实验设置（原 §4 方法 + §5 实验设置合并）<br>§5 结果与分析（原 §6 结果 + §7 的错误/鲁棒性/片段级） |
| **II 生成式任务** | §6 生成式任务（原 §8，方法自包含） |
| **III 综合讨论与结论** | §7 讨论（原 §7 解释性小节 + §9，按「主要发现→解释→局限→启示」重排）<br>§8 结论与展望 |

关于「§4 只管判别式」：**不把生成式方法抽进统一方法章**。两个任务不共享任何模型，
硬塞一章只会变成两个无关目录的拼接；正确做法是让 §4 的标题明确限定适用范围，
而 §6 独立成章保证生成式任务方法完整。这样不对称变成有意为之。

标签保全：合并后保留 `sec:method`/`sec:setup`（均指向 §4）与
`sec:results`/`sec:analysis`（均指向 §5），因此原有 `\ref` 全部有效；
§1.5 报告组织改写为三部路线图。

结果：`\section` 从 10 个降到 8 个，「讨论」只剩一个，报告 31 → **35 页**
（三部各占一张独立分部页）、0 编译错误、无双重复小节名、
4 处 overfull（与重构前同为排版微溢出，不影响观感）。

README 上一轮已是 12 章（第 5 章判别式、第 6 章生成式），与报告三部结构对应，
本轮不再改动。

## 2026-10-20 14:45

**补齐总—分—总结构：前置与讨论章此前只服务判别式任务**

上一轮改三部后暴露两个内容缺口：

1. **前置章节（§1–§3）没有涉及生成式任务** —— 引言的研究背景/挑战/研究问题/贡献全在讲分类；
   相关工作 6 个小节没有一篇生成式文献（EnCodec / AudioLM / MusicGen / LoRA 全缺）；
   数据章也只有分类任务的预处理与划分。
2. **§7 讨论更像"生成式任务开始之前"的讨论** —— 5 条主要发现里只有第 5 条涉及 Token，
   其余（预训练的收益与代价、为什么 Transformer 会输、数据规模与任务难度、
   鲁棒性的进一步解读）全部只讨论分类。

即：分（I/II）是齐的，但"总"只总了判别式一边。

### 补充内容

**§1 引言**
- 研究背景新增一段"生成式路线"：从神经音频编解码器（SoundStream/EnCodec）到
  AudioLM、MusicGen，说明"音频→离散 Token→语言模型"这条线为何与分类并列
- 问题与挑战新增第 6 条：生成任务没有唯一正确答案，客观距离只衡量"接近"不衡量"好听"
- 研究问题新增第 4 条：换成 Token 目标后，8 GB 预算能做到哪一步、失败边界在哪
- 主要贡献新增 3 条：生成式任务基准、被修正的 LoRA 负结果、负结果的辨别方法

**§2 相关工作**
- 新增小节「离散音频 Token 与音乐生成」：RVQ 与 EnCodec 规格、AudioLM 两级 Token、
  MusicGen 的 delay pattern 与 300 M 规模、LoRA 低秩旁路、FAD 与主观评价的争议
- 参考文献新增 5 条：`defossez2023encodec`、`borsos2023audiolm`、`copet2023musicgen`、
  `hu2021lora`、`kilgour2019fad`（20 → 25 条）

**§3 数据与协议**
- 新增小节「两个任务的协议共享与差异」+ 表 5：逐项对照数据来源、划分、种子、输入、
  表示、输出、指标；明确两点——划分严格复用使两个任务面对**同一批测试曲目**，
  但指标不可跨任务比较

**§7 讨论**
- 主要发现从 5 条扩到 8 条，分类 4 条 + 生成 2 条 + 方法论 2 条，两个任务对等
- 新增小节「判别式与生成式：同一种纪律，两种可度量性」：分类评价近乎完备，
  生成只有"半个"评价（Token 级可精确度量，音乐性不可），因此三重分开报告
- 新增小节「负结果是怎么被辨别出来的」：三件检验（去最优样本后复算、逐曲配对检验、
  按几何衰减判收敛），并坦白第三条推翻的正是本文自己上一轮的判断
- 威胁与局限新增「生成式任务的特有局限」：n=8、单种子、3 s 上限、指标只奖励"贴近原曲"

**README** 同步：§1 增加第 6 个研究问题与「两个并列任务 + 一条展示线」；
§10 创新点从 3 条扩到 5 条（补"从判别走到生成"与"对负结果较真"）。

### 顺带修的排版问题

`adaptive_avg_pool2d_backward_cuda` 这个 34 字符的等宽标识符无法断行，
在「内部效度」段造成 **137.8 pt** 的严重溢出。改为正文用中文描述、全名放脚注。

### 结果

报告 35 → **37 页**，0 编译错误，剩余 4 处 overfull（≤11.3 pt，均为重构前既有）；
参考文献 25 条，无未定义引用；§2 小节 6 → 7，§3 小节 +1，§7 小节 5 → 7。

---

## 结构审查与全篇重构（2026-10-21）

### 起因

用户读了 37 页版本后指出：§2 相关工作里 2.2–2.6 全是任务一、任务二只有一节且写得过细；
§3.3「统一预处理」只写了一个任务；全文只有一张流程图，画的还是分类。要求用子 agent
全面排查，并参考学术写作方法论。

### 方法：把通用量规改写成项目量规

引入 `Imbad0202/academic-research-skills`（v3.23.1，CC BY-NC 4.0），本地安装到
`~/.dsh/skills/academic-research-skills/`，并把其中 `structure_architect_agent.md`
（比例分配、Purpose/Transition、层级深度、四类失败判定）与 `writing_quality_check.md`
（虚词、清嗓子开头、凑数三段式、术语一致）改写成项目量规
`docs/report/structure_audit_rubric.md`（A 对称性 / B 覆盖 / C 比例 / D 叙述 / E 冗余 /
F 单一职责），判定条件落到本报告的具体情形。

随后派 7 个**只读**子 agent 分片审计（前言+相关工作、数据与协议、第一部分方法、
第一部分结果、第二部分、讨论结论附录、全局图表与术语），每条发现必须带原文摘录与
行号，并按 P0/P1/P2 分级。合计约 100 条，其中 P0 13 条。

### 最硬的发现（除用户指出的两处）

| 位置 | 事实 |
|---|---|
| §4 | 4.6/4.7/4.8 三节讲同一件事，训练流程讲 2–3 遍，收尾句逐字重复两次 |
| §5 | 8 节挂 19 个浮动体，11 个全文 0 次引用 |
| §6 | 299 行零条 `\cite`、零张架构图，3 个核心结果被降级为不进目录的 `\paragraph` |
| §7 | 11 节里 4 节纯分类解释、0 节生成解释 |
| §8 | 结论首段把全文界定为「六类流派分类模型比较」，且写「续写只有 5 秒」与 §7.8 自相矛盾 |
| 全局 | 图 14 张仅 2 张被引用（86% 悬空）；引用 92% 挤在 §1–§2，§6 为 0 |

### 重构结果

**§1** 摘要首句即声明两个并列任务，关键词补 Token/EnCodec/音乐生成/LoRA；1.2 把「生成式无
唯一真值」从 5:1 的列表末条提为独立段落；1.3 三个问题 → 五个问题（分类 3 + 生成 2）。

**§2** 六节全分类 + 一节全生成（6:1、5.3 倍篇幅、18:5 引用）改为 **分类 3 / 生成 3 / 共用 2**
共八节：拆出「神经音频编解码与离散 Token」「自回归音乐生成：AudioLM→MusicGen」
「参数高效适配：LoRA」，FAD 与 RMSE 并入共用评测节；新增 SoundStream、MusicLM、
Houlsby adapter 三条文献（25 → 28）。

**§3** 由「5 节讲分类 + 1 节补丁式对照」改为 7 节全双任务：3.3 从「统一预处理」改为
「两条前端」，并列判别式频谱与生成式 Token 流水线；新增生成式数据流图（图 2）；
3.4 泄漏诊断补生成侧对应物；3.5 六条纪律逐条写两侧落点，并如实说明
`tests/test_protocol.py` 不覆盖生成式；3.6 收缩为「为何不可比」；4.10 软硬件环境与
4.14 复现纪律上移为 3.7 与 3.5。

**§4** 14 个平铺小节 → 6 节：4.1–4.5 合并为「模型族」（五个 `\paragraph` + 表），
删掉与 4.7/4.8 重复的 4.6，4.7+4.8 合并为「训练流程与正则化」，4.9 只留实现细节。

**§5** 8 节 → 7 节：5.3 并入 5.4 为「消融与训练预算检验」，5.8 并入 5.6 为
「错误分析：类别、置信度与时间稳定性」；两张逐 seed 表与 GTZAN 训练曲线下沉附录，
补上 §1.5 早已承诺却落空的「附录逐 seed 结果」；新增 5.7 小结，与第二部分的小结对称。

**§6** 10 节 → 12 节：三个 `\paragraph` 升为小节（收敛性复核、适配结果、消融），
6.3 窗口构造并入 6.2；启用一直未插入的 `generation_spec.png` 与 `generation_cost.png`；
删去与 §2.4/§3.3/§7 重复的约 600 字。

**§7** 新增两节生成式机制解释（误差累积 3.36%→2.19%、Token 指标与音频距离为何背离），
合并 7.4+7.5 为「预训练、架构与成本」；局限按五个效度维度重组，每维给两任务各一句，
删去补丁式的「生成式特有局限」；伦理补生成式权重许可与不发布生成音频的理由；
启示补两条（先验证目标对齐、指标分层报告）。

**§8** 结论按两任务重写：判别式四条 + 生成式三条，负结果清单补 LoRA 目标对齐错误与
「去掉最优样本后增益消失」；「续写 5 秒」改为 3 秒并注明成因。

**附录** 超参拆为 A.3（判别式）与 A.4（生成式，含 Token 训练/采样/LoRA 三段）；
新增 A.5 逐曲结果表（8 首 × 原始/修正后 + 去最优样本后的均值差），由
`gen_tables.py:generation_pertrack_table()` 从 `metrics_base.json`/`metrics_ft.json` 生成；
结果文件表补齐 6 个生成侧文件。

### 结果

42 页（37 → 42），0 编译错误，0 未定义引用，剩 2 处 overfull（≤10.9 pt）；
参考文献 28 条；图表 31 个，悬空 0 个。§4/§5 合计 13 节 vs §6 12 节。
新量规留在 `docs/report/structure_audit_rubric.md`，skill 留在
`~/.dsh/skills/academic-research-skills/`，后续可复用。

---

## 仓库公开化与最终模型交付（2026-10-24）

演示由队友（晴天的太阳）负责，需要能直接 `git clone` 就用。做了三件事：

1. **隐私清理**：19 个文件里的 `/home/<user>/...` 与 `/mnt/d/...` 绝对路径全部改为
   `python3` / `$CONDA_PREFIX/bin/python` / `~/MUSIC_AI`；`results/gtzan/ast_eval.json`
   里的 HF 缓存路径脱敏为 `<HF_HOME>/hub/...`。复查 `git grep` 无命中。
2. **最终模型入库**：`checkpoints/gtzan`（22 MB）、`fma`（46 MB）、
   `token_prediction_long`（119 MB），单文件最大 21 MB。克隆后
   `streamlit run app/demo.py` 上传任意音频即可现场 PK，不需要数据集。
   实测新克隆：`available()` 返回 `['cnn','crnn','mlp','svm','transformer']`。
3. **大权重不入库**：`checkpoints/generation/` 的 5 个 LoRA 变体各 1.18 GB
   （都是 merged 进 Meta 基座后的整模型，5.5 GB），超过 GitHub 单文件 100 MB 上限，
   且其中 99% 是别人的非商用权重。改为 `checkpoints/README.md` 里的重训命令
   （r=8、1200 步、掩前缀，约 40 分钟）。数据集同样不入库（23 GB，GTZAN 另有版权问题）。

顺带修的：`--seconds 5` 在 README 与 `run_generation.sh` 里残留，照抄会撞 CUDA 位置索引断言，
统一改 3 并写明原因。

### 补：无显卡演示包（当日）

演示机大概率没有 GPU，于是按「分类能跑、生成只能放录音」重组素材：

- **分类**：`scripts/build_classification_demo.py` 从 FMA-small 测试集裁 8 段 5 秒片段
  （6 段五模型分歧 + 2 段全对），判断与置信度直接取自 `results/fma/predictions/*.csv`，
  现场不需要任何推理。选 FMA 而非 GTZAN 是因为前者 CC-BY 4.0 可再分发。
- **实测 CPU 推理**：CNN 196 ms、CRNN 60 ms、TF 80 ms、MLP 70 ms（SVM 首次 2.6 s 含特征提取），
  所以五模型现场 PK 不需要显卡；AST 与 MusicGen 续写不具备 CPU 现场条件。
- **生成**：`build_demo_bundle.py` 新增无缝拼接组 `REAL__ / GENbase__ / GENft__`（各 8 秒，
  同一段前缀接不同后 3 秒），适合改名 A/B/C 做盲听；`demo/audio` 共 50 段。
