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
  续写受 3 秒长度限制。
