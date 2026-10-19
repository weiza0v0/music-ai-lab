# 阶段 0：二期扩展环境与数据审计

> 目的：在写任何二期代码之前，先确认环境、数据与既有产物，避免重复下载、覆盖一期结果，
> 或在一个跑不动的硬件假设上开始大规模训练。
> 审计时间：2026-10-19 19:05–19:20（Asia/Shanghai）。

## 1. 结论摘要

| 项目 | 结论 | 依据 |
|---|---|---|
| GPU | 可用，RTX 4060 Laptop **8 GB**（8188 MiB），空载 | `nvidia-smi` |
| CUDA / PyTorch | torch 2.12.1+cu130，`cuda_available=True` | 运行时检查 |
| Python | 3.12.13（conda `DataAnalysis`） | `sys.version` |
| 处理器 / 内存 | 16 核 / **7.9 GB** 内存（可用约 2.5 GB，特征化时需控制并发） | `nproc`, `free` |
| 磁盘 | D: 剩余 **1.4 TB** | `df -h` |
| EnCodec 32 kHz | 可加载、可编码、可解码，**帧率 50 Hz、4 码本、2048 词表** | 实测 encode/decode |
| 预处理音频 | FMA-small 7,994 首可解码，artist-disjoint 划分可直接复用 | `data/splits/fma/splits.csv` |
| MusicGen-small | 见第 6 节（先推理，再评估微调） | 阶段二 |

**判定：环境足以支撑阶段一（EnCodec Token 预测）与阶段二的原始模型推理**；
微调是否可行取决于第 6 节的实测量存/耗时，未验证前不开始。

## 2. 软件环境

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.12.1+cu130 | 一期已在用，不升级 |
| torchaudio | 2.11.0+cu130 | 音频 I/O 备选 |
| transformers | 5.16.1 | 提供 `EncodecModel`、`MusicgenForConditionalGeneration` |
| librosa | 1.0.0 | 解码 mp3、重采样、Log-Mel |
| soundfile | 0.14.0（libsndfile 1.2.2） | mp3 支持不完整，个别文件需 librosa 回退 |
| numpy / pandas / scipy | 2.4.6 / 3.0.3 / 1.18.0 | |
| scikit-learn | 1.9.0 | 仅一期使用 |
| matplotlib | 3.11.0 | 出图 |

缺失但二期可能需要：`peft`（LoRA）、`accelerate`、`audiocraft`。
处理原则：**优先用 transformers 已支持的路径**，不为一两个便利函数升级/降级 torch；
若阶段二必须用 `audiocraft`，另建独立 conda 环境并锁版本，不动现有环境。

### 2.1 已知环境坑（复核时复现）

- `no_proxy` 里同时含 `::1` 与 `[::1]` 会让 `httpx`（transformers 下载器）抛
  `InvalidURL: Invalid port: ':1]'`。解决：下载前 `export NO_PROXY=localhost,127.0.0.1`。
- HF 未登录会提示 `unauthenticated requests`，不影响下载速度（镜像可达）。
- WSL 下 librosa 解码个别 mp3 会触发 `libmpg123: dequantization failed`，
  该窗口记为失败并跳过，不污染统计。

## 3. 数据

| 项目 | 值 |
|---|---|
| 数据集 | FMA-small（8 流派 × 1,000 首，30 s，mp3） |
| 可解码 | 7,994 / 8,000（6 首上游校验失败） |
| 采样率 | 44.1k / 48k / 22.05k 混合，统一重采样到 **32 kHz 单声道** |
| 划分 | artist-disjoint 5,596 / 1,199 / 1,199（seed 42），可直接复用 |
| 许可 | FMA 音频 CC-BY 4.0（仅用于课程研究） |

阶段一的窗口设计（详见 `README` 第七节）：每首取 2 个 **6 s** 窗口
（前 5 s 前缀 + 后 1 s 目标），窗口继承曲目所属划分，**不同曲目的窗口不跨集合**。
预计 15,988 个窗口、每个窗口 4×300 个 Token。

## 4. 既有产物（不得覆盖）

| 目录 | 内容 |
|---|---|
| `results/gtzan/`、`results/fma/`、`results/fma_leak/`、`results/fma_ablation/` | 一期分类结果 |
| `checkpoints/gtzan/`、`checkpoints/fma/`、`checkpoints/fma_leak/` | 一期权重 |
| `docs/Music_AI_Lab_Technical_Report.pdf` | 技术报告（写作时为 24 页，二期完成后 30 页） |

二期使用独立目录 `results/token_prediction/`、`results/generation/`、
`checkpoints/token_prediction/`、`checkpoints/generation/`，与一期零交叉。

## 5. EnCodec 兼容性实测

| 项目 | 实测值 |
|---|---|
| 检查点 | `facebook/encodec_32khz` |
| 参数 | 56.9 M（权重约 228 MB） |
| 采样率 | 32,000 Hz，单声道 |
| 码本 | 4 个，每码本 2,048 词表 |
| 帧率 | **50 Hz**（1 s = 50 帧，6 s = 300 帧） |
| 编码输出 | `audio_codes` 形状 `[B, n_chunks, 4, T]`，取 `[:, 0]` 得 `[B, 4, T]` |
| 解码 | `audio_codes` → `audio_values [B, 1, T×640]`，往返一致 |
| 编码吞吐 | 8×6 s 一批 ≈ 0.34 s（fp32，8 GB 卡） |
| 显存 | 峰值约 0.33 GB（batch 2），batch ≥ 128 会 OOM |

结论：**编码器输出与解码器输入格式互操作，可支撑"预测 Token → 解码成音频"的链路**。
该结论只适用于 `facebook/encodec_32khz`，不外推到其他 EnCodec 检查点。

## 6. MusicGen-small 可行性（阶段二前置）

二期末补记：本节在阶段二启动时留作占位，实际在二期执行中完成测量，回填如下。

| 项目 | 实测值 |
|---|---|
| 检查点 | `facebook/musicgen-small`（32 kHz，EnCodec 4 码本） |
| 首次加载耗时 | **17.2 s**（含权重下载后的本地加载） |
| LoRA 权重加载耗时 | **5.8 s** |
| 峰值显存（3 s 生成） | **1,356–1,431 MB**（均值约 1,393 MB） |
| 峰值显存（LoRA 训练） | **1,295 MB** |
| 4 s 生成耗时 | **未实测**——超过约 150 帧（>3 s）会触发位置索引越界断言，故统一取 3 s |
| 3 s 生成耗时 | **3.0–4.2 s**（均值 3.7 s），含前缀编码 0.38–0.58 s |
| 前缀续写接口 | **可用**：`decoder.generate` 直连 + 显式 `decoder_start_token_id`，自带 delay pattern |

结论：**8 GB 卡足以推理，也足以做 LoRA 微调**；但受位置编码上限约束，
续写统一取 3 秒，该限制已如实写进报告与日志。



## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 8 GB 显存不足以微调 MusicGen | 先推理；微调只做 LoRA 且限时，OOM 即记录失败与日志，不改结论 |
| 内存 7.9 GB，特征化并发过高会 OOM | 编码阶段 10 worker + 小 batch；训练阶段单进程 |
| 显存被编码与生成同时占用 | 两阶段串行执行，不并行 |
| 长任务被主机重启打断 | 编码按 shard 断点续跑；训练按 epoch 存最优权重 |
| Token 预测准确率高 ≠ 音乐好听 | 两类结果分开报告，解码音频只作定性对照 |
