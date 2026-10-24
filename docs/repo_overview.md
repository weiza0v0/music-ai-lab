# 仓库概况

> 读者：第一次接触本仓库的人（或 agent）。目标是在 5 分钟内知道**这是什么、东西在哪、
> 怎么跑、哪些结论能说、哪些坑别踩**。
> 详细论证见技术报告 `docs/Music_AI_Lab_Technical_Report.pdf`；演示见 `docs/handoff_demo.md`。

---

## 1 一句话

在一个统一、可复现的协议下，用**同一份按艺术家分离的数据划分**做两件事：
**判别式**（音乐流派分类，GTZAN 985 首 + FMA-small 7,994 首，六个模型对比）与
**生成式**（EnCodec 离散 Token 预测 + MusicGen 前缀续写 + LoRA 适配）。

课程项目，但按论文的纪律做：三种子、验证集选型、测试集只评一次、负结果全部保留。

---

## 2 两个任务与最终结论

### 2.1 判别式：谁更强、贵多少

GTZAN（测试 147 首）/ FMA-small（测试 1,199 首），Macro-F1 为 3 种子均值：

| 模型 | GTZAN | FMA-small | 单片段延迟 | 权重体积 | 一句话 |
|---|---|---|---|---|---|
| SVM（人工特征） | 0.699 | 0.440 | 0.07 ms | 1.2 MB | 便宜；脏输入下最稳 |
| MLP（人工特征） | 0.630 | 0.422 | 0.02 ms | 1.1 MB | 比 SVM 更弱，种子波动大 |
| CNN | 0.753 | 0.470 | 0.76 ms | 2.4 MB | 干净输入上的自建冠军 |
| CRNN（CNN+GRU） | 0.723 | **0.492** | 0.76 ms | 1.7 MB | 数据够了才反超 CNN |
| 轻量 Transformer | 0.542 | 0.378 | 0.26 ms | 1.2 MB | 千首级严重过拟合 |
| **AST（冻结 AudioSet 主干）** | **0.814** | **0.590** | 204 / 231 ms | 346 MB | 预训练一出手就赢，但贵 ~270 倍延迟 |

三条值得讲的对比：
- **预训练是单点收益最大的因素**：比最好的自建模型高 6.1 / 9.8 个百分点。
- **自建模型的排序随数据规模翻转**：GTZAN 上 CNN 领先 CRNN 3 个点，FMA 上 CRNN 反超 2.2 个点。
- **泄漏能把结论整块翻转**：同一批特征，只把划分从「按艺术家」换成「按曲目」，
  CNN 的 Macro-F1 从 0.470 跳到 0.568（+9.9），81.0% 的测试曲与训练集共享艺术家。

### 2.2 生成式：能预测，但离"好听"很远

Token 预测（EnCodec 32 kHz，4 码本 × 50 Hz，5 秒前缀 → 1 秒目标）：

| 模型 | teacher-forced Top-1 | free-running Top-1 | NLL |
|---|---|---|---|
| 均匀分布 | 0.049% | 0.049% | 7.62 |
| 频率先验（训练集最高频 Token） | 1.16% | 1.16% | — |
| Persistence（重复上一帧） | 1.61% | 1.61% | — |
| Bigram（n-gram） | 1.71% | 1.71% | — |
| LSTM | 2.78% | 2.08% | 6.559 |
| Transformer（4 层） | **3.36%** | **2.19%** | **6.522** |

MusicGen-small 前缀续写（5 秒前缀 → 3 秒，8 首测试曲，Log-Mel RMSE 越低越好）：

| 配置 | RMSE | 相对原始 | 配对检验 |
|---|---|---|---|
| 原始模型 | 19.23 | — | — |
| + LoRA r=8，1200 步 | **17.91** | **−1.32** | **t = −3.36，显著** |
| + LoRA 3600 步 | 17.71 | −1.52 | t = −0.47，不显著 |
| + LoRA 全窗口 | 17.64 | −1.59 | t = −0.33，不显著 |
| + LoRA r=16 | 18.59 | −0.64 | t = −1.06，不显著 |

结论只有一句：**修好目标对齐之后，r=8 的 LoRA 能稳定改善客观距离（7/8 首改善，去掉
最优样本后仍有 −0.99）；步数、窗口、秩三个旋钮都不是有效旋钮。**

---

## 3 目录地图

| 路径 | 内容 | 入库 |
|---|---|---|
| `src/` | 全部代码：`data/`（清洗与划分）、`train_classical.py`、`train_cnn.py`、`evaluate.py`、`predict.py`、`timeline.py`、`generation/`（EnCodec Token、训练、MusicGen、LoRA） | ✅ |
| `app/demo.py` | Streamlit「Music AI 鉴定所」：上传音频 → 五个模型现场 PK + 滑动窗口时间轴 | ✅ |
| `scripts/` | 一键脚本与专项脚本：`run_all.sh`、`run_fma.sh`、`run_token.sh`、`run_generation.sh`、`overnight_queue.sh`、`build_demo_bundle.py`、`build_arena_table.py` 等 | ✅ |
| `configs/` | `dataset.yaml`、`baseline.yaml`、`experiments.yaml` | ✅ |
| `data/` | `raw/`（原始音频）、`features/`（缓存）、`splits/`（划分 CSV，唯一入库的部分） | ❌ 只入库 `splits/` |
| `checkpoints/` | **最终模型入库**：`gtzan/`（22 MB）、`fma/`（46 MB）、`token_prediction_long/`（119 MB）；`generation/` 的 MusicGen LoRA（5.5 GB）不入库，见 `checkpoints/README.md` | ✅ 仅最终模型 |
| `results/` | **全部结果**：`metrics.csv`、逐样本预测、混淆矩阵、图、`token_prediction/`、`generation/` | ✅（290 个文件） |
| `demo/` | 演示素材：`audio/`（56 段 mp3，含整曲 30 秒与 8/12 秒长样例）、`classification/`（8 段 5 秒 + 8 段 15 秒 + 五模型判断表）、`arena_pk.md`（擂台表） | ✅ |
| `docs/` | 技术报告（`report/` 源码 + PDF）、`research_log.md`（逐次实验日志）、`experiment_log.md`、数据卡、审计报告、`handoff_demo.md` | ✅ |

体积：仓库 `.git` 约 250 MB（含 187 MB 最终权重）；`data/` 23 GB 与 MusicGen 的 5.5 GB 合并权重**不入库**，
复现方式见 `checkpoints/README.md`。

---

## 4 环境

| 项 | 值 |
|---|---|
| 系统 | WSL2 Ubuntu 22.04 |
| Python | `$CONDA_PREFIX/bin/python`（3.12.13） |
| 框架 | torch 2.12.1+cu130、torchaudio 2.11.0、transformers 5.16.1、librosa 1.0.0 |
| GPU | RTX 4060 Laptop 8 GB |
| LaTeX | xelatex + ctexart |

跑任何涉及模型或 HF 下载的命令前，这三行基本是必需品：

```bash
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export NUMBA_CACHE_DIR=/tmp/numba_cache      # 仅在受写限制的沙箱里需要
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 权重已在本地缓存时用，避免联网卡住
```

---

## 5 怎么跑

### 只看结果 / 做演示（不需要权重与数据）

```bash
git clone https://github.com/weiza0v0/music-ai-lab.git
cd music-ai-lab          # 克隆下来约 66 MB，只要 demo/ 与 results/ 就够做演示

# 可选：重算演示素材（已提交，一般不用重跑）
python3 scripts/build_arena_table.py     # 结果 → demo/arena_pk.md（需 pandas）
python3 scripts/build_demo_bundle.py     # wav → demo/audio/*.mp3（需 ffmpeg）
python3 scripts/build_classification_demo.py   # FMA 测试片段 → demo/classification/（需 ffmpeg + 本地 data/）
```

这两个脚本只读 `results/` 里已入库的文件，**不需要 GPU、不需要 5.9 GB 权重**；
用系统 `python3`（实测 3.13 + pandas 2.3）就能跑，输出与仓库里的版本逐字节一致。

### 看技术报告

```bash
cd docs/report && bash build.sh          # 需 xelatex；产物 main.pdf（42 页）
```

`build.sh` 会先跑 `gen_tables.py`，**所有表格与正文数字都是从 `results/` 现算的**，
不是手抄——改一个结果文件，重编译即可让报告同步。

### 从零重跑全部实验

见 `README.md` §2 的 12 步（数据下载 → 特征缓存 → 基线 → FMA 主实验 → Token 预测 →
MusicGen → LoRA → 报告）。单次全流程在本机约 2–3 天，其中 EnCodec 全量编码与 LoRA
微调占大头。

### 现场 Demo

```bash
python3 -m streamlit run app/demo.py
```

判别式与 Token 预测的**最终权重已在仓库里**，上传一段音频即可现场出结果；
只有生成式续写需要先重训 LoRA（见 `checkpoints/README.md`）。

---

## 6 现在的状态

已完成：

- 判别式任务：6 模型 × 2 数据集 × 3 种子，含泄漏对照、正则化消融、训练预算消融、
  计算成本、错误分析、七种扰动鲁棒性、片段级时间轴。
- 生成式任务：Token 预测（5 个模型 × 3 种子）、解码回音频、MusicGen 前缀续写、
  LoRA 适配与三个旋钮消融、逐曲配对检验。
- 技术报告 42 页（总—分—总三部分），28 条参考文献，图表 31 个，
  0 编译错误、0 未定义引用、0 悬空图表。
- 演示素材：`demo/audio/` 56 段（前缀/真实/原始/适配、A/B 串联、无缝盲听、整曲 30 秒、
  8/12 秒长样例）、`demo/classification/` 16 段 + 判断表、擂台表、全部图。
  **分类 CPU 可现场跑；生成只能放录音，稳定口径是 3 秒（实测可到 12 秒）。**

未做（如被问到，如实说）：

- 更大规模的生成（长于 3 秒的续写）——解码器直连路径的位置索引上限；
- 主观听感评分的规模化评测（只有少量试听）；
- 生成式任务的多种子方差（只有单一种子，已在报告 §7.9 列为局限）；
- AST 的微调（只做了冻结主干的线性探针）。

---

## 7 已知的坑

1. **`audiocraft` 装不上**：它 pin 了 `torch==2.1.0` 与 `av==11.0.0`，后者没有 Python 3.12
   的 wheel（`Failed to build 'av'`）。本项目改用 `transformers` 的
   `MusicgenForConditionalGeneration` + `EncodecModel` 直连解码器。
2. **`generate()` 超过约 150 帧会 CUDA 断言**（位置索引越界）→ 续写统一截到 3 秒。
   不要为了"看起来更长"改成 5 秒，那会崩。
3. **`httpx.InvalidURL: Invalid port`**：`NO_PROXY` 只设大写不够，必须同时设小写
   `no_proxy`，否则 `[::1]` 会被当成端口。
4. **numba 报 `cannot cache function '__o_fold'`**：不是代码 bug，是 `~/.cache/numba`
   不可写；设 `NUMBA_CACHE_DIR` 即可。
5. **EnCodec 输出第一维是分段数不是批大小**：把 `audio_codes[:, 0]` 当批维会**静默丢样本**
   （每 512 个窗口只写 1 行），不报错。本项目踩过，见报告 §6.2。
6. **`MusicgenForCausalLM` 的标签布局**：`input_ids` 要 `(B·K, T)` 扁平、`labels` 要
   `(B, T, K)`；传错不报错，只会让 loss 停在偏高值上——这是本项目最贵的一个坑，
   完整复盘见报告 §6.10。

---

## 8 想快速了解"为什么"，读这几节

| 问题 | 报告章节 |
|---|---|
| 两个任务为什么能并排做、指标为什么不能比 | §3.6 图 5、§7.2 |
| 数据泄漏怎么毁掉结论 | §3.4 表 4 |
| 预训练到底赢在哪里、贵多少 | §7.6、§5.4 |
| 小 Transformer 为什么输 | §7.6 |
| 误差累积：3.36% 为什么掉到 2.19% | §7.4 |
| 为什么 NLL 降了音频没变好 | §7.5 |
| 负结果是怎么被识别出来的 | §7.3 |
| 全部限制 | §7.9 |
