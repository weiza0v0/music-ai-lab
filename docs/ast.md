# AST 迁移学习（P1）：冻结 AudioSet 主干 + 线性分类头

> 本文件由 `src/train_ast.py` 在实验结束时自动生成，所有数字来自本次真实运行，
> 可追溯到 `results/gtzan/ast_metrics.json` 与 `results/gtzan/ast_eval.json`。
> 生成时间：2026-10-19T13:10:06+08:00；seed=42。

## 一句话结论

在 GTZAN 985 首、按曲目固定的 70/15/15 划分上，冻结 AST 主干、只训练 7,690 个参数（768→10 逻辑回归）后，测试集 n=147 的 Accuracy = **0.8231**，Macro-F1 = **0.8135**。相对 CNN：Accuracy +6.8 个百分点，Macro-F1 +6.0 个百分点。

注意：AST 使用了 AudioSet 预训练权重，属于「迁移学习组」，不能把收益直接解释为
Transformer 架构优于 CNN；它多花的是预训练数据、下载体积与推理时间。

## 权重来源

- HuggingFace 模型：`MIT/ast-finetuned-audioset-10-10-0.4593`
- 预训练数据：AudioSet（YouTube 音频片段，527 类，约 2M 片段）；
  模型卡报告 AudioSet mAP 0.4593，节点名中的 `10-10` 表示训练时 10 秒片段、10 个 patch。
- 架构：AST base，patch 16×16，12 层 Transformer，hidden=768，主干参数量 86,187,264。
- 授权：BSD-3-Clause（以模型卡为准）。
- 权重文件：`model.safetensors`，346.4 MB；首次运行下载到默认 HF 缓存 `~/.cache/huggingface/hub`。
- 本次加载模型耗时 6.6 s。

## 预处理

1. `librosa.load(..., sr=16000, mono=True)`，取每首曲目**中心 5 秒**（与 CNN 等模型的中心片段约定一致）。
2. 短于 5 秒补零；再在右侧补零到 **10.24 秒**（163840 个采样点 = 1024 帧 × 10 ms），以满足 AST 的固定输入长度。
3. `ASTFeatureExtractor`：16 kHz、128 Mel bins、n_fft=1024、hop=160、`do_normalize=True`（mean=-4.2677393，std=4.5689974），输出 `(1024, 128)`。
4. 主干输出取 `pooler_output`（768 维）= (CLS + distillation token) / 2，这正是原始 AudioSet 分类头使用的表征；不做任何微调。
5. 所有曲目的 embedding 先缓存到 `data/features/gtzan/ast_embeddings.npz`（git 忽略），后续实验直接复用。

## 冻结策略

- `model.eval()` + 所有参数 `requires_grad_(False)`；
- 前向统一放在 `torch.inference_mode()` 下，不构建计算图；
- 训练集只有 689 首，微调 86M 参数极易过拟合，因此只训练线性头；
- 线性头 = `StandardScaler`（**只在 train 上拟合**）+ `LogisticRegression`；
- C 在验证集上从 [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] 中按 Macro-F1 选择，选中 C=3；
- 测试集只在 C 与标准化器确定后评估一次。

## 结果

| 划分 | n | Accuracy | Macro-F1 |
|---|---|---|---|
| val（用于选 C） | 149 | 0.8591 | 0.8591 |
| test（只评一次） | 147 | **0.8231** | **0.8135** |

逐类 F1（test，按 F1 升序）：

| 流派 | F1 | 测试样本数 |
|---|---|---|
| rock | 0.417 | 15 |
| pop | 0.786 | 14 |
| reggae | 0.800 | 15 |
| blues | 0.812 | 15 |
| disco | 0.839 | 15 |
| metal | 0.839 | 14 |
| hiphop | 0.867 | 14 |
| country | 0.875 | 15 |
| jazz | 0.933 | 15 |
| classical | 0.968 | 15 |

最容易混淆的类别：rock → metal（3 首）；rock → blues（3 首）；rock → country（2 首）；reggae → pop（2 首）；reggae → disco（2 首）。混淆矩阵与逐类 F1 图见 `results/gtzan/figures/ast.png`，完整矩阵见 `results/gtzan/ast_eval.json`。

### 与 CNN baseline 对比

| 模型 | 预训练 | Accuracy (test) | Macro-F1 | 训练耗时 | 单片段推理 | 参数量 | 模型大小 |
|---|---|---|---|---|---|---|---|
| CNN（3 seeds，均值±标准差） | 无（从头训练） | 0.7551 ± 0.0068 | 0.7530 ± 0.0062 | 57.53 s | 0.76 ms | 592,554 | 2.39 MB |
| AST（seed 42，本实验） | AudioSet（冻结） | 0.8231 | 0.8135 | 23.42 s（仅分类头） | 204.06 ms | 86,194,954 | 346.45 MB |

CNN 数据取 `results/gtzan/metrics.csv` 中 3 个 seed 的均值±标准差；AST 只有 seed 42。
两者的训练耗时不可直接比较：CNN 是从头训练 30 epoch，AST 只训练线性头，但 AST 的 embedding 提取是一次性额外成本（见下）。

## 额外成本（真实测量）

- 权重下载/缓存：346.4 MB（safetensors）。
- 985 首 embedding 提取：213.9 s（约 217.1 ms/首，含 CPU 解码/重采样/特征化与 GPU 前向；本次重新计算）。
- 其中 Mel 特征化 16.9 s，GPU 前向 192.1 s；提取阶段峰值显存约 1140 MB。
- 分类头训练：23.42 s（9 个 C 的验证集网格搜索 + 最终 train 重拟合；最终重拟合 863 ms）。
- 单片段推理：204.1 ms（batch=1，中位数，n=28；包含特征化 + 主干前向 + 分类头，不含 wav 解码）。
- 参数量：总计 86,194,954，其中可训练 7,690（主干 86,187,264 全部冻结）。
- 模型大小：主干 346.4 MB + 分类头 51 KB = 346.45 MB。

## 局限

- **采样率域差异**：GTZAN 原始为 22.05 kHz，AST 要求 16 kHz。重采样会丢掉 8 kHz 以上的高频内容，对金属、古典等依赖高频音色的流派可能不利。
- **补零到 10.24 s**：每首只有 5 秒真实音频，其余是静音。AST 的位置编码与注意力会「看到」一半空白输入，和 AudioSet 的 10 秒连续片段不同。
- **预训练域差异**：AudioSet 来自 YouTube，含大量非音乐声音；GTZAN 是 30 秒、录音质量较统一的音乐片段，二者分布不同。
- **只做线性探针**：没有微调，也没有数据增强或多片段投票；这测量的是「冻结表征的线性可分性」，不是 AST 的上限。
- **单 seed**：AST 只跑了 seed 42；线性头本身方差小，但没有多 seed 的稳定性估计。
- **GTZAN 本身的问题**：985 首、10 类、存在标签噪声与重复曲目（已按 MD5 去重），绝对准确率不宜过度解读。
- **成本**：86.2M 参数、346 MB 权重、约 0.20 s/片段，与 CNN 的 0.59M 参数、2.4 MB、0.76 ms 相比，部署代价高 2–3 个数量级。

## 复现命令

```bash
cd /mnt/d/Desktop/MUSIC_AI
PY=/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python

# 首次运行会下载约 346 MB 权重到默认 HF 缓存；之后直接复用
$PY -m src.train_ast

# 忽略 embedding 缓存，强制重新提取（约 3–4 分钟，RTX 4060）
$PY -m src.train_ast --force-extract

# 查看验证集 C 网格与最终结果
cat results/gtzan/ast_metrics.json results/gtzan/ast_eval.json
```

脚本会写入：`data/features/gtzan/ast_embeddings.npz`（缓存）、`results/gtzan/ast_metrics.json`、`results/gtzan/ast_eval.json`、`results/gtzan/figures/ast.png`（如未加 `--no-figure`）和本文件；不会修改 `results/gtzan/metrics.csv`、README 或任何已有训练脚本。

## 环境与排障记录

- 环境：Python 3.12 conda `DataAnalysis`，torch 2.12.1+cu130，transformers 5.16.1，GPU RTX 4060 Laptop 8 GB。
- 首次加载 `AutoFeatureExtractor` 时遇到 `InvalidURL: Invalid port: ':1]'`。根因不是网络，而是本机 `NO_PROXY` 里含有 带方括号的 `[::1]`，`huggingface_hub 1.30` 解析该列表时出错；脚本在导入 transformers 前临时移除 `NO_PROXY/no_proxy`，代理仍正常使用。
- `AutoModel` 加载时会提示 `classifier.*` 为 UNEXPECTED：这是预期行为，因为加载的是 backbone `ASTModel`，不包含 AudioSet 的 527 类分类头。
