# 权重说明

仓库只提交**最终模型**，中间实验与生成式基座不入库（GitHub 单文件 100 MB 上限，基座一个变体就 1.18 GB）。

## 已入库（clone 下来就能用）

| 目录 | 内容 | 体积 | 用途 |
|---|---|---|---|
| `gtzan/` | SVM / MLP / CNN / CRNN / Transformer × 3 种子，AST 特征与线性头 | 22 MB | 判别式任务的 GTZAN 全部模型 |
| `fma/` | 同上，FMA-small 版 | 46 MB | 判别式任务的主实验模型 |
| `token_prediction_long/` | LSTM、Transformer × 3 种子（45 轮，报告采用版本） | 119 MB | 生成式任务的 Token 预测模型 |

单个文件最大约 21 MB，都在 GitHub 限制内。有了这三个目录，
`app/demo.py`（音乐 AI 鉴定所）**上传一段音频就能跑**，不需要数据集：

```bash
python3 -m streamlit run app/demo.py
```

## 未入库（按需复现）

| 内容 | 体积 | 为什么不入库 | 怎么得到 |
|---|---|---|---|
| `generation/musicgen-small-lora*` | 每个 1.18 GB | 是我们的 LoRA 增量**合并进 Meta 基座**后的整模型，5 个变体共 5.5 GB，远超单文件上限 | 见下 |
| `gtzan_long/`、`fma_long/` | 各 16 MB | 训练预算消融（30 轮 vs 60 轮）的中间结果，报告已有数字 | `bash scripts/overnight_queue.sh` |
| `fma_leak/`、`fma_ablation/` | 2.3 MB / 空 | 泄漏对照与正则化消融 | `scripts/run_fma_extra.sh` |

### 复现最终 LoRA 适配（报告采用的那一个）

先确保本地有 `facebook/musicgen-small`（约 5.5 GB，会自动从 Hugging Face 下载），然后：

```bash
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
PYTHON=<你的解释器> bash -c '
python -m src.generation.finetune_lora \
  --steps 1200 --rank 8 --alpha 16 \
  --batch-size 2 --mask-prefix \
  --out musicgen-small-lora-v2 --result-name finetune_v2.json'
```

- 8 GB 显存（RTX 4060 Laptop）上约 40 分钟，脚本自带 `--max-minutes 45` 兜底；
- 参数就是「三个旋钮都不如修好对齐重要」里那个**唯一显著**的配置（r=8、1200 步、掩前缀）；
- 复现后可直接用：

```bash
python -m src.generation.run_musicgen \
  --model checkpoints/generation/musicgen-small-lora-v2 \
  --tag ft --n 8 --seconds 3
```

### 如果只想听效果、不想训

`results/generation/audio/` 里已经存了当时跑出来的 wav（8 首测试曲 × 前缀/真实/原始/适配），
`demo/audio/` 里是转好的 mp3。**演示不需要权重。**

## 为什么不再分发整模型

`musicgen-small-lora-v2` 里 99% 的字节是 Meta 的 MusicGen-small（CC-BY-NC 4.0），
我们训的只有 2.36 M 参数的 LoRA 增量（fp16 约 4.7 MB）。要把整模型传上 GitHub
Release 或 Hugging Face 技术上可行，但那是「再分发别人 1.18 GB 的非商用权重」，
而我们的增量本身就是 `W' − W`、秩 ≤ 8，用重训命令复现比传输更干净。
详见技术报告 §7.10 与 §6.9。
