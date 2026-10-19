# 夜间训练交接说明（2026-10-20 00:45 起）

## 这次在跑什么，为什么

复核发现**两类系统性欠训练**，都不是调参刷分，而是原有预算把模型停在了
"还在变好"的位置上：

### 1. LoRA 负结果的真正原因是维度 bug，不是欠训练

`MusicgenForCausalLM` 要求 `input_ids` 是 `(B*K, T)`、`labels` 是 `(B, T, K)`。
旧代码把 `(B, K, T)` 同时当两者传入，库内部按 `labels.shape[1]` 截取 logits，
loss 只在错位目标上算。实测对照（原始模型无训练）：

| 标签布局 | loss |
|---|---|
| 旧：`(B,K,T)` 直接当 labels | **11.08** |
| 修正：`(B,T,K)` | **5.46** |
| 均匀分布基准 ln(2048) | 7.62 |

旧记录的起始 loss 10.42 与此完全吻合——它高于随机，说明从第一步就没在对齐。

顺带修掉两个静默错误：
- `GENRE_TEXT` 缺 `Experimental` / `Instrumental` / `International` 三个 FMA 流派，
  会被降级成泛化的 `"music"`，等于丢条件信息；
- 文本条件缓存以 `genres[0]` 为键，同批其余样本会拿到上一批的整块张量（错配）。

### 2. 一期 Token 模型被早停砍早了

seed42 历史显示 val_nll 到最后一轮仍在降：

| 模型 | ep10 | ep11 | ep12（停） |
|---|---|---|---|
| LSTM | 6.8146 | 6.8091 | **6.8060** |
| Transformer | 6.7249 | 6.7157 | **6.7127** |

`--epochs 15` + `--patience 3` + 只训 12 轮就停，明显没训够。

### 3. 部分分类模型的 best_epoch 撞在上限

| 数据集 | 模型 | best_epoch | 实际轮数 |
|---|---|---|---|
| GTZAN | CNN seed123 | **30** | 30 |
| GTZAN | CRNN seed42 | **30** | 30 |
| FMA | CRNN seed42 | **30** | 30 |

最后一轮仍在改善 → 加长到 60 轮。

## 怎么跑的

```bash
# 已 nohup 挂起，日志在 results/logs/overnight/
bash scripts/overnight_queue.sh
ONLY=token_long bash scripts/overnight_queue.sh   # 只跑某一项
```

队列串行抢 GPU，每项有哨兵文件，已完成的自动跳过，可重复执行。
**结果写进 `*_long` 目录，不覆盖报告在用的数字**，便于对照后再决定采纳。

作业顺序（优先级从高到低）：
1. `lora_v2` — 修正后的 LoRA 训练（已由外部进程先跑）
2. `eval_v2` — 用同一批前缀重评 base vs ft
3. `token_long` — LSTM/Transformer × 3 seeds × 45 epochs
4. `gtzan_long_{cnn,crnn,transformer}` + `gtzan_long_eval`
5. `fma_long_{cnn,crnn,transformer}` + `fma_long_eval`
6. `lora_r16` — rank 16 消融
7. `lora_full` — 不掩前缀（全窗口 loss）对照

预计总时长约 10 小时，7 点时应完成前 4 项左右。

## 验收时看什么

```bash
cd /mnt/d/Desktop/MUSIC_AI
cat results/logs/overnight/status.tsv          # 每项 OK/FAIL
cat results/logs/overnight/summary.md          # 汇总（队列结束时生成）
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -af overnight_queue                      # 还在跑吗
```

### 判断标准

**LoRA v2**
- `results/generation/finetune_v2.json` 的 `history` 末值：**显著低于 7.62** 才算学到了；
- 对比 `finetune_v1_negative.json`（旧：10.42 → 7.65，等于均匀分布）；
- `metrics_ft.json` 的 `logmel_rmse` 均值 vs `metrics_base.json` 的 19.23：
  需**且**去掉单曲后仍成立（旧版去掉 `000995` 后只剩 −0.16 dB，是噪声）；
- 逐曲看是否还有多数变差。

**Token long**
- `results/token_prediction_long/metrics.csv` 对比 `results/token_prediction/metrics.csv`；
- 重点看 `free_top1` / `nll`，以及 `best_epoch` 是否还撞在 45 上限（若撞上说明仍未训够）。

**分类 long**
- 对比 `best_epoch` 是否远离 60、test `macro_f1` 是否提升；
- 若提升 < 1 pp，说明原预算其实够用，如实写结论，不要硬凑"改进"。

### 铁律

- 负结果照实写。修好维度 bug 后若 LoRA 仍然没赢，就报告"修正后仍无增益"，
  并说明是哪一层限制（数据量/步数/3 秒长度）；
- 不覆盖报告正在引用的 v1 数字。旧产物已备份：
  `finetune_v1_negative.json`、`metrics_ft_v1_negative.json`、
  `audio/*_ft_v1.wav`；
- 采纳新数字进报告前，必须确认是**同协议**（同划分、同 seed、同评估口径）。

## 已改动的文件

- `src/generation/finetune_lora.py` — 修标签维度、补 3 个流派文本、
  修文本缓存错配、加 `--batch-size` / `--mask-prefix` / `--result-name`
- `scripts/overnight_queue.sh` — 新增夜间队列
