# 演示音频包

由 `scripts/build_demo_bundle.py` 从 `results/generation/audio/` 与
`results/token_prediction/audio/` 生成，**不是重新合成的**——每个文件都对应报告里的一次实测。

所有片段只留 1\,dB 余量（`volume=-1dB`），**不做响度归一**。早期版本用过
`loudnorm I=-18 LUFS`，它把 5 秒的短片段压低约 11\,dB，听感就是「前 5 秒几乎没声音」。
保留相对音量是有意的：真实录音的前缀本来就比续写段落安静一些。

## 一、前缀续写（3 秒）

| 文件 | 内容 |
|---|---|
| `000010_Pop__prompt.mp3` | 000010_Pop — 5 秒前缀（模型输入） |
| `000010_Pop__true.mp3` | 000010_Pop — 真实后 3 秒（参考答案） |
| `000010_Pop__base.mp3` | 000010_Pop — MusicGen-small 原始续写 |
| `000010_Pop__ft.mp3` | 000010_Pop — LoRA 适配后续写 |
| `000182_Rock__prompt.mp3` | 000182_Rock — 5 秒前缀（模型输入） |
| `000182_Rock__true.mp3` | 000182_Rock — 真实后 3 秒（参考答案） |
| `000182_Rock__base.mp3` | 000182_Rock — MusicGen-small 原始续写 |
| `000182_Rock__ft.mp3` | 000182_Rock — LoRA 适配后续写 |
| `000704_International__prompt.mp3` | 000704_International — 5 秒前缀（模型输入） |
| `000704_International__true.mp3` | 000704_International — 真实后 3 秒（参考答案） |
| `000704_International__base.mp3` | 000704_International — MusicGen-small 原始续写 |
| `000704_International__ft.mp3` | 000704_International — LoRA 适配后续写 |
| `000814_Folk__prompt.mp3` | 000814_Folk — 5 秒前缀（模型输入） |
| `000814_Folk__true.mp3` | 000814_Folk — 真实后 3 秒（参考答案） |
| `000814_Folk__base.mp3` | 000814_Folk — MusicGen-small 原始续写 |
| `000814_Folk__ft.mp3` | 000814_Folk — LoRA 适配后续写 |
| `000995_Experimental__prompt.mp3` | 000995_Experimental — 5 秒前缀（模型输入） |
| `000995_Experimental__true.mp3` | 000995_Experimental — 真实后 3 秒（参考答案） |
| `000995_Experimental__base.mp3` | 000995_Experimental — MusicGen-small 原始续写 |
| `000995_Experimental__ft.mp3` | 000995_Experimental — LoRA 适配后续写 |
| `001482_Electronic__prompt.mp3` | 001482_Electronic — 5 秒前缀（模型输入） |
| `001482_Electronic__true.mp3` | 001482_Electronic — 真实后 3 秒（参考答案） |
| `001482_Electronic__base.mp3` | 001482_Electronic — MusicGen-small 原始续写 |
| `001482_Electronic__ft.mp3` | 001482_Electronic — LoRA 适配后续写 |
| `007481_Hip-Hop__prompt.mp3` | 007481_Hip-Hop — 5 秒前缀（模型输入） |
| `007481_Hip-Hop__true.mp3` | 007481_Hip-Hop — 真实后 3 秒（参考答案） |
| `007481_Hip-Hop__base.mp3` | 007481_Hip-Hop — MusicGen-small 原始续写 |
| `007481_Hip-Hop__ft.mp3` | 007481_Hip-Hop — LoRA 适配后续写 |
| `014568_Instrumental__prompt.mp3` | 014568_Instrumental — 5 秒前缀（模型输入） |
| `014568_Instrumental__true.mp3` | 014568_Instrumental — 真实后 3 秒（参考答案） |
| `014568_Instrumental__base.mp3` | 014568_Instrumental — MusicGen-small 原始续写 |
| `014568_Instrumental__ft.mp3` | 014568_Instrumental — LoRA 适配后续写 |

## 二、A/B 串联（一条听完四段，适合直接插幻灯片）

| 文件 | 内容 |
|---|---|
| `AB__000010_Pop.mp3` | 000010_Pop — 【串联】前缀 → 真实后 3 秒 → 原始续写 → LoRA 续写（每段间隔 0.6s） |
| `AB__000995_Experimental.mp3` | 000995_Experimental — 【串联】前缀 → 真实后 3 秒 → 原始续写 → LoRA 续写（每段间隔 0.6s） |
| `AB__001482_Electronic.mp3` | 001482_Electronic — 【串联】前缀 → 真实后 3 秒 → 原始续写 → LoRA 续写（每段间隔 0.6s） |

## 二·B、无缝拼接（前缀 + 3 秒，无静音间隔）

同一段 5 秒前缀，分别接上「真实后 3 秒」「MusicGen 原始续写」「LoRA 续写」。
**最适合做 A/B 盲听**：三条都以同一段音乐开头，只有后半段不同。

| 文件 | 内容 |
|---|---|
| `REAL__000010_Pop.mp3` | 000010_Pop — 【无缝】5 秒前缀 + 真实后 3 秒 |
| `GENbase__000010_Pop.mp3` | 000010_Pop — 【无缝】5 秒前缀 + MusicGen 原始续写 |
| `GENft__000010_Pop.mp3` | 000010_Pop — 【无缝】5 秒前缀 + LoRA 适配后续写 |
| `REAL__000995_Experimental.mp3` | 000995_Experimental — 【无缝】5 秒前缀 + 真实后 3 秒 |
| `GENbase__000995_Experimental.mp3` | 000995_Experimental — 【无缝】5 秒前缀 + MusicGen 原始续写 |
| `GENft__000995_Experimental.mp3` | 000995_Experimental — 【无缝】5 秒前缀 + LoRA 适配后续写 |
| `REAL__001482_Electronic.mp3` | 001482_Electronic — 【无缝】5 秒前缀 + 真实后 3 秒 |
| `GENbase__001482_Electronic.mp3` | 001482_Electronic — 【无缝】5 秒前缀 + MusicGen 原始续写 |
| `GENft__001482_Electronic.mp3` | 001482_Electronic — 【无缝】5 秒前缀 + LoRA 适配后续写 |

## 二·C、整曲与更长的续写样例

主结果统一用 3 秒续写（这是 8 首测试曲都能稳定跑完的口径）。
下面这几条是**给人听**的补充：整曲 30 秒让你听到音乐本身，
`LONG__` 是我们在 000995 这一条上额外跑的长样例（4/6/8/12 秒都试过）。

| 文件 | 内容 |
|---|---|
| `FULL30__000010_Pop.mp3` | 000010_Pop — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__000182_Rock.mp3` | 000182_Rock — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__000704_International.mp3` | 000704_International — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__000814_Folk.mp3` | 000814_Folk — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__000995_Experimental.mp3` | 000995_Experimental — 【整曲】原始 30 秒（人听的上下文） |
| `LONG__000995_Experimental__base8s.mp3` | 000995_Experimental — 【长样例】5 秒前缀 + 原始模型 续写 8 秒 |
| `LONG__000995_Experimental__lora8s.mp3` | 000995_Experimental — 【长样例】5 秒前缀 + LoRA 续写 8 秒 |
| `LONG__000995_Experimental__base12s.mp3` | 000995_Experimental — 【长样例】5 秒前缀 + 原始模型 续写 12 秒 |
| `FULL30__001482_Electronic.mp3` | 001482_Electronic — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__007481_Hip-Hop.mp3` | 007481_Hip-Hop — 【整曲】原始 30 秒（人听的上下文） |
| `FULL30__014568_Instrumental.mp3` | 014568_Instrumental — 【整曲】原始 30 秒（人听的上下文） |

## 三、Token 预测（1 秒）

| 文件 | 内容 |
|---|---|
| `TOK__001482_Electronic__pred.mp3` | 001482 Electronic — 模型预测的 1 秒（1 秒） |
| `TOK__001482_Electronic__true.mp3` | 001482 Electronic — 真实的 1 秒（1 秒） |
| `TOK__000995_Experimental__pred.mp3` | 000995 Experimental — 模型预测的 1 秒（1 秒） |
| `TOK__000995_Experimental__true.mp3` | 000995 Experimental — 真实的 1 秒（1 秒） |
| `TOK__000010_Pop__pred.mp3` | 000010 Pop — 模型预测的 1 秒（1 秒） |
| `TOK__000010_Pop__true.mp3` | 000010 Pop — 真实的 1 秒（1 秒） |

## 四、每条曲目的客观距离（Log-Mel RMSE，越低越好）

| 曲目 | 流派 | 原始 | +LoRA | 差值 | 讲什么 |
|---|---|---|---|---|---|
| 000010 | Pop | 12.90 | 11.51 | -1.39 | 典型样本：小幅稳定改善 |
| 000995 | Experimental | 20.85 | 17.19 | -3.65 | 改善最大：−3.65 dB，最适合现场听出差别 |
| 001482 | Electronic | 27.13 | 27.46 | +0.33 | **反而变差** +0.33 dB：8 首里唯一退步的一首，用来讲“指标不是万能的” |

全 8 首的逐曲表见技术报告附录 A.5（表 29）；均值 19.23 → 17.91，8 首中 7 首改善，
去掉改善最大的一首后仍有 −0.99 dB。

## 五、播放建议

1. **先放 `AB__000995_Experimental.mp3`**：听众能在 12 秒内听出「前缀 → 真实 → 原始 → 适配」
   四段的差别，这是全场最有说服力的一条。
2. 再放 `AB__001482_Electronic.mp3`：**这一条适配后更差**。主动放它，然后解释为什么
   （8 首样本、配对检验、指标只奖励贴近原曲），比只放好消息可信得多。
3. **盲听玩法**：连放 `REAL__` / `GENbase__` / `GENft__` 三条（各 8 秒、同前缀），
   让观众猜哪条是真实录音。三条文件名已暴露答案，现场可改名成 A/B/C 再放。
4. 若时间紧，只放 `AB__000995` 一条就够。

## 六、注意

- 这些都是**真实后 3 秒**可对照的样本，不是自由创作。别把它说成“AI 作曲”。
- 32 kHz 单声道原始采样，转 mp3 只为便于插入幻灯片；报告里的测量全部在 wav 上完成。
- 生成音频的版权与许可问题见技术报告 §7.10：**不要公开发布这批音频**，课堂演示可用。
