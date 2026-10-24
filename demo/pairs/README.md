# 原曲 / 生成曲 对照包

每个文件的名字就是说明：开头数字表示播放顺序，后面是「曲目号_流派」与这一段是什么。

**播放顺序**：1 听原曲 → 2 听真实后续 → 3 听原始模型续写 → 4 听 LoRA 续写。
2/3/4 三条的**前 5 秒完全相同**（同一段前缀），只有后面的 3 秒不同。

| 曲目 | 流派 | 艺术家 — 曲名 | 原始模型 RMSE | LoRA RMSE | 差值 |
|---|---|---|---|---|---|
| `000995` | Experimental | Kevin Shields — No Good Deed Goes Unpunished（改善最大，另有 8/12 秒长样例） | 20.85 | 17.19 | -3.65 |
| `000704` | International | Fósforo — Mano De Dios | 15.34 | 13.70 | -1.64 |
| `000010` | Pop | Kurt Vile — Freeway | 12.90 | 11.51 | -1.39 |
| `000814` | Folk | Hall Of Fame — Motion Passings | 29.23 | 28.04 | -1.19 |
| `000182` | Rock | Ariel Pink's Haunted Graffiti — Jules Lost His Jewels | 16.73 | 15.57 | -1.16 |
| `014568` | Instrumental | Ergo Phizmiz — Monkey Dance | 14.74 | 13.71 | -1.03 |
| `007481` | Hip-Hop | BrokeMC — Seeing You Out | 16.91 | 16.09 | -0.81 |
| `001482` | Electronic | Rainbro — Reindeer Dance（**反而变差**，主动放它更可信） | 27.13 | 27.46 | +0.33 |

RMSE 是「生成频谱与真实后 3 秒的距离」，越低越好；8 首均值 19.23 → 17.91。

## 文件清单

- `1_000010_Pop_原曲30秒.mp3`
- `1_000182_Rock_原曲30秒.mp3`
- `1_000704_International_原曲30秒.mp3`
- `1_000814_Folk_原曲30秒.mp3`
- `1_000995_Experimental_原曲30秒.mp3`
- `1_001482_Electronic_原曲30秒.mp3`
- `1_007481_Hip-Hop_原曲30秒.mp3`
- `1_014568_Instrumental_原曲30秒.mp3`
- `2_000010_Pop_前缀5秒+真实3秒.mp3`
- `2_000182_Rock_前缀5秒+真实3秒.mp3`
- `2_000704_International_前缀5秒+真实3秒.mp3`
- `2_000814_Folk_前缀5秒+真实3秒.mp3`
- `2_000995_Experimental_前缀5秒+真实3秒.mp3`
- `2_001482_Electronic_前缀5秒+真实3秒.mp3`
- `2_007481_Hip-Hop_前缀5秒+真实3秒.mp3`
- `2_014568_Instrumental_前缀5秒+真实3秒.mp3`
- `3_000010_Pop_前缀5秒+原始模型续写3秒.mp3`
- `3_000182_Rock_前缀5秒+原始模型续写3秒.mp3`
- `3_000704_International_前缀5秒+原始模型续写3秒.mp3`
- `3_000814_Folk_前缀5秒+原始模型续写3秒.mp3`
- `3_000995_Experimental_前缀5秒+原始模型续写3秒.mp3`
- `3_001482_Electronic_前缀5秒+原始模型续写3秒.mp3`
- `3_007481_Hip-Hop_前缀5秒+原始模型续写3秒.mp3`
- `3_014568_Instrumental_前缀5秒+原始模型续写3秒.mp3`
- `4_000010_Pop_前缀5秒+LoRA续写3秒.mp3`
- `4_000182_Rock_前缀5秒+LoRA续写3秒.mp3`
- `4_000704_International_前缀5秒+LoRA续写3秒.mp3`
- `4_000814_Folk_前缀5秒+LoRA续写3秒.mp3`
- `4_000995_Experimental_前缀5秒+LoRA续写3秒.mp3`
- `4_001482_Electronic_前缀5秒+LoRA续写3秒.mp3`
- `4_007481_Hip-Hop_前缀5秒+LoRA续写3秒.mp3`
- `4_014568_Instrumental_前缀5秒+LoRA续写3秒.mp3`
- `5_000010_Pop_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_000182_Rock_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_000704_International_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_000814_Folk_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_000995_Experimental_前缀5秒+原始模型续写12秒.mp3`
- `5_001482_Electronic_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_007481_Hip-Hop_四段串联_前缀-真实-原始-LoRA.mp3`
- `5_014568_Instrumental_四段串联_前缀-真实-原始-LoRA.mp3`
- `6_000995_Experimental_前缀5秒+LoRA续写8秒.mp3`
- `7_000995_Experimental_四段串联_前缀-真实-原始-LoRA.mp3`

## 说明

- 每条 2/3/4 都以同一段 5 秒前缀开头，所以可以直接连着放，听众只需注意后 3 秒。
- 只有 `000995` 有多于 3 秒的续写：那是唯一跑通过更长续写的曲目（见 `demo/README.md` §3.5）。
- 音频未做响度归一，只留 1 dB 余量；真实录音的前缀本来就比续写段安静一些。
- 素材与 `demo/audio/` 里的同名文件是同一份音频，这里只是换成了自解释的中文名。
