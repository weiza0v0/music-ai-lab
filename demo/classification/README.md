# 分类演示包（无需 GPU）

8 段 5 秒片段，全部取自 FMA-small 的**测试集**（模型没见过），
并把五个模型当时给出的判断一起附上：现场不需要任何推理。

**授权**：FMA-small 音频为 CC-BY 4.0（Defferrard et al., ISMIR 2017），
允许再分发，条件是署名。逐曲的曲名与艺术家见下表。GTZAN 音频不随仓库分发
（版权归各权利人），所以这批示例用 FMA。

所有片段统一裁成**中心 5 秒**、22 050 Hz 单声道、响度对齐 −18 LUFS —— 
与模型评估时的输入口径一致，因此下面表格里的判断就是模型对这段音频的真实输出。

另有一份 `<曲目>_context15.mp3`：同一首曲目的**中心 15 秒**，只给人耳听上下文用。
模型不消费它——评估与上表判断用的始终是中心 5 秒。

## 一、五个模型意见不一致（擂台用）

同一段音频，五个模型给出不同答案。这是最值得讲的一组：分歧集中在流派边界模糊的样本上。

| 音频文件 | 真值 | 曲目 | SVM | MLP | CNN | CRNN | TF | 答对 |
|---|---|---|---|---|---|---|---|---|
| `073776_International.mp3` | International | StrangeZero — Different Feelings | Instrumental ✗ 23% | Folk ✗ 35% | Experimental ✗ 43% | Electronic ✗ 91% | Pop ✗ 29% | 0/5 |
| `014576_Instrumental.mp3` | Instrumental | Ergo Phizmiz — Funeral March For A Marionette | Experimental ✗ 26% | Instrumental ✓ 33% | Folk ✗ 53% | Pop ✗ 30% | International ✗ 37% | 1/5 |
| `047835_Pop.mp3` | Pop | Révolution Musicale — He Wears Clogs For All Time | Pop ✓ 27% | International ✗ 27% | Folk ✗ 55% | Instrumental ✗ 52% | Experimental ✗ 26% | 1/5 |
| `117944_International.mp3` | International | mildtape — Whispers remix | Electronic ✗ 31% | Pop ✗ 35% | Folk ✗ 35% | International ✓ 73% | Hip-Hop ✗ 30% | 1/5 |
| `122201_Experimental.mp3` | Experimental | Alas de Mosca — Film noir | Instrumental ✗ 22% | Hip-Hop ✗ 25% | Pop ✗ 35% | Rock ✗ 28% | Experimental ✓ 40% | 1/5 |
| `000182_Rock.mp3` | Rock | Ariel Pink's Haunted Graffiti — Jules Lost His Jewels | International ✗ 19% | Hip-Hop ✗ 26% | Pop ✗ 24% | Hip-Hop ✗ 34% | Experimental ✗ 24% | 0/5 |

## 二、五个模型全都答对且一致（对照组）

干净、典型的样本上五个模型没有分歧——差别只出现在边界样本上。

| 音频文件 | 真值 | 曲目 | SVM | MLP | CNN | CRNN | TF | 答对 |
|---|---|---|---|---|---|---|---|---|
| `000814_Folk.mp3` | Folk | Hall Of Fame — Motion Passings | Folk ✓ 74% | Folk ✓ 99% | Folk ✓ 96% | Folk ✓ 85% | Folk ✓ 58% | 5/5 |
| `000995_Experimental.mp3` | Experimental | Kevin Shields — No Good Deed Goes Unpunished | Experimental ✓ 71% | Experimental ✓ 57% | Experimental ✓ 97% | Experimental ✓ 98% | Experimental ✓ 71% | 5/5 |

## 三、现场怎么用

### 方式 1：只播放（完全不需要环境）

按上表顺序放 mp3，先问「你们觉得这是什么流派」，再公布五个模型的答案与置信度。
分歧组里最值得放的是**人工特征模型对、CNN 错**的那几条：
说明卷积看到的是局部音色纹理，而不是人耳理解的风格。

### 方式 2：现场跑（CPU 就够，不需要显卡）

五个自建/传统模型在纯 CPU 上单段推理约 **0.1–0.3 秒**（实测：CNN 196 ms、
CRNN 60 ms、TF 80 ms、MLP 70 ms；SVM 首次调用含特征提取约 2.6 s，之后很快）。
所以笔记本没有显卡也能现场 PK：

```bash
git clone https://github.com/weiza0v0/music-ai-lab.git && cd music-ai-lab
pip install -r requirements.txt
# 这一包是 FMA 的片段，用 FMA 的权重（8 类）
MUSIC_CKPT=checkpoints/fma python3 -m streamlit run app/demo.py
```

上传任意一段音乐即可；想复现上表，就上传本目录里对应的 mp3。

> AST 不在这个包里：它需要 346 MB 权重与约 200 ms/段（CPU 上更慢），
> 现场只讲它的榜上成绩与成本对比即可。

## 四、数字出处

- 判断与置信度：`results/fma/predictions/{model}_seed42_test.csv`
- 全量指标：`results/fma/metrics.csv`；逐类 F1 见报告 §5.5
- 这张表由 `scripts/build_classification_demo.py` 生成，可重跑核对
