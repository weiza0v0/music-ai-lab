# 错误分析（P0）

数据：GTZAN 测试集 147 首，模型为 seed 42 的 checkpoint。
复现命令：`python -m src.error_analysis`。

## 1. 各模型最容易混淆的流派对

| 模型 | Top-3 混淆对（真实→预测，错误数） |
|---|---|
| SVM | hiphop→pop (3)；disco→rock (3)；disco→hiphop (2) |
| MLP | jazz→blues (4)；disco→metal (4)；rock→disco (3) |
| CNN | reggae→pop (3)；blues→reggae (2)；hiphop→pop (2) |
| CRNN | reggae→pop (4)；rock→blues (3)；rock→disco (3) |
| Transformer | hiphop→reggae (5)；rock→blues (5)；metal→blues (4) |

## 2. 置信度：错的时候有多自信？

| 模型 | 正确样本平均置信度 | 错误样本平均置信度 | 错误数 | 其中 ≥0.9 的自信错误 |
|---|---|---|---|---|
| SVM | 67.1% | 44.4% | 43 | 0 |
| MLP | 85.8% | 68.4% | 53 | 12 |
| CNN | 84.4% | 64.9% | 37 | 3 |
| CRNN | 83.7% | 60.1% | 38 | 4 |
| Transformer | 88.5% | 71.3% | 64 | 16 |

> 置信度不等于正确率：模型在错误样本上依然给出很高的概率，
> 这正是「AI 会自信地犯错」的现场素材。

## 3. 典型错误案例（CNN）

| 曲目 | 真实 | 预测 | 置信度 | 文件 |
|---|---|---|---|---|
| reggae.00088 | reggae | pop | 83.1% | `data/raw/gtzan/genres/reggae/reggae.00088.wav` |
| reggae.00069 | reggae | pop | 61.1% | `data/raw/gtzan/genres/reggae/reggae.00069.wav` |
| blues.00077 | blues | reggae | 81.8% | `data/raw/gtzan/genres/blues/blues.00077.wav` |
| blues.00008 | blues | reggae | 79.1% | `data/raw/gtzan/genres/blues/blues.00008.wav` |
| jazz.00003 | jazz | classical | 99.6% | `data/raw/gtzan/genres/jazz/jazz.00003.wav` |

听感对照建议：按上表顺序播放这几首，先让同学判断「像不像」，再揭晓模型的判断。
波形/频谱对照见 `results/figures/error_cases.png`。

## 4. 结论与解释

- **rock 是最难的类**：CNN 的 rock F1 仅 0.429，主要被误判为 blues（2 首）、disco（2 首）。
- **classical 最稳**：F1 0.933，器乐音色与动态范围独特，几乎所有模型都能分对。
- **最主要的混淆对**：reggae→pop（3 首）；blues→reggae（2 首）；hiphop→pop（2 首）；这些流派共享节奏型或音色，「流派」本身是模糊标签。
- 轻量 Transformer 的错误更分散：训练 loss 到 0、验证抬头，且高置信错误比例明显更高（见上表），过拟合没有换来稳定判断。
- 这些错误不是「模型太笨」，而是**标签边界模糊 + 30 秒片段信息有限**；
  汇报时把它作为「AI 是否真正理解音乐」的论据，而不是只报准确率。

## 5. 产物

- `results/error_analysis.csv`：逐模型混淆对计数
- `results/error_analysis.json`：混淆对、置信度统计、典型样例
- `results/figures/error_confusions.png`、`results/figures/error_cases.png`
