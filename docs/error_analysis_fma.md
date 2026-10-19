# 错误分析（P0）

数据：FMA-small 测试集 1199 首，模型为 seed 42 的 checkpoint。
复现命令：`python -m src.error_analysis`。

## 1. 各模型最容易混淆的流派对

| 模型 | Top-3 混淆对（真实→预测，错误数） |
|---|---|
| SVM | Hip-Hop→Electronic (30)；Pop→International (30)；International→Hip-Hop (26) |
| MLP | Pop→International (34)；Experimental→Pop (30)；Pop→Hip-Hop (26) |
| CNN | Pop→Folk (52)；Instrumental→Folk (45)；Pop→Rock (33) |
| CRNN | Pop→Rock (29)；Pop→Electronic (25)；Pop→International (24) |
| Transformer | Instrumental→Experimental (41)；Hip-Hop→Electronic (34)；Folk→Instrumental (31) |

## 2. 置信度：错的时候有多自信？

| 模型 | 正确样本平均置信度 | 错误样本平均置信度 | 错误数 | 其中 ≥0.9 的自信错误 |
|---|---|---|---|---|
| SVM | 52.2% | 39.6% | 664 | 1 |
| MLP | 64.0% | 50.8% | 668 | 33 |
| CNN | 68.7% | 51.9% | 630 | 23 |
| CRNN | 80.4% | 63.4% | 610 | 90 |
| Transformer | 49.7% | 40.8% | 728 | 1 |

> 置信度不等于正确率：模型在错误样本上依然给出很高的概率，
> 这正是「AI 会自信地犯错」的现场素材。

## 3. 典型错误案例（CNN）

| 曲目 | 真实 | 预测 | 置信度 | 文件 |
|---|---|---|---|---|
| 016158 | Pop | Folk | 98.9% | `data/fma/fma_small/016/016158.mp3` |
| 104780 | Pop | Folk | 96.7% | `data/fma/fma_small/104/104780.mp3` |
| 108881 | Instrumental | Folk | 92.8% | `data/fma/fma_small/108/108881.mp3` |
| 095724 | Instrumental | Folk | 89.8% | `data/fma/fma_small/095/095724.mp3` |

听感对照建议：按上表顺序播放这几首，先让同学判断「像不像」，再揭晓模型的判断。
波形/频谱对照见 `results/fma/figures/error_cases.png`。

## 4. 结论与解释

- **Pop 是最难的类**：CNN 的 Pop F1 仅 0.239，主要被误判为 Folk（52 首）、Rock（33 首）。
- **Hip-Hop 最稳**：F1 0.674，器乐音色与动态范围独特，几乎所有模型都能分对。
- **最主要的混淆对**：Pop→Folk（52 首）；Instrumental→Folk（45 首）；Pop→Rock（33 首）；这些流派共享节奏型或音色，「流派」本身是模糊标签。
- 轻量 Transformer 的错误更分散：训练 loss 到 0、验证抬头，且高置信错误比例明显更高（见上表），过拟合没有换来稳定判断。
- 这些错误不是「模型太笨」，而是**标签边界模糊 + 30 秒片段信息有限**；
  汇报时把它作为「AI 是否真正理解音乐」的论据，而不是只报准确率。

## 5. 产物

- `results/fma/error_analysis.csv`：逐模型混淆对计数
- `results/fma/error_analysis.json`：混淆对、置信度统计、典型样例
- `results/fma/figures/error_confusions.png`、`results/fma/figures/error_cases.png`
