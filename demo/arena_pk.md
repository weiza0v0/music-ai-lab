# 擂台表（GTZAN 测试集，seed 42）

由 `scripts/build_arena_table.py` 从 `results/gtzan/predictions/*.csv` 直接生成，
只用已提交的预测文件，**不需要权重与音频**。✓ 表示答对，百分比是该模型的置信度。

## 一、五个模型意见最不一致的 8 首

这 8 首是「擂台」的核心：同一段音频，五个模型的判断各不相同，正好用来讲
归纳偏置与错误模式。

| 曲目 | 真值 | SVM | MLP | CNN | CRNN | TF | 答对 |
|---|---|---|---|---|---|---|---|
| `reggae.00049` | reggae | country ✗ 30% | disco ✗ 76% | rock ✗ 46% | pop ✗ 37% | country ✗ 25% | 0/5 |
| `disco.00069` | disco | rock ✗ 53% | rock ✗ 51% | disco ✓ 58% | blues ✗ 50% | jazz ✗ 89% | 1/5 |
| `hiphop.00055` | hiphop | country ✗ 32% | disco ✗ 41% | reggae ✗ 64% | reggae ✗ 33% | hiphop ✓ 92% | 1/5 |
| `metal.00008` | metal | hiphop ✗ 34% | hiphop ✗ 93% | metal ✓ 85% | rock ✗ 56% | disco ✗ 86% | 1/5 |
| `rock.00014` | rock | reggae ✗ 50% | reggae ✗ 98% | hiphop ✗ 60% | disco ✗ 31% | rock ✓ 74% | 1/5 |
| `classical.00099` | classical | classical ✓ 82% | classical ✓ 100% | rock ✗ 77% | country ✗ 70% | jazz ✗ 46% | 2/5 |
| `disco.00065` | disco | hiphop ✗ 30% | metal ✗ 31% | disco ✓ 99% | disco ✓ 71% | reggae ✗ 79% | 2/5 |
| `reggae.00097` | reggae | jazz ✗ 38% | jazz ✗ 53% | country ✗ 53% | country ✗ 44% | blues ✗ 96% | 0/5 |

这批曲目上各模型答对数：SVM 1/8、MLP 1/8、CNN 3/8、CRNN 1/8、TF 2/8

## 二、五个模型全都答对且一致的 4 首（对照组）

用来对比：干净、典型的样本上五个模型没有分歧，差别只出现在边界样本上。

| 曲目 | 真值 | SVM | MLP | CNN | CRNN | TF | 答对 |
|---|---|---|---|---|---|---|---|
| `blues.00012` | blues | blues ✓ 65% | blues ✓ 98% | blues ✓ 85% | blues ✓ 97% | blues ✓ 97% | 5/5 |
| `blues.00013` | blues | blues ✓ 73% | blues ✓ 93% | blues ✓ 75% | blues ✓ 99% | blues ✓ 99% | 5/5 |
| `blues.00014` | blues | blues ✓ 45% | blues ✓ 98% | blues ✓ 58% | blues ✓ 65% | blues ✓ 90% | 5/5 |
| `blues.00019` | blues | blues ✓ 70% | blues ✓ 100% | blues ✓ 98% | blues ✓ 99% | blues ✓ 60% | 5/5 |

## 三、怎么用

1. 幻灯片上只放 **第一节那张表的 3–4 行**（放满 8 行会看不清），其余口述。
2. 讲的时候按列读：CNN 与 CRNN 的判断常常相邻（都吃局部时频结构），
   TF 容易给出低置信度的离谱答案，SVM 在边界样本上反而稳。
3. 全量混淆矩阵见 `results/gtzan/figures/confusion_matrices.png`，
   逐类 F1 见 `results/gtzan/figures/leaderboard.png` 与技术报告 §5.6。
4. AST 不在这张表里（它的预测未按样本落盘），但它在两个数据集上都是第一，
   放在「预训练一出手就赢」那一页讲。
