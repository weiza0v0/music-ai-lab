# 数据卡：GTZAN（本次 MVP 使用的版本）

## 来源与获取

- 数据集：GTZAN Genre Collection（Tzanetakis, Essl & Cook, 2001）
- 本次实际下载：HuggingFace `marsyas/gtzan` 的 `data/genres.tar.gz`
  （https://huggingface.co/datasets/marsyas/gtzan ，1.17 GiB）
- 内容：10 个流派 × 100 首 = 1000 首，每首约 30 秒，22,050 Hz、单声道、16-bit WAV

## 授权与使用边界

- 该数据集长期用于学术研究；**音频版权仍归各曲目权利人**，不要公开再分发音频文件。
- 本仓库不提交 `data/raw/`（见 `.gitignore`），只提交划分清单与统计结果。
- 用于课堂演示/课程汇报可以，公开发布模型权重前请自行核对授权。

## 已知质量问题（实测）

| 问题 | 本次实测 | 处理 |
|---|---|---|
| 损坏文件 | `jazz.00054` 无法解码 | 剔除 |
| 重复音频 | 28 首与其他曲目 MD5 完全相同 | 每组只保留 1 首，共剔除 14 首 |
| 标签噪声 | 学界公认（如 rock/metal 边界、disco/pop 混用） | 保留，作为错误分析素材 |
| AppleDouble 垃圾 | 1000 个 `._*.wav`（macOS 元数据，非音频） | 审计时过滤并删除 |
| 类别不均衡 | 基本均衡（每类 100） | 分层划分 |

## 本次使用的子集

- 清洗后 **985 首**：train 689 / val 149 / test 147（按曲目分层 70/15/15，seed=42）。
- 每首取**中心 5 秒**片段（22,050 Hz，110,250 采样点）。
- 划分文件：`data/splits/splits.csv`；统计：`data/splits/stats.json`。
- 特征缓存：`data/processed/features.npz`（209 维人工特征 + 64×216 Log-Mel）。

## 引用

```bibtex
@inproceedings{tzanetakis2002gtzan,
  title={GTZAN genre collection},
  author={Tzanetakis, George and Cook, Perry},
  year={2002}
}
```

相关讨论：Sturm, B. L. (2013). *The GTZAN dataset: Its contents, its faults, their
effects on evaluation, and its future use.* arXiv:1306.1461.
