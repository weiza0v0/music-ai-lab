# deck/ · 两套演示（HTML + PPT）

同一个内容、同一套设计，两种交付形态。**28 页，实测 27 分钟。**

```
deck/
├── html/                      ← 方案 A：网页版（点击播放、键盘翻页）
│   ├── index.html             单文件，样式与脚本内联
│   ├── assets/figures/*.png   8 张图（6 张重绘 + 1 张原始频谱图 + 1 张分类榜）
│   └── audio/*.mp3            6 段音频，断网也能播
├── ppt/                       ← 方案 B：原生 PPT
│   ├── Music_AI_Lab_demo.pptx 3.4 MB，28 页，音频已内嵌
│   └── Music_AI_Lab_demo.pdf  1.2 MB，同版备份（教室电脑打不开 PPT 时用）
├── assets/
│   ├── poster/play.png        音频按钮的圆形海报帧
│   └── audio/                 6 段音频的原件备份（PPT 换机器失效时手动放）
├── scripts/
│   ├── make_figures.py        从 results/ 现算并重绘全部插图
│   └── build_pptx.py          生成 pptx（改文案改这里，重跑即可）
└── render/                    两套的渲染验收图（拼图 + 单页）
```

## 怎么用

**HTML**：双击 `html/index.html`（Chrome / Edge 都行）。
`←/→` 翻页、`空格` 播当前页音频、`F` 全屏、`N` 看备注、`R` 揭晓盲听答案、`Esc` 关备注。
第 22 页的「揭晓答案」是按钮，点了才显示 A/B/C 的真相。

**PPT**：打开 `ppt/Music_AI_Lab_demo.pptx`，**放映模式**下点圆环按钮即播（编辑模式点不动）。
万一某台机器内嵌音频不响，用 `assets/audio/` 里的同名 mp3 手动放，页面角上写了文件名。

## 音频 cue

| 页 | 文件 | 时长 | 用途 |
|---|---|---|---|
| 09 | `073776_International.mp3` | 5.1 s | 六模型全错的那段 |
| 21 | `AB__000995_Experimental.mp3` | 16.4 s | 前缀→真实→原始→适配 四段串联 |
| 22 | `REAL__` / `GENbase__` / `GENft__000995` | 各 8.0 s | 盲听 A/B/C（A 是真录音） |
| 23 | `AB__001482_Electronic.mp3` | 16.4 s | 适配后反而更差的那一条 |

主用合计 62 秒。P22 答案：**A = 真人录音，B = 原始模型，C = LoRA 适配后**（只写在备注里，页面上不显示）。

## 设计约束（改版时别破坏）

- 底色只有两种：`#FAF7F2` 纸白、`#F3EEE6` 浅赭。**没有深色页**（用户明确要求）。
- 红 `#C0512F` 只表示"错 / 变差 / 失败"，绿 `#2F5B4F` 只表示"对 / 变好 / 冠军"，金 `#B08442` 一页最多一次。
- 一页一个结论；表格字号不小于 19px（PPT 里约 11.5 pt）；图整张插入不裁剪。
- 不用 emoji、渐变、发光、圆角卡片堆叠。
- 字体：HTML 用 Noto Serif SC / Noto Sans SC / JetBrains Mono（本机缺字体时回退宋体、黑体）；
  PPT 用 Times New Roman + 宋体（标题）、Arial + 黑体（正文）、Consolas（数字）——这三个 Windows 全自带，换机器不跑版。

## 验收记录

| 检查 | 结果 |
|---|---|
| HTML `deck_lint` | **P0 = 0**；剩 1 条 P1「连续三页同主题」——因为要求全浅色，**有意保留** |
| HTML `deck_render` | 28 页全部独立渲染，人工逐页看过，无溢出、无字体回退异常 |
| PPT `audit-pptx-geometry.py` | ✅ 无框重叠、无溢出风险 |
| PPT `audit-pptx-design.py` | 无渐变、无真 emoji（15 条告警是 `−` 减号、`≫`、`✓✗` 被判成 emoji，属误报）；配色即下方 6 色 + 4 个混色灰，全部来自同一套 token |
| PPT `pptx-preview` 渲染 | 28 页逐页看过：标题未贴正文、无文字出框、来源行未被挤出画布 |
| 音频 | 6 个文件都在包内，HTML 侧用相对路径；PPT 侧内嵌且另有原件备份 |

## 数据来源

所有数字读自 `results/**/metrics.csv`、`results/gtzan/robustness.csv`、
`results/token_prediction_long/metrics.csv`、`results/logs/overnight/*.log`，
图由 `scripts/make_figures.py` 现场重绘，**没有手填数字**。重跑：

```bash
python3 deck/scripts/make_figures.py     # 重绘 6 张图
python3 deck/scripts/build_pptx.py       # 重新生成 pptx
```
