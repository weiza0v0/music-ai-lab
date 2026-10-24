#!/usr/bin/env python3
"""生成 28 页原生 .pptx（每个字都是可编辑文本框，音频内嵌）。

设计坐标一律用 1600×900 的「设计像素」写，脚本换算成英寸/磅：
    1 px = 13.333/1600 in = 1/120 in
    1 px = 0.6 pt
这样 HTML 版与 PPT 版的版式能对齐，改一处两边同步。

几何铁律（见 od-deck §6.6）：
  * 相邻文字框不许重叠：上一框底 ≤ 下一框顶
  * 框高 = 行数 × 字号/72 × 行高 × 1.08（中文字更宽松）
  * 中文字体两个槽位都要填：run.font.name + a:ea
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "deck/html/assets/figures"
AUD = ROOT / "deck/html/audio"
POSTER = ROOT / "deck/assets/poster"
OUT = ROOT / "deck/ppt/Music_AI_Lab_demo.pptx"

# ---------------- palette ----------------
INK = RGBColor(0x1C, 0x1A, 0x17)
MUTED = RGBColor(0x8A, 0x81, 0x7A)
RULE = RGBColor(0xD8, 0xD0, 0xC6)
RULE_SOFT = RGBColor(0xE8, 0xE1, 0xD7)
ACCENT = RGBColor(0xC0, 0x51, 0x2F)
ACCENT2 = RGBColor(0x2F, 0x5B, 0x4F)
FLAG = RGBColor(0xB0, 0x84, 0x42)
BG = RGBColor(0xFA, 0xF7, 0xF2)
BGALT = RGBColor(0xF3, 0xEE, 0xE6)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)

# (拉丁字体, 中文字体) —— 中英分槽，否则 SimHei 顶替拉丁字会掉字距
FACE = {
    "SERIF": ("Times New Roman", "SimSun"),      # 标题：宋体 + Times
    "SANS": ("Arial", "SimHei"),                 # 正文：黑体 + Arial
    "MONO": ("Consolas", "SimHei"),              # 数字：等宽
}
SERIF, SANS, MONO = "SERIF", "SANS", "MONO"

PXIN = 13.333 / 1600.0
PXPT = 0.6


def X(px: float) -> Emu:
    return Inches(px * PXIN)


def FS(px: float) -> Pt:
    return Pt(round(px * PXPT, 1))


def set_font(run, face: str, size_px: float, color: RGBColor, bold=False):
    latin, ea = FACE.get(face, (face, face))
    run.font.name = latin
    run.font.size = FS(size_px)
    run.font.color.rgb = color
    run.font.bold = bold
    rPr = run._r.get_or_add_rPr()
    for tag, tf in (("a:ea", ea), ("a:cs", ea)):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", tf)


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(X(x), X(y), X(w), X(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb, tf


ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}


def para(tf, runs, *, size_px, font=SANS, color=INK, bold=False,
         align="l", spacing=1.32, space_before=0, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = ALIGN[align]
    p.line_spacing = spacing
    if space_before:
        p.space_before = FS(space_before)
    if isinstance(runs, str):
        runs = [(runs, color, bold, font)]
    for t, c, b, f in runs:
        r = p.add_run()
        r.text = t
        set_font(r, f or font, size_px, c if c is not None else color, b)
    return p


def text(slide, x, y, w, content, *, size_px, font=SANS, color=INK, bold=False,
         align="l", spacing=1.32, lines=None):
    """单块文字；高度按「行数 × 字号 × 行高 × 1.08」自动算足。"""
    n = lines if lines else (content.count("\n") + 1)
    # 统一按 px 空间算高度：行数 × 字号(px) × 行高 × 1.08，再留 15% 余量
    h_px = n * size_px * spacing * 1.08 * 1.15
    tb, tf = textbox(slide, x, y, w, h_px)
    chunks = content.split("\n")
    for i, ch in enumerate(chunks):
        para(tf, ch, size_px=size_px, font=font, color=color, bold=bold,
             align=align, spacing=spacing, first=(i == 0))
    return y + h_px


def hline(slide, x, y, w, color=RULE, thick_px=1):
    """用连接线画分隔线：渲染成真正的发丝线，不会被当成小矩形加粗。"""
    from pptx.enum.shapes import MSO_CONNECTOR
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(x), X(y), X(x + w), X(y))
    ln.line.color.rgb = color
    ln.line.width = Pt(0.75)
    return ln


def no_effects(sh):
    """去掉自选图形自带的 theme 样式引用（LibreOffice 会照 effectRef 画阴影）。"""
    el = sh._element
    spPr = el.spPr
    for tag in ("a:effectLst", "a:effectDag"):
        for sub in spPr.findall(qn(tag)):
            spPr.remove(sub)
    spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    style = el.find(qn("p:style"))
    if style is not None:
        el.remove(style)
    return sh


def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X(x), X(y), X(w), X(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    no_effects(sh)
    return sh


def slide_new(prs, bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bgshape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bgshape.fill.solid()
    bgshape.fill.fore_color.rgb = bg
    bgshape.line.fill.background()
    bgshape.shadow.inherit = False
    # 页眉
    text(s, 88, 34, 900, "MUSIC AI LAB", size_px=14, font=MONO, color=MUTED)
    return s


def footer(s, n, total, src=""):
    if src:
        text(s, 88, 826, 1100, src, size_px=14, font=MONO, color=MUTED)
    text(s, 1130, 826, 160, f"{n:02d} / {total}", size_px=14, font=MONO,
         color=MUTED, align="r")


def title(s, txt, y=150, size_px=48):
    return text(s, 88, y, 1420, txt, size_px=size_px, font=SERIF, bold=True, spacing=1.34)


def table(s, x, y, w, cols, rows, *, size_px=22, head_size=16, row_h=64,
          aligns=None, head=True, emphasize=(), col_w=None, line_col=RULE,
          pad=16, bg_em=BGALT):
    """手绘细线表格：只用横线，无竖线（和纸面设计一致）。"""
    n = len(cols) if cols else len(rows[0])
    aligns = aligns or ["l"] * n
    col_w = col_w or [w / n] * n
    cy = y
    if head and cols:
        for j, c in enumerate(cols):
            cx = x + sum(col_w[:j])
            text(s, cx + ((0 if j == 0 else pad) if aligns[j] == "l" else pad), cy, col_w[j], c,
                 size_px=head_size, font=SANS, color=MUTED, align=aligns[j])
        cy += head_size * 1.32 * 1.08 + 12
        hline(s, x, cy, w, line_col)
        cy += 0
    for i, row in enumerate(rows):
        cells = row if isinstance(row, (list, tuple)) else [row]
        if i in emphasize:
            rect(s, x - 8, cy, w + 16, row_h, bg_em)
        for j, cell in enumerate(cells):
            cx = x + sum(col_w[:j])
            cw = col_w[j]
            if isinstance(cell, tuple):
                txt, color, bold, font = (list(cell) + [None, None, None])[:4]
                color = color or INK
                font = font or SANS
            else:
                txt, color, bold, font = cell, INK, False, SANS
            padx = (0 if j == 0 else pad) if aligns[j] == "l" else pad
            text(s, cx + padx, cy + (row_h - size_px * 1.32 * 1.08) / 2, cw, txt,
                 size_px=size_px, font=font, color=color, bold=bool(bold),
                 align=aligns[j])
        cy += row_h
        hline(s, x, cy, w, RULE_SOFT if i < len(rows) - 1 else line_col)
    return cy


def picture(slide, x, y, w, path, *, caption=None, plate=True):
    """插图；返回整块（含白底与图注）的底边 y，方便往下排版。"""
    im = Image.open(path)
    iw, ih = im.size
    pad = 18
    inner_w = w - 2 * pad
    inner_h = inner_w * ih / iw
    cap_h = (12 + 14 * 1.32 * 1.08) if caption else 0
    box_h = pad + inner_h + cap_h + pad
    if plate:
        rect(slide, x, y, w, box_h, SURFACE, RULE)
    slide.shapes.add_picture(str(path), X(x + pad), X(y + pad), width=X(inner_w))
    if caption:
        text(slide, x + pad, y + pad + inner_h + 12, inner_w, caption,
             size_px=14, font=MONO, color=MUTED)
    return y + box_h


def playbtn(slide, x, y, audio, *, size_px=76, label=None, tc="0:00"):
    """内嵌音频，海报帧是圆形播放键；放映时点一下即播。"""
    mime = "audio/mpeg"
    slide.shapes.add_movie(str(audio), X(x), X(y), X(size_px), X(size_px),
                           poster_frame_image=str(POSTER / "play.png"),
                           mime_type=mime)
    if label:
        text(slide, x + size_px + 26, y + 8, 700, label, size_px=17, color=MUTED)
    if tc:
        text(slide, x + size_px + 26, y + 8 + 24, 700, tc, size_px=17, font=MONO,
             color=MUTED)


# ======================================================================
TOTAL = 28
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ---- 01 封面 ----
s = slide_new(prs)
text(s, 88, 196, 900, "课程项目 · 音乐 AI · 2026", size_px=14, color=MUTED)
yy = text(s, 88, 236, 1300, "一个项目，两个任务，", size_px=84, font=SERIF, bold=True, spacing=1.18)
yy = text(s, 88, yy, 1300, "五个负结果", size_px=84, font=SERIF, bold=True, spacing=1.18)
yy = text(s, 88, yy + 10, 1200, "音乐流派分类 × 音频 Token 预测", size_px=30, color=MUTED,
          spacing=1.4)
yy = text(s, 88, yy + 18, 1200, "GTZAN 985 · FMA-small 7,994 · 6 模型 · 3 种子 · 42 页报告",
          size_px=18, font=MONO, color=MUTED)
yy += 26
hline(s, 88, yy, 760, RULE)
yy += 26
text(s, 88, yy, 1200, "组员　梁煜炜 · 贾智涵 · 王薪羽 · 陈泽玉", size_px=18, font=MONO, color=MUTED)
footer(s, 1, TOTAL, "2026.10")

# ---- 02 两个任务 ----
s = slide_new(prs)
text(s, 88, 250, 1420, "同一批音乐，模型能听懂多少？", size_px=76, font=SERIF, bold=True,
     spacing=1.25)
text(s, 88, 470, 620, "p(流派 | 音频)", size_px=26, font=MONO, spacing=1.3)
text(s, 88, 470 + 26 * 1.3 * 1.08 + 14, 620, "判别式　答对几成　Macro-F1",
     size_px=16, color=MUTED)
text(s, 820, 470, 700, "p(下一个 Token | 前 5 秒)", size_px=26, font=MONO, spacing=1.3)
text(s, 820, 470 + 26 * 1.3 * 1.08 + 14, 700, "生成式　离真相多远　Top-1 / Log-Mel RMSE",
     size_px=16, color=MUTED)
hline(s, 88, 650, 1424, RULE)
text(s, 88, 676, 1200, "同一份数据划分 · 同一批测试曲目 · 指标不可比", size_px=20, color=MUTED)
footer(s, 2, TOTAL)

# ---- 03 数据 ----
s = slide_new(prs)
title(s, "数据从哪来：985 首，和 7,994 首", 150, 44)
table(s, 88, 268, 1424, ["", "GTZAN", "FMA-small"], [
    [("曲目", None, True), ("985", INK, False, MONO), ("7,994", INK, False, MONO)],
    [("类别", None, True), ("10", INK, False, MONO), ("8", INK, False, MONO)],
    [("训练 / 验证 / 测试", None, True), ("690 / 147 / 147", INK, False, MONO),
     ("5,596 / 1,199 / 1,199", INK, False, MONO)],
    [("模型输入", None, True), ("中心 5 秒", INK, False, MONO), ("中心 5 秒", INK, False, MONO)],
    [("采样率", None, True), ("22.05 kHz", INK, False, MONO), ("22.05 kHz", INK, False, MONO)],
], size_px=22, row_h=76, aligns=["l", "r", "r"], col_w=[560, 432, 432])
text(s, 88, 700, 1200, "素材本身是 30 秒，但模型全程只听中心 5 秒。", size_px=18, color=MUTED)
footer(s, 3, TOTAL, "来源：docs/repo_overview.md §2 · 报告 §3.4")

# ---- 04 泄漏 ----
s = slide_new(prs, BGALT)
title(s, "一个很容易做错的划分", 150, 44)
text(s, 88, 250, 520, "0.470", size_px=150, font=MONO, color=MUTED, spacing=1.0)
text(s, 600, 300, 120, "→", size_px=56, color=MUTED)
cy = text(s, 760, 250, 620, "0.568", size_px=150, font=MONO, color=ACCENT2, spacing=1.0) + 30
table(s, 88, cy, 1000, None, [
    [("同一批特征", None, True), ("209 维人工特征，一个都没换", MUTED)],
    [("同一个 CNN", None, True), ("同一份权重、同一个种子", MUTED)],
    [("换的只是划分", None, True), ("「按艺术家划分」→「按曲目划分」", MUTED)],
], size_px=22, row_h=70, aligns=["l", "l"], col_w=[340, 660])
text(s, 88, cy + 70 * 3 + 34, 1200, "81.0% 的测试曲与训练集共享艺术家",
     size_px=22, color=ACCENT, bold=True)
footer(s, 4, TOTAL, "来源：报告 §3.4 表 4")

# ---- 05 规则 ----
s = slide_new(prs)
title(s, "三条比赛规则", 150, 44)
items = [("01", "三个种子，报均值", "42 / 123 / 2024，不挑最好的那一次"),
         ("02", "验证集选型，测试集只评一次", "评完不许回头改参数"),
         ("03", "按艺术家划分", "同一个人只出现在训练或测试一边")]
cy = 268
for i, (num, big, small) in enumerate(items):
    text(s, 88, cy, 80, num, size_px=20, font=MONO, color=MUTED)
    b = text(s, 180, cy - 4, 1200, big, size_px=31, bold=True)
    e = text(s, 180, b + 4, 1200, small, size_px=19, color=MUTED)
    cy = e + 30
    if i < len(items) - 1:
        hline(s, 88, cy - 18, 1200, RULE_SOFT)
footer(s, 5, TOTAL, "来源：data/splits/*/splits.csv（全模型共用）")

# ---- 06 六位选手 ----
s = slide_new(prs)
title(s, "六位选手", 150, 44)
table(s, 88, 254, 1424, ["选手", "输入", "参数量", "体积", "定位"], [
    [("SVM", None, True), ("209 维人工特征", None, False, MONO), ("—", None, False, MONO),
     ("1.2 MB", None, False, MONO), ("老派，抗造", MUTED)],
    [("MLP", None, True), ("209 维人工特征", None, False, MONO), ("8.8 万", None, False, MONO),
     ("1.1 MB", None, False, MONO), ("略弱于 SVM", MUTED)],
    [("CNN", None, True), ("64×216 log-Mel", None, False, MONO), ("59 万", None, False, MONO),
     ("2.4 MB", None, False, MONO), ("自建冠军（GTZAN）", MUTED)],
    [("CRNN", None, True), ("同上 + BiGRU", None, False, MONO), ("42 万", None, False, MONO),
     ("1.7 MB", None, False, MONO), ("数据够了才反超", MUTED)],
    [("轻量 Transformer", None, True), ("同上 + 位置编码", None, False, MONO),
     ("65 万", None, False, MONO), ("1.2 MB", None, False, MONO), ("千首级过拟合", MUTED)],
    [("AST", None, True), ("128×1024 log-Mel", None, False, MONO),
     ("8,600 万（冻结）", None, False, MONO), ("346 MB", None, False, MONO),
     ("预训练，两榜第一", MUTED)],
], size_px=21, row_h=64, aligns=["l", "l", "r", "r", "l"], col_w=[300, 360, 300, 220, 244])
text(s, 88, 254 + 64 * 6 + 40, 1300, "AST 为冻结主干 + 线性头（7,690 参数），不是微调。",
     size_px=18, color=MUTED)
footer(s, 6, TOTAL, "来源：results/gtzan/metrics.csv · 报告 §5.1")

# ---- 07 成绩单 ----
s = slide_new(prs)
title(s, "六位选手的成绩单", 150, 44)
picture(s, 88, 254, 690, FIG / "fig_gtzan_f1.png", caption="GTZAN · 985 首 · 10 类")
ny = picture(s, 822, 254, 690, FIG / "fig_fma_f1.png", caption="FMA-small · 7,994 首 · 8 类") + 26
b1 = text(s, 88, ny, 300, ".814", size_px=52, font=MONO, color=ACCENT2)
text(s, 88, b1 + 2, 300, "AST · GTZAN", size_px=16, color=MUTED)
b2 = text(s, 460, ny, 300, ".590", size_px=52, font=MONO, color=ACCENT2)
text(s, 460, b2 + 2, 300, "AST · FMA-small", size_px=16, color=MUTED)
b3 = text(s, 832, ny, 340, ".542", size_px=52, font=MONO, color=ACCENT)
text(s, 832, b3 + 2, 400, "轻量 Transformer · 两榜垫底", size_px=16, color=MUTED)
footer(s, 7, TOTAL, "来源：results/gtzan/metrics.csv · results/fma/metrics.csv（3 种子均值）")

# ---- 08 代价 ----
s = slide_new(prs, BGALT)
title(s, "冠军的代价", 150, 44)
picture(s, 88, 250, 800, FIG / "fig_cost.png",
        caption="延迟与体积，双对数坐标（GTZAN 测试集实测）")
b1 = text(s, 950, 300, 420, "270×", size_px=54, font=MONO, color=ACCENT, bold=True)
text(s, 950, b1, 420, "延迟：0.76 → 204 ms", size_px=17, color=MUTED)
b2 = text(s, 950, 470, 420, "145×", size_px=54, font=MONO, color=ACCENT, bold=True)
text(s, 950, b2, 420, "体积：2.4 → 346 MB", size_px=17, color=MUTED)
text(s, 950, 640, 420, "离线批处理可接受，\n实时交互不可用。", size_px=19, color=MUTED)
footer(s, 8, TOTAL, "来源：报告 §5.4 · results/gtzan/metrics.csv")

# ---- 09 擂台 + 音频 ----
s = slide_new(prs)
title(s, "同一首歌，六种判断", 132, 44)
text(s, 88, 196, 1400, "六个模型、同一批曲目、各自投票——内部管这个设定叫「PVP 擂台」",
     size_px=18, color=MUTED)
rows = []
for name, truth, cells in (
    ("073776", "International",
     [("Instrumental ✗ 23%", ACCENT), ("Folk ✗ 35%", ACCENT), ("Experimental ✗ 43%", ACCENT),
      ("Electronic ✗ 91%", ACCENT), ("Pop ✗ 29%", ACCENT)]),
    ("classical.00099", "classical",
     [("classical ✓ 82%", ACCENT2), ("classical ✓ 100%", ACCENT2), ("rock ✗ 77%", ACCENT),
      ("country ✗ 70%", ACCENT), ("jazz ✗ 46%", ACCENT)]),
    ("blues.00012", "blues",
     [("blues ✓ 65%", ACCENT2), ("blues ✓ 98%", ACCENT2), ("blues ✓ 85%", ACCENT2),
      ("blues ✓ 97%", ACCENT2), ("blues ✓ 97%", ACCENT2)]),
):
    rows.append([(name, INK, False, MONO), (truth, MUTED)] + [(t, c) for t, c in cells])
table(s, 88, 250, 1424, ["曲目", "真值", "SVM", "MLP", "CNN", "CRNN", "TF"], rows,
      size_px=19, head_size=15, row_h=70,
      aligns=["l", "l", "l", "l", "l", "l", "l"],
      col_w=[196, 176, 214, 190, 244, 214, 190])
playbtn(s, 88, 560, AUD / "073776_International.mp3", size_px=76,
        label="5 秒 · FMA 测试集 · 模型没见过", tc="")
footer(s, 9, TOTAL, "来源：results/fma/predictions/*_seed42_test.csv")

# ---- 10 归纳偏置 ----
s = slide_new(prs)
title(s, "模型不是白纸", 150, 44)
table(s, 88, 254, 1424, ["模型", "生来就信的假设", "数据少时", "数据多时"], [
    [("SVM / MLP", None, True), ("自己没有先验，但输入的人工特征里全是", None), ("稳，天花板低", MUTED), ("天花板低", MUTED)],
    [("CNN", None, True), ("局部性 + 平移等变", INK, True), ("强", None), ("强", None)],
    [("CRNN", None, True), ("局部性 + 循环层的记忆", INK, True), ("稍弱", None), ("更强", None)],
    [("轻量 Transformer", None, True), ("只有位置编码，几乎没有结构先验", INK, True),
     ("严重过拟合", ACCENT), ("待观察", MUTED)],
    [("AST", None, True), ("架构同上，预训练把先验灌进权重", INK, True),
     ("最强", ACCENT2), ("最强", ACCENT2)],
], size_px=21, row_h=76, aligns=["l", "l", "l", "l"], col_w=[320, 644, 240, 220],
   emphasize=(1, 2))
text(s, 88, 254 + 76 * 5 + 44, 1400, "先验越少，样本效率越低；先验越少，天花板越高。这是一枚硬币的两面。",
     size_px=18, color=MUTED)
footer(s, 10, TOTAL, "来源：报告 §7.6")

# ---- 11 三条证据 ----
s = slide_new(prs)
title(s, "先验越少，越吃数据", 150, 44)
table(s, 88, 264, 1424, None, [
    [("0.753 > 0.723 ≫ 0.542", INK, True, MONO), ("GTZAN 985 首：CNN 赢，轻量 TF 差 21 个点", MUTED)],
    [("0.492 > 0.470", INK, True, MONO), ("FMA 8 倍数据：CRNN 反超 CNN", MUTED)],
    [("0.814 / 0.590", INK, True, MONO), ("同一个 Transformer 架构，预训练后两榜第一", MUTED)],
], size_px=24, row_h=96, aligns=["l", "l"], col_w=[470, 954])
text(s, 88, 264 + 96 * 3 + 66, 1400, "问题不在架构，在数据与先验。", size_px=30,
     font=SERIF, bold=True, color=FLAG)
footer(s, 11, TOTAL, "来源：results/gtzan/metrics.csv · results/fma/metrics.csv")

# ---- 12 过拟合曲线 ----
s = slide_new(prs)
title(s, "那条过分自信的曲线", 150, 44)
cy = picture(s, 88, 236, 1090, FIG / "fig_curves.png",
             caption="GTZAN 训练 / 验证损失（细线为三个种子，粗线为均值）")
text(s, 88, cy + 26, 1400,
     "30 轮加到 60 轮：GTZAN +0.75 / +1.36，FMA −0.5 左右 —— 再训也这样。",
     size_px=19, color=MUTED)
footer(s, 12, TOTAL, "来源：results/logs/overnight/*.log · 报告 §5.3 表 13")

# ---- 13 鲁棒性 ----
s = slide_new(prs, BGALT)
title(s, "换个听法就翻车", 150, 44)
cy = picture(s, 88, 232, 1090, FIG / "fig_robust.png",
             caption="七种扰动下的准确率（n=147，不重新训练；seed 42 权重）；剪成 3 秒时 CRNN 掉 53.1 点、SVM 只掉 12.9 点")
text(s, 88, cy + 26, 1400,
     "0 dB 噪声下 CNN 从第一名掉到倒数第一：0.748 → 0.122，掉 62.6 个点；同一条件下 SVM 只掉 40.8 个点。",
     size_px=21, color=INK)

footer(s, 13, TOTAL, "来源：results/gtzan/robustness.csv（35 行）")

# ---- 14 旋钮 ----
s = slide_new(prs)
title(s, "能拧的旋钮，都拧过了", 150, 44)
table(s, 88, 268, 1424, ["试过什么", "结果"], [
    [("SpecAugment + 标签平滑", None, True), ("FMA 上最大 +0.7 个点，落在种子波动内", MUTED)],
    [("30 轮 → 60 轮", None, True), ("两个数据集互有胜负", MUTED)],
    [("学习率 / 批大小扫描", None, True), ("无稳定收益", MUTED)],
], size_px=22, row_h=80, aligns=["l", "l"], col_w=[560, 864])
text(s, 88, 268 + 80 * 3 + 60, 1400, "剩下的差距，不是调参能补的。", size_px=26, color=ACCENT)
footer(s, 14, TOTAL, "来源：报告 §5.3 表 13")

# ---- 15 转场 ----
s = slide_new(prs)
text(s, 88, 246, 1420, "它知道这是什么。", size_px=76, font=SERIF, bold=True, spacing=1.25)
y2 = 246 + 76 * 1.25 * 1.08 + 18
text(s, 88, y2, 1420, "它不知道接下来会响什么。", size_px=76, font=SERIF, spacing=1.25,
     color=MUTED)
st = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, X(88), X(y2 + 76 * 1.25 * 1.08 * 0.55),
                        X(760), X(2))
st.fill.solid(); st.fill.fore_color.rgb = RULE; st.line.fill.background(); st.shadow.inherit = False
y3 = y2 + 76 * 1.25 * 1.08 + 46
hline(s, 88, y3, 820, RULE)
text(s, 88, y3 + 26, 1200, "p(流派 | 音频)　　→　　p(下一个 Token | 前 5 秒)",
     size_px=20, font=MONO, color=MUTED)
footer(s, 15, TOTAL)

# ---- 16 符号化 ----
s = slide_new(prs)
title(s, "把声音拆成符号", 150, 44)
specs = [("采样率", "32 kHz"), ("帧率", "50 Hz（每帧 20 ms）"),
         ("码本", "4 个"), ("每帧符号", "4 个 Token")]
cy = 268
for k, v in specs:
    text(s, 88, cy, 300, k, size_px=19, color=MUTED)
    text(s, 400, cy, 400, v, size_px=21, font=MONO, align="r")
    cy += 21 * 1.32 * 1.08 + 18
    hline(s, 88, cy - 6, 712, RULE_SOFT)
text(s, 88, cy + 24, 712, "「预测音频」变成「预测下一个符号」——交叉熵直接能用。",
     size_px=19, color=MUTED)
bars = [("码本 0", 1.00, RGBColor(0x40, 0x3D, 0x3A)),
        ("码本 1", 0.72, RGBColor(0x82, 0x7F, 0x7C)),
        ("码本 2", 0.46, RGBColor(0xBC, 0xB9, 0xB4)),
        ("码本 3", 0.24, RGBColor(0xE4, 0xE1, 0xDC))]
by = 268
for lb, wf, col in bars:
    text(s, 880, by + 4, 110, lb, size_px=15, font=MONO, color=MUTED)
    rect(s, 1000, by, 512 * wf, 28, col, RULE)
    by += 46
text(s, 880, by + 12, 620, "残差矢量量化：第 1 个码本画轮廓，后 3 个逐级补细节 → 串成整数序列",
     size_px=15, color=MUTED)
footer(s, 16, TOTAL, "来源：报告 §6.2")

# ---- 17 两条流水线 ----
s = slide_new(prs, BGALT)
title(s, "两条流水线", 150, 44)
def flow(y, boxes, tag):
    x = 88
    for i, b in enumerate(boxes):
        w = 60 + len(b) * 15
        rect(s, x, y, w, 62, SURFACE, RULE)
        text(s, x + 18, y + 18, w - 24, b, size_px=19, font=MONO)
        x += w
        if i < len(boxes) - 1:
            text(s, x + 8, y + 16, 40, "→", size_px=22, color=MUTED)
            x += 44
    text(s, x + 16, y + 20, 420, tag, size_px=16, color=MUTED)

flow(258, ["5 秒前缀", "4×250 Token", "1 秒 · 4×50 Token"], "① 自己训的小模型")
flow(360, ["5 秒前缀（波形）", "3 秒波形"], "② MusicGen-small + 我们的 LoRA")
hline(s, 88, 470, 1424, RULE)
table(s, 88, 500, 1424, None, [
    [("① 指标", None, True), ("下一个符号猜对没有（Top-1）", MUTED)],
    [("② 指标", None, True), ("频谱离真实多远（Log-Mel RMSE）", MUTED)],
], size_px=21, row_h=64, aligns=["l", "l"], col_w=[380, 1044])
text(s, 88, 500 + 64 * 2 + 40, 1400, "这两个数不能互相比，也不能跟分类的 0.814 比。",
     size_px=19, color=ACCENT)
footer(s, 17, TOTAL, "来源：报告 §3.6 · §6.2")

# ---- 18 基线 ----
s = slide_new(prs)
title(s, "先立四个「笨」基线", 150, 44)
table(s, 88, 254, 1424, ["基线", "做法", "Top-1"], [
    [("均匀分布", None, True), ("词表内等概率瞎猜", MUTED), ("0.049%", None, False, MONO)],
    [("频率先验", None, True), ("永远猜训练集最高频符号", MUTED), ("1.16%", None, False, MONO)],
    [("Persistence", None, True), ("把上一帧原封不动重复一遍", MUTED), ("1.62%", None, False, MONO)],
    [("Bigram", None, True), ("数相邻符号谁跟着谁", MUTED), ("1.71%", None, False, MONO)],
    [("我们的模型", None, True), ("Transformer / LSTM", None),
     ("2.19% / 2.08%", INK, True, MONO)],
], size_px=21, row_h=74, aligns=["l", "l", "r"], col_w=[340, 764, 320],
   emphasize=(4,))
text(s, 88, 254 + 74 * 5 + 46, 1400, "打不过 Persistence，就说明它在假装学习。",
     size_px=19, color=ACCENT)
footer(s, 18, TOTAL, "来源：results/token_prediction_long/metrics.csv")

# ---- 19 Token 成绩 ----
s = slide_new(prs)
title(s, "学会了，但只学会一点点", 150, 44)
pic_bottom = picture(s, 88, 250, 760, FIG / "fig_token_top1.png", caption="free-running Top-1（3 种子均值）")
b1 = text(s, 900, 250, 600, "3.36%", size_px=44, font=MONO, color=INK)
text(s, 900, b1, 600, "看着答案考试（teacher-forced）", size_px=16, color=MUTED)
b2 = text(s, 900, 400, 620, "2.19%", size_px=64, font=MONO, color=FLAG)
text(s, 900, b2, 620, "自己往下接（free-running）", size_px=16, color=MUTED)
b3 = text(s, 900, 576, 600, "−35%", size_px=44, font=MONO, color=ACCENT)
text(s, 900, b3, 620, "一步之差", size_px=16, color=MUTED)
text(s, 88, pic_bottom + 26, 1400,
     "2.19% 是瞎猜的 45 倍，也是一百个符号里猜对两个。", size_px=19, color=MUTED)
footer(s, 19, TOTAL, "来源：results/token_prediction_long/metrics.csv")

# ---- 20 码本 ----
s = slide_new(prs)
title(s, "码本也有难易之分", 150, 44)
cy = picture(s, 88, 232, 820, FIG / "fig_codebook.png", caption="四个码本各自的 free-running Top-1")
table(s, 88, cy + 18, 1424, None, [
    [("码本 0 → 码本 3", None, True), ("2.8% → 1.5%，由粗到细逐级下降", MUTED)],
    [("这说明", None, True), ("结构对得上 RVQ；如果是在瞎猜，四个码本应该一样烂", MUTED)],
], size_px=20, row_h=52, aligns=["l", "l"], col_w=[380, 1044])
footer(s, 20, TOTAL, "来源：results/token_prediction_long/metrics.csv")

# ---- 21 听四段 + 音频 ----
s = slide_new(prs)
title(s, "请听四段", 140, 44)
tags = [("01", "前缀", "模型看到的 5 秒"), ("02", "真实", "唱片里原本的后 3 秒"),
        ("03", "原始续写", "MusicGen-small"), ("04", "适配后续写", "+ LoRA r=8")]
tx = 88
for n_, t_, d_ in tags:
    hline(s, tx, 232, 318, RULE)
    text(s, tx, 246, 318, n_, size_px=15, font=MONO, color=MUTED)
    text(s, tx, 286, 318, t_, size_px=25, bold=True)
    text(s, tx, 286 + 25 * 1.3 * 1.08 + 8, 318, d_, size_px=16, color=MUTED)
    tx += 358
playbtn(s, 88, 470, AUD / "AB__000995_Experimental.mp3", size_px=76,
        label="段间隔 0.6 秒 · 000995 Experimental", tc="0:16")
b = text(s, 760, 462, 480, "20.85 → 17.19", size_px=44, font=MONO, align="r")
text(s, 700, b, 540, "Log-Mel RMSE · 八首里改善最大（−3.65）",
     size_px=16, color=MUTED, align="r")
footer(s, 21, TOTAL, "来源：demo/audio/README.md §4 · 报告 §6.9")

# ---- 22 盲听 ----
s = slide_new(prs, BGALT)
title(s, "三条里只有一条是真的", 140, 44)
text(s, 88, 204, 1400, "同一段 5 秒前缀，只有后半段不同", size_px=20, color=MUTED)
centers = [340, 700, 1060]
for cx, letter, f in zip(centers, "ABC",
                         ["REAL__000995_Experimental.mp3",
                          "GENbase__000995_Experimental.mp3",
                          "GENft__000995_Experimental.mp3"]):
    playbtn(s, cx - 38, 300, AUD / f, size_px=76)
    text(s, cx - 40, 396, 80, letter, size_px=30, font=MONO, align="c")
    text(s, cx - 70, 448, 140, "0:08", size_px=15, font=MONO, color=MUTED, align="c")
rect(s, 560, 530, 320, 58, SURFACE, RULE)
text(s, 560, 548, 320, "揭晓答案", size_px=19, align="c")
text(s, 940, 548, 520, "（点击或按页面说明口头公布）", size_px=15, color=MUTED)
footer(s, 22, TOTAL, "三个文件已重命名，页面不显示原始文件名")

# ---- 23 反例 ----
s = slide_new(prs)
title(s, "这一条，改完反而更差", 150, 44)
playbtn(s, 88, 260, AUD / "AB__001482_Electronic.mp3", size_px=76,
        label="同样四段：前缀 → 真实 → 原始 → 适配", tc="0:16")
b = text(s, 88, 400, 1200, "27.13 → 27.46", size_px=76, font=MONO, color=ACCENT)
text(s, 88, b + 6, 1200, "8 首测试曲里唯一退步的一首（+0.33）", size_px=21, color=MUTED)
footer(s, 23, TOTAL, "来源：demo/audio/README.md §4")

# ---- 24 标签布局 ----
s = slide_new(prs)
title(s, "最贵的一课：标签写错了", 150, 44)
table(s, 88, 254, 1424, ["loss", "是什么情况"], [
    [("11.08", ACCENT, True, MONO), ("错误布局：比均匀分布还差", INK, True)],
    [("5.46", ACCENT2, True, MONO), ("正确布局", INK, True)],
    [("7.62", MUTED, False, MONO), ("均匀分布（乱猜的上界）", MUTED)],
], size_px=30, head_size=16, row_h=76, aligns=["r", "l"], col_w=[260, 1164])
hline(s, 88, 254 + 76 * 3 + 40, 1424, RULE)
table(s, 88, 254 + 76 * 3 + 70, 1424, None, [
    [("症状", None, True), ("不报错，loss 只停在偏高的值上，看起来像「还没训够」", MUTED)],
    [("改对之后", None, True), ("19.23 → 17.91，7/8 首改善，t = −3.36", MUTED)],
], size_px=20, row_h=60, aligns=["l", "l"], col_w=[300, 1124])
footer(s, 24, TOTAL, "来源：报告 §6.10")

# ---- 25 三旋钮 ----
s = slide_new(prs, BGALT)
title(s, "三个旋钮，全都不灵", 150, 44)
table(s, 88, 254, 1424, ["配置", "RMSE", "相对原始", "配对检验"], [
    [("原始模型", None, True), ("19.23", None, False, MONO), ("—", MUTED, False, MONO), ("—", MUTED)],
    [("LoRA r=8，1200 步", None, True), ("17.91", INK, True, MONO), ("−1.32", ACCENT2, True, MONO),
     ("t = −3.36，显著", ACCENT2, True)],
    [("LoRA 3600 步", None, True), ("17.71", None, False, MONO), ("−1.52", None, False, MONO),
     ("t = −0.47，不显著", MUTED)],
    [("LoRA 全窗口", None, True), ("17.64", None, False, MONO), ("−1.59", None, False, MONO),
     ("t = −0.33，不显著", MUTED)],
    [("LoRA r=16", None, True), ("18.59", None, False, MONO), ("−0.64", None, False, MONO),
     ("t = −1.06，不显著", MUTED)],
], size_px=21, row_h=72, aligns=["l", "r", "r", "l"], col_w=[400, 200, 240, 584],
   emphasize=(1,))
text(s, 88, 254 + 72 * 5 + 44, 1400, "NLL 一直在降，音频距离没跟着降。", size_px=19, color=ACCENT)
footer(s, 25, TOTAL, "来源：报告 §6.9 / §6.11")

# ---- 26 三秒 ----
s = slide_new(prs)
title(s, "为什么只有 3 秒", 150, 44)
table(s, 88, 254, 1424, ["续写长度", "耗时（RTX 4060）", "情况"], [
    [("3 秒", None, True), ("5.6 s", None, False, MONO), ("8/8 曲目稳定，主结果口径", None)],
    [("6 秒", None, True), ("16.3 s", None, False, MONO), ("可生成", MUTED)],
    [("8 秒", None, True), ("18.3 s", None, False, MONO), ("可生成", MUTED)],
    [("12 秒", None, True), ("35.4 s", None, False, MONO), ("可生成", MUTED)],
    [("16 / 20 秒", ACCENT, True), ("—", None, False, MONO), ("崩：位置索引越界", ACCENT)],
], size_px=21, row_h=70, aligns=["l", "r", "l"], col_w=[320, 400, 704], emphasize=(0,))
b = text(s, 88, 254 + 70 * 5 + 40, 1400,
         "崩的原因不在长度：pad/bos 的 id 是 2048，而词表只到 2047。", size_px=19, color=MUTED)
text(s, 88, b + 4, 1400, "3 秒是「八首都能稳定跑完」的口径，不是能力上限。",
     size_px=19, color=MUTED)
footer(s, 26, TOTAL, "来源：demo/README.md §3.5")

# ---- 27 指标陷阱 ----
s = slide_new(prs)
title(s, "指标不等于好听", 150, 44)
cy = picture(s, 88, 232, 780, FIG / "generation_spec.png",
             caption="前缀 | 真实 | 原始续写 | 适配后续写（Log-Mel）")
text(s, 930, 300, 460, "指标", size_px=17, color=MUTED)
text(s, 930, 328, 460, "Log-Mel RMSE——离原曲多近", size_px=20, color=INK)
text(s, 930, 400, 460, "没做", size_px=17, color=MUTED)
text(s, 930, 428, 460, "规模化主观听感评分", size_px=20, color=INK)
text(s, 930, 520, 460, "所以我们只说「客观距离变近了」，不说「变好听了」。",
     size_px=19, color=ACCENT)
footer(s, 27, TOTAL, "来源：报告 §6.7 / §7.9")

# ---- 28 结案 ----
s = slide_new(prs)
title(s, "结案", 132, 44)
lines = [("判别式：最大的单点收益是预训练，不是架构", 36, INK, True),
         ("模型排序会随数据规模翻转：985 首 CNN 赢，7,994 首 CRNN 反超", 36, INK, True),
         ("泄漏能让结论虚高十个点：0.470 → 0.568", 36, INK, True),
         ("生成式：能预测，学得很少，修好对齐后有可测改善，三个旋钮无效", 26, MUTED, False)]
cy = 226
for txt_, sz, col, bold in lines:
    rect(s, 90, cy + sz * 0.42, 11, 11, INK)
    cy = text(s, 130, cy, 1360, txt_, size_px=sz, color=col, bold=bold) + 12
hline(s, 88, cy + 10, 1424, RULE)
cy += 44
b = text(s, 88, cy, 1400,
         "五个负结果都留在报告里：正则化无效 · 轻量 TF 小数据崩 · 噪声下 CNN 垫底 · LoRA 初版失效 · 三旋钮无效",
         size_px=18, color=MUTED)
text(s, 88, b + 30, 1400, "我们没把「跑通」当成「做好」。", size_px=26, font=SERIF, bold=True)
footer(s, 28, TOTAL, "梁煜炜 · 贾智涵 · 王薪羽 · 陈泽玉　｜　谢谢")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print("saved", OUT, f"{OUT.stat().st_size/1024/1024:.1f} MB", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
