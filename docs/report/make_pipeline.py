from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

for f in ("/mnt/c/Windows/Fonts/msyh.ttc", "/home/weizai0v0/.fonts/wqy-microhei.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
C = "#c2703d"; G = "#6b8ea8"; D = "#333333"

def box(x, y, w, h, text, color="#ffffff", edge="#888888", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                fc=color, ec=edge, lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=D)

def arrow(x1, y1, x2, y2, color="#777777"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=color))

box(0.15, 2.55, 1.55, 0.85, "GTZAN 985 首\nFMA-small 8,000 首")
box(2.05, 2.55, 1.75, 0.85, "中心 5 秒片段\n22,050 Hz 单声道")
box(4.2, 3.35, 2.0, 0.8, "209 维人工特征\n(MFCC/谱/节奏)", color="#eef3f7")
box(4.2, 1.75, 2.0, 0.8, "64×216 Log-Mel\n(增益归一化)", color="#fbefe9")
box(6.55, 3.35, 1.75, 0.8, "SVM / MLP", color="#eef3f7")
box(6.55, 2.45, 1.75, 0.8, "CNN / CRNN /\n轻量 Transformer", color="#fbefe9")
box(6.55, 1.5, 1.75, 0.75, "16 kHz 补零 10.24 s\n→ 冻结 AST + 线性头", color="#f7f2e8")
box(8.75, 2.35, 1.95, 1.55, "统一协议\n按曲目 70/15/15\n验证集选型\n测试集单次评估\nAcc / Macro-F1 / 成本")
box(4.2, 0.35, 4.1, 0.62, "同一划分、同一批 5 秒中心片段、预处理只在训练集拟合", color="#f2f2f2")

arrow(1.70, 2.95, 2.05, 2.95)
arrow(3.80, 2.95, 4.2, 3.55)
arrow(3.80, 2.85, 4.2, 2.15)
arrow(6.20, 3.72, 6.55, 3.72)
arrow(6.20, 2.80, 6.55, 2.80)
arrow(6.20, 1.90, 6.55, 1.85)
arrow(8.30, 3.72, 8.75, 3.35)
arrow(8.30, 2.80, 8.75, 3.05)
arrow(8.30, 1.85, 8.75, 2.75)
ax.set_title("图 1  实验流程：统一数据协议下的六类模型对比", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figures/pipeline.png", dpi=180, bbox_inches="tight")
print("saved pipeline.png")
