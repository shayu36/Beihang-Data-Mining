"""Render the shared normal-reference pipeline using the report's Chinese typeface."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname=str(ROOT / "fonts" / "SimHei.ttf"))
INK = "#172d40"
MUTED = "#486273"
LINE = "#628093"
WIDTH, HEIGHT = 7.2, 2.0


def main():
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()

    def label(x, y, text, size=9, align="center", color=INK):
        ax.text(x, y, text, ha=align, va="center", fontproperties=FONT, fontsize=size, color=color)

    def box(x, y, w, h, title, note="", fill="#f0f5f7", edge="#acc0cb"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                               linewidth=0.65, facecolor=fill, edgecolor=edge)
        ax.add_patch(patch)
        if note:
            label(x + w / 2, y + h * 0.65, title, 9.6)
            label(x + w / 2, y + h * 0.29, note, 7.7, color=MUTED)
        else:
            label(x + w / 2, y + h / 2, title, 9.1)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8.5,
                                     lw=0.9, edgecolor=LINE, facecolor=LINE, shrinkA=0, shrinkB=0))

    label(0.04, 1.83, "输入与描述子", 10.4, "left")
    label(3.10, 1.83, "同参检测器", 10.4, "left")
    label(7.15, 1.83, "各域分别用正常样本建库和校准", 8.1, "right", MUTED)
    for start, end in [(0.05, 2.70), (3.10, 7.15)]:
        ax.plot([start, end], [1.66, 1.66], lw=0.55, color="#bdcbd2")

    top, bottom = 1.35, 0.79
    for y, source, descriptors in [
        (top, "冻结的图像编码器", "256 个局部向量"),
        (bottom, "一条六维记录", "1 个六维向量"),
    ]:
        box(0.04, y - 0.19, 1.27, 0.38, source)
        arrow((1.32, y), (1.52, y))
        box(1.54, y - 0.19, 1.16, 0.38, descriptors)
        ax.plot([2.70, 2.88], [y, y], lw=0.9, color=LINE)

    mid = (top + bottom) / 2
    ax.plot([2.88, 2.88], [bottom, top], lw=0.9, color=LINE)
    arrow((2.88, mid), (3.08, mid))
    core = {"fill": "#eaf4f0", "edge": "#8eb1a3"}
    box(3.10, mid - 0.32, 1.18, 0.64, "5 近邻距离", "对每个向量计算", **core)
    arrow((4.29, mid), (4.48, mid))
    box(4.50, mid - 0.32, 1.19, 0.64, "最高 2% 均值", "图像 6 块，表格 1 项", **core)
    arrow((5.70, mid), (5.89, mid))
    box(5.91, mid - 0.32, 1.24, 0.64, "正常校准上尾秩", "p ≤ 0.05 判为异常", **core)

    ax.plot([0.05, 7.15], [0.49, 0.49], lw=0.55, color="#bdcbd2")
    label(0.05, 0.25, "可选适配", 8.8, "left", MUTED)
    label(1.20, 0.25, "图像　局部距离与全局 SVM 分数融合", 8.6, "left")
    label(4.30, 0.25, "表格　用方向描述子刻画属性联合偏离", 8.6, "left")

    prefix = ROOT / "figures" / "shared-pipeline"
    prefix.parent.mkdir(exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(prefix.with_suffix(".png"), dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
