"""Chinese diagram + editable equations + their identical LaTeX source.

Canonical mathematical source: course-report/sec/task3-method.tex.
Validation selection: course-report/sec/task3-experiments.tex.
No training or empirical results are changed by this visual reconstruction.
"""
from pathlib import Path
import re
import json
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from native_math import equation, M

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course-report/figure-source/模型主图_中文版.pptx"
TEX = ROOT / "course-report/figure-source/模型主图_数学符号与公式.tex"
METHOD = ROOT / "course-report/sec/task3-method.tex"
EXPERIMENT = ROOT / "course-report/sec/task3-experiments.tex"
W, H = 1920, 1080
EMU = Inches(16) / W
PT = 16 * 72 / W
INK = "19232C"


def source_eq(path, label):
    text = path.read_text(encoding="utf-8")
    for body in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S):
        if rf"\label{{{label}}}" in body:
            body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
            return re.sub(r"\s+", " ", body)
    raise ValueError(f"Equation not found: {label}")


norm = source_eq(METHOD, "eq:standardization").split(r"\qquad")
ridge_parts = source_eq(METHOD, "eq:ridge_prediction").split(r"\qquad")
fusion_parts = source_eq(METHOD, "eq:fusion").split(r"\qquad")

# These LaTeX strings are the only mathematical authoring source for both files.
FORM = {}


def add(key, title, latex, reference):
    FORM[key] = {"title": title, "latex": latex.strip().rstrip(",."), "source": reference}


add("window", "输入窗口与维度", r"\mathbf{X}_i\in\mathbb{R}^{L\times F},\quad L=12,\ F=21", "eq:window；实验配置")
add("standardize_x", "逐特征标准化", norm[0], "eq:standardization，第一式")
add("standardize_y", "目标标准化", norm[1], "eq:standardization，第二式")
add("tensor", "标准化输入窗口", r"\widetilde{\mathbf{X}}_i\in\mathbb{R}^{12\times21}", "eq:standardization；L=12、F=21")
add("hidden", "GRU 隐藏维度", r"H=128", "实验配置")
add("dropout", "层间及回归头 Dropout", r"\mathcal{D}_{p},\quad p=0.2", "eq:stacking、eq:head_output；实验配置")
add("head_width", "回归头结构", r"H\rightarrow d\rightarrow1,\quad d=64", "eq:head_hidden、eq:head_output；实验配置")
add("last_state", "取第二层最后位置状态", r"\mathbf{h}_{i,L}^{(2,k)}", "eq:head_hidden")
for k in range(1, 6):
    add(f"member_{k}", f"第 {k} 个成员的标准化预测",
        rf"f_{{\theta_{k}}}(\widetilde{{\mathbf{{X}}}}_i)", "eq:head_output，k 取具体编号")
add("ensemble", "五成员预测均值", source_eq(METHOD, "eq:ensemble"), "eq:ensemble（逐式复制）")
add("member_count", "成员数量", r"K=5", "实验配置")
add("g", "GRU 集成预测", r"g_i", "eq:ensemble")
add("flatten", "按行展平的向量", ridge_parts[0], "eq:ridge_prediction，第一式")
add("flatten_dimension", "展平后的维度", r"LF=252", "L=12、F=21")
add("ridge", "岭回归预测", ridge_parts[1], "eq:ridge_prediction，第二式")
add("alpha", "岭回归正则系数", r"\alpha=1", "实验配置")
add("r", "岭回归预测标量", r"r_i", "eq:ridge_prediction")
add("fusion", "标准化空间内的加权融合", fusion_parts[0], "eq:fusion，第一式")
add("weight_domain", "融合权重取值区间", fusion_parts[1], "eq:fusion，第二式")
add("weight_g", "GRU 集成权重", r"w^*=0.95", "实验配置与 results/metrics.json")
add("weight_r", "岭回归权重", r"1-w^*=0.05", "由 w*=0.95 得到")
add("fused", "选定权重下的标准化融合预测", r"\widehat{\widetilde{y}}_i(w^*)", "eq:fusion，取 w=w*")
add("inverse", "目标浓度还原", r"\widehat{y}_i=\mu_y+s_y\widehat{\widetilde{y}}_i(w^*)", "eq:final_prediction；用 eq:ensemble、eq:fusion 等价写成紧凑形式")
add("output", "单步 CO₂ 浓度预测", r"\widehat{y}_i", "eq:final_prediction，单位 ppm")
add("gru_loss", "GRU 成员训练损失", source_eq(METHOD, "eq:gru_loss"), "eq:gru_loss（逐式复制）")
add("ridge_loss", "岭回归训练目标", source_eq(METHOD, "eq:ridge_loss"), "eq:ridge_loss（逐式复制，仅删除 split 排版环境）")
add("select", "验证集选择全局融合权重", source_eq(EXPERIMENT, "eq:fusion_selection"), "实验部分 eq:fusion_selection（逐式复制）")
add("grid", "候选权重集合", r"\mathcal{W}=\{0,0.05,\ldots,0.95,1\}", "实验配置与训练细节")
add("multiply", "标量乘法节点", r"\times", "加权融合运算")
add("plus", "求和节点", r"+", "加权融合运算")
add("co2", "化学符号与单位", r"\mathrm{CO}_{2},\quad\mathrm{ppm}", "任务定义")
add("relu", "回归头激活函数", r"\operatorname{ReLU}", "eq:head_hidden")

# Normalize only TeX line-breaking markup, keeping the exact mathematical terms.
FORM["ridge_loss"]["latex"] = re.sub(
    r"\\begin\{split\}|\\end\{split\}|&|\\\\", "", FORM["ridge_loss"]["latex"]
).strip()

SYMBOLS = [
 (r"i", "监督样本索引"), (r"t", "观测行索引"), (r"\tau", "窗口内部位置索引"),
 (r"j", "特征索引"), (r"j_*", "CO₂ 目标特征索引"), (r"k", "GRU 成员索引，1 至 K"),
 (r"L", "窗口长度，本模型为 12"), (r"F", "特征数，本模型为 21"),
 (r"H", "GRU 隐藏维度，本模型为 128"), (r"d", "回归头隐藏维度，本模型为 64"),
 (r"K", "GRU 成员数，本模型为 5"), (r"p", "Dropout 丢弃概率，本模型为 0.2"),
 (r"\mathbf{X}_i", "第 i 个多变量输入窗口"), (r"X_{i,\tau,j}", "窗口内位置 τ 的第 j 个特征值"),
 (r"\widetilde{\mathbf{X}}_i", "标准化后的输入窗口"),
 (r"\widetilde{X}_{i,\tau,j}", "标准化后的逐特征数值"),
 (r"y_i", "真实 CO₂ 浓度，单位 ppm"), (r"\widetilde{y}_i", "标准化目标值"),
 (r"\widehat{y}_i", "反标准化后的预测浓度，单位 ppm"),
 (r"\widehat{\widetilde{y}}_i(w)", "权重为 w 时的标准化融合预测"),
 (r"\mu_j,\ s_j", "第 j 个特征的训练均值和标准差"),
 (r"\mu_y,\ s_y", "目标 CO₂ 特征的训练均值和标准差"),
 (r"\theta_k", "第 k 个 GRU 及其回归头的参数集合"),
 (r"f_{\theta_k}", "第 k 个完整非线性预测器"),
 (r"f_{\theta_k}(\widetilde{\mathbf{X}}_i)", "第 k 个成员的标准化标量预测"),
 (r"\mathbf{h}_{i,L}^{(2,k)}", "第 k 个成员的第二层 GRU 在最后位置 L 的隐藏状态"),
 (r"\mathcal{D}_p", "丢弃概率为 p 的 Dropout 算子"),
 (r"g_i", "K 个 GRU 预测值的算术平均"), (r"r_i", "岭回归分支的标准化预测"),
 (r"\mathbf{q}_i", "按行展平后的 LF 维特征向量"),
 (r"\operatorname{vec}_{\mathrm{row}}", "按行展平算子"),
 (r"\boldsymbol{\beta},\ b", "岭回归系数向量与截距"),
 (r"\alpha", "岭回归 L2 正则系数，本模型为 1"),
 (r"w,\ 1-w", "GRU 集成与岭回归的组合权重"),
 (r"w^*", "验证集 MSE 最小的全局权重，保存值为 0.95"),
 (r"\mathcal{W}", "离散候选权重集合，共 21 个候选值"),
 (r"\mathcal{I}_{\mathrm{tr}}", "训练样本索引集"),
 (r"M_{\mathrm{tr}}=|\mathcal{I}_{\mathrm{tr}}|", "训练样本数"),
 (r"\mathcal{I}_{\mathrm{va}}", "验证样本索引集"),
 (r"|\mathcal{I}_{\mathrm{va}}|", "验证样本数"),
 (r"\mathcal{L}_{\mathrm{G}}(\theta_k)", "GRU 成员的标准化均方误差"),
 (r"\mathcal{L}_{\mathrm{R}}(\boldsymbol{\beta},b)", "岭回归残差平方和与 L2 正则项之和"),
 (r"\mathbb{R}^{L\times F},\ \mathbb{R}^{LF}", "实数矩阵空间与实数向量空间"),
 (r"(\cdot)^\top", "转置"), (r"\lVert\boldsymbol{\beta}\rVert_2^2", "系数向量的平方 L2 范数"),
 (r"\sum", "求和"), (r"\arg\min", "使目标函数最小的自变量取值"),
 (r"\in,\ [0,1],\ \{\cdots\}", "集合隶属、闭区间、离散集合"),
 (r"\times,\ +,\ -", "标量乘法、加法、减法"),
 (r"\operatorname{ReLU}", "回归头使用的 ReLU 激活函数"),
 (r"\mathrm{CO}_{2},\ \mathrm{ppm}", "二氧化碳及浓度单位")
]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(16), Inches(9)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
shown = []


def u(v):
    return round(v * EMU)


def no_effect(s):
    for ref in s._element.xpath("./p:style/a:effectRef"):
        ref.set("idx", "0")
    s._element.spPr.append(OxmlElement("a:effectLst"))


def group(name, parent=None):
    g = (slide.shapes if parent is None else parent.shapes).add_group_shape()
    g.name = name
    return g


def box(g, name, x, y, w, h, fill="FFFFFF", color="65757C", radius=True,
        dash=False, width=1.6, circle=False):
    kind = MSO_SHAPE.OVAL if circle else (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    q = g.shapes.add_shape(kind, u(x), u(y), u(w), u(h))
    q.name = name
    no_effect(q)
    if radius and not circle:
        q.adjustments[0] = 0.13
    if fill:
        q.fill.solid()
        q.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        q.fill.background()
    if color:
        q.line.color.rgb = RGBColor.from_string(color)
        q.line.width = Pt(width * PT)
        if dash:
            q.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        q.line.fill.background()
    return q


def txt(g, name, x, y, w, h, value, size=26, bold=False, color=INK,
        align=PP_ALIGN.CENTER):
    q = g.shapes.add_textbox(u(x), u(y), u(w), u(h))
    q.name = name
    tf = q.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    r = p.add_run()
    r.text = value
    r.font.name = "Microsoft YaHei"
    r.font.size = Pt(size * PT)
    r.font.bold = bold
    r.font.color.rgb = RGBColor.from_string(color)
    for tag in ("a:ea", "a:cs"):
        el = OxmlElement(tag)
        el.set("typeface", "Microsoft YaHei")
        r._r.get_or_add_rPr().append(el)
    return q


def eq(g, key, x, y, w, h, size=26, latex=None):
    latex = FORM[key]["latex"] if latex is None else latex
    if key not in shown:
        shown.append(key)
    return equation(g.shapes, f"公式_{key}", latex, u(x), u(y), u(w), u(h), size * PT)


def line(g, name, x1, y1, x2, y2, arrow=False, color=INK, dash=False, width=2.4):
    q = g.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, u(x1), u(y1), u(x2), u(y2))
    q.name = name
    no_effect(q)
    q.line.color.rgb = RGBColor.from_string(color)
    q.line.width = Pt(width * PT)
    if dash:
        q.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        ar = OxmlElement("a:tailEnd")
        for k, v in {"type": "triangle", "w": "med", "len": "med"}.items():
            ar.set(k, v)
        q._element.spPr.get_or_add_ln().append(ar)
    return q


def path(g, name, pts, arrow=False, **kwargs):
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        line(g, f"{name}_{k}", *a, *b, arrow=arrow and k == len(pts) - 2, **kwargs)


def labelbox(g, name, x, y, w, h, value, size=26, fill="FFFFFF", color=INK):
    q = box(g, name, x, y, w, h, fill, color)
    txt(g, name + "_文字", x + 5, y + 3, w - 10, h - 6, value, size, True)
    return q


bg = group("01_分区背景")
box(bg, "输入分区", 25, 62, 370, 690, "EEF5FC", "6597C3", dash=True)
box(bg, "GRU分区", 425, 62, 824, 450, "ECF7F0", "399E7D", dash=True)
box(bg, "岭回归分区", 425, 535, 824, 217, "FFF7E1", "CA9A30", dash=True)
box(bg, "融合分区", 1280, 62, 615, 690, "F4EFFA", "9D77C1", dash=True)
box(bg, "训练分区", 25, 783, 1870, 263, "FCEFF4", "D57F9A", dash=True)
title = group("02_中文标题")
txt(title, "主标题", 410, 8, 855, 42, "GRU 多成员集成与岭回归融合", 34, True)
txt(title, "输入标题", 45, 80, 330, 42, "多变量观测输入", 31, True)
txt(title, "GRU标题", 447, 76, 540, 42, "五个独立训练的 GRU 预测器", 30, True, align=PP_ALIGN.LEFT)
txt(title, "岭回归标题", 447, 546, 550, 40, "岭回归线性分支", 30, True, align=PP_ALIGN.LEFT)
txt(title, "融合标题", 1300, 80, 575, 42, "加权融合与 CO₂ 浓度预测", 30, True)
txt(title, "训练标题", 47, 794, 650, 37, "训练目标与验证集选权", 29, True, align=PP_ALIGN.LEFT)

inputs = group("03_输入与共享标准化")
eq(inputs, "window", 46, 126, 328, 45, 23)
txt(inputs, "历史目标", 48, 174, 324, 33, "包含历史 CO₂ 浓度", 25)
grid = group("原始窗口_12行21列", inputs)
for row in range(12):
    for col in range(21):
        fill = "E4D7F0" if col == 20 else ("E2EEF8" if (row + col) % 3 else "F9FCFF")
        box(grid, f"观测_{row+1}_{col+1}", 78 + 13 * col, 239 + 12.8 * row,
            13, 12.8, fill, "567084", radius=False, width=0.6)
txt(inputs, "变量轴", 99, 205, 240, 29, "21 个变量", 22)
txt(inputs, "记录轴", 32, 275, 35, 90, "记\n录", 22)
line(inputs, "输入到标准化", 215, 394, 215, 427, arrow=True)
labelbox(inputs, "共享标准化", 52, 428, 316, 43, "使用训练统计量标准化", 24)
eq(inputs, "standardize_x", 40, 475, 340, 72, 23)
eq(inputs, "standardize_y", 45, 552, 330, 67, 24)
tensor = group("标准化窗口张量", inputs)
for off in [27, 18, 9, 0]:
    box(tensor, f"张量层_{off}", 69+off, 659-off, 253, 68, "DEECF8", "47667D", radius=False)
eq(tensor, "tensor", 77, 674, 237, 41, 26)
flow = group("04_共享输入连线")
path(flow, "共享标准化输入", [(322, 692), (409, 692), (409, 358), (455, 358)], arrow=True)
line(flow, "输入至岭回归", 409, 620, 470, 620, arrow=True)

gru = group("05_GRU成员与均值")
box(gru, "循环层配置", 477, 128, 209, 79, "F8FCF9", "80998B")
txt(gru, "层数", 484, 131, 98, 31, "两层 GRU", 21, True)
eq(gru, "hidden", 580, 131, 99, 31, 22)
eq(gru, "dropout", 485, 166, 193, 32, 22)
box(gru, "回归头配置", 715, 128, 283, 79, "F8FCF9", "80998B")
eq(gru, "head_width", 721, 132, 271, 32, 22)
eq(gru, "last_state", 722, 169, 99, 31, 21)
eq(gru, "relu", 818, 169, 77, 31, 20)
eq(gru, "dropout", 888, 169, 103, 31, 19, latex=r"\mathcal{D}_{p}")
ys = [238, 282, 326, 370, 414]
line(gru, "成员共享输入总线", 455, ys[0], 455, ys[-1])
line(gru, "预测聚合总线", 1010, ys[0], 1010, ys[-1])
for k, y in enumerate(ys, 1):
    member = group(f"GRU成员_{k}", gru)
    line(member, "进入GRU", 455, y, 480, y, arrow=True)
    labelbox(member, "GRU", 480, y-20, 125, 40, f"GRU {k}", 25, "D8EDDF")
    line(member, "GRU到回归头", 605, y, 652, y, arrow=True)
    labelbox(member, "回归头", 652, y-20, 145, 40, f"回归头 {k}", 24, "E6F2E9")
    line(member, "回归头输出", 797, y, 825, y, arrow=True)
    eq(member, f"member_{k}", 830, y-24, 168, 47, 24)
    line(member, "预测到总线", 998, y, 1010, y)
line(gru, "总线到均值", 1010, 326, 1032, 326, arrow=True)
box(gru, "均值聚合器", 1032, 242, 195, 166, "F9FCFA", INK)
txt(gru, "均值标题", 1041, 252, 177, 36, "预测均值", 26, True)
eq(gru, "g", 1046, 298, 167, 44, 32)
eq(gru, "member_count", 1046, 360, 167, 31, 24)
eq(gru, "ensemble", 492, 431, 724, 66, 24)

ridge = group("06_岭回归分支")
labelbox(ridge, "按行展平", 470, 597, 130, 46, "按行展平", 24)
line(ridge, "展平到向量", 600, 620, 628, 620, arrow=True)
eq(ridge, "flatten", 631, 590, 345, 58, 24)
line(ridge, "向量到回归", 979, 620, 1004, 620, arrow=True)
labelbox(ridge, "岭回归", 1004, 597, 136, 46, "岭回归", 25)
line(ridge, "线性预测输出", 1140, 620, 1170, 620, arrow=True)
eq(ridge, "r", 1173, 594, 52, 47, 30)
eq(ridge, "flatten_dimension", 656, 671, 177, 40, 26)
eq(ridge, "ridge", 842, 664, 289, 51, 28)
eq(ridge, "alpha", 1117, 674, 110, 38, 25)

fusion = group("07_加权融合与还原")
txt(fusion, "凸组合说明", 1300, 130, 575, 30, "两条分支在标准化空间内进行凸组合", 23)
eq(fusion, "fusion", 1296, 164, 583, 65, 31)
eq(fusion, "weight_domain", 1684, 239, 168, 35, 23)
eq(fusion, "g", 1250, 275, 57, 41, 28)
path(fusion, "集成到权重", [(1227, 314), (1355, 314)], arrow=True)
path(fusion, "线性到权重", [(1227, 620), (1264, 620), (1264, 567), (1355, 567)], arrow=True)
for tag, y, key in [("GRU", 314, "weight_g"), ("岭回归", 567, "weight_r")]:
    eq(fusion, key, 1299, y-68, 158, 42, 25)
    line(fusion, tag+"权重端口", 1377, y-26, 1377, y-22, width=2)
    box(fusion, tag+"乘法节点", 1355, y-22, 44, 44, "FFFFFF", INK, circle=True, width=2)
    eq(fusion, "multiply", 1360, y-20, 34, 40, 32)
path(fusion, "GRU加权输出", [(1399, 314), (1460, 314), (1460, 384), (1492, 384)], arrow=True)
path(fusion, "岭回归加权输出", [(1399, 567), (1460, 567), (1460, 412), (1492, 412)], arrow=True)
box(fusion, "求和节点", 1488, 376, 44, 44, "FFFFFF", INK, circle=True, width=2)
eq(fusion, "plus", 1493, 378, 34, 39, 32)
line(fusion, "求和到融合预测", 1532, 398, 1560, 398, arrow=True)
box(fusion, "融合预测", 1560, 362, 303, 73, "FFFFFF", INK)
txt(fusion, "融合预测标签", 1567, 364, 289, 27, "标准化融合预测", 21, True)
eq(fusion, "fused", 1571, 391, 281, 40, 29)
line(fusion, "融合到还原", 1710, 436, 1710, 472, arrow=True)
labelbox(fusion, "反标准化", 1570, 473, 281, 44, "目标浓度反标准化", 25)
eq(fusion, "inverse", 1493, 524, 380, 72, 28)
line(fusion, "还原到输出", 1710, 601, 1710, 630, arrow=True)
box(fusion, "输出标量", 1696, 631, 28, 28, "FFFFFF", INK, circle=True, width=2)
eq(fusion, "output", 1743, 625, 59, 38, 28)
txt(fusion, "预测结果", 1516, 668, 360, 34, "下一条记录的 CO₂ 浓度", 25, True)
txt(fusion, "单位", 1613, 705, 188, 27, "单位：ppm", 23)
box(fusion, "选定权重", 1312, 677, 169, 47, "EBDDFA", "8653BA")
eq(fusion, "weight_g", 1318, 681, 157, 39, 26)

train = group("08_独立训练与验证集选权")
box(train, "GRU损失分区", 44, 845, 568, 179, "FFF8FA", "9F8993")
box(train, "岭回归损失分区", 632, 845, 568, 179, "FFF8FA", "9F8993")
box(train, "验证选权分区", 1220, 845, 656, 179, "FFF8FA", "9F8993")
txt(train, "GRU损失标题", 59, 851, 538, 33, "GRU 成员独立拟合", 26, True)
txt(train, "岭回归损失标题", 647, 851, 538, 33, "岭回归独立拟合", 26, True)
txt(train, "验证选权标题", 1235, 851, 626, 33, "验证集选择全局融合权重", 26, True)
eq(train, "gru_loss", 57, 890, 542, 87, 22)
eq(train, "ridge_loss", 646, 890, 540, 87, 21)
eq(train, "select", 1232, 885, 632, 99, 22)
txt(train, "GRU优化说明", 64, 984, 528, 28, "Adam 优化 · 验证集早停", 22)
txt(train, "岭回归说明", 650, 984, 532, 28, "训练集拟合 · 截距不参与正则化", 22)
eq(train, "grid", 1240, 985, 616, 29, 23)
line(train, "验证集决定权重", 1397, 844, 1397, 725, arrow=True,
     color="8653BA", dash=True, width=2.6)
legend = group("09_连线图例")
line(legend, "预测数据流示意", 1497, 809, 1564, 809, arrow=True)
txt(legend, "数据流标签", 1571, 793, 95, 31, "预测数据流", 19)
line(legend, "验证控制示意", 1690, 809, 1754, 809, arrow=True, color="8653BA", dash=True)
txt(legend, "控制流标签", 1763, 793, 97, 31, "验证集选权", 19)

slide.notes_slide.notes_text_frame.text = (
    "图中文字与符号以 course-report/sec/task3-method.tex 为准；验证选权式来自 "
    "course-report/sec/task3-experiments.tex。第 i 个样本、第 k 个成员的预测为 "
    "f_{theta_k}(X_tilde_i)，均值为 g_i，岭回归为 r_i，融合预测为 y_hat_tilde_i(w)。"
    "目标还原使用 mu_y 和 s_y；sigma 在论文方法中表示 Sigmoid，故不作标准差符号。"
    "图中的反标准化公式是论文 eq:final_prediction 结合 eq:ensemble、eq:fusion 的等价紧凑形式。"
    "模型包含五个独立 GRU，Ridge 独立拟合；验证搜索决定权重，不是端到端联合损失。"
    "所有文字、模块、箭头和数学公式均为可编辑的 PowerPoint 原生对象；公式使用 OMML。"
)
prs.core_properties.title = "GRU 多成员集成与岭回归融合：中文模型主图"
prs.core_properties.subject = "与当前论文方法符号一致的一页可编辑主图"
prs.core_properties.author = ""
prs.save(OUT)

# A standalone UTF-8 LaTeX document. Each figure equation has its source label,
# and the same literal LaTeX used to construct the Office equation is retained.
tex = [
    r"% !TeX program = xelatex",
    r"% UTF-8；中文主图对应的全部数学符号与公式。",
    r"% 图中文字依据：course-report/sec/task3-method.tex",
    r"% 验证选权依据：course-report/sec/task3-experiments.tex",
    r"% 编译：xelatex 模型主图_数学符号与公式.tex",
    r"\documentclass[UTF8,a4paper,11pt]{ctexart}",
    r"\usepackage[margin=22mm]{geometry}",
    r"\usepackage{amsmath,amssymb,bm,longtable,booktabs,array}",
    r"\allowdisplaybreaks",
    r"\setlength{\parindent}{0pt}",
    r"\title{模型主图：数学符号与公式 LaTeX 代码}",
    r"\author{}\date{}",
    r"\begin{document}\maketitle",
    r"\section{符号与论文保持一致}",
    r"本文件对应一页中文模型主图。输入、预测、损失函数与索引采用当前论文方法部分的记号；验证集选权来自实验部分。图中以 $s_j,s_y$ 表示标准差，$\sigma$ 在论文方法中表示 Sigmoid，不用于标准差。图中 $g_i,r_i$ 均为标准化预测，输出 $\widehat y_i$ 的单位为 ppm。",
    r"\section{图内全部公式与数学标签}",
]
for n, (key, info) in enumerate(FORM.items(), 1):
    tex += [f"% F{n:02d} [{key}] {info['title']}",
            "% 对应论文：" + info["source"],
            r"\subsection*{" + f"F{n:02d}：{info['title']}" + "}",
            r"\[", info["latex"], r"\]", ""]
tex += [r"\section{完整输出与紧凑图示的等价关系}",
        "% 论文原式 eq:final_prediction，未改动数学内容。",
        r"\[", source_eq(METHOD, "eq:final_prediction"), r"\]",
        r"将 $g_i=K^{-1}\sum_{k=1}^{K}f_{\theta_k}(\widetilde{\mathbf X}_i)$ 和 $\widehat{\widetilde y}_i(w)=wg_i+(1-w)r_i$ 代入，得到图中的紧凑式：",
        r"\[", FORM["inverse"]["latex"], r"\]",
        r"\section{全部符号对照表}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.37\linewidth}p{0.57\linewidth}}",
        r"\toprule 符号 & 含义 \\ \midrule \endhead"]
for symbol, meaning in SYMBOLS:
    tex += [f"% LaTeX: {symbol}", f"${symbol}$ & {meaning} " + r"\\"]
tex += [r"\bottomrule\end{longtable}", r"\end{document}"]
tex_text = ("\n".join(tex) + "\n").replace("CO₂", r"CO$_2$").replace("τ", r"$\tau$")
TEX.write_text(tex_text, encoding="utf-8")

# Keep compact validation in stdout; no separate report is needed.
def descendants(shapes):
    for s in shapes:
        if s.shape_type == 6:
            yield from descendants(s.shapes)
        else:
            yield s

loaded = Presentation(OUT)
assert len(loaded.slides) == 1
objects = list(descendants(loaded.slides[0].shapes))
equations = [s for s in objects if s._element.findall(f".//{{{M}}}oMath")]
assert len(equations) >= 35
assert not any(s.shape_type == 13 for s in objects)
assert all(s.left >= 0 and s.top >= 0 and
           s.left + s.width <= loaded.slide_width+5 and
           s.top + s.height <= loaded.slide_height+5 for s in objects)
assert abs(json.loads((ROOT/"results/metrics.json").read_text())["best_weight"]-.95)<1e-9
assert not any(chr(65533) in value for value in [TEX.read_text(encoding="utf-8")])
print(f"PPTX: {OUT}")
print(f"LaTeX: {TEX}")
print(f"1 slide; {len(objects)} native objects; {len(equations)} editable equations; 0 pictures.")
print(f"LaTeX contains {len(FORM)} equation/label entries and {len(SYMBOLS)} symbol definitions.")
