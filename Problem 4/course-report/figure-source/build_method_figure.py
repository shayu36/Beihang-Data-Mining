"""Editable Task4 method figure, using native PowerPoint objects.

Visual references, inspected at Figure 2 in each supplied PDF:
  AutoDrivingPaper/02_SPSC_AAAI2026.pdf -- dashed computation groups.
  AutoDrivingPaper/03_I2-World_ICCV2025.pdf -- representation/operator alignment.
  AutoDrivingPaper/04_QueryOcc_CVPR2026.pdf -- compact left-to-right graph.
The figure uses Task4's own data flow. Miniature trees, histograms and contours
are conceptual illustrations. All text, matrices, illustrations and connectors
are native shapes. No reference-paper artwork is embedded.
"""
from pathlib import Path
import argparse

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

REPORT = Path(__file__).resolve().parents[1]
OUTPUT = REPORT / "figure-source/Task4_方法主图_可编辑.pptx"
INK = "23303E"
BLUE, GREEN, ORANGE, PURPLE = "3B6B99", "368578", "C98A42", "77539D"
LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, LIGHT_PURPLE = "EDF3FA", "EDF5F0", "FCF4E7", "F2EDF8"

presentation = Presentation()
presentation.slide_width = Inches(16.8)
presentation.slide_height = Inches(7.0)
slide = presentation.slides.add_slide(presentation.slide_layouts[6])


def rgb(value):
    return RGBColor.from_string(value)


def remove_theme_effects(shape):
    shape._element.spPr.append(OxmlElement("a:effectLst"))
    for reference in shape._element.xpath(".//a:effectRef"):
        reference.set("idx", "0")


def box(x, y, width, height, fill="FFFFFF", line=INK, rounded=False, name=""):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if rounded else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(1)
    if rounded:
        shape.adjustments[0] = 0.08
    shape.name = name or "module"
    remove_theme_effects(shape)
    return shape


def text(value, x, y, width, height, size=23, bold=False, color=INK, align=PP_ALIGN.CENTER, name=""):
    minimum_height = (size * 1.2 + 1) / 72
    actual_height = max(height, minimum_height)
    y -= (actual_height - height) / 2
    height = actual_height
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    shape.name = name or value
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    east_asian = OxmlElement("a:ea")
    east_asian.set("typeface", "SimHei" if bold else "SimSun")
    run._r.get_or_add_rPr().append(east_asian)
    return shape


def math_text(parts, x, y, width, height, size=23, name="equation"):
    shape = text("", x, y, width, height, size=size, name=name)
    shape.top -= Pt(1.5)
    shape.height += Pt(3)
    paragraph = shape.text_frame.paragraphs[0]
    for value, baseline in parts:
        run = paragraph.add_run()
        run.text = value
        run.font.name = "Times New Roman"
        run.font.size = Pt(size if baseline == 0 else size * 0.7)
        run.font.color.rgb = rgb(INK)
        if baseline:
            run._r.get_or_add_rPr().set("baseline", str(baseline))
    return shape


def line(x1, y1, x2, y2, color=INK, arrow=False, width=1.3):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    remove_theme_effects(shape)
    if arrow:
        ending = OxmlElement("a:tailEnd")
        ending.set("type", "triangle")
        ending.set("w", "sm")
        ending.set("len", "sm")
        shape._element.spPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ln").append(ending)
    return shape


def ellipse(x, y, width, height, fill, stroke, angle=0, line_width=1):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,
        Inches(x), Inches(y), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(line_width)
    shape.rotation = angle
    remove_theme_effects(shape)
    return shape


def grid(x, y, width, height, headings, colors, rows=4):
    columns = len(headings)
    cell_width = width / columns
    cell_height = height / (rows + 1)
    for row in range(rows + 1):
        for col in range(columns):
            fill = colors[col]
            cell = box(x + col * cell_width, y + row * cell_height,
                       cell_width, cell_height, fill=fill, line="81909D")
            cell.line.width = Pt(0.6)
            label = headings[col] if row == 0 else "·"
            text(label, x + col * cell_width, y + row * cell_height,
                 cell_width, cell_height, size=21 if row == 0 else 18)



# Redesigned after the three supplied Figure 2 references: normal-only fitting
# above, three parallel scoring lanes below, and branch-wise rank operators.
# All illustrative distributions and trees are schematic, not measured data.

def dashed(shape):
    for old in shape._element.spPr.xpath("./a:ln/a:prstDash"):
        old.getparent().remove(old)
    node = OxmlElement("a:prstDash")
    node.set("val", "dash")
    shape._element.spPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ln").insert(1, node)
    return shape


def route(points, color=INK, dash=False, width=1.3):
    for index, (a, b) in enumerate(zip(points, points[1:])):
        edge = line(*a, *b, color=color, arrow=index == len(points)-2, width=width)
        if dash:
            dashed(edge)


def matrix(x, y, w, h, fill, stroke, cols=3, rows=4):
    # Empty native cells indicate a batch of features or scores.
    for row in range(rows):
        for col in range(cols):
            cell=box(x+col*w/cols, y+row*h/rows, w/cols, h/rows, fill=fill, line=stroke)
            cell.line.width=Pt(.65)


def symbol(base, sub, x, y, w, h=.3, size=25):
    return math_text([(base,0),(sub,-22000)],x,y,w,h,size=size)


# Compact fitting strip and formula annotations have no unused outer margin.
box(.10,.10,7.80,1.64,fill="F7FAFD",line="90AAC2",rounded=True,name="normal-sample-fitting")
text("正常样本拟合",.30,.20,2.58,.37,size=28,bold=True,align=PP_ALIGN.LEFT)
box(8.10,.10,8.60,1.64,fill="FBFAFD",line="C9BCD9",rounded=True,name="fusion-equations")
text("秩归一化",8.34,.22,3.54,.35,size=25,bold=True)
text("等权融合",12.18,.22,4.18,.35,size=25,bold=True)

# The exact formulas are live text, with genuine subscripts and fraction rules.
symbol("q","i,k",8.50,.96,.64,size=25)
text("=",9.17,.96,.34,.31,size=25)
math_text([("r",0),("i,k",-22000),(" − 1",0)],9.58,.76,1.58,.35,size=25)
line(9.58,1.13,11.16,1.13,width=1)
text("M − 1",9.58,1.20,1.58,.30,size=25)
symbol("S","i",12.14,.96,.54,size=25)
text("=",12.72,.96,.32,.31,size=25)
math_text([("q",0),("i,I",-22000),(" + q",0),("i,H",-22000),(" + q",0),("i,M",-22000)],13.09,.76,3.24,.35,size=25)
line(13.10,1.13,16.32,1.13,width=1)
text("3",14.42,1.20,.60,.30,size=25)

# Fitting data and normalization operators, aligned with the scoring row.
matrix(.33,.88,.73,.65,"D8EADC",GREEN,cols=6)
symbol("X","tr",.28,.59,.83,size=24)
line(1.06,1.20,1.36,1.20,arrow=True)
box(1.36,.94,1.29,.53,fill="FFFFFF",line="8998A7")
text("特征子集",1.36,.94,1.29,.53,size=21)
line(2.65,1.20,2.90,1.20,arrow=True)
box(2.90,.80,1.30,.80,fill="FFFFFF",line="8998A7")
text("标准化",2.90,.84,1.30,.31,size=23)
text("μ, σ",2.90,1.17,1.30,.29,size=24)
line(4.20,1.20,4.55,1.20,arrow=True)
matrix(4.55,.88,.72,.65,"D8EADC",GREEN,cols=4)
symbol("Z","tr",4.48,.59,.86,size=24)
route([(5.27,1.20),(5.57,1.20),(5.57,2.61),(5.88,2.61)],color=GREEN,dash=True,width=1.6)
text("拟合检测器",5.82,1.00,1.76,.36,size=23,color=GREEN)

# Headings follow the same left-to-right order as the computation graph.
text("共享特征",.30,2.02,2.62,.38,size=28,bold=True,align=PP_ALIGN.LEFT)
text("多视角异常评分",5.70,2.02,2.90,.38,size=27,bold=True)
text("分支秩转换",8.83,2.02,3.13,.38,size=27,bold=True)
text("融合与判断",12.45,2.02,4.00,.38,size=27,bold=True)
dashed(box(5.88,2.48,2.66,4.42,line="8E99A2",rounded=True,name="three-fitted-detectors"))
box(9.30,2.48,1.96,4.42,fill="F6F7F8",line="D5D9DE",rounded=True,name="branch-wise-rank-transform")

# Test flow uses the same feature subset and normal-sample statistics.
cy=4.62
matrix(.33,cy-.47,.73,.94,"DCEAF7",BLUE,cols=6)
symbol("X","te",.25,cy-.86,.90,size=26)
line(1.06,cy,1.36,cy,arrow=True)
box(1.36,cy-.29,1.29,.58,fill="FFFFFF",line="8998A7")
text("特征子集",1.36,cy-.29,1.29,.58,size=21)
line(2.65,cy,2.90,cy,arrow=True)
box(2.90,cy-.29,1.30,.58,fill="FFFFFF",line="8998A7")
text("标准化",2.90,cy-.29,1.30,.58,size=23)
line(4.20,cy,4.55,cy,arrow=True)
matrix(4.55,cy-.47,.72,.94,"DCEAF7",BLUE,cols=4)
symbol("Z","te",4.48,cy-.86,.86,size=26)
route([(3.55,1.60),(3.55,cy-.29)],color=GREEN,dash=True,width=1.6)
text("复用统计量",3.70,2.79,1.55,.34,size=19,color=GREEN)

# Dedicated fan-out and collection lanes avoid crossings with annotations.
lanes=[3.22,4.62,6.02]
line(5.27,cy,5.57,cy)
line(5.57,lanes[0],5.57,lanes[-1])
for yy in lanes:
    line(5.57,yy,5.88,yy,arrow=True)

# Isolation tree: the colored root-to-leaf path is a schematic example.
text("Isolation Forest",6.00,2.66,2.40,.34,size=25,bold=True)
nodes=[(6.92,3.09),(6.45,3.37),(7.40,3.37),(6.15,3.65),(6.73,3.65),(7.13,3.65),(7.68,3.65)]
for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
    hi=(a,b) in [(0,2),(2,6)]
    line(*nodes[a],*nodes[b],color=BLUE if hi else "7D8890",width=2.3 if hi else 1.1)
for j,(xx,yy) in enumerate(nodes):
    ellipse(xx-.065,yy-.065,.13,.13,"A8CAE5" if j in [0,2,6] else "FFFFFF",BLUE)
text("隔离路径",6.24,3.75,1.91,.28,size=21,color=BLUE)

# Histogram: low-density bin, no artificial numerical observations.
text("HBOS",6.00,4.16,2.40,.33,size=25,bold=True)
base=5.12
for j,h in enumerate([.08,.21,.40,.52,.33,.15,.05]):
    box(6.23+j*.22,base-h,.22,h,fill="C3DECE",line=GREEN)
line(6.13,base,7.99,base,arrow=True,width=1.1)
line(6.13,base,6.13,4.41,arrow=True,width=1.1)
ellipse(7.55,4.90,.10,.10,GREEN,GREEN)
text("边缘密度",6.24,5.19,1.91,.28,size=21,color=GREEN)

# Robust covariance: contours and distant point, all native vector shapes.
text("MCD",6.00,5.58,2.40,.33,size=25,bold=True)
ellipse(6.21,6.03,1.50,.42,"FAEDD5",ORANGE,angle=340)
ellipse(6.50,6.10,.93,.24,"F1D5A6",ORANGE,angle=340)
for xx,yy in [(6.72,6.27),(6.94,6.24),(7.16,6.15)]:
    ellipse(xx,yy,.06,.06,BLUE,BLUE)
ellipse(7.88,5.99,.10,.10,ORANGE,ORANGE)
line(7.06,6.24,7.88,6.04,color=ORANGE,width=1)
text("稳健距离",6.24,6.57,1.91,.28,size=21,color=ORANGE)

# Each branch retains its own batch vector and its own rank calculation.
for yy,k,fill,stroke in zip(lanes,["I","H","M"],["DCEAF7","DEEEE4","F7E8CD"],[BLUE,GREEN,ORANGE]):
    line(8.54,yy,8.77,yy,arrow=True)
    matrix(8.78,yy-.38,.40,.76,fill,stroke,cols=1,rows=4)
    symbol("s",k,8.70,yy-.74,.55,size=25)
    line(9.18,yy,9.48,yy,arrow=True)
    box(9.48,yy-.40,1.60,.80,fill="FFFFFF",line="A4ADB6")
    text("平均秩",9.48,yy-.35,1.60,.32,size=23)
    text("归一化",9.48,yy+.04,1.60,.30,size=22)
    line(11.08,yy,11.48,yy,arrow=True)
    matrix(11.49,yy-.38,.40,.76,fill,stroke,cols=1,rows=4)
    symbol("q",k,11.41,yy-.74,.55,size=25)

# Three normalized score vectors feed a single arithmetic mean.
route([(11.89,lanes[0]),(12.36,lanes[0]),(12.36,4.39),(12.82,4.39)])
route([(11.89,lanes[1]),(12.78,lanes[1])])
route([(11.89,lanes[2]),(12.36,lanes[2]),(12.36,4.85),(12.82,4.85)])
ellipse(12.78,4.20,.84,.84,LIGHT_PURPLE,PURPLE,line_width=1.4)
text("均值",12.78,4.20,.84,.84,size=24)
line(13.62,cy,14.00,cy,arrow=True)
matrix(14.00,cy-.52,.45,1.04,"DED1EE",PURPLE,cols=1,rows=4)
text("S",13.96,cy-.89,.53,.31,size=28)
line(14.45,cy,14.87,cy,arrow=True)
box(14.87,cy-.30,1.45,.60,fill="FFFFFF",line="8998A7")
text("阈值 τ",14.87,cy-.30,1.45,.60,size=24)
line(15.60,cy+.30,15.60,5.45,arrow=True)
box(14.76,5.47,1.67,.68,fill=LIGHT_PURPLE,line="B39BC9",rounded=True)
text("正常 / 患病",14.76,5.47,1.67,.68,size=24,bold=True)
text("异常分数",13.49,5.32,1.46,.33,size=22,color=PURPLE)

# Compact legend explains the two line semantics, not a second model.
line(.38,6.22,.96,6.22,arrow=True)
text("样本与评分",1.10,6.04,1.98,.36,size=23,align=PP_ALIGN.LEFT)
dashed(line(.38,6.73,.96,6.73,color=GREEN,arrow=True,width=1.6))
text("拟合与统计量",1.10,6.55,2.54,.36,size=23,color=GREEN,align=PP_ALIGN.LEFT)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
presentation.save(OUTPUT)
print(OUTPUT)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export",action="store_true",help="Export PDF and PNG with installed Microsoft PowerPoint")
arguments=parser.parse_args()
if arguments.export:
    import win32com.client
    application=win32com.client.Dispatch("PowerPoint.Application")
    document=application.Presentations.Open(str(OUTPUT),ReadOnly=True,Untitled=False,WithWindow=False)
    try:
        document.SaveAs(str(REPORT/"figures/method-overview.pdf"),32)
        document.Slides(1).Export(str(REPORT/"figures/method-overview.png"),"PNG",3360,1400)
    finally:
        document.Close()
