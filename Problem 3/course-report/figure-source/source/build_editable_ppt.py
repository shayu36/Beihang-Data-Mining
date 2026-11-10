"""Reconstruct the verified model overview as one native, editable PPT slide.

All panels, matrix cells, tensor slabs, module boxes, arrows and equations are
PowerPoint shapes/text. No source screenshot is embedded. Coordinates follow
the approved 1774 x 887 diagram, at a physical slide size of 16 x 8 inches.
"""

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course-report/figure-source/模型主图_可编辑.pptx"
W, H = 1774, 887
SCALE = 16 * 914400 / W
FONT_SCALE = 16 * 72 / W
INK = "171B20"
BLUE = "5285B5"
TEAL = "299775"
AMBER = "C99223"
PURPLE = "8E65B9"
PINK = "CD668A"

prs = Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(8)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")


def u(v):
    return round(v * SCALE)


def rgb(s):
    return RGBColor.from_string(s)


def remove_theme_effects(shape):
    """Avoid inheriting the default Office theme's shadows on diagram objects."""
    for ref in shape._element.xpath("./p:style/a:effectRef"):
        ref.set("idx", "0")
    sppr = shape._element.spPr
    for old in list(sppr):
        if old.tag.endswith("}effectLst") or old.tag.endswith("}effectDag"):
            sppr.remove(old)
    sppr.append(OxmlElement("a:effectLst"))


def group(name, parent=None):
    shapes = slide.shapes if parent is None else parent.shapes
    g = shapes.add_group_shape()
    g.name = name
    return g


def box(parent, name, x, y, w, h, fill="FFFFFF", stroke="555D63", radius=True,
        width=1.8, dashed=False, kind=None):
    shapes = parent.shapes if parent is not None else slide.shapes
    kind = kind or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    q = shapes.add_shape(kind, u(x), u(y), u(w), u(h))
    q.name = name
    remove_theme_effects(q)
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        q.adjustments[0] = 0.15
    if fill is None:
        q.fill.background()
    else:
        q.fill.solid()
        q.fill.fore_color.rgb = rgb(fill)
    if stroke is None:
        q.line.fill.background()
    else:
        q.line.color.rgb = rgb(stroke)
        q.line.width = Pt(width * FONT_SCALE)
        if dashed:
            q.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return q


def text(parent, name, x, y, w, h, value, size=26, bold=False, italic=False,
         color=INK, align=PP_ALIGN.CENTER, font="Times New Roman"):
    """Value is text or [(text, baseline, relative_size), ...] for live math."""
    shapes = parent.shapes if parent is not None else slide.shapes
    q = shapes.add_textbox(u(x), u(y), u(w), u(h))
    q.name = name
    tf = q.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = p.space_after = Pt(0)
    p.line_spacing = 1.0
    spans = [(value, 0, 1)] if isinstance(value, str) else value
    for word, baseline, relative in spans:
        run = p.add_run()
        run.text = word
        run.font.name = font
        run.font.size = Pt(size * relative * FONT_SCALE)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = rgb(color)
        if baseline:
            run._r.get_or_add_rPr().set("baseline", str(baseline))
    return q


def line(parent, name, x1, y1, x2, y2, color=INK, width=2.5,
         arrow=False, dashed=False):
    shapes = parent.shapes if parent is not None else slide.shapes
    q = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, u(x1), u(y1), u(x2), u(y2))
    q.name = name
    remove_theme_effects(q)
    q.line.color.rgb = rgb(color)
    q.line.width = Pt(width * FONT_SCALE)
    if dashed:
        q.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        ln = q._element.spPr.get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
        ln.append(tail)
    return q


def path(parent, name, points, arrow=False, **kw):
    for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
        line(parent, f"{name}.{k + 1}", *a, *b,
             arrow=arrow and k == len(points) - 2, **kw)


def labeled_box(parent, name, x, y, w, h, value, size=25, fill="FFFFFF",
                stroke="555D63", bold=False, italic=False, width=1.8):
    q = box(parent, name + ".shape", x, y, w, h, fill, stroke, width=width)
    text(parent, name + ".label", x + 6, y + 2, w - 12, h - 4,
         value, size, bold, italic)
    return q


def sub(a, b, rest=""):
    return [(a, 0, 1), (b, -25000, 0.7), (rest, 0, 1)]


# Backgrounds are native independent shapes beneath the editable components.
bg = group("00_Panel_backgrounds")
box(bg, "Input panel", 20, 28, 371, 636, "EFF5FC", BLUE, width=1.6, dashed=True)
box(bg, "GRU panel", 423, 64, 752, 407, "EBF7F1", TEAL, width=1.6, dashed=True)
box(bg, "Ridge panel", 431, 488, 745, 174, "FFF6DF", AMBER, width=1.6, dashed=True)
box(bg, "Fusion panel", 1198, 28, 558, 636, "F4EFFB", PURPLE, width=1.6, dashed=True)
box(bg, "Training panel", 16, 692, 1746, 181, "FCEDF1", PINK, width=1.6, dashed=True)

titles = group("01_Region_titles")
text(titles, "Input title", 30, 34, 351, 38, "Multivariate input", 32, True, True)
text(titles, "Architecture title", 422, 6, 754, 45,
     "GRU ensemble with a linear branch", 35, True, True)
text(titles, "GRU title", 456, 70, 490, 38, "Five GRU regressors", 31, True, True,
     align=PP_ALIGN.LEFT)
text(titles, "Ridge title", 454, 491, 310, 41, "Ridge branch", 31, True, True,
     align=PP_ALIGN.LEFT)
text(titles, "Fusion title", 1210, 33, 534, 39,
     "Weighted fusion and forecast", 31, True, True)
text(titles, "Training title", 39, 695, 650, 38,
     "Training and validation", 31, True, True, align=PP_ALIGN.LEFT)

inputs = group("02_Input_and_standardization")
text(inputs, "Input dimensions", 50, 79, 312, 31, "12 records × 21 variables", 27, True)
text(inputs, "History label", 55, 113, 302, 31, "CO₂ history included", 27, True, True)
text(inputs, "Matrix column label", 120, 145, 200, 29, "variables", 22, True, True)
matrix = group("Input_matrix_12_by_21", inputs)
for row in range(12):
    for col in range(21):
        fill = "E8D8F1" if col == 20 else ("DCECF7" if (row + col) % 3 else "F8FBFE")
        box(matrix, f"Cell_r{row + 1:02d}_c{col + 1:02d}",
            88 + col * 13, 202 + row * 14.25, 13, 14.25,
            fill, "40454A", radius=False, width=0.65)
box(matrix, "Matrix perimeter", 88, 202, 273, 171, None, INK, radius=False, width=1.8)
path(inputs, "Column bracket", [(96, 195), (96, 184), (355, 184), (355, 195)], width=1.6)
path(inputs, "Row bracket", [(81, 204), (71, 204), (71, 372), (81, 372)], width=1.6)
row_label = text(inputs, "Matrix row label", 12, 275, 84, 28, "records", 22, True, True)
row_label.rotation = 270
line(inputs, "Matrix to scaler", 220, 374, 220, 416, arrow=True)
labeled_box(inputs, "Train fitted scaler", 65, 417, 309, 39,
            "Train-fitted standardization", 24, bold=True)
norm = group("Standardization_formula", inputs)
text(norm, "Normalization lhs", 138, 474, 72, 38, "X̃ =", 30, italic=True)
text(norm, "Normalization numerator", 210, 464, 94, 29, "X − μ", 29, italic=True)
line(norm, "Normalization fraction bar", 207, 497, 304, 497, width=1.6)
text(norm, "Normalization denominator", 227, 499, 55, 30, "σ", 28, italic=True)
tensor = group("Standardized_window_tensor", inputs)
for offset in [30, 20, 10, 0]:
    box(tensor, f"Tensor slab {offset // 10}", 75 + offset, 573 - offset,
        241, 73, "DFEDF8", "344C5E", radius=False, width=1.7)
text(tensor, "Standardized tensor dimensions", 89, 585, 211, 44,
     [("X̃ ∈ ℝ", 0, 1), ("12×21", 30000, 0.67)], 31, italic=True)

# Top-level prediction paths are separate selectable connectors.
flow = group("03_Shared_input_prediction_flow")
path(flow, "Shared standardized input", [(336, 592), (412, 592), (412, 318), (466, 318)],
     arrow=True)
line(flow, "Input to Ridge", 412, 573, 515, 573, arrow=True)

gru = group("04_Five_GRU_regressors_and_mean")
box(gru, "GRU architecture annotation", 502, 111, 181, 59, "F4FAF7", "64716D", width=1.4)
text(gru, "GRU depth and width", 506, 114, 173, 28, "2 layers · H = 128", 21, True)
text(gru, "GRU dropout", 509, 140, 167, 25, "Dropout 0.2", 21, True)
box(gru, "Regression head annotation", 747, 111, 261, 59, "F4FAF7", "64716D", width=1.4)
text(gru, "Head architecture", 749, 114, 257, 27, "Last state → 128 → 64 → 1", 20, True)
text(gru, "Head activation and dropout", 751, 140, 253, 25, "ReLU · Dropout 0.2", 21, True)
ys = [205, 259, 313, 367, 421]
line(gru, "Input bus", 466, ys[0], 466, ys[-1])
line(gru, "Ensemble output bus", 964, ys[0], 964, ys[-1])
for k, y in enumerate(ys, 1):
    member = group(f"GRU_member_{k}", gru)
    line(member, "Shared input to GRU", 466, y, 522, y, arrow=True)
    gnode = labeled_box(member, f"GRU {k}", 522, y - 20, 141, 40,
                        f"GRU {k}", 28, fill="D6EFDC", stroke=INK, bold=True)
    head = labeled_box(member, f"Head {k}", 716, y - 20, 141, 40,
                       f"Head {k}", 28, fill="E4F2E8", stroke=INK, bold=True)
    conn = line(member, "GRU to head", 663, y, 716, y, arrow=True)
    conn.begin_connect(gnode, 3)
    conn.end_connect(head, 1)
    line(member, "Head to scalar", 857, y, 900, y, arrow=True)
    text(member, f"Scalar g{k}", 904, y - 25, 42, 39, sub("g", str(k)), 31, italic=True)
    line(member, "Scalar to mean bus", 945, y, 964, y)
line(gru, "Bus to mean", 964, 313, 1004, 313, arrow=True)
box(gru, "Mean aggregation", 1004, 234, 146, 144, "F7FAF8", INK, width=1.8)
text(gru, "Mean title", 1010, 240, 134, 37, "Mean", 30, True, True)
mean = group("Prediction_mean_formula", gru)
text(mean, "Mean lhs", 1016, 303, 40, 36, "g =", 27, italic=True)
text(mean, "Mean fraction numerator", 1056, 295, 20, 24, "1", 23)
line(mean, "Mean fraction bar", 1054, 321, 1079, 321, width=1.5)
text(mean, "Mean fraction denominator", 1056, 323, 20, 26, "5", 23)
text(mean, "Sum", 1080, 300, 34, 47, "Σ", 42, font="Cambria Math")
text(mean, "Sum upper bound", 1087, 281, 20, 22, "5", 19)
text(mean, "Sum lower bound", 1073, 346, 50, 22, "k = 1", 18, italic=True)
text(mean, "Sum term", 1113, 306, 34, 40, sub("g", "k"), 27, italic=True)

ridge = group("05_Ridge_branch")
labeled_box(ridge, "Flatten", 516, 550, 137, 46, "Flatten", 28, bold=True, stroke=INK)
line(ridge, "Flatten to vector", 653, 573, 687, 573, arrow=True)
text(ridge, "Flattened vector", 689, 544, 115, 45,
     [("q ∈ ℝ", 0, 1), ("252", 30000, 0.67)], 30, italic=True)
line(ridge, "Vector to Ridge", 795, 573, 838, 573, arrow=True)
labeled_box(ridge, "Ridge regressor", 838, 550, 188, 46, "Ridge, α = 1", 28,
            bold=True, stroke=INK)
line(ridge, "Ridge to scalar", 1026, 573, 1095, 573, arrow=True)
text(ridge, "Ridge scalar r", 1104, 546, 32, 43, "r", 33, italic=True)
text(ridge, "Ridge affine equation", 840, 606, 200, 38,
     [("r = β", 0, 1), ("T", 30000, 0.7), ("q + b", 0, 1)], 29, italic=True)

fusion = group("06_Weighted_fusion_and_CO2_forecast")
line(fusion, "Ensemble to GRU multiplier", 1150, 260, 1270, 260, arrow=True)
text(fusion, "Ensemble scalar g", 1207, 224, 36, 35, "g", 30, italic=True)
path(fusion, "Ridge to multiplier", [(1137, 573), (1224, 573), (1224, 518), (1271, 518)],
     arrow=True)
for name, y, weight in [("GRU", 260, "0.95"), ("Ridge", 518, "0.05")]:
    box(fusion, name + " multiplication", 1271, y - 22, 44, 44, "FFFFFF", INK,
        kind=MSO_SHAPE.OVAL, width=2.2)
    text(fusion, name + " multiplication symbol", 1275, y - 23, 36, 44, "×", 35)
    text(fusion, name + " coefficient", 1260, y - 78, 68, 37, weight, 30)
    line(fusion, name + " coefficient port", 1293, y - 41, 1293, y - 22, width=2)

# Separate ports into addition avoid an ambiguous junction between predictions.
path(fusion, "Scaled GRU to addition", [(1315, 260), (1369, 260), (1369, 294), (1409, 294)],
     arrow=True)
path(fusion, "Scaled Ridge to addition", [(1315, 518), (1382, 518), (1382, 326), (1409, 326)],
     arrow=True)
box(fusion, "Addition", 1403, 288, 44, 44, "FFFFFF", INK, kind=MSO_SHAPE.OVAL, width=2.2)
text(fusion, "Addition symbol", 1407, 286, 36, 47, "+", 35)
line(fusion, "Addition to fusion value", 1447, 310, 1485, 310, arrow=True)
labeled_box(fusion, "Fused standardized prediction", 1485, 276, 246, 53,
            "z = 0.95g + 0.05r", 28, italic=True, stroke=INK)
line(fusion, "Fusion to inverse scaling", 1608, 330, 1608, 377, arrow=True)
labeled_box(fusion, "Inverse standardization", 1480, 377, 264, 51,
            "Inverse standardization", 24, bold=True, stroke=INK)
text(fusion, "Inverse scaling formula", 1512, 436, 207, 43,
     [("ŷ = μ", 0, 1), ("y", -25000, 0.7), (" + σ", 0, 1),
      ("y", -25000, 0.7), ("z", 0, 1)], 31, italic=True)
line(fusion, "Inverse scaling to output", 1610, 484, 1610, 529, arrow=True)
box(fusion, "CO2 scalar output", 1595, 530, 30, 30, "FFFFFF", INK,
    kind=MSO_SHAPE.OVAL, width=2.2)
text(fusion, "Forecast label", 1470, 565, 282, 34, "Next-record CO₂", 29, True)
text(fusion, "Forecast unit", 1510, 597, 202, 31, "(ppm)", 28, True)
labeled_box(fusion, "Validation-selected weight", 1258, 566, 144, 48,
            "w* = 0.95", 31, fill="EBDCF9", stroke="7D3DC0", italic=True)

training = group("07_Training_and_validation_objectives")
box(training, "GRU fitting panel", 49, 742, 418, 116, "FFF5F7", "6A6870", width=1.5)
text(training, "GRU fitting title", 64, 743, 388, 35, "GRU fitting", 28, True, True)
text(training, "GRU MSE", 72, 776, 372, 43,
     [("L", 0, 1), ("k", -25000, 0.7), (" = mean[(g", 0, 1),
      ("k", -25000, 0.7), (" − ỹ)", 0, 1), ("2", 30000, 0.7), ("]", 0, 1)],
     28, italic=True)
text(training, "GRU optimization", 59, 819, 398, 32,
     "Adam · validation early stopping", 24, bold=True)
box(training, "Ridge fitting panel", 496, 742, 438, 116, "FFF5F7", "6A6870", width=1.5)
text(training, "Ridge fitting title", 512, 743, 406, 35, "Ridge fitting", 28, True, True)
text(training, "Ridge objective", 511, 776, 408, 43,
     [("L", 0, 1), ("r", -25000, 0.7), (" = Σ(r − ỹ)", 0, 1),
      ("2", 30000, 0.7), (" + α‖β‖", 0, 1), ("2", -25000, 0.7),
      ("2", 30000, 0.7)], 28, italic=True)
text(training, "Ridge fit data", 512, 819, 406, 32, "Training set · α = 1", 24, bold=True)
box(training, "Validation selection panel", 967, 742, 452, 116, "FFF5F7", "6A6870", width=1.5)
text(training, "Selection title", 976, 743, 434, 35,
     "Validation weight selection", 28, True, True)
text(training, "Validation search objective", 977, 777, 432, 39,
     [("w* = argmin", 0, 1), ("w", -25000, 0.7), (" MSE", 0, 1),
      ("val", -25000, 0.66), ("(wg + (1−w)r)", 0, 1)], 25, italic=True)
text(training, "Candidate grid", 983, 817, 420, 33,
     "w ∈ {0, 0.05, …, 1}", 27, italic=True)
line(training, "Validation chooses fusion coefficient", 1314, 741, 1314, 615,
     color="7D3DC0", width=2.6, arrow=True, dashed=True)
legend = group("08_Connector_legend")
box(legend, "Legend panel", 1452, 750, 290, 80, "FFFFFF", "6A6870", width=1.5)
line(legend, "Prediction flow sample", 1474, 773, 1544, 773, arrow=True)
text(legend, "Prediction flow meaning", 1554, 756, 180, 32, "Prediction flow", 23)
line(legend, "Control flow sample", 1474, 807, 1544, 807,
     color="7D3DC0", arrow=True, dashed=True)
text(legend, "Control flow meaning", 1550, 790, 188, 32, "Validation control", 23)

slide.notes_slide.notes_text_frame.text = (
    "Current model overview. Input: 12 records and 21 numerical variables, including historical CO2. "
    "The training-fitted standardizer feeds five separately trained two-layer GRUs (hidden 128, "
    "dropout 0.2; head 128 -> 64 -> 1, ReLU and dropout 0.2) and a separate 252-dimensional "
    "Ridge branch (alpha=1). The five scalar GRU predictions are averaged. Validation MSE "
    "selects w from 0 to 1 in increments of 0.05. The saved w is 0.95. "
    "z=0.95*g+0.05*r, followed by inverse scaling to CO2 concentration in ppm. "
    "GRU fitting, Ridge fitting and validation weight selection are separate stages, not a joint loss. "
    "The picture represents the current computation and does not assert that the code's string-sorted "
    "windows are chronologically ordered. OT in code maps to CO2 (ppm) in the original dataset. "
    "All diagram content is native editable PowerPoint geometry and text."
)
prs.core_properties.title = "CO2 forecasting: GRU ensemble and Ridge fusion"
prs.core_properties.subject = "Editable reconstruction of the current model overview"
prs.core_properties.author = ""
prs.save(OUT)


def flattened(shapes):
    for shape in shapes:
        if shape.shape_type == 6:
            yield from flattened(shape.shapes)
        else:
            yield shape


check = Presentation(OUT)
assert len(check.slides) == 1
items = list(flattened(check.slides[0].shapes))
assert all(s.shape_type != 13 for s in items), "Unexpected embedded picture"
words = "\n".join(s.text for s in items if s.has_text_frame)
for k in range(1, 6):
    assert f"GRU {k}" in words and f"Head {k}" in words
for label in ["Ridge, α = 1", "0.95", "0.05", "Next-record CO₂", "(ppm)",
              "GRU fitting", "Ridge fitting", "Validation weight selection"]:
    assert label in words, label
assert all(s.left >= 0 and s.top >= 0 for s in items)
assert all(s.left + s.width <= check.slide_width + 20 and
           s.top + s.height <= check.slide_height + 20 for s in items)
print(f"Saved: {OUT}")
print(f"Slides: 1; native objects: {len(items)}; text objects: "
      f"{sum(bool(s.has_text_frame and s.text) for s in items)}; pictures: 0")
