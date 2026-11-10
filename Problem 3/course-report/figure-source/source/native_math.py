"""Native editable Office equations, built from the exact supplied LaTeX."""
from copy import deepcopy
from pathlib import Path
import re
import unicodedata
from lxml import etree
from latex2mathml.converter import convert
from pptx.oxml.xmlchemy import OxmlElement
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Pt

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
_transform = etree.XSLT(etree.parse(str(XSL)))


def equation(shapes, name, latex, x, y, w, h, size_pt=18, color="171B20"):
    """Add a native a14:m/OMML equation to a DrawingML text shape."""
    shape = shapes.add_textbox(x, y, w, h)
    shape.name = name
    shape._element.nvSpPr.cNvPr.set("descr", "LaTeX: " + latex)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.space_before = p.space_after = Pt(0)
    p.font.name = "Cambria Math"
    p.font.size = Pt(size_pt)

    # Preserve upright labels through latex2mathml, whose mathrm conversion may
    # omit mathvariant on single Latin letters. This affects typography only.
    converter_latex = re.sub(r"\\mathrm\{([^{}]+)\}", r"\\operatorname{\1}", latex)
    converter_latex = converter_latex.replace(r"\arg\min", r"\operatorname{argmin}")
    mathml = etree.fromstring(convert(converter_latex, display="block").encode("utf-8"))
    ns = {"mml": "http://www.w3.org/1998/Math/MathML"}
    # PowerPoint can omit supplementary-plane script letters in compact limits.
    # Store the base letter plus native script styling instead of that codepoint.
    for node in mathml.iter():
        token = node.text or ""
        if len(token) == 1 and "SCRIPT CAPITAL" in unicodedata.name(token, ""):
            node.text = unicodedata.normalize("NFKD", token)
            node.set("mathvariant", "script")
    for operator in mathml.findall(".//mml:mo", ns):
        if (operator.text or "").isascii() and (operator.text or "").isalpha():
            operator.set("mathvariant", "normal")
    for over in mathml.findall(".//mml:mover", ns):
        if len(over) == 2 and "".join(over[1].itertext()) in {"^", "~", "¯", "→"}:
            over.set("accent", "true")
    transformed = _transform(mathml).getroot()
    math = transformed if transformed.tag == f"{{{M}}}oMath" else transformed.find(f"{{{M}}}oMath")
    if math is None:
        raise ValueError(f"Missing native math: {latex}")
    math = deepcopy(math)
    # Word's XSL produces WordprocessingML formatting. PowerPoint uses a:rPr.
    for run in math.findall(f".//{{{M}}}r"):
        props = OxmlElement("a:rPr")
        props.set("sz", str(round(size_pt * 100)))
        fill = OxmlElement("a:solidFill")
        c = OxmlElement("a:srgbClr")
        c.set("val", color)
        fill.append(c)
        props.append(fill)
        for tag in ("a:latin", "a:ea", "a:cs"):
            face = OxmlElement(tag)
            face.set("typeface", "Cambria Math")
            props.append(face)
        run.insert(1 if len(run) and run[0].tag == f"{{{M}}}rPr" else 0, props)
    for ctrl in math.findall(f".//{{{M}}}ctrlPr"):
        for child in list(ctrl):
            ctrl.remove(child)
        props = OxmlElement("a:rPr")
        props.set("sz", str(round(size_pt * 100)))
        face = OxmlElement("a:latin")
        face.set("typeface", "Cambria Math")
        props.append(face)
        ctrl.append(props)
    mathpara = etree.Element(f"{{{M}}}oMathPara", nsmap={"m": M})
    para_props = etree.SubElement(mathpara, f"{{{M}}}oMathParaPr")
    just = etree.SubElement(para_props, f"{{{M}}}jc")
    just.set(f"{{{M}}}val", "center")
    mathpara.append(math)
    wrapper = etree.Element(f"{{{A14}}}m", nsmap={"a14": A14})
    wrapper.append(mathpara)
    p._p.append(wrapper)
    return shape
