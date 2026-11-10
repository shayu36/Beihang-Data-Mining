"""Verify the single-slide Chinese figure and its native equation sources."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
with ZipFile(ROOT / "course-report/figure-source/模型主图_中文版.pptx") as z:
    slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    assert len(slides) == 1
    tree = etree.fromstring(z.read(slides[0]))
    assert not tree.findall(".//p:pic", NS)
    equations = tree.findall(".//m:oMath", NS)
    assert len(equations) == 39
    symbols = "".join(tree.xpath("//m:t/text()", namespaces=NS))
    assert "\\" not in symbols, "Unconverted TeX command in native math"
    sources = tree.xpath("//p:cNvPr/@descr", namespaces=NS)
    tex = (ROOT / "course-report/figure-source/模型主图_数学符号与公式.tex").read_text(encoding="utf-8")
    for source in sources:
        if source.startswith("LaTeX: "):
            latex = source[len("LaTeX: "):]
            assert latex in tex, f"Equation absent from source list: {latex}"
    selector = tree.xpath("//p:sp[p:nvSpPr/p:cNvPr[contains(@descr, 'arg')]]", namespaces=NS)
    assert len(selector) == 1
    assert selector[0].xpath(".//m:lim//m:r[m:rPr/m:scr[@m:val='script']]/m:t[text()='W']", namespaces=NS)
    print("Verified native script W in the weight-selection lower limit.")
    # Key objectives and the expanded output remain copied from the paper.
    for filename, labels in (
        ("task3-method.tex", ["ensemble", "gru_loss", "ridge_loss", "final_prediction"]),
        ("task3-experiments.tex", ["fusion_selection"]),
    ):
        paper = (ROOT / "course-report/sec" / filename).read_text(encoding="utf-8")
        for label in labels:
            matches = [body for body in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", paper, re.S)
                       if rf"\label{{eq:{label}}}" in body]
            assert len(matches) == 1
            body = re.sub(r"\\label\{[^}]*\}", "", matches[0]).strip()
            body = re.sub(r"\s+", " ", body).rstrip(",.")
            if label == "ridge_loss":
                body = re.sub(r"\\begin\{split\}|\\end\{split\}|&|\\\\", "", body).strip()
            assert body in tex, f"Paper equation differs: {label}"
    print("Verified training, fusion-selection and output equations against current paper sources.")
print(f"Verified: {len(slides)} slide, {len(equations)} native equations, no raster images; all equation sources are in the LaTeX file.")
