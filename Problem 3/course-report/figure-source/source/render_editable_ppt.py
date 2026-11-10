"""Render the delivered slide using installed Microsoft PowerPoint, no window."""
from pathlib import Path
import argparse
import pythoncom
import win32com.client

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--ppt", type=Path, default=ROOT / "course-report/figure-source/模型主图_可编辑.pptx")
parser.add_argument("--preview", type=Path)
parser.add_argument("--width", type=int, default=2880)
args = parser.parse_args()
PPT = args.ppt.resolve()
BUILD = Path(__file__).resolve().parent.parent / "build"
BUILD.mkdir(exist_ok=True)
PREVIEW = (args.preview or BUILD / "ppt_preview.png").resolve()
PREVIEW.parent.mkdir(exist_ok=True, parents=True)

pythoncom.CoInitialize()
try:
    win32com.client.GetActiveObject("PowerPoint.Application")
    already_running = True
except pythoncom.com_error:
    already_running = False

app = win32com.client.DispatchEx("PowerPoint.Application")
presentation = None
try:
    presentation = app.Presentations.Open(str(PPT), True, False, False)
    assert presentation.Slides.Count == 1
    height = round(args.width * presentation.PageSetup.SlideHeight / presentation.PageSetup.SlideWidth)
    presentation.Slides.Item(1).Export(str(PREVIEW), "PNG", args.width, height)
    print(f"PowerPoint rendered one slide to {PREVIEW}")
finally:
    if presentation is not None:
        presentation.Close()
    if not already_running:
        app.Quit()
    pythoncom.CoUninitialize()
