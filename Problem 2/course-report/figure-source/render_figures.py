"""Public entry point for the eight verified-data report figures."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("render_verified_figures.py")), run_name="__main__")
