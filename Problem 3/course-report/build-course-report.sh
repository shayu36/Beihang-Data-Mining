#!/bin/sh
set -eu
report_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$report_dir"
mkdir -p build-course-report/cache

# 子目录结构：样式在 styles/，文献在 bib/，字体度量在 fonts/。
# 用绝对路径设置搜索路径，避免相对路径在 -outdir 下失效。
export TEXINPUTS="$report_dir/styles:$report_dir/fonts:${TEXINPUTS:-}"
export BIBINPUTS="$report_dir/bib:${BIBINPUTS:-}"
export BSTINPUTS="$report_dir/bib:${BSTINPUTS:-}"
export TFMFONTS="$report_dir/fonts:${TFMFONTS:-}"
export VFFONTS="$report_dir/fonts:${VFFONTS:-}"

# Keep temporary files and font caches together; no changes to system TeX.
export XDG_CACHE_HOME="$report_dir/build-course-report/cache"
latexmk -norc -Werror -pdf \
  -pdflatex='pdflatex -no-shell-escape %O %S' \
  -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build-course-report course-report.tex
cp build-course-report/course-report.pdf course-report.pdf
printf '\nPDF: %s/course-report.pdf\n' "$report_dir"
