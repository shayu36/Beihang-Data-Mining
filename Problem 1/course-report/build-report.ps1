param()
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
        throw 'XeLaTeX is required. Install TeX Live or MiKTeX and add its binaries to PATH.'
    }
    if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
        throw 'latexmk is required to resolve references and bibliography.'
    }
    $buildDirectory = Join-Path $PSScriptRoot 'build-course-report'
    New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build-course-report course-report.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed; inspect build-course-report/course-report.log.' }
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'course-report.pdf') -Destination (Join-Path $PSScriptRoot 'course-report.pdf') -Force
    Write-Output (Join-Path $PSScriptRoot 'course-report.pdf')
} finally {
    Pop-Location
}
