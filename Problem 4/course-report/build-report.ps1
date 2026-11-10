$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & latexmk -pdf -quiet -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build-course-report course-report.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build-course-report\course-report.pdf') -Destination (Join-Path $PSScriptRoot 'course-report.pdf') -Force
    Write-Output (Join-Path $PSScriptRoot 'course-report.pdf')
} finally {
    Pop-Location
}
