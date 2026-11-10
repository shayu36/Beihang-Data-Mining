param(
    [ValidateSet('Auto', 'Tectonic', 'XeLaTeX', 'pdfLaTeX')]
    [string]$Engine = 'Auto',
    [string]$TectonicPath = ''
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $buildDirectory = Join-Path $PSScriptRoot 'build-course-report'
    New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
    if (-not $TectonicPath) {
        $tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
        if ($tectonicCommand) { $TectonicPath = $tectonicCommand.Source }
    }
    if ($Engine -eq 'Auto') {
        if ($TectonicPath) { $Engine = 'Tectonic' }
        elseif (Get-Command latexmk -ErrorAction SilentlyContinue) { $Engine = 'XeLaTeX' }
        else { throw 'Install TeX Live/MiKTeX with latexmk, or provide -TectonicPath to tectonic.exe.' }
    }
    if ($Engine -eq 'Tectonic') {
        if (-not $TectonicPath) { throw 'Tectonic not found. Use -TectonicPath to specify tectonic.exe.' }
        & $TectonicPath --keep-logs --keep-intermediates -Z search-path=styles -Z search-path=fonts -Z search-path=bib --outdir $buildDirectory course-report.tex
    } elseif ($Engine -eq 'pdfLaTeX') {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build-course-report course-report.tex
    } else {
        & latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build-course-report course-report.tex
    }
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'course-report.pdf') -Destination (Join-Path $PSScriptRoot 'course-report.pdf') -Force
    Write-Output (Join-Path $PSScriptRoot 'course-report.pdf')
} finally {
    Pop-Location
}
