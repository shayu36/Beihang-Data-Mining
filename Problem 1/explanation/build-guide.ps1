$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    for ($pass = 0; $pass -lt 3; $pass++) {
        & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build all-tasks-explained.tex
        if ($LASTEXITCODE -ne 0) { throw 'Guide compilation failed.' }
    }
    Copy-Item -LiteralPath 'build/all-tasks-explained.pdf' -Destination 'all-tasks-explained.pdf' -Force
} finally {
    Pop-Location
}
