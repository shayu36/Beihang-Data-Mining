param(
    [string]$Python = 'python',
    [string]$Weights = '',
    [ValidateScript({ $_ -gt 0 })][int]$Threads = 4,
    [string]$OutputRoot = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'course_extension1'),
    [string]$CompareRun = '',
    [string]$Images = '',
    [string]$Thyroid = ''
)

$ErrorActionPreference = 'Stop'

function Find-CourseArchive {
    param([string]$Given, [string]$FileName, [string]$Option)
    if ($Given) {
        if (-not (Test-Path -LiteralPath $Given -PathType Leaf)) { throw "File for $Option was not found: $Given" }
        return (Get-Item -LiteralPath $Given).FullName
    }
    $directory = Get-Item -LiteralPath $PSScriptRoot
    while ($null -ne $directory) {
        $candidate = Join-Path $directory.FullName $FileName
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        $directory = $directory.Parent
    }
    throw "$FileName was not found above the code directory. Supply $Option with the course ZIP path."
}

function Invoke-PythonStep {
    param([string]$ScriptName, [string[]]$Arguments = @())
    Write-Host "Running $ScriptName"
    & $pythonPath (Join-Path $PSScriptRoot $ScriptName) @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$ScriptName exited with code $LASTEXITCODE" }
}

$pythonPath = (Get-Command -Name $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$imageArchive = Find-CourseArchive $Images 'Image_Anomaly_Detection.zip' '-Images'
$thyroidArchive = Find-CourseArchive $Thyroid 'thyroid.zip' '-Thyroid'
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runDir = Join-Path $outputPath ('run_' + $stamp)
$extraDir = Join-Path $outputPath ('supplements_' + $stamp)
if ((Test-Path -LiteralPath $runDir) -or (Test-Path -LiteralPath $extraDir)) {
    throw 'The timestamped result directory already exists. Run again to create a fresh one.'
}
$correctionDir = Join-Path $extraDir 'correction'
$sharedDir = Join-Path $extraDir 'shared'
$nestedFile = Join-Path $extraDir 'nested_image_cv.json'
$challengeFile = Join-Path $extraDir 'training_challenge.json'
$dataArgs = @('--images', $imageArchive, '--thyroid', $thyroidArchive)
$threadArgs = @('--threads', $Threads.ToString())
$weightArgs = @()
if ($Weights) { $weightArgs = @('--weights', (Get-Item -LiteralPath $Weights).FullName) }
$compareArgs = @()
if ($CompareRun) { $compareArgs = @('--compare', (Get-Item -LiteralPath $CompareRun).FullName) }
$referenceArgs = @()
foreach ($relative in @('..\results\correction\predictions.csv', 'course-report\results\correction\predictions.csv', '..\course-report-extension1\results\correction\predictions.csv')) {
    $reference = Join-Path $PSScriptRoot $relative
    if (Test-Path -LiteralPath $reference -PathType Leaf) {
        $referenceArgs = @('--reference', (Get-Item -LiteralPath $reference).FullName)
        break
    }
}

$previousEncoding = $env:PYTHONIOENCODING
$previousConsoleEncoding = [Console]::OutputEncoding
try {
    $env:PYTHONIOENCODING = 'utf-8'
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    Write-Host "Primary training output: $runDir"
    Write-Host "Supplementary experiments: $extraDir"
    Invoke-PythonStep -ScriptName 'test_core.py'
    Invoke-PythonStep -ScriptName 'run.py' -Arguments (@('--output', $runDir) + $dataArgs + $threadArgs + $weightArgs)
    Invoke-PythonStep -ScriptName 'verify.py' -Arguments (@('--run', $runDir, '--raw') + $dataArgs + $threadArgs + $weightArgs + $compareArgs)
    Invoke-PythonStep -ScriptName 'compare_correction.py' -Arguments (@('--thyroid', $thyroidArchive, '--output', $correctionDir) + $threadArgs)
    Invoke-PythonStep -ScriptName 'verify_correction.py' -Arguments (@('--thyroid', $thyroidArchive, '--run', $correctionDir,
        '--output', (Join-Path $correctionDir 'verification.json')) + $referenceArgs)
    Invoke-PythonStep -ScriptName 'shared_reference.py' -Arguments (@('--run', $runDir, '--output', $sharedDir) + $dataArgs + $threadArgs)
    Invoke-PythonStep -ScriptName 'nested_image_cv.py' -Arguments (@('--features', (Join-Path $runDir 'features.npz'), '--output', $nestedFile) + $threadArgs)
    Invoke-PythonStep -ScriptName 'train_only_challenge.py' -Arguments @('--thyroid', $thyroidArchive, '--output', $challengeFile)
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    [Console]::OutputEncoding = $previousConsoleEncoding
}
Write-Output "Results and models: $runDir"
Write-Output "Three-feature comparison and model: $correctionDir"
Write-Output "Shared detector metrics: $sharedDir"
Write-Output "Nested image validation: $nestedFile"
Write-Output "Thyroid training challenge: $challengeFile"
