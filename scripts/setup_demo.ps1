param(
    [ValidateSet("3.11", "3.12")]
    [string]$PythonVersion = "3.12",
    [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvironmentRoot = Join-Path $ProjectRoot ".venv-demo"
$PythonExecutable = Join-Path $EnvironmentRoot "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) not found. Install Python $PythonVersion and retry."
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    Write-Host "Creating demo environment at $EnvironmentRoot"
    & py "-$PythonVersion" -m venv $EnvironmentRoot
}

Write-Host "Installing Document HRM dependencies..."
& $PythonExecutable -m pip install --upgrade pip
& $PythonExecutable -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

if (-not $SkipWarmup) {
    Write-Host "Warming the semantic encoder for the first demo run..."
    & $PythonExecutable (Join-Path $ProjectRoot "scripts\warmup_document_hrm_demo.py")
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start the demo with: .\start_demo.cmd"
