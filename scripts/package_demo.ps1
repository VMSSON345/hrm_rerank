param(
    [string]$OutputName = "finhrm-local-demo-v12-runtime.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $ProjectRoot "dist"
$Destination = Join-Path $DistRoot $OutputName
$ChecksumPath = "$Destination.sha256"

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination
}
if (Test-Path -LiteralPath $ChecksumPath) {
    Remove-Item -LiteralPath $ChecksumPath
}

$PackageItems = @(
    "DEMO.md",
    "README.md",
    "strategy.md",
    "requirements.txt",
    "start_demo.cmd",
    "config/financial_identities.yaml",
    "models/document_hrm_V8_candidate.pt",
    "web_demo",
    "src/hrm_finance_qa",
    "scripts/serve_document_hrm.py",
    "scripts/warmup_document_hrm_demo.py",
    "scripts/setup_demo.ps1",
    "scripts/start_demo.ps1",
    "scripts/setup_demo.sh",
    "scripts/start_demo.sh",
    "scripts/setup_mineru_venv.sh",
    "scripts/run_mineru_api.sh"
)

Push-Location $ProjectRoot
try {
    & tar.exe `
        -a -c -f $Destination `
        "--exclude=*/__pycache__/*" `
        "--exclude=*.pyc" `
        @PackageItems
    if ($LASTEXITCODE -ne 0) {
        throw "Không thể tạo gói demo (tar exit code $LASTEXITCODE)."
    }
} finally {
    Pop-Location
}

$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
"$Hash *$OutputName" | Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Write-Host "Đã tạo: $Destination"
Write-Host "SHA-256: $Hash"
