param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Port = 8765,
    [string]$MinerUApiUrl = "http://127.0.0.1:8000",
    [string]$ModelPath = "",
    [string]$ProgramRerankerPath = "",
    [switch]$TypedProgramSynthesis,
    [switch]$FinancialIdentityCatalog,
    [switch]$CandidateRecallV124,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExecutable = Join-Path $ProjectRoot ".venv-demo\Scripts\python.exe"
$Checkpoint = if ($ModelPath) {
    if ([System.IO.Path]::IsPathRooted($ModelPath)) {
        $ModelPath
    } else {
        Join-Path $ProjectRoot $ModelPath
    }
} else {
    Join-Path $ProjectRoot "models\document_hrm_V8_candidate.pt"
}
$ServerScript = Join-Path $ProjectRoot "scripts\serve_document_hrm.py"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Chưa có môi trường demo. Chạy scripts\setup_demo.ps1 một lần trước."
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Không tìm thấy checkpoint V8: $Checkpoint"
}
if ($ProgramRerankerPath) {
    $ProgramReranker = if ([System.IO.Path]::IsPathRooted($ProgramRerankerPath)) {
        $ProgramRerankerPath
    } else {
        Join-Path $ProjectRoot $ProgramRerankerPath
    }
    if (-not (Test-Path -LiteralPath $ProgramReranker)) {
        throw "Khong tim thay program-reranker sidecar: $ProgramReranker"
    }
}

$Arguments = @(
    $ServerScript,
    "--model-path", $Checkpoint,
    "--device", $Device,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--mineru-api-url", $MinerUApiUrl
)
if ($ProgramRerankerPath) {
    $Arguments += @("--program-reranker-path", $ProgramReranker)
}
if ($TypedProgramSynthesis) {
    $Arguments += "--typed-program-synthesis"
}
if ($FinancialIdentityCatalog) {
    $Arguments += "--financial-identity-catalog"
}
if ($CandidateRecallV124) {
    $Arguments += "--candidate-recall-v12-4"
}
if (-not $NoBrowser) {
    $Arguments += "--open-browser"
}

Write-Host "Khởi động FinHRM local tại http://127.0.0.1:$Port"
Write-Host "Nhấn Ctrl+C để dừng."
& $PythonExecutable @Arguments
