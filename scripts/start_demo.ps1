$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxRoot = (& wsl.exe -- wslpath -u $projectRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Không chuyển được đường dẫn workspace sang WSL.' }
& wsl.exe -- bash "$linuxRoot/scripts/start_demo.sh"
exit $LASTEXITCODE
