param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$giProjectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $giProjectDirectory
try {
    & $Python -u run_server_experiment.py --stage smoke
    if ($LASTEXITCODE -ne 0) { throw 'GI preparation check failed' }
} finally {
    Pop-Location
}
