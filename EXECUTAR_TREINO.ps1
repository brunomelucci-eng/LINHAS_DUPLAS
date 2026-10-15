$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Config = Join-Path $PSScriptRoot "configs\train\rtx3060_70epochs.yaml"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente virtual Python nao encontrado em $Python"
}

if (-not (Test-Path -LiteralPath $Config)) {
    throw "Arquivo de configuracao nao encontrado em $Config"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 1/2. PREPARACAO E HIGIENIZACAO DO DATASET (MANIFESTOS)     " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $Python "scripts\prepare_manifests.py"

if ($LASTEXITCODE -ne 0) {
    throw "Erro durante a preparacao dos manifestos (Exit code: $LASTEXITCODE)"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " 2/2. INICIANDO TREINAMENTO ROWGRAPHNET (70 EPOCAS - RTX 3060)" -ForegroundColor Green
Write-Host " Config: $Config" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Green

& $Python "train.py" --config $Config

if ($LASTEXITCODE -ne 0) {
    throw "Erro durante o treinamento (Exit code: $LASTEXITCODE)"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " TREINAMENTO CONCLUIDO COM SUCESSO!                         " -ForegroundColor Green
Write-Host " Checkpoints salvos em checkpoints/exp_rtx3060_tile256_70ep " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
