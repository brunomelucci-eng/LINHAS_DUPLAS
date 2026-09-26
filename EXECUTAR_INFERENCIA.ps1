[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Imagem,

    [Parameter(Mandatory = $true)]
    [string]$Talhao,

    [string]$Saida
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Config = Join-Path $PSScriptRoot "configs\inference_portable.yaml"
$Modelo = Join-Path $PSScriptRoot "modelos\best_23_08_2026_12_imagens_run_001.pt"

foreach ($Arquivo in @($Python, $Config, $Modelo, $Imagem, $Talhao)) {
    if (-not (Test-Path -LiteralPath $Arquivo)) {
        throw "Arquivo nao encontrado: $Arquivo"
    }
}

if ([string]::IsNullOrWhiteSpace($Saida)) {
    $Carimbo = Get-Date -Format "yyyyMMdd_HHmmss"
    $Saida = Join-Path $PSScriptRoot "outputs\inferencia\linhas_$Carimbo.gpkg"
}
$Saida = [System.IO.Path]::GetFullPath($Saida)
$PastaSaida = Split-Path -Parent $Saida
New-Item -ItemType Directory -Path $PastaSaida -Force | Out-Null

& $Python scripts\predict.py `
    --config $Config `
    --checkpoint $Modelo `
    --orthomosaic $Imagem `
    --roi $Talhao `
    --output $Saida

if ($LASTEXITCODE -ne 0) {
    throw "A inferencia ou o pos-processamento falhou. Consulte outputs\runs\<run_id>\logs\execution.log."
}

Write-Host "Inferencia concluida. Produto operacional sem serrilhado: $Saida"
Write-Host "O GPKG canonico completo esta em outputs\runs\<run_id>\vectors."
