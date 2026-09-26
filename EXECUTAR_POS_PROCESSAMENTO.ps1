[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$VetoresBrutos,

    [Parameter(Mandatory = $true)]
    [string]$Probabilidade,

    [Parameter(Mandatory = $true)]
    [string]$Talhao,

    [Parameter(Mandatory = $true)]
    [string]$Saida
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $Python scripts\refine_existing_vectors.py `
    --config configs\inference_portable.yaml `
    --input-vectors $VetoresBrutos `
    --input-layer post_inference_lines `
    --probability-raster $Probabilidade `
    --roi $Talhao `
    --output $Saida

if ($LASTEXITCODE -ne 0) { throw "O pos-processamento falhou." }
Write-Host "Pos-processamento concluido: $Saida"
