[CmdletBinding()]
param(
    [string]$Config = "configs\inference_rowgraphnet.yaml",
    [string]$Modelo = "modelos\best_rowgraphnet_16_imagens.pt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$Jobs = @(
    @{
        Nome   = "Area 12"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\12\12.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\12\12.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\12\linhas_12.gpkg"
    },
    @{
        Nome   = "Area 13"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\13\result_13.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\13\13.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\13\linhas_13.gpkg"
    },
    @{
        Nome   = "Area 15"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\15\15.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\15\VETORES\15.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\15\linhas_15.gpkg"
    },
    @{
        Nome   = "Area 16"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\16.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\16.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\linhas_16.gpkg"
    }
)

$Total = $Jobs.Count
$Indice = 1

foreach ($Job in $Jobs) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "[$Indice/$Total] Executando inferencia: $($Job.Nome)" -ForegroundColor Green
    Write-Host "Modelo: $Modelo" -ForegroundColor DarkGray
    Write-Host "Imagem: $($Job.Imagem)"
    Write-Host "Talhao: $($Job.Talhao)"
    Write-Host "Saida:  $($Job.Saida)"
    Write-Host "========================================================`n" -ForegroundColor Cyan

    & powershell -ExecutionPolicy Bypass -File .\EXECUTAR_INFERENCIA.ps1 `
        -Imagem $Job.Imagem `
        -Talhao $Job.Talhao `
        -Saida  $Job.Saida `
        -Config $Config `
        -Modelo $Modelo

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Falha na execucao da $($Job.Nome). Interrompendo fila."
        exit $LASTEXITCODE
    }

    $Indice++
}

Write-Host "`n[SUCESSO] Todas as $($Total) inferencias da fila foram concluidas!" -ForegroundColor Green
