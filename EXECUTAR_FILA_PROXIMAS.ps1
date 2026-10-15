$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Config = Join-Path $PSScriptRoot "configs\inference_rowgraphnet.yaml"
$Modelo = Join-Path $PSScriptRoot "modelos\best_rowgraphnet_16_imagens.pt"

# Validacao de arquivos base
foreach ($Req in @($Python, $Config, $Modelo)) {
    if (-not (Test-Path -LiteralPath $Req)) {
        throw "Arquivo essencial nao encontrado: $Req"
    }
}

# Funcao inteligente para verificar se ha outra inferencia em execucao
function Get-OtherPredictProcesses {
    $currentPid = $PID
    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -like "python*" -and
        $_.ProcessId -ne $currentPid -and
        $_.CommandLine -like "*predict.py*"
    })
    if ($null -eq $procs) {
        return @()
    }
    return @($procs)
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " FILA INTELIGENTE DE INFERENCIA - PROXIMAS AREAS " -ForegroundColor Cyan
Write-Host " Modelo: $Modelo" -ForegroundColor DarkGray
Write-Host " Config: $Config" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Espera inteligente ate que a fila anterior termine 100%
$waitingReported = $false
while ($true) {
    $activeProcs = @(Get-OtherPredictProcesses)
    if ($activeProcs.Length -gt 0) {
        $waitingReported = $true
        $pidsStr = ($activeProcs | ForEach-Object { $_.ProcessId }) -join ", "
        $timestamp = (Get-Date).ToString("HH:mm:ss")
        Write-Host "[$timestamp] [AGUARDANDO] Fila anterior ainda ativa (PID: $pidsStr). Aguardando conclusao..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
    } else {
        if ($waitingReported) {
            # Debounce de 15 segundos para garantir que o loop anterior nao estava apenas trocando de area
            Write-Host "Verificando finalizacao da fila anterior (debounce 15s)..." -ForegroundColor DarkGray
            Start-Sleep -Seconds 15
            $checkAgain = @(Get-OtherPredictProcesses)
            if ($checkAgain.Length -gt 0) {
                continue
            }
            Write-Host "`n>>> Fila anterior 100% concluida! Liberando execucao do proximo lote...`n" -ForegroundColor Green
        } else {
            Write-Host "Nenhuma inferencia em andamento detectada. Iniciando imediatamente!`n" -ForegroundColor Green
        }
        break
    }
}

# 2. Lista das proximas areas configuradas
$Jobs = @(
    @{
        Nome   = "Area 14"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\14\14.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\14\14_TALHAO.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\14\linhas_14.gpkg"
    },
    @{
        Nome   = "Area 8"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\8\result_8.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\8\8.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\8\linhas_8.gpkg"
    },
    @{
        Nome   = "Area 16"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\16.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\16.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\16\linhas_16.gpkg"
    },
    @{
        Nome   = "Area 7"
        Imagem = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\7\result_7.gpkg"
        Talhao = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\7\7_TALHAO.geojson"
        Saida  = "C:\Users\BRUNO\Documents\PROJETOS\DUPLO\areas_novas\7\linhas_7.gpkg"
    }
)

$Total = $Jobs.Count
$Index = 1

foreach ($Job in $Jobs) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " [$Index/$Total] Processando: $($Job.Nome)" -ForegroundColor Green
    Write-Host " Imagem: $($Job.Imagem)"
    Write-Host " Talhao: $($Job.Talhao)"
    Write-Host " Saida : $($Job.Saida)"
    Write-Host "============================================================" -ForegroundColor Cyan

    # Se a saida ja existir (ex: Area 16 se ja foi concluida pelo outro terminal), pula sem retrabalho
    if (Test-Path -LiteralPath $Job.Saida) {
        $saidaItem = Get-Item -LiteralPath $Job.Saida
        $tamanhoMb = [math]::Round($saidaItem.Length / 1MB, 2)
        Write-Host ">>> [PULANDO] Arquivo de saida ja existe ($tamanhoMb MB): $($Job.Saida)`n" -ForegroundColor DarkYellow
        $Index++
        continue
    }

    if (-not (Test-Path -LiteralPath $Job.Imagem)) {
        throw "Imagem nao encontrada: $($Job.Imagem)"
    }
    if (-not (Test-Path -LiteralPath $Job.Talhao)) {
        throw "Talhao nao encontrado: $($Job.Talhao)"
    }

    $saidaDir = Split-Path -Parent $Job.Saida
    if ($saidaDir -and -not (Test-Path -LiteralPath $saidaDir)) {
        New-Item -ItemType Directory -Path $saidaDir -Force | Out-Null
    }

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    & $Python (Join-Path $PSScriptRoot "scripts\predict.py") `
        --config $Config `
        --checkpoint $Modelo `
        --orthomosaic $Job.Imagem `
        --roi $Job.Talhao `
        --output $Job.Saida

    if ($LASTEXITCODE -ne 0) {
        throw "Falha na inferencia de $($Job.Nome). Codigo de saida: $LASTEXITCODE"
    }

    $Stopwatch.Stop()
    $tempoMin = [math]::Round($Stopwatch.Elapsed.TotalMinutes, 1)
    Write-Host ">>> Concluido com sucesso: $($Job.Nome) em $tempoMin min!`n" -ForegroundColor Green

    $Index++
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host " TODAS AS AREAS DO SEGUNDO LOTE FORAM CONCLUIDAS! " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
