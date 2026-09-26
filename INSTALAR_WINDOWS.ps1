[CmdletBinding()]
param(
    [ValidateSet("CUDA", "CPU")]
    [string]$Modo = "CUDA"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$PythonBase = $null
$PythonArgs = @()

$KnownPaths = @(
    "py.exe",
    "$env:LocalAppData\Programs\Python\Python312\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:LocalAppData\Programs\Python\Launcher\py.exe",
    "python.exe"
)

foreach ($candidate in $KnownPaths) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        $target = $cmd.Source
        if ($target -like "*WindowsApps*") { continue }
        if ($candidate -like "*py.exe*") {
            $test = & $target -3.12 -c "import sys; print(sys.version_info[:2] == (3, 12))" 2>$null
            if ($test -eq "True") {
                $PythonBase = $target
                $PythonArgs = @("-3.12")
                break
            }
        } else {
            $test = & $target -c "import sys; print(sys.version_info[:2] == (3, 12))" 2>$null
            if ($test -eq "True") {
                $PythonBase = $target
                $PythonArgs = @()
                break
            }
        }
    }
}

if (-not $PythonBase) {
    throw "Python 3.12 nao encontrado. Instale Python 3.12 x64 e marque Add Python to PATH."
}

& $PythonBase @PythonArgs -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 x64'"
if ($LASTEXITCODE -ne 0) {
    throw "Este pacote foi validado com Python 3.12 x64."
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $PythonBase @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente .venv." }
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Falha ao atualizar pip/setuptools/wheel." }

if ($Modo -eq "CUDA") {
    & $Python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
}
else {
    & $Python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar PyTorch no modo $Modo." }

& $Python -m pip install -r requirements-portable.txt
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar requirements-portable.txt." }

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "O ambiente possui dependencias incompatíveis." }

& $Python verificar_instalacao.py
if ($LASTEXITCODE -ne 0) { throw "A verificacao final da instalacao falhou." }

Write-Host ""
Write-Host "Instalacao concluida. Use .\EXECUTAR_INFERENCIA.ps1 para processar uma imagem."
