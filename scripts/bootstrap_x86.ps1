param(
    [string]$Python = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$LoadOnly
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
& $Python -m venv .venv
$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install torch torchvision --index-url $TorchIndexUrl
& $VenvPython -m pip install -e ".[workbench,inference,dev]"
& $VenvPython -m fair_agent.cli doctor
if ($LoadOnly) {
    & $VenvPython scripts/smoke_models.py --load-only
} else {
    & $VenvPython scripts/smoke_models.py
}

Write-Host ""
Write-Host "AgileAgent 已准备就绪。使用以下命令启动界面："
Write-Host "  .venv\Scripts\python.exe -m fair_agent.cli refresh"
Write-Host "  .venv\Scripts\python.exe -m fair_agent.cli serve"
