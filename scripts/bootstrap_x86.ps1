param(
    [string]$Python = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$LoadOnly
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$CommandArgs
    )
    & $Executable @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败：$Executable $($CommandArgs -join ' ')"
    }
}

$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Invoke-Checked -Executable $Python -CommandArgs @("-m", "venv", ".venv")
}
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "pip", "install", "--upgrade", "pip")
& $VenvPython -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "torchvision", "--index-url", $TorchIndexUrl)
}
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "pip", "install", "-e", ".[workbench,inference,dev]")
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "fair_agent.cli", "doctor")
if ($LoadOnly) {
    Invoke-Checked -Executable $VenvPython -CommandArgs @("scripts/smoke_models.py", "--load-only")
} else {
    Invoke-Checked -Executable $VenvPython -CommandArgs @("scripts/smoke_models.py")
}
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "fair_agent.cli", "refresh")
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "fair_agent.cli", "decide")

Write-Host ""
Write-Host "AgileAgent 已准备就绪，正在启动工作台。按 Ctrl+C 可停止服务。"
Invoke-Checked -Executable $VenvPython -CommandArgs @("-m", "fair_agent.cli", "serve")
