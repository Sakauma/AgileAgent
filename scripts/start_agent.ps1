param(
    [string]$Python
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if (-not $Python) {
    $Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到已配置的 Python：$Python`n请先运行 scripts/bootstrap_x86.ps1。"
}

function Invoke-Checked {
    param([string[]]$CommandArgs)
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败：$Python $($CommandArgs -join ' ')"
    }
}

Invoke-Checked -CommandArgs @("-m", "fair_agent.cli", "doctor")
Invoke-Checked -CommandArgs @("-m", "fair_agent.cli", "refresh")
Invoke-Checked -CommandArgs @("-m", "fair_agent.cli", "decide")

Write-Host ""
Write-Host "正在启动 AgileAgent 工作台：http://localhost:8501"
Write-Host "按 Ctrl+C 可停止服务。"
Invoke-Checked -CommandArgs @("-m", "fair_agent.cli", "serve")
