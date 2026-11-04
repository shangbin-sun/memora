# Memora eval one-click via WSL.
# Usage:
#   .\scripts\run-eval-via-wsl.ps1
#   .\scripts\run-eval-via-wsl.ps1 --limit 20
#   .\scripts\run-eval-via-wsl.ps1 --datasets cockpit-pref-recall.json
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsPass
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$wslRoot = (wsl wslpath -a $root).Trim()
$argLine = ($ArgsPass | ForEach-Object {
    if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join ' '

$cmd = "sed -i 's/\r$//' '$wslRoot/scripts/run_eval.sh' '$wslRoot/scripts/eval_memory.py'; chmod +x '$wslRoot/scripts/run_eval.sh'; cd '$wslRoot' && ./scripts/run_eval.sh $argLine"
Write-Host "WSL: $cmd"
wsl bash -lc $cmd
exit $LASTEXITCODE
