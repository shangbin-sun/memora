# 主机无 Rust：通过 WSL 执行 memora JNI 打包。
# Usage:
#   .\scripts\build-jni-via-wsl.ps1 linux
#   .\scripts\build-jni-via-wsl.ps1 windows
#   .\scripts\build-jni-via-wsl.ps1 android arm64-v8a
#   .\scripts\build-jni-via-wsl.ps1 all
$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    throw "未找到 wsl。请先安装 WSL 并在其中配置 Rust。"
}

$RootWin = Split-Path -Parent $PSScriptRoot
# D:\workspace\... -> /mnt/d/workspace/...
$drive = $RootWin.Substring(0, 1).ToLowerInvariant()
$pathRest = $RootWin.Substring(2).Replace("\", "/")
$RootWsl = "/mnt/$drive$pathRest"

$argsList = $args
if ($argsList.Count -eq 0) {
    $argsList = @("linux")
}

$joined = ($argsList | ForEach-Object { $_ }) -join " "
Write-Host "WSL: cd $RootWsl && ./scripts/build-jni.sh $joined"
wsl -e bash -lc "cd '$RootWsl' && chmod +x scripts/build-jni.sh scripts/build-jni-linux.sh scripts/build-jni-android.sh 2>/dev/null; ./scripts/build-jni.sh $joined"
if ($LASTEXITCODE -ne 0) {
    throw "WSL build failed with exit $LASTEXITCODE"
}
