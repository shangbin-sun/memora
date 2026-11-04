# DEPRECATED: 主机无 Rust。请改用 WSL：
#   .\scripts\build-jni-via-wsl.ps1 windows
# 或在 WSL 内：
#   ./scripts/build-jni.sh windows
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\build-jni-via-wsl.ps1" windows
