#!/usr/bin/env bash
# 兼容旧入口：转发到 build-jni.sh linux
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/build-jni.sh" linux "${1:-x86_64}"
