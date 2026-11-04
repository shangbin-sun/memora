#!/usr/bin/env bash
# 兼容旧入口：转发到 build-jni.sh android
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/build-jni.sh" android "${1:-arm64-v8a}" &&
cp -r target/aarch64-linux-android/release/libmemora.so /mnt/d/workspace/beiqi/android/sprite/app/src/main/jniLibs/arm64-v8a/libmemora.so
