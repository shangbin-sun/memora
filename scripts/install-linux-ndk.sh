#!/usr/bin/env bash
set -euo pipefail
NDK_VER="27.3.13750724"
URL="https://dl.google.com/android/repository/android-ndk-r27d-linux.zip"
DEST="$HOME/Android/Sdk/ndk/$NDK_VER"
ZIP="/tmp/android-ndk-r27d-linux.zip"

echo "HOME=$HOME"
mkdir -p "$(dirname "$DEST")"
ls -la "$(dirname "$DEST")" || true

if [[ -x "$DEST/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" ]]; then
  echo "already ok: $DEST"
  exit 0
fi

rm -rf "$DEST"

if [[ ! -f "$ZIP" ]]; then
  echo "download $URL"
  curl -L --fail -o "$ZIP" "$URL"
fi
ls -lh "$ZIP"

echo "unzip..."
rm -rf /tmp/android-ndk-r27d
unzip -q "$ZIP" -d /tmp
mv /tmp/android-ndk-r27d "$DEST"
test -x "$DEST/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
"$DEST/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" --version | head -2
echo "OK $DEST"
