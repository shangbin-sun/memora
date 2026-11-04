#!/usr/bin/env bash
# Memora JNI 打包入口 —— 请在 WSL 内执行（主机无 Rust）。
# Usage:
#   ./scripts/build-jni.sh linux [x86_64|aarch64]
#   ./scripts/build-jni.sh windows
#   ./scripts/build-jni.sh android [arm64-v8a|armeabi-v7a|x86_64]
#   ./scripts/build-jni.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF_NATIVES_ROOT="$ROOT/../../java/workflow/src/main/resources/natives"
cd "$ROOT"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found. Install Rust inside WSL, then re-run." >&2
  exit 1
fi

copy_to() {
  local dest_dir="$1"
  local src="$2"
  local dest_name="$3"
  mkdir -p "$dest_dir"
  cp -f "$src" "$dest_dir/$dest_name"
  echo "OK  $src -> $dest_dir/$dest_name"
}

build_linux() {
  local arch="${1:-x86_64}"
  case "$arch" in
    x86_64|amd64) dest_arch="x86_64"; target="" ;;
    aarch64|arm64)
      dest_arch="aarch64"
      target="aarch64-unknown-linux-gnu"
      rustup target add "$target" >/dev/null
      ;;
    *) echo "unsupported linux arch: $arch"; exit 1 ;;
  esac

  if [[ -n "$target" ]]; then
    cargo build -p memora --release --features jni --target "$target"
    so="$ROOT/target/$target/release/libmemora.so"
  else
    cargo build -p memora --release --features jni
    so="$ROOT/target/release/libmemora.so"
  fi
  copy_to "$WF_NATIVES_ROOT/linux-${dest_arch}" "$so" "libmemora.so"
}

build_windows() {
  local target="x86_64-pc-windows-gnu"
  if ! command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    echo "error: mingw-w64 not found. In WSL: sudo apt-get install -y mingw-w64" >&2
    exit 1
  fi
  rustup target add "$target" >/dev/null
  cargo build -p memora --release --features jni --target "$target"
  dll="$ROOT/target/$target/release/memora.dll"
  copy_to "$WF_NATIVES_ROOT/windows-x86_64" "$dll" "memora.dll"
}

# Resolve ANDROID_NDK_HOME for cargo-ndk (WSL often has Windows SDK under /mnt/c).
resolve_android_ndk() {
  local host_tag="linux-x86_64"
  case "$(uname -s)" in
    Darwin*) host_tag="darwin-x86_64" ;;
    MINGW*|MSYS*|CYGWIN*) host_tag="windows-x86_64" ;;
  esac

  ndk_has_host_toolchain() {
    local dir="$1"
    [[ -x "$dir/toolchains/llvm/prebuilt/$host_tag/bin/clang" ]] \
      || [[ -x "$dir/toolchains/llvm/prebuilt/$host_tag/bin/clang.exe" ]]
  }

  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "$ANDROID_NDK_HOME" ]]; then
    if ndk_has_host_toolchain "$ANDROID_NDK_HOME"; then
      return 0
    fi
    echo "warning: ANDROID_NDK_HOME=$ANDROID_NDK_HOME has no $host_tag toolchain; searching..." >&2
  fi
  if [[ -n "${NDK_HOME:-}" && -d "$NDK_HOME" ]] && ndk_has_host_toolchain "$NDK_HOME"; then
    export ANDROID_NDK_HOME="$NDK_HOME"
    return 0
  fi

  pick_ndk_dir() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    if ndk_has_host_toolchain "$dir"; then
      export ANDROID_NDK_HOME="$dir"
      return 0
    fi
    local latest ve
    # Prefer newest version that has the current host toolchain.
    while IFS= read -r ver; do
      ver="${ver%/}"
      if ndk_has_host_toolchain "$ver"; then
        export ANDROID_NDK_HOME="$ver"
        return 0
      fi
    done < <(ls -1d "$dir"/*/ 2>/dev/null | sort -V -r)
    return 1
  }

  local -a candidates=()
  [[ -n "${ANDROID_HOME:-}" ]] && candidates+=("$ANDROID_HOME/ndk" "$ANDROID_HOME/ndk-bundle")
  [[ -n "${ANDROID_SDK_ROOT:-}" ]] && candidates+=("$ANDROID_SDK_ROOT/ndk" "$ANDROID_SDK_ROOT/ndk-bundle")
  candidates+=(
    "$HOME/Android/Sdk/ndk"
    "$HOME/Android/Sdk/ndk-bundle"
    "$HOME/workspace/android-ndk-r27d"
    "$HOME/workspace/android-ndk"
    /opt/android-ndk
    /usr/lib/android-ndk
  )
  # Also accept a direct NDK root (not nested under ndk/<ver>).
  for c in "$HOME/workspace"/android-ndk*; do
    [[ -d "$c" ]] && candidates+=("$c")
  done

  # Windows NDK under /mnt/c only works for windows-x86_64 host, not WSL/Linux.
  if [[ "$host_tag" == "windows-x86_64" ]]; then
    local win_use
    win_user="$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r' || true)"
    if [[ -n "$win_user" ]]; then
      candidates+=("/mnt/c/Users/${win_user}/AppData/Local/Android/Sdk/ndk")
    fi
    # shellcheck disable=SC2086
    candidates+=(/mnt/c/Users/*/AppData/Local/Android/Sdk/ndk)
  fi

  local c
  for c in "${candidates[@]}"; do
    pick_ndk_dir "$c" && return 0
  done
  return 1
}

build_android() {
  local abi="${1:-arm64-v8a}"
  case "$abi" in
    arm64-v8a|armeabi-v7a|x86_64) ;;
    *) echo "unsupported android abi: $abi"; exit 1 ;;
  esac
  if ! command -v cargo-ndk >/dev/null 2>&1; then
    echo "error: cargo-ndk not found. In WSL: cargo install cargo-ndk" >&2
    exit 1
  fi
  if ! resolve_android_ndk; then
    echo "error: Could not find a Linux-host NDK (need toolchains/llvm/prebuilt/linux-x86_64)." >&2
    echo "Windows Android Studio NDK under /mnt/c/... only has windows-x86_64 and cannot be used from WSL." >&2
    echo "Install a Linux NDK, then set ANDROID_NDK_HOME, e.g.:" >&2
    echo "  mkdir -p \"\$HOME/Android/Sdk/ndk\" && cd /tmp" >&2
    echo "  curl -L -o ndk.zip https://dl.google.com/android/repository/android-ndk-r27d-linux.zip" >&2
    echo "  unzip -q ndk.zip && mv android-ndk-r27d \"\$HOME/Android/Sdk/ndk/27.3.13750724\"" >&2
    echo "  export ANDROID_NDK_HOME=\"\$HOME/Android/Sdk/ndk/27.3.13750724\"" >&2
    exit 1
  fi
  echo "ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
  local out="/tmp/memora-jni-android-$$"
  rm -rf "$out"
  cargo ndk -t "$abi" -o "$out" build -p memora --release --features jni
  local so
  so="$(find "$out" -name 'libmemora.so' | head -n 1)"
  if [[ -z "$so" ]]; then
    echo "error: libmemora.so not found under $out" >&2
    exit 1
  fi
  copy_to "$WF_NATIVES_ROOT/android-${abi}" "$so" "libmemora.so"
  rm -rf "$out"
}

cmd="${1:-}"
case "$cmd" in
  linux)
    build_linux "${2:-x86_64}"
    ;;
  windows)
    build_windows
    ;;
  android)
    build_android "${2:-arm64-v8a}"
    ;;
  all)
    build_linux x86_64
    if command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
      build_windows
    else
      echo "skip windows: mingw-w64 not installed"
    fi
    ;;
  *)
    echo "Usage: $0 {linux [arch]|windows|android [abi]|all}"
    exit 1
    ;;
esac
