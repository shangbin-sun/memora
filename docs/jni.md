# Memora JNI 多平台构建（**仅在 WSL 中执行**）

> 主机 Windows **无 Rust 环境**。所有 `cargo` / `cargo-ndk` 打包请在 **WSL** 内完成，产物拷到
> `java/workflow/src/main/resources/natives/<platform>/`，供 JVM / Android 运行时按平台加载。

启用 feature `jni` 后，`memora` 以 `cdylib` 导出 `com.aiwf.memory.memora.MemoraNative`。

## 环境准备（WSL）

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Windows 交叉编译（可选，产出 memora.dll）
sudo apt-get update
sudo apt-get install -y mingw-w64
rustup target add x86_64-pc-windows-gnu

# Android（可选）
cargo install cargo-ndk
# 需要 **Linux 宿主** NDK（含 linux-x86_64 工具链）。
# Windows Android Studio 装的 NDK 只有 windows-x86_64，不能在 WSL 里用。
# 安装示例：
#   mkdir -p ~/Android/Sdk/ndk && cd /tmp
#   curl -L -o ndk.zip https://dl.google.com/android/repository/android-ndk-r27d-linux.zip
#   unzip -q ndk.zip && mv android-ndk-r27d ~/Android/Sdk/ndk/27.3.13750724
#   export ANDROID_NDK_HOME=~/Android/Sdk/ndk/27.3.13750724
# 脚本也会自动探测 ~/Android/Sdk/ndk/<version>
```

仓库路径示例（若 Windows 盘符为 D:）：

```bash
cd /mnt/d/workspace/beiqi/rust/memora
```

## 一键构建（推荐）

```bash
chmod +x scripts/*.sh
./scripts/build-jni.sh linux          # → natives/linux-x86_64/libmemora.so
./scripts/build-jni.sh windows        # → natives/windows-x86_64/memora.dll（需 mingw）
./scripts/build-jni.sh android arm64-v8a
./scripts/build-jni.sh all            # linux + windows（有工具链则编）；android 需另传 abi
```

从 **PowerShell 主机**只负责唤起 WSL（不跑 cargo）：

```powershell
wsl -e bash -lc 'cd /mnt/d/workspace/beiqi/rust/memora && ./scripts/build-jni.sh linux'
wsl -e bash -lc 'cd /mnt/d/workspace/beiqi/rust/memora && ./scripts/build-jni.sh windows'
```

也可：

```powershell
.\scripts\build-jni-via-wsl.ps1 linux
.\scripts\build-jni-via-wsl.ps1 windows
.\scripts\build-jni-via-wsl.ps1 android arm64-v8a
```

## 产物目录

| 平台 | 资源路径 |
|------|----------|
| Linux x86_64 | `natives/linux-x86_64/libmemora.so` |
| Linux aarch64 | `natives/linux-aarch64/libmemora.so` |
| Windows x86_64 | `natives/windows-x86_64/memora.dll` |
| Android arm64 | `natives/android-arm64-v8a/libmemora.so` |
| Android armv7 | `natives/android-armeabi-v7a/libmemora.so` |
| Android x86_64 | `natives/android-x86_64/libmemora.so` |

完整路径相对于：`java/workflow/src/main/resources/`。

## 运行时加载

Java 侧 `NativeLibraryLoader` 按 `os.name` + `os.arch`（Android 再看 ABI）从 classpath 解压并 `System.load`。

## 无状态 JNI（无需 create）

| Java | Rust |
|------|------|
| `normalizeFact(bodyJson)` | `temporal::normalize_fact` → `{content,event_at_ms,importance}` |
| `checkWrite(content, actor)` | `write_filter::check_write` → `{ok,reason?}` |
| `promptBundle(version)` | `prompts::bundle` ← `prompts/v1/*` |

`remember` / `revise` 在引擎内也会调用 `check_write`；凭证始终拒绝，`dream_job`/`stm_promote` 仅跳过内容安全词表。
