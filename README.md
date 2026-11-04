# Memora

可嵌入 Rust 应用的本地优先记忆引擎：短期工作记忆（moka）、长期记忆（SQLite + FTS5 + HNSW）、艾宾浩斯遗忘、梦境总结、审计与 `.memora` 导入导出。

`memora` 是纯库 crate：不启动 HTTP、不捆绑 Web UI。独立运行请用 workspace 里的 `memora-server`。

## 作为库使用

```rust
use std::sync::Arc;
use memora::{HashEmbedder, MemoryEngine, MemoryService, RememberInput, RecallQuery};

# async fn demo() -> memora::Result<()> {
let engine = MemoryEngine::builder()
    .db_path("memora.db")
    .embedder(Arc::new(HashEmbedder::new(64)))
    .build()
    .await?;

let id = engine.remember(RememberInput::episodic("用户偏好 SQLite")).await?;
let hits = engine.recall(RecallQuery { text: "SQLite".into(), ..Default::default() }).await?;
let _ = (id, hits);
# Ok(())
# }
```

只编译 / 打包库（不含 server 与 Web）：

```bash
cargo build -p memora
cargo package -p memora --list
```

`Cargo.toml` 的 `default-members` 指向库本身，工作区根目录执行 `cargo build` / `cargo test` 也不会带上 server。

## JNI（Java / Android / Windows / Linux）

> **在 WSL 中打包**（Windows 主机无 Rust）。见 [docs/jni.md](docs/jni.md)。

```bash
# WSL
cd /mnt/d/workspace/beiqi/rust/memora
./scripts/build-jni.sh linux     # libmemora.so
./scripts/build-jni.sh windows   # memora.dll（需 mingw-w64）
./scripts/build-jni.sh android arm64-v8a
```

主机 PowerShell 仅唤起 WSL：

```powershell
.\scripts\build-jni-via-wsl.ps1 linux
```

导出 `com.aiwf.memory.memora.MemoraNative`（create / destroy / remember / recall / list / …，以及无 handle 的
`normalizeFact` / `checkWrite` / `promptBundle`）。产物落到
`java/workflow/src/main/resources/natives/<platform>/`。

库内配套模块（宿主可直接 `use memora::…`）：

| 模块 | 作用 |
|------|------|
| `temporal` | 相对日期归一化（`Asia/Shanghai`）、`event_at_ms` / 衰减权重 |
| `write_filter` | 凭证 + 内容安全写入门禁（`remember`/`revise` 内置） |
| `prompts` | 版本化 curator 提示词（`prompts/v1/`，`CURRENT_VERSION`） |

运行集成样例：

```bash
cargo run --example embedded
```

## MCP / Skill（供其他 Agent 调用）

```bash
cargo build -p memora-mcp --release
// 编译，zig做连接器，编译musl静态产物
cargo zbuild -p memora-mcp  --target aarch64-unknown-linux-musl
```

stdio MCP 进程内嵌入记忆引擎，暴露 `remember` / `recall` / `forget` / `tick` / `dream` 等工具。Skill 说明见 [skills/memora-memory/SKILL.md](skills/memora-memory/SKILL.md)，MCP 接入见 [docs/mcp.md](docs/mcp.md)。

源码仓库不包含预编译二进制或 Skill 发布压缩包。运行上面的 `cargo build -p memora-mcp --release` 后，产物位于 `target/release/memora-mcp`（Windows 为 `memora-mcp.exe`）；制作 Skill 安装包时，将目标平台的可执行文件复制到 `skills/memora-memory/`，再压缩该目录。

## 独立 Web 服务

```bash
cargo run -p memora-server
```

默认监听 `http://127.0.0.1:8787`，打开即可在页面上写入、召回、解释、tick / dream、导入导出。写入默认进入 STM，并标记长期记忆「待巩固」；`dream` 时按批分类巩固。

嵌入后端由 `MEMORA_EMBEDDER` / `--embedder` 选择：`auto`（有 `LLM_API_KEY` 走 LLM embeddings，否则 Hash）、`hash`、`llm`/`qianfan`（OpenAI 兼容 `/embeddings`，由 `LLM_EMBED_URL` / `LLM_EMBED_MODEL` / `LLM_EMBED_DIM` 指定）。

常用参数：

```bash
cargo run -p memora-server -- --bind 127.0.0.1:8787 --db data/memora.db --namespace default --embedder llm
```

| 参数 / 环境变量 | 含义 |
| --- | --- |
| `--bind` / `MEMORA_BIND` | 监听地址 |
| `--db` / `MEMORA_DB` | SQLite 路径 |
| `--namespace` / `MEMORA_NAMESPACE` | 默认命名空间 |
| `--embedder` / `MEMORA_EMBEDDER` | `auto` \| `hash` \| `llm` \| `qianfan` |
| `--hash-embedder` | 强制哈希嵌入（等同 `embedder=hash`） |
| `--tick-interval` / `MEMORA_TICK_INTERVAL` | 后台 tick 秒数，默认 600；0 关闭 |
| `--dream-interval` / `MEMORA_DREAM_INTERVAL` | 后台 dream 秒数，默认 180；0 关闭 |
