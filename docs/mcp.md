# Memora MCP + Skill

供 Cursor / Claude 等 Agent 通过 [Model Context Protocol](https://modelcontextprotocol.io/) 读写本地记忆。

## 组件

| 路径 | 作用 |
| --- | --- |
| `mcp/`（`memora-mcp`） | stdio MCP server，进程内嵌入 `MemoryEngine` |
| `.cursor/skills/memora-memory/` | Agent Skill：何时 remember / recall |
| `.cursor/mcp.json.example` | Cursor MCP 配置样例 |

## 快速开始

```bash
# WSL
cd /mnt/d/workspace/beiqi/rust/memora
cargo build -p memora-mcp --release
```

将 `.cursor/mcp.json.example` 复制为 `.cursor/mcp.json`，按本机路径改 `command` / `MEMORA_DB`，并在 `env` 里填写嵌入参数（`LLM_API_KEY`、`LLM_EMBED_URL`、`LLM_EMBED_MODEL`、`LLM_EMBED_DIM`）。这些值会覆盖 `.env`。经 WSL 启动时必须带上 `WSLENV`，否则变量进不了 Linux 进程。然后在 Cursor 里启用 MCP 并重启 Agent。

Skill 位于项目 `.cursor/skills/memora-memory/`，打开本仓库时 Agent 可自动发现。

## 一次性子命令

Hook 不走 MCP 握手。无子命令时仍是 stdio 服务。`recall` / `remember` 读环境变量（与服务相同），向 stdout 打一行 JSON 后退出。`--text` / `--content` 省略时从 stdin 读取。

```bash
printf '%s' '用户原话' | memora-mcp recall --top-k 5
printf '%s' '用户：…\n助手：…' | memora-mcp remember
```

召回成功形如 `{"ok":true,"hits":[{"text":"[主驾 · 记录于2026-10-26 11:20] 我有点头疼"}]}`。成员和创建时间由核心召回层生成；时间固定 UTC+8，表示记忆记录时间，不是事件发生时间。one-shot 的 context 回退路径也优先使用 `hits[].display_text`，原始记忆正文不被改写。`remember` 马上返回 `{"ok":true,"accepted":true}`，写入在后台完成。失败退出码非 0，并打 `{"ok":false,"error":"..."}`。

## 写入完成语义与召回性能

进程内 `MemoryEngine::remember/revise` 及对应 MCP 写入返回，表示记忆与持久图谱任务已提交，不表示图谱已构建；图谱由默认后台 worker 处理，失败保留任务并重试。one-shot 上述 `accepted` 仍只是已接收写入任务，不等同于事务完成。查询向量按 namespace、模型/空间和完整查询缓存与合并在途请求；记忆修改后仍重新检索、检查权限，不直接复用旧候选。抽取仍由 LLM 理解，不增加关键词过滤。

## 工具一览

`remember` · `recall` · `get_memory` · `list_memories` · `revise_memory` · `forget_memory` · `restore_memory` · `explain_memory` · `tick` · `dream` · `status`

详细环境变量与 WSL 启动方式见 Skill 内 [reference.md](../.cursor/skills/memora-memory/reference.md)。
