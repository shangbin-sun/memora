---
name: memora-memory
description: >-
  Memora long-term memory maintenance. Cursor hooks already recall before
  each turn and remember the final answer. Use this skill to revise, forget,
  inspect, tick, or dream. Do not call recall or remember yourself on a
  normal turn.
---

# Memora Memory

本地优先记忆。读写只走 MCP 服务器 `memora`，禁止自建文件、笔记或平行存储代替。

每一轮的 `recall` 和问答 `remember` 由 Cursor Hook 完成（发提示词前召回，回答结束后写入）。正常回合不要再自己调用这两个工具，也不要因为 Hook 已经注入了召回结果再补一次。

用户纠正或收回旧说法时，用 `revise_memory` 或 `forget_memory`。需要巩固、去重、抽规律时用 `dream`，遗忘维护用 `tick`。查看用 `get_memory` / `list_memories` / `explain_memory` / `status`。

Hook 写入的正文只有这两段，不含工具轨迹：

```text
用户：<本轮用户 query 原文>
助手：<本轮最终回答原文>
```

密码、API Key、token、私钥出现时，对应段落写成「已脱敏」。

## 未安装时自动安装配置

只有在需要 `revise_memory` / `forget_memory` / `dream`，且当前工具列表里没有它们时，才做下面的安装。已可用则跳过本节。不要为了每轮召回去检查或调用 `recall`。

不可用时 **不要询问用户是否安装**，立刻做完下面的步骤。

### 1. 定位二进制

本技能目录 = 本 `SKILL.md` 所在目录，与宿主把技能包放在哪里无关。

使用本技能目录下的 `memora-mcp`。`Permission denied` 时对该文件执行 `chmod +x`，然后重试。文件不存在就告诉用户技能包里缺少 `memora-mcp`，然后停止安装。

### 2. 写入 MCP 配置

写入当前宿主用来加载 MCP 的配置（已有 `mcpServers` 的那个文件或设置界面）。没有就按该宿主的方式新建。只合并 `mcpServers.memora`，保留其他 server。不要假设配置文件的固定路径。

环境变量：

| 变量 | 值 |
| --- | --- |
| `MEMORA_DB` | 本机可写的 sqlite 路径，例如技能目录旁的 `memora.db`（经 WSL 启动时用 `/mnt/...` 路径） |
| `MEMORA_NAMESPACE` | `default` |
| `MEMORA_EMBEDDER` | **已有 MCP 配置或进程环境里的值原样保留，禁止改成 `hash`。** 只有从未配置过时：已有 `LLM_API_KEY` 用 `llm`，否则用 `hash` |
| `LLM_*` | 仅从本机已有 MCP 配置或 `.env` 复制。没有就省略。不要写占位符，不要提交密钥，不要在命令行或日志里打印密钥 |

`command` 填 `memora-mcp` 的绝对路径，`args` 为 `[]`。数据库路径用本机可写的绝对路径。经 WSL 启动 Linux 二进制时，`command` 用 `wsl`，`args` 为 `["-e", "<WSL 内的 memora-mcp 绝对路径>"]`，并设置 `WSLENV`，把 `MEMORA_*` 与 `LLM_*` 都列进去且以 `/u` 结尾，否则环境变量进不了 WSL 进程。

嵌入器只在 **进程启动时** 选定。MCP 配置里的 `env` 只对由宿主拉起的那个进程生效。手动执行二进制时，写在命令前面的 `MEMORA_EMBEDDER=hash` 会盖掉配置；这种进程写进去的 `embedding_model` 是 `hash-ngram-v1`。禁止为了测试或补记再开一个带 `MEMORA_EMBEDDER=hash` 的一次性进程。改完配置后要重启 MCP，让新进程读到 `llm`。可用 `status` 核对：`embedder` 应为嵌入模型名（例如 `Qwen3-Embedding-8B`），而不是 `hash-ngram-v1`。`llm` 模式缺少 `LLM_API_KEY` 会直接启动失败，不会悄悄退回 hash；只有 `auto` 才会在没有密钥时退回 hash。

### 3. 安装后立刻再用

配置写完后重新发现 `memora` 工具。能看到 `revise_memory` / `forget_memory` 即可。不要为了验收去调用 `recall` 或 `remember`。

若配置已写入但本会话仍没有这些工具，用一句话告诉用户：在当前宿主里启用并重载 MCP 服务器 `memora`。在工具真正可用之前，不要声称已经改过记忆。

## 其他工具

| 场景 | 工具 |
| --- | --- |
| 每轮召回与问答写入 | Cursor Hook，不要自己调用 `recall` / `remember` |
| 纠正或收回 | `revise_memory` 或 `forget_memory` |
| 遗忘维护 | `tick` |
| 巩固、去重、抽规律 | `dream` |

- `remember`：`content` 必填。可选 `kind`（`episodic` \| `semantic` \| `pattern` \| `summary`）、`importance`（0–1）、`entities`、`session_id`、`namespace`、`pin_to_stm`、`member_id`、`domains`。
- `recall`：`text` 必填。可选 `top_k`、`namespace`、`session_id`、`member_id`、`domains`。
- `upsert_member` / `list_members` / `remove_member`：家庭成员名册。
- `get_memory` / `list_memories` / `explain_memory`：查看。
- `status`：命名空间与嵌入器。

## 怎么改记忆

只在用户纠正、收回，或需要查看 / 巩固时动工具。不要把执行过程写进去。

```text
用户：上次说的默认命名空间改成 sprite，不要再记成 default。
助手：已用 revise_memory 改掉那条命名空间记忆。
```
