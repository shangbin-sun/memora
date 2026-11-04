//! MCP tools wrapping [`MemoryEngine`].

use memora::{
    DreamOptions, ForgetReason, HouseholdMember, ListQuery, MemberRole, MemoryEngine, MemoryKind,
    MemoryService, MemoryState, MemoryUpdate, RecallQuery, RememberInput,
};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Clone)]
pub struct MemoraMcp {
    engine: MemoryEngine,
    embedder_fallback: String,
    tool_router: ToolRouter<Self>,
}

impl MemoraMcp {
    pub fn new(engine: MemoryEngine, embedder_fallback: String) -> Self {
        Self {
            engine,
            embedder_fallback,
            tool_router: Self::tool_router(),
        }
    }
}

fn text_json(value: impl serde::Serialize) -> Result<CallToolResult, McpError> {
    let s = serde_json::to_string_pretty(&value)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![Content::text(s)]))
}

fn map_err(e: memora::MemoraError) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn parse_uuid(id: &str) -> Result<Uuid, McpError> {
    Uuid::parse_str(id.trim()).map_err(|e| McpError::invalid_params(format!("invalid id: {e}"), None))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberArgs {
    /// Memory content to store (required)
    pub content: String,
    #[serde(default)]
    pub summary: Option<String>,
    /// episodic | semantic | pattern | summary
    #[serde(default)]
    pub kind: Option<String>,
    /// 0~1, default 0.5
    #[serde(default)]
    pub importance: Option<f64>,
    /// 0~1, default 0.8
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub entities: Option<Vec<String>>,
    /// Pin into short-term working memory
    #[serde(default)]
    pub pin_to_stm: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    /// Household member id. When set, the memory is attributed to that member.
    #[serde(default)]
    pub member_id: Option<String>,
    /// Life domains: medical, education, travel, lifestyle, diet, finance, social, household
    #[serde(default)]
    pub domains: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallArgs {
    /// Natural-language query
    pub text: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Filter to this household member id
    #[serde(default)]
    pub member_id: Option<String>,
    /// Filter to any of these domains
    #[serde(default)]
    pub domains: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IdArgs {
    /// Memory UUID
    pub id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListArgs {
    #[serde(default)]
    pub namespace: Option<String>,
    /// active | archived | forgotten | superseded | all
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    /// Substring match on content / summary
    #[serde(default)]
    pub keyword: Option<String>,
    /// Filter by household member id
    #[serde(default)]
    pub member_id: Option<String>,
    /// Exact Chinese tag, e.g. 医疗健康
    #[serde(default)]
    pub tag: Option<String>,
    /// Default 10
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReviseArgs {
    pub id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub importance: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub entities: Option<Vec<String>>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetArgs {
    pub id: String,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DreamArgs {
    /// Lookback window in ms (default 24h)
    #[serde(default)]
    pub lookback_ms: Option<i64>,
    #[serde(default)]
    pub candidate_limit: Option<usize>,
    #[serde(default)]
    pub consolidate_limit: Option<usize>,
    #[serde(default)]
    pub archive_sources: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EmptyArgs {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MemberArgs {
    pub member_id: String,
    pub display_name: String,
    /// self | spouse | child | parent | other
    pub role: String,
    #[serde(default)]
    pub aliases: Option<Vec<String>>,
    #[serde(default)]
    pub age_band: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MemberIdArgs {
    pub member_id: String,
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NamespaceArgs {
    #[serde(default)]
    pub namespace: Option<String>,
}

#[tool_router]
impl MemoraMcp {
    #[tool(
        description = "Store a long-term memory asynchronously. Returns as soon as the write is queued; the id is assigned in the background. Use for durable facts, preferences, decisions, or events the agent should remember later."
    )]
    async fn remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.content.trim().is_empty() {
            return Err(McpError::invalid_params("content is required", None));
        }
        let mut input = RememberInput::episodic(args.content);
        if let Some(k) = args.kind.as_deref() {
            input.kind = MemoryKind::parse(k).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        }
        input.summary = args.summary;
        input.importance = args.importance.unwrap_or(0.5).clamp(0.0, 1.0);
        input.confidence = args.confidence.unwrap_or(0.8).clamp(0.0, 1.0);
        input.session_id = args.session_id;
        input.namespace = args.namespace;
        input.entities = args.entities.unwrap_or_default();
        input.pin_to_stm = args.pin_to_stm.unwrap_or(false);
        input.actor = args.actor;
        input.member_id = args.member_id;
        input.domains = args.domains.unwrap_or_default();
        let engine = self.engine.clone();
        tokio::spawn(async move {
            match engine.remember(input).await {
                Ok(id) => tracing::info!(%id, "async remember finished"),
                Err(e) => tracing::warn!(error = %e, "async remember failed"),
            }
        });
        text_json(serde_json::json!({ "accepted": true }))
    }

    #[tool(
        description = "Semantic + keyword recall. Call before answering when prior user facts/preferences may exist."
    )]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> Result<CallToolResult, McpError> {
        if args.text.trim().is_empty() {
            return Err(McpError::invalid_params("text is required", None));
        }
        let result = self
            .engine
            .recall(RecallQuery {
                text: args.text,
                namespace: args.namespace,
                session_id: args.session_id,
                top_k: args.top_k,
                member_id: args.member_id,
                domains: args.domains.unwrap_or_default(),
                ..Default::default()
            })
            .await
            .map_err(map_err)?;
        text_json(result)
    }

    #[tool(description = "Fetch one memory by UUID.")]
    async fn get_memory(
        &self,
        Parameters(args): Parameters<IdArgs>,
    ) -> Result<CallToolResult, McpError> {
        let id = parse_uuid(&args.id)?;
        let rec = self
            .engine
            .get(id)
            .await
            .map_err(map_err)?
            .ok_or_else(|| McpError::invalid_params(format!("not found: {id}"), None))?;
        text_json(rec)
    }

    #[tool(description = "List memories with optional keyword/member/tag/state filters. Default limit 10, newest first.")]
    async fn list_memories(
        &self,
        Parameters(args): Parameters<ListArgs>,
    ) -> Result<CallToolResult, McpError> {
        let state = match args.state.as_deref() {
            None | Some("") => Some(MemoryState::Active),
            Some("all") => None,
            Some(s) => Some(
                MemoryState::parse(s).map_err(|e| McpError::invalid_params(e.to_string(), None))?,
            ),
        };
        let kind = match args.kind.as_deref() {
            None | Some("") | Some("all") => None,
            Some(s) => Some(
                MemoryKind::parse(s).map_err(|e| McpError::invalid_params(e.to_string(), None))?,
            ),
        };
        let page = self
            .engine
            .list(ListQuery {
                namespace: args.namespace,
                state,
                kind,
                keyword: args.keyword,
                member_id: args.member_id,
                tag: args.tag,
                limit: args.limit.or(Some(10)),
                offset: args.offset,
            })
            .await
            .map_err(map_err)?;
        text_json(page)
    }

    #[tool(description = "Revise an existing memory (content/summary/importance/entities).")]
    async fn revise_memory(
        &self,
        Parameters(args): Parameters<ReviseArgs>,
    ) -> Result<CallToolResult, McpError> {
        let id = parse_uuid(&args.id)?;
        self.engine
            .revise(
                id,
                MemoryUpdate {
                    content: args.content,
                    summary: args.summary,
                    importance: args.importance,
                    confidence: args.confidence,
                    metadata: None,
                    entities: args.entities,
                    reason: args.reason,
                    contradicts: vec![],
                    ..MemoryUpdate::default()
                },
            )
            .await
            .map_err(map_err)?;
        text_json(serde_json::json!({ "ok": true, "id": id }))
    }

    #[tool(description = "Soft-forget a memory (mark forgotten). Prefer over delete.")]
    async fn forget_memory(
        &self,
        Parameters(args): Parameters<ForgetArgs>,
    ) -> Result<CallToolResult, McpError> {
        let id = parse_uuid(&args.id)?;
        let reason = args
            .reason
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "user".into());
        self.engine
            .forget(id, ForgetReason::User { reason })
            .await
            .map_err(map_err)?;
        text_json(serde_json::json!({ "ok": true, "id": id }))
    }

    #[tool(description = "Restore a forgotten/archived memory to active.")]
    async fn restore_memory(
        &self,
        Parameters(args): Parameters<IdArgs>,
    ) -> Result<CallToolResult, McpError> {
        let id = parse_uuid(&args.id)?;
        self.engine.restore(id).await.map_err(map_err)?;
        text_json(serde_json::json!({ "ok": true, "id": id }))
    }

    #[tool(description = "Explain why a memory was recalled / its provenance.")]
    async fn explain_memory(
        &self,
        Parameters(args): Parameters<IdArgs>,
    ) -> Result<CallToolResult, McpError> {
        let id = parse_uuid(&args.id)?;
        text_json(self.engine.explain(id).await.map_err(map_err)?)
    }

    #[tool(
        description = "Run forgetting schedule (Ebbinghaus decay): archive low-retention, forget after grace, purge old forgotten."
    )]
    async fn tick(
        &self,
        Parameters(_args): Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, McpError> {
        text_json(self.engine.tick().await.map_err(map_err)?)
    }

    #[tool(
        description = "Dream: classify pending consolidations, archive low-value, merge near-duplicates, extract patterns. Heavier than tick."
    )]
    async fn dream(
        &self,
        Parameters(args): Parameters<DreamArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut opts = DreamOptions::default();
        if let Some(ms) = args.lookback_ms {
            opts.lookback_ms = ms;
        }
        opts.candidate_limit = args.candidate_limit;
        opts.consolidate_limit = args.consolidate_limit;
        if let Some(a) = args.archive_sources {
            opts.archive_sources = a;
        }
        text_json(self.engine.dream(opts).await.map_err(map_err)?)
    }

    #[tool(description = "Engine status: namespace, embedder, summarizer, classifier.")]
    async fn status(
        &self,
        Parameters(_args): Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, McpError> {
        text_json(serde_json::json!({
            "namespace": self.engine.namespace(),
            "embedder": self.engine.embedder_model().unwrap_or(&self.embedder_fallback),
            "summarizer": self.engine.summarizer_id(),
            "classifier": self.engine.classifier_id(),
        }))
    }

    #[tool(description = "Register or update a household member (id, display name, role, aliases).")]
    async fn upsert_member(
        &self,
        Parameters(args): Parameters<MemberArgs>,
    ) -> Result<CallToolResult, McpError> {
        let role = MemberRole::parse(&args.role).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let member = HouseholdMember {
            member_id: args.member_id,
            display_name: args.display_name,
            role,
            aliases: args.aliases.unwrap_or_default(),
            age_band: args.age_band,
        };
        self.engine
            .upsert_member(args.namespace.as_deref(), member)
            .await
            .map_err(map_err)?;
        text_json(serde_json::json!({ "ok": true }))
    }

    #[tool(description = "List household members in the namespace.")]
    async fn list_members(
        &self,
        Parameters(args): Parameters<NamespaceArgs>,
    ) -> Result<CallToolResult, McpError> {
        let members = self
            .engine
            .list_members(args.namespace.as_deref())
            .await
            .map_err(map_err)?;
        text_json(members)
    }

    #[tool(description = "Remove a household member by id. Does not delete their memories.")]
    async fn remove_member(
        &self,
        Parameters(args): Parameters<MemberIdArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.engine
            .remove_member(args.namespace.as_deref(), &args.member_id)
            .await
            .map_err(map_err)?;
        text_json(serde_json::json!({ "ok": true }))
    }
}

/// Shown to the model when the MCP server connects.
const MEMORA_INSTRUCTIONS: &str = r#"Memora 是这个家庭的本地长期记忆。每一轮用户发言都必须调用本服务，不要因为内容琐碎、像闲聊或似乎和记忆无关就跳过。

每轮顺序：先 recall（用这句话里的关键词做查询），再回答。然后把这轮用户新说的每一条数据都 remember：事实、偏好、习惯、计划、健康、学业、出行、饮食、家庭事件、纠正。一句话里有多件事就多次 remember，一条只写一件事，正文要脱离对话也能读懂。只有「好的」「嗯」这类没有新信息的回复可以只 recall、不 remember。不要存密码、密钥或其他机密。

家庭名册：upsert_member 登记成员（member_id、display_name、role、aliases）。role 只能是 self、spouse、child、parent、other。别名（爸爸、小宝）用来自动判断这条记忆属于谁。list_members 查看名册，remove_member 只删名册、不删已有记忆。

写入：remember 的 content 必填。重要事实把 importance 设到 0.6 以上。已经知道是谁时传入 member_id，或用 actor 标出说话人。领域用 domains，取值 medical、education、travel、lifestyle、diet、finance、social、household；不传则按别名和用词自动归类，不确定会标成 unknown。改正用 revise_memory；收回用 forget_memory；恢复用 restore_memory。

读取：recall 用自然语言。可用 member_id、domains 限定到某位家人或某一类。get_memory、list_memories、explain_memory 用来核对单条记忆。medical 属于敏感记忆，回答里只在被问到时使用。

维护：tick 做遗忘曲线整理，dream 做巩固、去重，并按成员和领域归纳。不必每次对话都调用。status 查看当前命名空间和嵌入后端。"#;

#[tool_handler]
impl ServerHandler for MemoraMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(MEMORA_INSTRUCTIONS.into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
