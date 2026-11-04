use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;

pub type MemoryId = Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Pattern,
    Summary,
}

impl MemoryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Pattern => "pattern",
            Self::Summary => "summary",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "pattern" => Ok(Self::Pattern),
            "summary" => Ok(Self::Summary),
            other => Err(crate::error::MemoraError::InvalidState(format!(
                "unknown kind: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryState {
    Active,
    Archived,
    Forgotten,
    Superseded,
}

impl MemoryState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Archived => "archived",
            Self::Forgotten => "forgotten",
            Self::Superseded => "superseded",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "archived" => Ok(Self::Archived),
            "forgotten" => Ok(Self::Forgotten),
            "superseded" => Ok(Self::Superseded),
            other => Err(crate::error::MemoraError::InvalidState(format!(
                "unknown state: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Supports,
    Contradicts,
    Summarizes,
    DerivedFrom,
    Supersedes,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Summarizes => "summarizes",
            Self::DerivedFrom => "derived_from",
            Self::Supersedes => "supersedes",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "supports" => Ok(Self::Supports),
            "contradicts" => Ok(Self::Contradicts),
            "summarizes" => Ok(Self::Summarizes),
            "derived_from" => Ok(Self::DerivedFrom),
            "supersedes" => Ok(Self::Supersedes),
            other => Err(crate::error::MemoraError::InvalidState(format!(
                "unknown relation: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberInput {
    pub content: String,
    pub summary: Option<String>,
    pub kind: MemoryKind,
    pub importance: f64,
    pub confidence: f64,
    pub session_id: Option<String>,
    pub namespace: Option<String>,
    pub entities: Vec<String>,
    pub source_hash: Option<String>,
    pub metadata: serde_json::Value,
    pub pin_to_stm: bool,
    pub stm_key: Option<String>,
    pub parent_id: Option<MemoryId>,
    pub actor: Option<String>,
    /// When true, write as already consolidated (no dream re-classify queue).
    #[serde(default)]
    pub skip_consolidation: bool,
    /// Explicit household member. Skips name matching for the subject.
    #[serde(default)]
    pub member_id: Option<String>,
    /// Explicit life-domain labels (`diet`, `medical`, …). Kept for classifier compat.
    #[serde(default)]
    pub domains: Vec<String>,
}

impl RememberInput {
    pub fn episodic(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            summary: None,
            kind: MemoryKind::Episodic,
            importance: 0.5,
            confidence: 0.8,
            session_id: None,
            namespace: None,
            entities: Vec::new(),
            source_hash: None,
            metadata: serde_json::json!({}),
            pin_to_stm: false,
            stm_key: None,
            parent_id: None,
            actor: None,
            skip_consolidation: false,
            member_id: None,
            domains: Vec::new(),
        }
    }
}

fn initial_revision() -> i64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    /// Optimistic concurrency token for content, facets and lifecycle changes.
    #[serde(default = "initial_revision")]
    pub revision: i64,
    pub kind: MemoryKind,
    pub state: MemoryState,
    pub content: String,
    pub summary: Option<String>,
    pub importance: f64,
    pub confidence: f64,
    pub salience: f64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub last_accessed_at_ms: Option<i64>,
    pub next_review_at_ms: Option<i64>,
    pub review_count: i64,
    pub source_hash: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_version: i64,
    pub embedding_pending: bool,
    #[serde(default)]
    pub consolidation_pending: bool,
    pub parent_id: Option<MemoryId>,
    pub namespace: String,
    pub session_id: Option<String>,
    pub entities: Vec<String>,
    pub metadata: serde_json::Value,
    /// Single searchable tag stored as a column (e.g. `医疗健康`).
    #[serde(default)]
    pub tag: String,
    /// Searchable household member ids for this memory.
    #[serde(default)]
    pub subject_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallQuery {
    pub text: String,
    pub namespace: Option<String>,
    pub session_id: Option<String>,
    pub kinds: Vec<MemoryKind>,
    pub time_from_ms: Option<i64>,
    pub time_to_ms: Option<i64>,
    pub top_k: Option<usize>,
    pub query_embedding: Option<Vec<f32>>,
    /// Local-only fallback without calling a remote embedding service.
    pub fts_only: bool,
    /// Keep hits whose subjects include this household member.
    pub member_id: Option<String>,
    /// Keep hits tagged with any of these legacy English domains.
    pub domains: Vec<String>,
    /// Keep hits with this exact Chinese tag (e.g. `医疗健康`).
    pub tag: Option<String>,
    /// When set, filter graph hits by privacy for this viewer member.
    pub viewer_member_id: Option<String>,
    /// Skip knowledge-graph recall path (memory-only).
    pub skip_graph: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListQuery {
    pub namespace: Option<String>,
    pub state: Option<MemoryState>,
    pub kind: Option<MemoryKind>,
    /// Substring match on content / summary (case-insensitive for ASCII).
    pub keyword: Option<String>,
    /// Keep rows whose `subject_ids` contain this member id.
    pub member_id: Option<String>,
    /// Exact match on the Chinese `tag` column.
    pub tag: Option<String>,
    /// Page size; default 10.
    pub limit: Option<usize>,
    /// Row offset; default 0.
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPage {
    pub items: Vec<MemoryRecord>,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub bm25_rank: Option<usize>,
    pub bm25_score: Option<f64>,
    pub vector_rank: Option<usize>,
    pub l2: Option<f32>,
    pub rrf: f64,
    pub decay: f64,
    pub confidence: f64,
    pub importance: f64,
    pub access_boost: f64,
    pub final_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledMemory {
    pub memory: MemoryRecord,
    pub scores: ScoreBreakdown,
    pub from_stm: bool,
    /// Recall presentation including subjects and creation time; raw memory is unchanged.
    #[serde(default)]
    pub display_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextRole {
    StmFact,
    Atomic,
    Summary,
    ExpandedSource,
    /// Structured knowledge-graph fact rendered as natural language.
    GraphFact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub memory_id: MemoryId,
    pub role: ContextRole,
    pub text: String,
    pub tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub items: Vec<ContextItem>,
    pub estimated_tokens: usize,
    pub budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub hits: Vec<RecalledMemory>,
    pub stm_hits: Vec<MemoryRecord>,
    pub context: ContextPack,
    /// Second recall path: structured family knowledge-graph facts.
    #[serde(default)]
    pub graph_hits: Vec<crate::graph::RecalledGraphFact>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryUpdate {
    /// Optional foreground compare-and-set token, used by JNI metadata overlays.
    pub expected_revision: Option<i64>,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub importance: Option<f64>,
    pub confidence: Option<f64>,
    pub metadata: Option<serde_json::Value>,
    pub entities: Option<Vec<String>>,
    pub reason: Option<String>,
    pub contradicts: Vec<MemoryId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ForgetReason {
    User { reason: String },
    Decay,
    Purge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVersion {
    pub id: MemoryId,
    pub memory_id: MemoryId,
    pub version: i64,
    pub operation: String,
    pub before_json: Option<String>,
    pub after_json: String,
    pub actor: String,
    pub reason: Option<String>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelation {
    pub from_id: MemoryId,
    pub to_id: MemoryId,
    pub relation: RelationKind,
    pub weight: f64,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: MemoryId,
    pub seq: i64,
    pub memory_id: Option<MemoryId>,
    pub event_type: String,
    pub payload_json: String,
    pub prev_hash: Option<String>,
    pub hash: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExplanation {
    pub memory: MemoryRecord,
    pub versions: Vec<MemoryVersion>,
    pub relations: Vec<MemoryRelation>,
    pub recent_audit: Vec<AuditEvent>,
    pub last_scores: Option<ScoreBreakdown>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportRequest {
    pub namespace: Option<String>,
    pub include_forgotten: bool,
    pub include_vectors: bool,
    /// Omit memories tagged medical / sensitivity=high.
    pub exclude_sensitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportMode {
    Merge,
    ReplaceNamespace,
    DryRun,
    Reembed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportReport {
    pub inserted: usize,
    pub updated: usize,
    pub skipped: usize,
    pub requeued_embed: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TickReport {
    pub promoted: usize,
    pub reviewed: usize,
    pub archived: usize,
    pub forgotten: usize,
    /// Physically deleted after sitting in `forgotten` past the grace window.
    pub purged: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamOptions {
    pub lookback_ms: i64,
    pub candidate_limit: Option<usize>,
    /// Archive (never purge) clustered sources. False keeps them active.
    pub archive_sources: bool,
    /// Max pending memories to classify in this dream run.
    pub consolidate_limit: Option<usize>,
}

impl Default for DreamOptions {
    fn default() -> Self {
        Self {
            lookback_ms: 24 * 60 * 60 * 1000,
            candidate_limit: None,
            // Keep source memories searchable; summaries are additive, not a replace.
            archive_sources: false,
            consolidate_limit: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamReport {
    pub input_ids: Vec<MemoryId>,
    pub output_ids: Vec<MemoryId>,
    pub clusters: usize,
    pub patterns: usize,
    pub summarizer_id: String,
    pub consolidated: usize,
    pub archived_low_value: usize,
    pub classifier_id: String,
    /// Memories removed by exact-content merge/purge (physical delete).
    #[serde(default)]
    pub merged: usize,
    /// Near-duplicate / exact groups processed.
    #[serde(default)]
    pub dedup_groups: usize,
    /// Physically deleted superseded/archived exact-content clones.
    #[serde(default)]
    pub purged: usize,
}

#[async_trait::async_trait]
pub trait MemoryService: Send + Sync {
    async fn remember(&self, input: RememberInput) -> Result<MemoryId>;
    async fn recall(&self, query: RecallQuery) -> Result<RecallResult>;
    async fn revise(&self, id: MemoryId, update: MemoryUpdate) -> Result<()>;
    async fn forget(&self, id: MemoryId, reason: ForgetReason) -> Result<()>;
    async fn explain(&self, id: MemoryId) -> Result<MemoryExplanation>;
    async fn export(&self, req: ExportRequest) -> Result<Vec<u8>>;
    async fn import(&self, archive: &[u8], mode: ImportMode) -> Result<ImportReport>;
}
