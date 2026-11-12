//! Memora: embeddable local-first memory engine.
//!
//! Host applications construct a [`MemoryEngine`], optionally inject an
//! [`Embedder`] / [`Summarizer`] / [`MemoryClassifier`], then call [`MemoryService`] plus `tick` / `dream`.
//!
//! ```ignore
//! let engine = MemoryEngine::builder().db_path("memora.db").build().await?;
//! let id = engine.remember(RememberInput::episodic("用户偏好 SQLite")).await?;
//! let _ = engine.recall(RecallQuery { text: "SQLite".into(), ..Default::default() }).await?;
//! engine.tick().await?;
//! engine.dream(DreamOptions::default()).await?;
//! ```

mod candidate_recall;
pub mod chunks;
pub mod question_turn;
mod classify;
mod config;
mod core;
mod dreaming;
mod embedding;
mod engine;
mod error;
mod exchange;
mod extraction_plan;
mod forgetting;
pub mod graph;
mod household;
mod long_extract;
mod edge_table;
mod model_queue;
mod performance;
mod prompts;
mod qianfan;
mod query_embedding;
mod query_rewrite;
mod recall_display;
pub mod reconcile;
mod reconcile_protocol;
mod retrieval;
mod short_term;
mod store;
mod temporal;
mod types;
mod write_filter;

#[cfg(feature = "jni")]
mod jni_bridge;

pub use classify::{ClassifyItem, ClassifyOutput, HeuristicClassifier, MemoryClassifier};
pub use config::Config;
pub use dreaming::ExtractiveSummarizer;
pub use embedding::{Embedder, HashEmbedder, SummarizeInput, SummarizeOutput, Summarizer};
pub use engine::{EngineBuilder, MemoryEngine};
pub use error::{MemoraError, Result};
pub use graph::{
    EmptyGraphExtractor, EntityType, FactObject, FactPolarity, FactStatus, GrantInput,
    GraphExtractor, GraphOverview, GraphOverviewQuery, HeuristicGraphExtractor, IngestReport,
    KgEntity, KgFact, KgGrant, PersonGraph, PersonGraphQuery, PrivacyLevel, RecalledGraphFact,
    RelationType, RevokeGrantInput,
};
pub use household::{
    complete_owned_content, ground_owner_from_source, infer_content_owner, is_shared_member_id,
    HouseholdMember, MemberRole, MemoryTag, DEFAULT_TAGS, PUBLIC_MEMBER_ID,
};
pub use query_rewrite::contextual_query;
pub use prompts::{bundle as prompt_bundle, latest as prompt_latest, output_budget, PromptBundle};
pub use qianfan::QianfanGraphExtractor;
pub use qianfan::{
    QianfanChat, QianfanClassifier, QianfanConfig, QianfanEmbedder, QianfanSummarizer,
};
pub use recall_display::render_memory_snippet;
pub use reconcile::{
    prepare_reconcile, reconcile_turn, resume_reconcile, EvidenceSource, ExtractionProgress,
    ReconcileCheckpoint, ReconcileOutcome, ReconcileReport, ReconcileRequest, RemainingRange,
};
pub use temporal::{
    adjusted_score, display_content, extract_event_at_ms, importance_for_event, normalize_fact,
    strip_event_marker, temporal_weight, today as temporal_today, NormalizedFact,
};
pub use types::{
    AuditEvent, ContextItem, ContextPack, ContextRole, DreamOptions, DreamReport, ExportRequest,
    ForgetReason, ImportMode, ImportReport, ListQuery, MemoryExplanation, MemoryId, MemoryKind,
    MemoryPage, MemoryRecord, MemoryRelation, MemoryService, MemoryState, MemoryUpdate,
    MemoryVersion, RecallQuery, RecallResult, RecalledMemory, RelationKind, RememberInput,
    ScoreBreakdown, TickReport,
};
pub use write_filter::{
    check_write, is_content_blocked, is_credential_secret, RejectReason, WriteCheck,
};

pub mod wrong_practice;

mod simple_preferences;
