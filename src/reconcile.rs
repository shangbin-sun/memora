//! Android-aligned turn curator: extract, evidence gate, ops, remember.
//!
//! Mirrors sprite `reconcileMemoraTurn` so the Web UI can exercise the same path.

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{
    prompt_latest, ForgetReason, HouseholdMember, MemberRole, MemoryEngine, MemoryService,
    MemoryUpdate, QianfanChat, RecallQuery, RememberInput, PUBLIC_MEMBER_ID,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct Fact {
    owner_id: String,
    content: String,
    tag: String,
    llm_event_at: Option<String>,
    event_at_ms: Option<i64>,
    importance: Option<f64>,
    evidence: String,
    graph_already_ingested: bool,
    context_provenance: Value,
    source_span: Value,
    root_source_id: String,
    source_id: String,
    source_kind: String,
    speaker_member_id: Option<String>,
    message_at_ms: i64,
    origin: FactOrigin,
}

/// Where a fact came from, kept apart so a fact can always be traced back.
#[derive(Clone, Debug, Default)]
struct FactOrigin {
    source_name: String,
    source_locator: Option<String>,
    file_path: Option<String>,
    session_id: Option<String>,
    message_id: Option<String>,
    /// Full user text for conversation facts; survives deleting the chat session.
    utterance: Option<String>,
    preserve_original: bool,
}

fn fact_key(fact: &Fact) -> String {
    // Independent sources and distinct event dates are never silently merged.
    json!([
        fact.owner_id,
        fact.tag,
        normalize_memory_text(&fact.content),
        fact.llm_event_at,
        fact.root_source_id
    ])
    .to_string()
}

fn extracted_item_key(key: &str, body: &str, item: &Value) -> String {
    json!([
        key,
        normalize_memory_text(body),
        item.get("tag"),
        item.get("owner").or_else(|| item.get("owner_name")),
        item.get("subjects"),
        item.get("event_at"),
        item.get("evidence"),
        item.get("role"),
        item.get("aliases")
    ])
    .to_string()
}

/// Host-supplied evidence (OCR / document parse / user text). Optional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub id: String,
    #[serde(default = "default_evidence_kind")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    pub text: String,
    #[serde(default)]
    pub locator: Option<String>,
    #[serde(default)]
    pub truncated: bool,
    /// Trusted host acknowledgement for this source.text only (not model output).
    /// Skip duplicate graph projection for NEW memories extracted from this source.
    #[serde(default)]
    pub graph_already_ingested: bool,
    /// Host opt-in: keep original excerpts behind this source's facts (txt/md only).
    #[serde(default)]
    pub preserve_original: bool,
    /// Real on-device path of the uploaded file, for provenance.
    #[serde(default)]
    pub file_path: Option<String>,
}

fn default_evidence_kind() -> String {
    "user_text".into()
}

#[derive(Debug, Deserialize)]
pub struct ReconcileRequest {
    /// User utterance (required). Same role as Android `query`.
    pub query: String,
    /// Optional assistant reply (context only; not a fact source).
    #[serde(default)]
    pub answer: String,
    /// Recent dialogue for coreference only.
    #[serde(default)]
    pub transcript: String,
    #[serde(default)]
    pub speaker_id: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// Chat message this turn belongs to, for provenance.
    #[serde(default)]
    pub message_id: Option<String>,
    /// Epoch ms for relative-date anchoring; default = now.
    #[serde(default)]
    pub message_at_ms: Option<i64>,
    /// Optional OCR / attachment / user-text evidence. Empty → synthesize
    /// one `user_text` source from `query` (Web default).
    #[serde(default)]
    pub evidence: Vec<EvidenceSource>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExtractionProgress {
    pub chunks_total: usize,
    pub chunks_completed: usize,
    pub failed_source_ids: Vec<String>,
    pub partial: bool,
    /// Bad individual records are isolated, not a reason to discard siblings.
    pub invalid_records: usize,
    /// Responses stopped by the provider's output budget, even if some lines are valid.
    pub truncated_source_ids: Vec<String>,
    pub planning_calls: usize,
    pub extraction_calls: usize,
    pub recovered_chunks: usize,
    pub truncated_attempts: usize,
    pub invalid_attempt_records: usize,
    /// Valid staged memory records, not a count of database writes.
    pub valid_records: usize,
    pub remaining_ranges: Vec<RemainingRange>,
}

/// Half-open Unicode-character offsets in the original source, not bytes.
#[derive(Debug, Clone, Serialize)]
pub struct RemainingRange {
    pub source_id: String,
    pub start: usize,
    pub end: usize,
    pub reason: String,
}

/// Private in-memory preparation state. Hosts must not serialize raw requests
/// by default or replay this after uncertain application failures.
pub struct ReconcileCheckpoint {
    request: ReconcileRequest,
    context: crate::long_extract::ExtractContext,
    sources: Vec<EvidenceSource>,
    staged: Vec<crate::long_extract::StagedSource>,
}

pub enum ReconcileOutcome {
    Complete(ReconcileReport),
    Pending {
        report: ReconcileReport,
        checkpoint: ReconcileCheckpoint,
    },
}

#[derive(Debug, Serialize)]
pub struct ReconcileReport {
    /// True means extraction is staged only: no memory/member/graph writes ran.
    pub staged: bool,
    pub skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
    pub speaker_id: String,
    pub extract_raw: Option<String>,
    pub registers: usize,
    pub memories_extracted: usize,
    pub memories_kept: usize,
    pub retractions: usize,
    pub ops_raw: Option<String>,
    pub written_ids: Vec<String>,
    pub revised_ids: Vec<String>,
    pub forgotten: usize,
    pub fallback_used: bool,
    pub warnings: Vec<String>,
    pub facts: Vec<Value>,
    /// One row per fact that reached storage: memory id, operation and full origin.
    pub provenance: Vec<Value>,
    /// Ids of original excerpts written or refreshed by this turn.
    pub chunk_ids: Vec<String>,
    /// Source coverage is explicit; partial extraction must not look complete.
    pub extraction: ExtractionProgress,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Host validation checks provenance and JSON shape, never speech meaning.
fn source_id_ok(sid: &str, source_id: &str) -> bool {
    !sid.is_empty() && sid == source_id
}

fn evidence_matches(item: &Value, source_id: &str, source_text: &str) -> bool {
    let Some(evidence) = item.get("evidence").and_then(Value::as_str) else {
        return false;
    };
    let Some(sid) = item.get("source_id").and_then(Value::as_str) else {
        return false;
    };
    !evidence.trim().is_empty() && source_id_ok(sid, source_id) && source_text.contains(evidence)
}

#[cfg(test)]
fn valid_extraction_item(key: &str, item: &Value) -> bool {
    let Some(object) = item.as_object() else {
        return false;
    };
    let required: &[&str] = match key {
        "register" => &["display_name", "role", "aliases", "source_id", "evidence"],
        "memories" => &[
            "owner",
            "content",
            "subjects",
            "tag",
            "event_at",
            "source_id",
            "evidence",
        ],
        "retractions" => &["owner", "content", "subjects", "source_id", "evidence"],
        _ => return false,
    };
    if object.len() != required.len() || !required.iter().all(|key| object.contains_key(*key)) {
        return false;
    }
    let string = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    if !string("source_id") || !string("evidence") {
        return false;
    }
    if key == "register" {
        return string("display_name")
            && object
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| {
                    matches!(role, "self" | "spouse" | "child" | "parent" | "other")
                })
            && object
                .get("aliases")
                .and_then(Value::as_array)
                .is_some_and(|aliases| {
                    aliases
                        .iter()
                        .all(|v| v.as_str().is_some_and(|v| !v.trim().is_empty()))
                });
    }
    if !string("owner")
        || !string("content")
        || !object
            .get("subjects")
            .and_then(Value::as_array)
            .is_some_and(|subjects| {
                subjects.len() == 1
                    && subjects[0].as_str() == object.get("owner").and_then(Value::as_str)
            })
    {
        return false;
    }
    key == "retractions"
        || (string("tag")
            && object.get("event_at").is_some_and(|v| {
                v.is_null()
                    || v.as_str()
                        .is_some_and(|date| validated_event_date(Some(date)).is_some())
            }))
}

/// ISO date syntax only. Birthdays, schedules, uncertainty and relative dates
/// are understood by the LLM; the host never classifies them with keywords.
fn validated_event_date(raw: Option<&str>) -> Option<(String, i64)> {
    use chrono::TimeZone;
    let raw = raw?;
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok()?;
    if date.format("%Y-%m-%d").to_string() != raw {
        return None;
    }
    let instant = chrono::FixedOffset::east_opt(8 * 3600)?
        .from_local_datetime(&date.and_hms_opt(0, 0, 0)?)
        .single()?;
    Some((raw.to_string(), instant.timestamp_millis()))
}

fn parse_json_object(raw: &str) -> Option<Value> {
    serde_json::from_str(raw.trim()).ok()
}

fn normalize_memory_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join("")
}

fn memory_text_similar(a: &str, b: &str) -> bool {
    normalize_memory_text(a) == normalize_memory_text(b)
}

/// A stored fact must read on its own: "不喝咖啡" alone says nothing about who, and
/// neither search nor the answering model can recover it from the row's owner column.
fn with_owner_name(content: &str, owner_id: &str, members: &[HouseholdMember]) -> String {
    if crate::household::is_shared_member_id(owner_id) {
        return content.to_string();
    }
    let member = members.iter().find(|m| m.member_id == owner_id);
    let display = member.map_or(owner_id, |m| m.display_name.as_str());
    let named = content.contains(display)
        || content.contains(owner_id)
        || member.is_some_and(|m| m.aliases.iter().any(|a| {
            a.chars().count() >= 2 && content.contains(a.as_str())
        }));
    if named {
        content.to_string()
    } else {
        format!("{display}{content}")
    }
}

fn roster_prompt(members: &[HouseholdMember]) -> String {
    // Host may keep shared placeholder rows for storage; never offer them as
    // extractable people in the LLM roster.
    let members: Vec<_> = members
        .iter()
        .filter(|m| !crate::household::is_shared_member_id(&m.member_id))
        .collect();
    if members.is_empty() {
        return "[]".into();
    }
    serde_json::to_string(
        &members
            .iter()
            .map(|m| {
                json!({
                    "member_id": m.member_id,
                    "display_name": m.display_name,
                    "role": m.role.as_str(),
                    // "我" is decided by speaker_id. Offering it as an alias makes the
                    // model attribute any unknown first person to this row.
                    "aliases": m.aliases.iter()
                        .filter(|a| !crate::household::is_unresolved_owner_id(a))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".into())
}

fn tags_prompt(tags: &[crate::household::MemoryTag]) -> String {
    if tags.is_empty() {
        return "[]".into();
    }
    serde_json::to_string(
        &tags
            .iter()
            .map(|t| {
                json!({
                    "tag_id": t.tag_id,
                    "display_name": t.display_name,
                    "aliases": t.aliases,
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".into())
}

/// Only unambiguous commands in the user's own text bypass model interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UserMemoryIntent {
    Decline,
    ForgetOwn(String),
}

pub(crate) fn user_memory_intent(text: &str) -> Option<UserMemoryIntent> {
    let mut text=text.trim();
    for _ in 0..3 {
        if let Some(rest)=["请", "麻烦", "帮我"].iter().find_map(|prefix|text.strip_prefix(prefix)) {
            text=rest.trim_start();
        } else {break;}
    }
    let no_store=["不要记住这句话", "不要记录这句话", "不要保存这句话", "别记住这句话", "别记这句话", "这句话不要记住", "这段话不要记住"];
    if no_store.iter().any(|prefix|text.starts_with(prefix))
        && !["但请记住", "但要记住", "不过请记住", "只记住", "但请记录"].iter().any(|marker|text.contains(marker)) {
        return Some(UserMemoryIntent::Decline);
    }
    let clean=text.trim_end_matches(|c:char|c.is_whitespace() || matches!(c,'。'|'！'|'!'|'？'|'?'|'.'));
    let mut topic=["忘掉我", "忘记我", "删除我"].iter().find_map(|prefix|clean.strip_prefix(prefix))?.trim();
    if topic.chars().any(|c|matches!(c,'，'|','|'；'|';'|'。'|'\n')) {return None;}
    topic=topic.strip_suffix('吧').unwrap_or(topic).trim();
    for suffix in ["这件事", "这个偏好", "这条记忆", "这一点", "这点"] {
        if let Some(rest)=topic.strip_suffix(suffix) {topic=rest.trim();break;}
    }
    if !["喜欢", "爱", "讨厌", "习惯", "不喜欢", "不爱", "不喝", "不吃"].iter().any(|verb|topic.starts_with(verb))
        || topic.contains("记住") || topic.contains("如果") || topic.chars().count()<3 {
        return None;
    }
    Some(UserMemoryIntent::ForgetOwn(topic.to_owned()))
}

#[cfg(test)]
mod user_intent_regressions {
    use super::*;
    #[test]
    fn explicit_no_store_is_not_reduced_to_a_storeable_subclause() {
        assert_eq!(user_memory_intent("不要记住这句话，我今天有点难过"),Some(UserMemoryIntent::Decline));
        assert_eq!(user_memory_intent("请不要记录这句话，我今天难过"),Some(UserMemoryIntent::Decline));
        assert!(user_memory_intent("“不要记住这句话”是一个例句").is_none());
        assert!(user_memory_intent("不要记住这句话，但请记住明天开会").is_none());
    }
    #[test]
    fn own_forgetting_is_narrow_and_preserves_its_topic() {
        assert_eq!(user_memory_intent("忘掉我喜欢游泳这件事"),Some(UserMemoryIntent::ForgetOwn("喜欢游泳".into())));
        assert_eq!(user_memory_intent("请帮我忘记我喜欢喝茶这个偏好吧"),Some(UserMemoryIntent::ForgetOwn("喜欢喝茶".into())));
        assert!(user_memory_intent("忘掉我老婆喜欢日料这件事").is_none());
        assert!(user_memory_intent("忘掉我喜欢游泳这件事，记住我喜欢跑步").is_none());
        assert!(user_memory_intent("我忘记了明天几点开会").is_none());
    }
    #[test]
    fn forgetting_never_matches_the_opposite_preference() {
        assert!(retraction_matches("王明周末喜欢游泳","王明喜欢游泳"));
        assert!(!retraction_matches("王明不喜欢游泳","王明喜欢游泳"));
        assert!(!retraction_matches("王明喜欢游泳","王明不喜欢游泳"));
    }
}

const RECONCILE_MAX_SOURCE_CHARS: usize = 120_000;
/// Longer text alongside an attachment is treated as real content, not a caption.
const ATTACHMENT_REMARK_CHARS: usize = 40;
const RECONCILE_EXTRACT_CONCURRENCY: usize = 2;
// Newline-aware cuts may use only half a chunk; retain the 120k input capacity.
const RECONCILE_MAX_CHUNKS: usize = 128;
const RECONCILE_OPS_MAX_PROMPT_CHARS: usize = 24_000;
const RECONCILE_OPS_MAX_FACTS: usize = 12;
const RECONCILE_OPS_MAX_HITS: usize = 48;
const RECONCILE_MAX_RETRACTIONS: usize = 32;

/// Poll independent work without batch barriers. Dropping this future aborts all
/// JoinSet tasks, so cancellation does not leave orphan LLM calls running.
async fn bounded_map<T, R, F, Fut>(
    items: Vec<T>,
    concurrency: usize,
    mut work: F,
) -> crate::Result<Vec<R>>
where
    T: Send + 'static,
    R: Send + 'static,
    F: FnMut(T) -> Fut,
    Fut: Future<Output = R> + Send + 'static,
{
    let count = items.len();
    let mut queued: VecDeque<_> = items.into_iter().enumerate().collect();
    let mut tasks = tokio::task::JoinSet::new();
    let mut results = Vec::with_capacity(count);
    loop {
        while tasks.len() < concurrency.max(1) {
            let Some((index, item)) = queued.pop_front() else {
                break;
            };
            let future = work(item);
            tasks.spawn(async move { (index, future.await) });
        }
        let Some(joined) = tasks.join_next().await else {
            break;
        };
        results.push(joined?);
    }
    results.sort_by_key(|(index, _)| *index);
    Ok(results.into_iter().map(|(_, value)| value).collect())
}

/// Validate resource bounds before planning or any business write. Keep original
/// evidence whole; semantic staging owns chunk boundaries and source offsets.
fn evidence_tasks(req: &ReconcileRequest) -> crate::Result<Vec<EvidenceSource>> {
    let sources: Vec<_> = req
        .evidence
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .cloned()
        .collect();
    let sources = if sources.is_empty() {
        vec![EvidenceSource {
            id: "user-text".into(),
            kind: "user_text".into(),
            name: "本轮用户原话".into(),
            text: req.query.clone(),
            locator: None,
            truncated: false,
            graph_already_ingested: false,
            preserve_original: false,
            file_path: None,
        }]
    } else {
        sources
    };
    if req.query.chars().count() > RECONCILE_MAX_SOURCE_CHARS
        || req.answer.chars().count() > RECONCILE_MAX_SOURCE_CHARS
        || req.transcript.chars().count() > RECONCILE_MAX_SOURCE_CHARS
    {
        return Err(crate::MemoraError::InvalidState(
            "reconcile context exceeds 120000 characters".into(),
        ));
    }
    if sources.len() > RECONCILE_MAX_CHUNKS {
        return Err(crate::MemoraError::InvalidState(
            "reconcile source count exceeds resource limit".into(),
        ));
    }
    let mut chars = 0usize;
    let mut ids = HashSet::new();
    for source in &sources {
        if source.id.trim().is_empty() || !ids.insert(source.id.clone()) {
            return Err(crate::MemoraError::InvalidState(
                "reconcile source IDs must be nonempty and unique".into(),
            ));
        }
        chars = chars.saturating_add(source.text.chars().count());
        if chars > RECONCILE_MAX_SOURCE_CHARS {
            return Err(crate::MemoraError::InvalidState(
                "reconcile total evidence exceeds 120000 characters".into(),
            ));
        }
    }
    Ok(sources)
}

/// Compatibility entry point for callers that do not retain continuation state.
/// Pending reports contain no writes. Hosts needing resume use prepare_reconcile.
pub async fn reconcile_turn(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    req: ReconcileRequest,
) -> crate::Result<ReconcileReport> {
    match prepare_reconcile(engine, chat, req).await? {
        ReconcileOutcome::Complete(report) | ReconcileOutcome::Pending { report, .. } => Ok(report),
    }
}

/// Best-effort entry for hosts that cannot resume (Android JNI). A failed block or
/// file must not throw away everything else: complete blocks are applied, and any
/// incomplete source loses its control records so it can never change members or
/// delete memories on partial context. The report stays `partial`, with the ranges.
pub async fn reconcile_turn_incremental(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    req: ReconcileRequest,
) -> crate::Result<ReconcileReport> {
    let checkpoint = match prepare_reconcile(engine, chat, req).await? {
        ReconcileOutcome::Complete(report) => return Ok(report),
        ReconcileOutcome::Pending { checkpoint, .. } => checkpoint,
    };
    let mut progress = staged_progress(&checkpoint);
    let ReconcileCheckpoint { request, context, sources, staged } = checkpoint;
    let mut kept_sources = Vec::new();
    let mut kept_staged = Vec::new();
    for (source, mut part) in sources.into_iter().zip(staged) {
        if !crate::long_extract::valid_staged_source(&source, &part) {
            continue;
        }
        for block in &mut part.parts {
            if source.truncated || !block.parsed.complete {
                block.parsed.registers.clear();
                block.parsed.retractions.clear();
            }
        }
        kept_sources.push(source);
        kept_staged.push(part);
    }
    progress.partial = true;
    let mut report =
        apply_staged(engine, chat, request, context, kept_sources, kept_staged, progress).await?;
    report.warnings.push("partial extraction applied; re-upload to fill remaining ranges".into());
    Ok(report)
}

pub async fn prepare_reconcile(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    mut req: ReconcileRequest,
) -> crate::Result<ReconcileOutcome> {
    let sources = evidence_tasks(&req)?;
    if sources.iter().all(|s| s.text.trim().is_empty()) {
        return Err(crate::MemoraError::InvalidState(
            "query is required (or non-empty evidence)".into(),
        ));
    }
    if sources.iter().all(|s| s.kind == "user_text")
        && sources.iter().all(|s| crate::question_turn::is_pure_question(&s.text))
    {
        return Ok(ReconcileOutcome::Complete(ReconcileReport {
            staged: false,
            skipped: true,
            skip_reason: Some("question_only".into()),
            speaker_id: req.speaker_id.clone().unwrap_or_default(),
            extract_raw: None,
            registers: 0,
            memories_extracted: 0,
            memories_kept: 0,
            retractions: 0,
            ops_raw: None,
            written_ids: vec![],
            revised_ids: vec![],
            forgotten: 0,
            fallback_used: false,
            warnings: vec![],
            facts: vec![],
            provenance: vec![],
            chunk_ids: vec![],
            extraction: ExtractionProgress::default(),
        }));
    }
    let members = engine.list_members(req.namespace.as_deref()).await?;
    if let Some(report) = crate::simple_preferences::remember(engine, &req, &members, &sources).await? {
        return Ok(ReconcileOutcome::Complete(report));
    }
    let tags = engine.list_tags(req.namespace.as_deref()).await?;
    // Freeze time and explicit speaker across continuations; never infer the
    // speaker from the UI default or a chunk's topic summary.
    req.message_at_ms = Some(req.message_at_ms.unwrap_or_else(now_ms));
    let context = crate::long_extract::ExtractContext {
        roster: roster_prompt(&members),
        tags: tags_prompt(&tags),
        speaker: req
            .speaker_id
            .as_deref()
            .map(str::trim)
            .filter(|id| members.iter().any(|m| m.member_id == *id))
            .unwrap_or("")
            .to_string(),
        transcript: if req.transcript.trim().is_empty() {
            format!("用户：{}\n助手：{}", req.query, req.answer)
        } else {
            req.transcript.clone()
        },
        message_at_ms: req.message_at_ms.unwrap(),
    };
    let staged = bounded_map(sources.clone(), RECONCILE_EXTRACT_CONCURRENCY, {
        let context = context.clone();
        let chat = chat.clone();
        move |source| crate::long_extract::stage_source(chat.clone(), source, context.clone())
    })
    .await?;
    finish_prepared(
        engine,
        chat,
        ReconcileCheckpoint {
            request: req,
            context,
            sources,
            staged,
        },
    )
    .await
}

/// Checkpoints are moved, never cloned by this API. An uncertain apply error
/// consumes the checkpoint: callers must not replay it automatically.
pub async fn resume_reconcile(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    checkpoint: ReconcileCheckpoint,
) -> crate::Result<ReconcileOutcome> {
    if checkpoint.sources.is_empty() || checkpoint.sources.len() != checkpoint.staged.len() {
        return Err(crate::MemoraError::InvalidState(
            "reconcile checkpoint source count mismatch".into(),
        ));
    }
    let ReconcileCheckpoint {
        request,
        context,
        sources,
        staged,
    } = checkpoint;
    let work = sources.clone().into_iter().zip(staged).collect();
    let staged = bounded_map(work, RECONCILE_EXTRACT_CONCURRENCY, {
        let context = context.clone();
        let chat = chat.clone();
        move |(source, previous)| {
            crate::long_extract::resume_source(chat.clone(), source, context.clone(), previous)
        }
    })
    .await?;
    finish_prepared(
        engine,
        chat,
        ReconcileCheckpoint {
            request,
            context,
            sources,
            staged,
        },
    )
    .await
}

fn mark_invalid_coverage(progress: &mut ExtractionProgress, source: &EvidenceSource) {
    progress.partial = true;
    progress.chunks_total += 1;
    progress.failed_source_ids.push(source.id.clone());
    progress.remaining_ranges.push(RemainingRange {
        source_id: source.id.clone(),
        start: 0,
        end: source.text.chars().count(),
        reason: "coverage_invalid".into(),
    });
}

fn staged_progress(checkpoint: &ReconcileCheckpoint) -> ExtractionProgress {
    let mut progress = ExtractionProgress::default();
    let mut valid = HashSet::new();
    for (index, source) in checkpoint.sources.iter().enumerate() {
        let Some(staged) = checkpoint.staged.get(index) else {
            mark_invalid_coverage(&mut progress, source);
            continue;
        };
        progress.planning_calls += staged.planning_calls;
        progress.extraction_calls += staged.extraction_calls;
        progress.recovered_chunks += staged.recovered_chunks;
        progress.truncated_attempts += staged.truncated_attempts;
        progress.invalid_attempt_records += staged.invalid_attempt_records;
        if !crate::long_extract::valid_staged_source(source, staged) {
            mark_invalid_coverage(&mut progress, source);
            continue;
        }
        for part in &staged.parts {
            progress.chunks_total += 1;
            progress.invalid_records += part.parsed.invalid_records;
            let complete = part.parsed.complete && !source.truncated;
            if complete {
                progress.chunks_completed += 1;
            } else {
                progress.partial = true;
                progress.failed_source_ids.push(source.id.clone());
                progress.remaining_ranges.push(RemainingRange {
                    source_id: source.id.clone(),
                    start: part.start,
                    end: part.end,
                    // Provider errors can contain echoed input or credentials;
                    // reports (including persisted job results) carry codes only.
                    reason: if source.truncated {
                        "source_incomplete"
                    } else {
                        "extraction_incomplete"
                    }
                    .into(),
                });
                if staged.truncated_attempts > 0 {
                    progress.truncated_source_ids.push(source.id.clone());
                }
            }
            for item in &part.parsed.memories {
                valid.insert(
                    json!([
                        source.id,
                        extracted_item_key(
                            "memories",
                            item["content"].as_str().unwrap_or(""),
                            item
                        )
                    ])
                    .to_string(),
                );
            }
        }
    }
    if checkpoint.staged.len() != checkpoint.sources.len() || checkpoint.sources.is_empty() {
        progress.partial = true;
    }
    progress.failed_source_ids.sort();
    progress.failed_source_ids.dedup();
    progress.truncated_source_ids.sort();
    progress.truncated_source_ids.dedup();
    progress.valid_records = valid.len();
    progress
}

async fn finish_prepared(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    checkpoint: ReconcileCheckpoint,
) -> crate::Result<ReconcileOutcome> {
    let progress = staged_progress(&checkpoint);
    if progress.partial {
        let report = ReconcileReport {
            staged: true, skipped: false, skip_reason: None,
            speaker_id: checkpoint.context.speaker.clone(), extract_raw: None,
            registers: 0, memories_extracted: progress.valid_records, memories_kept: 0,
            retractions: 0, ops_raw: None, written_ids: vec![], revised_ids: vec![], forgotten: 0,
            fallback_used: false, facts: vec![], provenance: vec![], chunk_ids: vec![],
            warnings: vec!["valid extraction records staged only; no memory/member/graph operations applied; resume unresolved ranges".into()],
            extraction: progress,
        };
        return Ok(ReconcileOutcome::Pending { report, checkpoint });
    }
    let ReconcileCheckpoint {
        request,
        context,
        sources,
        staged,
    } = checkpoint;
    let report = apply_staged(engine, chat, request, context, sources, staged, progress).await?;
    Ok(ReconcileOutcome::Complete(report))
}

async fn apply_staged(
    engine: &MemoryEngine,
    chat: &QianfanChat,
    req: ReconcileRequest,
    context: crate::long_extract::ExtractContext,
    sources: Vec<EvidenceSource>,
    staged: Vec<crate::long_extract::StagedSource>,
    extraction: ExtractionProgress,
) -> crate::Result<ReconcileReport> {
    let mut performance = crate::performance::StageTimer::new("reconcile_apply");
    let ns = req.namespace.as_deref();
    let explicit_speaker = context.speaker.as_str();
    let speaker_for_me = context.speaker.clone();
    let message_at_ms = context.message_at_ms;
    // This is intentionally AFTER every source has passed extraction coverage.
    ensure_public_member(engine, ns).await?;
    let rule_members = engine.list_members(ns).await?;
    let bundle = prompt_latest();
    let mut registers = Vec::new();
    let mut memories = Vec::new();
    let mut retractions = Vec::new();
    let mut seen = HashSet::new();
    let mut extract_raw_last: Option<String> = None;
    // Count validated records after exact source-local deduplication; overlapping
    // hard windows must not inflate the report with the same wire record.
    let mut memories_extracted = extraction.valid_records;
    let mut warnings = Vec::new();
    let mut part_text: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let turn_has_attachment = sources.iter().any(|source| source.kind != "user_text");
    for (root_source, staged_source) in sources.into_iter().zip(staged) {
        warnings.extend(staged_source.warnings);
        let source_intent=if root_source.kind=="user_text" {user_memory_intent(&root_source.text)}else{None};
        for staged_part in staged_source.parts {
            let part = staged_part.source;
            let mut parsed = staged_part.parsed;
            if part.kind=="user_text" && source_intent.is_none() {
                let rule_facts=crate::memory_rules::analyze(&part.text,&rule_members,Some(explicit_speaker));
                for f in rule_facts.facts {
                    if crate::memory_rules::rejection(&root_source.text,&f.evidence).is_some() {continue;}
                    if parsed.memories.iter().filter(|item|crate::memory_rules::review(&root_source.text,item["evidence"].as_str().unwrap_or(""),item["content"].as_str().unwrap_or("")).is_none()).any(|item| item["evidence"].as_str().is_some_and(|e|e.chars().count()>=4 && (f.evidence.contains(e)||e.contains(f.evidence.as_str())))) {continue;}
                    parsed.memories.push(json!({"owner":f.owner,"subjects":[f.owner],"content":f.content,"tag":f.tag,"evidence":f.evidence,"source_id":part.id,"event_at":null}));
                    memories_extracted+=1;
                    warnings.push("extract supplemented by grounded fact rules".into());
                }
            }
            extract_raw_last = staged_part.raw.or(extract_raw_last);
            if part.preserve_original && !matches!(&source_intent,Some(UserMemoryIntent::Decline)) {
                part_text.insert(part.id.clone(), part.text.clone());
            }
            let collect = |key: &str,
                           items: &[Value],
                           out: &mut Vec<Value>,
                           warnings: &mut Vec<String>,
                           seen: &mut HashSet<String>| {
                for item in items {
                    if matches!(&source_intent,Some(UserMemoryIntent::Decline))
                        || (matches!(&source_intent,Some(UserMemoryIntent::ForgetOwn(_))) && key!="retractions") {
                        continue;
                    }
                    if !evidence_matches(item, &part.id, &part.text) {
                        warnings.push(format!("extract drop invalid provenance: {key}"));
                        continue;
                    }
                    // Deletion always needs the user's own words. People may be introduced
                    // by an uploaded document, which is how a cold start learns who is who.
                    if key=="register" && part.kind=="user_text" && (crate::memory_rules::rejection(&root_source.text,item["evidence"].as_str().unwrap_or("")).is_some() || !crate::memory_rules::explicit_person_introduction(item["evidence"].as_str().unwrap_or(""),item["display_name"].as_str().unwrap_or(""))) {
                        warnings.push("extract drop nonfactual person registration".into());continue;
                    }
                    if key == "retractions" && part.kind != "user_text" {
                        continue;
                    }
                    if key == "register" && !matches!(part.kind.as_str(), "user_text" | "document") {
                        continue;
                    }
                    // "这是孩子买的课" only introduces the file. Stored as a fact it says
                    // nothing, and it outranks the real rows because it names the same words.
                    if matches!(key, "memories" | "register")
                        && turn_has_attachment
                        && part.kind == "user_text"
                        && part.text.chars().count() <= ATTACHMENT_REMARK_CHARS
                    {
                        continue;
                    }
                    if key == "memories" && part.kind == "user_text" {
                        let evidence = item["evidence"].as_str().unwrap_or("");
                        let content = item["content"].as_str().unwrap_or("");
                        let first_person=regex::Regex::new(r"^(?:我|我的)").unwrap().is_match(evidence.trim()) || part.text.trim().starts_with('我');
                        if first_person && !rule_members.iter().any(|m|m.member_id==explicit_speaker && !crate::household::is_shared_member_id(&m.member_id)) {
                            warnings.push("extract drop first person without verified speaker".into());continue;
                        }
                        if crate::memory_rules::review(&root_source.text,evidence,content).is_some()
                            || crate::question_turn::evidence_in_question(&part.text, evidence)
                            || crate::question_turn::describes_the_asking(content)
                        {
                            warnings.push("extract drop fact taken from a question".into());
                            continue;
                        }
                    }
                    let body = item
                        .get(if key == "register" {
                            "display_name"
                        } else {
                            "content"
                        })
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if seen.insert(
                        json!([root_source.id, extracted_item_key(key, body, item)]).to_string(),
                    ) {
                        let mut accepted = item.clone();
                        if key=="memories" && part.kind=="user_text" {
                            let evidence=item["evidence"].as_str().unwrap_or("");
                            let analysis=crate::memory_rules::analyze(&part.text,&rule_members,Some(explicit_speaker));
                            let matching:Vec<_>=analysis.facts.iter().filter(|f|evidence.chars().count()>=2 && !["喜欢","我的","最近","每天","每月"].contains(&evidence) && (f.evidence.contains(evidence)||evidence.contains(f.evidence.as_str()))).collect();
                            if matching.len()==1 {
                                let f=matching[0];
                                accepted["owner"]=json!(f.owner);accepted["subjects"]=json!([f.owner]);
                                accepted["content"]=json!(f.content);accepted["tag"]=json!(f.tag);
                                accepted["evidence"]=json!(f.evidence);accepted["event_at"]=Value::Null;accepted["_host_rule_validated"]=json!(true);
                                warnings.push("extract verified by grounded fact rules".into());
                            }
                        }

                        if key=="memories" && part.kind=="user_text" {
                            let evidence=accepted["evidence"].as_str().unwrap_or("");
                            let clear_self=evidence.trim().starts_with('我') && !regex::Regex::new(r"^我(?:的)?(?:妈妈|爸爸|老婆|老公|孩子|儿子|女儿|哥哥|姐姐|朋友|同事|老师|爷爷|奶奶|国|司|校|给|帮|替|跟|和|带)").unwrap().is_match(evidence.trim());
                            if clear_self {
                                if let Some(speaker)=rule_members.iter().find(|m|m.member_id==explicit_speaker) {
                                    let owner=accepted["owner"].as_str().unwrap_or("");
                                    if ["我","自己","本人",explicit_speaker].contains(&owner) {
                                        accepted["owner"]=json!(speaker.member_id);accepted["subjects"]=json!([speaker.member_id]);
                                        if let Some(rest)=accepted["content"].as_str().unwrap_or("").strip_prefix('我') {accepted["content"]=json!(format!("{}{rest}",speaker.display_name));}
                                    }
                                }
                            }
                            if !engine.rule_content_supported(accepted["evidence"].as_str().unwrap_or(""),accepted["content"].as_str().unwrap_or(""),&rule_members) {
                                warnings.push("extract drop unsupported noun or named entity".into());continue;
                            }
                            if let Some(tag)=crate::memory_rules::category_hint(accepted["evidence"].as_str().unwrap_or("")).filter(|_|accepted["_host_rule_validated"]!=true) {
                                if accepted["tag"].as_str()!=Some(tag) {accepted["tag"]=json!(tag);warnings.push("extract category verified by grounded fact rules".into());}
                            }
                            let owner=accepted["owner"].as_str().unwrap_or("");
                            let known=rule_members.iter().any(|m|m.member_id==owner || m.display_name==owner || m.aliases.iter().any(|a|a==owner));
                            if !known && !crate::household::is_shared_member_id(owner) && !crate::memory_rules::explicit_person_introduction(accepted["evidence"].as_str().unwrap_or(""),owner) {
                                warnings.push("extract drop new person without explicit introduction".into());continue;
                            }
                        }
                        let label=item.get(if key=="register" {"display_name"}else{"owner"}).and_then(Value::as_str).unwrap_or("");
                        accepted["_host_non_person_label"]=json!(root_source.kind!="user_text" && non_person_source_label(label,&root_source.text));
                        accepted["_host_graph_already_ingested"] =
                            json!(part.graph_already_ingested);
                        accepted["_host_source_id"] = json!(part.id);
                        accepted["_host_source_kind"] = json!(part.kind);
                        accepted["_host_root_source_id"] = json!(root_source.id);
                        accepted["_host_source_name"] = json!(part.name);
                        accepted["_host_source_locator"] = json!(part.locator);
                        accepted["_host_file_path"] = json!(part.file_path);
                        accepted["_host_preserve_original"] = json!(part.preserve_original);
                        // Full model summaries and raw citation units stay only
                        // in the ephemeral checkpoint: persisting them could
                        // leak unrelated subjects under this fact's ACL.
                        accepted["_host_context"] = context_references(&staged_part.context);
                        accepted["_host_span"] = json!({"start":staged_part.start,"end":staged_part.end,"unit":"unicode_character"});
                        out.push(accepted);
                    }
                }
            };
            collect(
                "register",
                &parsed.registers,
                &mut registers,
                &mut warnings,
                &mut seen,
            );
            collect(
                "memories",
                &parsed.memories,
                &mut memories,
                &mut warnings,
                &mut seen,
            );
            collect(
                "retractions",
                &parsed.retractions,
                &mut retractions,
                &mut warnings,
                &mut seen,
            );
        }
    }

    performance.mark("extract_validate");
    // Register household members from extract.
    if retractions.len() > RECONCILE_MAX_RETRACTIONS {
        warnings.push(format!(
            "retraction budget reached; {} excess items ignored",
            retractions.len() - RECONCILE_MAX_RETRACTIONS
        ));
        retractions.truncate(RECONCILE_MAX_RETRACTIONS);
    }
    for item in &registers {
        let name = item
            .get("display_name")
            .or_else(|| item.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if name.is_empty() || crate::household::is_shared_member_id(name)
            || crate::household::is_unresolved_owner_id(name) {
            continue;
        }
        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("other");
        let role = MemberRole::parse(role).unwrap_or(MemberRole::Other);
        let aliases = item
            .get("aliases")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // A new alias must land on the person already known, never replace their aliases.
        let known = engine.list_members(ns).await?;
        if item["_host_non_person_label"]==true && resolve_member_id(name,&known).is_none() {
            warnings.push("extract drop non-person source field used as a new member".into());continue;
        }
        let existing = known.iter().find(|m| {
            m.member_id == name || m.display_name == name || m.aliases.iter().any(|a| a == name)
        });
        let mut merged: Vec<String> = existing.map(|m| m.aliases.clone()).unwrap_or_default();
        for alias in aliases {
            let taken_by_other = known.iter().any(|m| {
                existing.map_or(true, |e| e.member_id != m.member_id)
                    && (m.member_id == alias || m.display_name == alias || m.aliases.iter().any(|a| a == &alias))
            });
            if !taken_by_other
                && !crate::household::is_unresolved_owner_id(&alias)
                && !merged.contains(&alias)
                && alias != name
            {
                merged.push(alias);
            }
        }
        let _ = engine
            .upsert_member(
                ns,
                HouseholdMember {
                    member_id: existing.map_or_else(|| name.to_string(), |m| m.member_id.clone()),
                    display_name: existing.map_or_else(|| name.to_string(), |m| m.display_name.clone()),
                    role: existing.map_or(role, |m| m.role),
                    aliases: merged,
                    age_band: None,
                },
            )
            .await;
    }
    let mut members = engine.list_members(ns).await?;
    let tags = engine.list_tags(ns).await?;

    let mut facts: Vec<Fact> = Vec::new();
    let mut fact_seen = HashSet::new();
    for item in &memories {
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let owner_id = match validated_owner(item, &members) {
            Some(owner) => owner,
            None => {
                warnings.push("extract drop ambiguous or ungrounded owner".into());
                continue;
            }
        };
        // The default driver row answers to "我" for UI enrollment only. Without a
        // verified speaker, "我" is nobody: that is what keeps an unknown voice's first
        // person out of someone else's memories.
        if owner_id != explicit_speaker
            && members.iter().any(|m| {
                m.member_id == owner_id
                    && m.aliases.iter().any(|a| crate::household::is_unresolved_owner_id(a))
            })
            && !item.get("evidence").and_then(Value::as_str).unwrap_or("").contains(owner_id.as_str())
        {
            warnings.push("extract drop first person without a verified speaker".into());
            continue;
        }
        // Content-derived owners may not be on the roster yet; roster is for
        // dedupe/merge only. Upsert so later facts and retractions can resolve.
        if !members.iter().any(|m| m.member_id == owner_id) {
            let _ = engine
                .upsert_member(
                    ns,
                    HouseholdMember {
                        member_id: owner_id.clone(),
                        display_name: owner_id.clone(),
                        role: MemberRole::Other,
                        aliases: vec![],
                        age_band: None,
                    },
                )
                .await;
            members = engine.list_members(ns).await?;
        }
        let evidence = item
            .get("evidence")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let tag = resolve_tag(item.get("tag").and_then(Value::as_str).unwrap_or(""), &tags, content);
        let event = validated_event_date(item.get("event_at").and_then(Value::as_str));
        let llm_event_at = event.as_ref().map(|(date, _)| date.clone());
        let event_at_ms = event.map(|(_, ms)| ms);
        let content = with_owner_name(content, &owner_id, &members);
        let content = preserve_birth_precision(&content, &evidence);
        let fact = Fact {
            owner_id,
            content,
            tag,
            llm_event_at,
            event_at_ms,
            importance: None,
            evidence,
            graph_already_ingested: item
                .get("_host_graph_already_ingested")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            context_provenance: item["_host_context"].clone(),
            source_span: item["_host_span"].clone(),
            root_source_id: item["_host_root_source_id"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            source_id: item["_host_source_id"].as_str().unwrap_or("").to_string(),
            source_kind: item["_host_source_kind"].as_str().unwrap_or("").to_string(),
            speaker_member_id: (!explicit_speaker.is_empty()).then(|| explicit_speaker.to_string()),
            message_at_ms,
            origin: FactOrigin {
                source_name: item["_host_source_name"].as_str().unwrap_or("").to_string(),
                source_locator: item["_host_source_locator"].as_str().map(str::to_string),
                file_path: item["_host_file_path"].as_str().map(str::to_string),
                session_id: req.session_id.clone(),
                message_id: req.message_id.clone(),
                utterance: (item["_host_source_kind"].as_str() == Some("user_text"))
                    .then(|| req.query.chars().take(2_000).collect()),
                preserve_original: item["_host_preserve_original"].as_bool().unwrap_or(false),
            },
        };
        if fact_seen.insert(fact_key(&fact)) {
            facts.push(fact);
        }
    }

    let mut forgotten = 0usize;
    for item in &retractions {
        let topic = item
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if topic.is_empty() {
            continue;
        }
        let Some(owner) = validated_owner(item, &members) else {
            warnings.push("ignored retraction without a unique roster owner".into());
            continue;
        };
        let hits = engine
            .recall(RecallQuery {
                text: topic.chars().take(240).collect(),
                namespace: req.namespace.clone(),
                top_k: Some(12),
                fts_only: true,
                skip_graph: true,
                member_id: Some(owner.clone()),
                ..Default::default()
            })
            .await?;
        let candidates: Vec<_> = hits
            .hits
            .into_iter()
            .filter(|h| retraction_matches(&h.memory.content, topic))
            .collect();
        let targets: Vec<_> = candidates
            .iter()
            .filter(|hit| hit.memory.subject_ids == [owner.clone()])
            .collect();
        // An omitted/ambiguous owner must never erase several people's copies.
        let buckets: HashSet<_> = targets
            .iter()
            .map(|h| {
                let mut owners = h.memory.subject_ids.clone();
                owners.sort();
                owners
            })
            .collect();
        if buckets.len() != 1 {
            continue;
        }
        for hit in targets {
            engine
                .forget(
                    hit.memory.id,
                    ForgetReason::User {
                        reason: "retraction".into(),
                    },
                )
                .await?;
            forgotten += 1;
        }
    }
    let mut written_ids = Vec::new();
    let mut revised_ids = Vec::new();
    let fallback_used = warnings.iter().any(|w|w.contains("grounded fact rules"));
    let mut ops_raw = None;
    performance.mark("validate_and_retract");

    if facts.is_empty() {
        return Ok(ReconcileReport {
            staged: false,
            skipped: false,
            skip_reason: None,
            speaker_id: speaker_for_me.clone(),
            extract_raw: extract_raw_last,
            registers: registers.len(),
            memories_extracted,
            memories_kept: 0,
            retractions: retractions.len(),
            ops_raw: None,
            written_ids,
            revised_ids,
            forgotten,
            fallback_used,
            warnings,
            facts: vec![],
            provenance: vec![],
            chunk_ids: vec![],
            extraction: extraction.clone(),
        });
    }

    // Batch query vectors and candidate reads. This internal path preserves the
    // same member/tag/lifecycle filters without building user-facing context.
    let queries = facts
        .iter()
        .map(|fact| RecallQuery {
            text: fact.content.chars().take(240).collect(),
            namespace: req.namespace.clone(),
            top_k: Some(4),
            member_id: Some(fact.owner_id.clone()),
            tag: if fact.tag.is_empty() || fact.tag == "其他" {
                None
            } else {
                Some(fact.tag.clone())
            },
            skip_graph: true,
            ..Default::default()
        })
        .collect();
    let mut hit_by_id = std::collections::BTreeMap::new();
    match engine.recall_candidates_batch(queries).await {
        Ok(results) => {
            for records in results {
                for record in records {
                    hit_by_id.entry(record.id.to_string()).or_insert(record);
                }
            }
        }
        Err(error) => warnings.push(format!("candidate recall failed: {error}")),
    }
    performance.mark("candidate_recall");
    let hits: Vec<_> = hit_by_id.values().cloned().collect();
    // Only facts with actual old candidates need model-assisted comparison.
    // Batch small, lossless JSON objects: never truncate fact text in ops prompts.
    let mut ops_prompts = Vec::new();
    let conversational: Vec<(usize, &Fact)> = facts
        .iter()
        .enumerate()
        .filter(|(_, fact)| fact.source_kind == "user_text")
        .collect();
    for batch in conversational.chunks(if chat.edge_profile() { 1 } else { RECONCILE_OPS_MAX_FACTS }) {
        let candidates: Vec<_> = hits
            .iter()
            .filter(|hit| batch.iter().any(|(_, fact)| same_correction_bucket(hit, fact)))
            .take(if chat.edge_profile() { 2 } else { RECONCILE_OPS_MAX_HITS })
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let facts_json: Value = batch.iter().map(|(index, f)| json!({"fact_index": index, "owner": f.owner_id, "content": f.content, "tag": f.tag, "event_at": f.llm_event_at, "subjects": [f.owner_id.clone()]})).collect();
        let hits_json: Value = candidates.iter().map(|h| json!({"id": h.id, "content": h.content, "tag": h.tag, "subjects": h.subject_ids})).collect();
        let prompt = format!(
            "{}\n新事实：{}\n旧记忆（按成员和标签过滤的候选）：{}\n",
            bundle.ops_rules, facts_json, hits_json
        );
        if prompt.chars().count() + bundle.ops_system.chars().count()
            > if chat.edge_profile() { 2400 } else { RECONCILE_OPS_MAX_PROMPT_CHARS }
        {
            warnings.push(
                "ops batch exceeds context budget; validated facts use deterministic write path"
                    .into(),
            );
            continue;
        }
        let budget = (900 + batch.len() * 120 + candidates.len() * 40).clamp(900, 4000) as u32;
        ops_prompts.push((prompt, budget));
    }
    performance.mark("ops_prepare");
    let ops_results = bounded_map(ops_prompts, 2, |(prompt, budget)| {
        let chat = chat.clone();
        async move {
            chat.complete_with_timeout(
                Some(bundle.ops_system),
                &prompt,
                Some(budget),
                Some(0.2),
                Duration::from_secs(20),
            )
            .await
        }
    })
    .await?;
    performance.mark("ops_model");
    let mut combined_ops = Vec::new();
    for result in ops_results {
        match result {
            Ok(raw) => {
                if let Some(root) = parse_json_object(&raw) {
                    if let Some(items) = root.get("ops").and_then(Value::as_array) {
                        combined_ops.extend(items.iter().cloned());
                    } else {
                        warnings
                            .push("ops response lacks ops array; deterministic fallback".into());
                    }
                } else {
                    warnings.push("ops response invalid JSON; deterministic fallback".into());
                }
            }
            Err(error) => warnings.push(format!("ops request failed: {error}")),
        }
    }
    let ops_text = json!({"ops": combined_ops}).to_string();
    if !combined_ops.is_empty() {
        ops_raw = Some(ops_text.clone());
    }

    performance.mark("ops_parse");
    // Reconciliation decisions below may only reference validated facts.
    let mut written_keys = HashSet::new();
    // Every fact that reached storage, so provenance and excerpts can reference it.
    let mut stored: std::collections::HashMap<String, (String, &'static str)> =
        std::collections::HashMap::new();
    if let Some(ops_root) = parse_json_object(&ops_text) {
        if let Some(Value::Array(ops)) = ops_root.get("ops") {
            for item in ops {
                let op = item
                    .get("op")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase();
                // Only evidence-checked, targeted retractions authorize deletion.
                if matches!(op.as_str(), "delete" | "forget") {
                    warnings
                        .push("ignored ops deletion: explicit targeted retraction required".into());
                    continue;
                }
                let Some(fact) = matched_op_fact(item, &facts, &members) else {
                    warnings.push(format!("ignored unmatched/ambiguous ops {op}"));
                    continue;
                };
                let key = fact_key(fact);
                if written_keys.contains(&key) {
                    continue;
                }
                let id = item.get("id").and_then(Value::as_str).unwrap_or("").trim();
                let target = hit_by_id.get(id).filter(|h| same_correction_bucket(h, fact));
                match op.as_str() {
                    "noop" | "keep" | "skip" => {
                        // The LLM decides semantic equivalence for an authorized
                        // candidate; host checks identity/bucket, not wording.
                        let same = target.or_else(|| {
                            hits.iter().find(|h| {
                                same_fact_bucket(h, fact)
                                    && memory_text_similar(&h.content, &fact.content)
                            })
                        });
                        if let Some(same) = same {
                            stored.insert(key.clone(), (same.id.to_string(), "noop"));
                            written_keys.insert(key);
                        }
                    }
                    "update" | "rewrite" | "overwrite" => {
                        if let Some(existing) = target {
                            match engine
                                .revise_deferred_graph(existing.id, fact_update(existing, fact)?)
                                .await
                            {
                                Ok(()) => {
                                    revised_ids.push(existing.id.to_string());
                                    stored.insert(key.clone(), (existing.id.to_string(), "revise"));
                                    written_keys.insert(key);
                                }
                                Err(e) => warnings.push(format!("ops revise failed: {e}")),
                            }
                        }
                        // Cross-owner/cross-tag/missing targets fall back to add.
                    }
                    "add" | "remember" => {
                        if let Some(same) = hits.iter().find(|h| {
                            same_fact_bucket(h, fact)
                                && memory_text_similar(&h.content, &fact.content)
                        }) {
                            stored.insert(key.clone(), (same.id.to_string(), "noop"));
                            written_keys.insert(key);
                            continue;
                        }
                        match remember_validated_fact(engine, fact, ns, req.session_id.as_deref())
                            .await
                        {
                            Ok(id) => {
                                stored.insert(key.clone(), (id.clone(), "add"));
                                written_ids.push(id);
                                written_keys.insert(key);
                            }
                            Err(e) => warnings.push(format!("ops remember failed: {e}")),
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Validated facts, never operation bodies, are the only write source.
    for fact in &facts {
        let key = fact_key(fact);
        if written_keys.contains(&key) {
            continue;
        }
        if fact.source_kind != "user_text" {
            let repeated = hits.iter().find(|h| {
                same_correction_bucket(h, fact)
                    && h.metadata.get("source").and_then(|s| s.get("kind")).and_then(Value::as_str)
                        == Some(fact.source_kind.as_str())
                    && h.metadata.get("source").and_then(|s| s.get("evidence")).and_then(Value::as_str)
                        == Some(fact.evidence.as_str())
            });
            if let Some(hit) = repeated {
                stored.insert(key.clone(), (hit.id.to_string(), "noop"));
                written_keys.insert(key);
                continue;
            }
        }
        let same = hits
            .iter()
            .find(|h| same_fact_bucket(h, fact) && memory_text_similar(&h.content, &fact.content));
        if let Some(hit) = same {
            match engine
                .revise_deferred_graph(hit.id, fact_update(hit, fact)?)
                .await
            {
                Ok(()) => {
                    revised_ids.push(hit.id.to_string());
                    stored.insert(key.clone(), (hit.id.to_string(), "revise"));
                    written_keys.insert(key);
                }
                Err(e) => warnings.push(format!("fallback revise failed: {e}")),
            }
        } else {
            match remember_validated_fact(engine, fact, ns, req.session_id.as_deref()).await {
                Ok(id) => {
                    stored.insert(key.clone(), (id.clone(), "add"));
                    written_ids.push(id);
                    written_keys.insert(key);
                }
                Err(e) => warnings.push(format!("fallback remember failed: {e}")),
            }
        }
    }
    performance.mark("write");
    let (chunk_writes, chunk_of_fact) = chunk_writes_for(&facts, &stored, &part_text, ns);
    let chunk_ids = match engine.save_chunks(chunk_writes).await {
        Ok(ids) => ids,
        Err(error) => {
            warnings.push(format!("original excerpts not saved: {error}"));
            Vec::new()
        }
    };
    let provenance: Vec<Value> = facts
        .iter()
        .filter_map(|fact| {
            let key = fact_key(fact);
            let (memory_id, op) = stored.get(&key)?;
            let chunk_id = chunk_of_fact.get(&key).and_then(|i| chunk_ids.get(*i));
            Some(fact_provenance(fact, memory_id, op, chunk_id))
        })
        .collect();
    performance.mark("chunks");
    let facts_out: Vec<Value> = facts
        .iter()
        .map(|f| {
            json!({
                "owner": f.owner_id,
                "content": f.content,
                "tag": f.tag,
                "event_at": f.llm_event_at,
                "event_at_ms": f.event_at_ms,
                "importance": f.importance,
                "evidence": f.evidence,
            })
        })
        .collect();

    Ok(ReconcileReport {
        staged: false,
        skipped: false,
        skip_reason: None,
        speaker_id: speaker_for_me,
        extract_raw: extract_raw_last,
        registers: registers.len(),
        memories_extracted,
        memories_kept: facts.len(),
        retractions: retractions.len(),
        ops_raw,
        written_ids,
        revised_ids,
        forgotten,
        fallback_used,
        warnings,
        facts: facts_out,
        provenance,
        chunk_ids,
        extraction,
    })
}

async fn ensure_public_member(engine: &MemoryEngine, ns: Option<&str>) -> crate::Result<()> {
    let members = engine.list_members(ns).await?;
    if members
        .iter()
        .any(|m| m.member_id == PUBLIC_MEMBER_ID || m.display_name == PUBLIC_MEMBER_ID)
    {
        return Ok(());
    }
    engine
        .upsert_member(
            ns,
            HouseholdMember {
                member_id: PUBLIC_MEMBER_ID.into(),
                display_name: PUBLIC_MEMBER_ID.into(),
                role: MemberRole::Other,
                aliases: vec!["大家".into(), "全家".into(), "家庭".into()],
                age_band: None,
            },
        )
        .await?;
    Ok(())
}
fn validated_owner(item: &Value, members: &[HouseholdMember]) -> Option<String> {
    let owner = item.get("owner")?.as_str()?.trim();
    if owner.is_empty() || crate::household::is_unresolved_owner_id(owner) {
        return None;
    }
    let subjects = item.get("subjects")?.as_array()?;
    if subjects.len() != 1 || subjects[0].as_str()?.trim() != owner {
        return None;
    }
    let owner = crate::household::canonical_owner_id(owner).ok()?;
    if item["_host_non_person_label"]==true && resolve_member_id(&owner,members).is_none() {return None;}
    // An unknown explicit name may be registered; an ambiguous known alias may not.
    if !members.iter().any(|m| m.member_id == owner)
        && members.iter().filter(|m| m.display_name == owner || m.aliases.iter().any(|a| a == &owner)).count() > 1 {
        return None;
    }
    // Prefer a unique roster hit (id / display_name / alias) for merge; otherwise
    // keep the content-derived label — roster is not a whitelist.
    Some(resolve_member_id(&owner, members).unwrap_or(owner))
}

fn matched_op_fact<'a>(
    item: &Value,
    facts: &'a [Fact],
    members: &[HouseholdMember],
) -> Option<&'a Fact> {
    if let Some(index) = item.get("fact_index").and_then(Value::as_u64) {
        return facts.get(usize::try_from(index).ok()?);
    }
    let content = item.get("content").and_then(Value::as_str)?.trim();
    if content.is_empty() {
        return None;
    }
    let mut owners = Vec::new();
    for key in ["owner", "owner_name"] {
        if let Some(value) = item.get(key) {
            let token = value.as_str()?.trim();
            if !token.is_empty() {
                owners.push(resolve_member_id(token, members)?);
            }
        }
    }
    if let Some(value) = item.get("subjects") {
        for value in value.as_array()? {
            let token = value
                .as_str()
                .or_else(|| value.get("member_id").and_then(Value::as_str))?;
            owners.push(resolve_member_id(token, members)?);
        }
    }
    owners.sort();
    owners.dedup();
    if owners.len() > 1 {
        return None;
    }
    let mut candidates = facts.iter().filter(|fact| {
        normalize_memory_text(&fact.content) == normalize_memory_text(content)
            && owners.first().map_or(true, |owner| owner == &fact.owner_id)
            && item
                .get("tag")
                .and_then(Value::as_str)
                .map_or(true, |tag| tag == fact.tag)
            && item
                .get("event_at")
                .map_or(true, |date| date.as_str() == fact.llm_event_at.as_deref())
    });
    let fact = candidates.next()?;
    // No owner + identical content for multiple people must never pick first.
    if candidates.next().is_some() {
        return None;
    }
    Some(fact)
}

/// A correction may move a date ("周三改到周四"), so only person and tag must match.
/// Whether it is the same event is the ops model's call; dedupe stays on same_fact_bucket.
fn same_correction_bucket(memory: &crate::types::MemoryRecord, fact: &Fact) -> bool {
    memory.subject_ids == [fact.owner_id.clone()] && (memory.tag.is_empty() || memory.tag == fact.tag)
}

/// The model cannot quote a stored row it never saw: "忘掉我喜欢打羽毛球" names the fact
/// in the user's words. Match either direction within that person's rows.
fn retraction_matches(stored: &str, topic: &str) -> bool {
    let stored = normalize_memory_text(stored);
    let topic = normalize_memory_text(topic);
    if ["喜欢","习惯","爱"].iter().any(|verb|stored.contains(verb) && topic.contains(verb)
        && stored.contains(&format!("不{verb}"))!=topic.contains(&format!("不{verb}"))) {return false;}
    stored == topic
        || (topic.chars().count() >= 4
            && (is_subsequence(&topic, &stored) || is_subsequence(&stored, &topic)))
}

/// 周明在学吉他 is 周明最近在学吉他 minus a word: same fact, not a substring.
fn is_subsequence(short: &str, long: &str) -> bool {
    let mut chars = long.chars();
    short.chars().all(|wanted| chars.any(|c| c == wanted))
}

fn same_fact_bucket(memory: &crate::types::MemoryRecord, fact: &Fact) -> bool {
    let mut owners = memory.subject_ids.clone();
    if owners.is_empty() {
        if let Some(values) = memory.metadata.get("subjects").and_then(Value::as_array) {
            owners = values
                .iter()
                .filter_map(|s| s.get("member_id").and_then(Value::as_str))
                .map(str::to_string)
                .collect();
        }
    }
    owners.sort();
    owners.dedup();
    owners == [fact.owner_id.clone()]
        && (memory.tag.is_empty() || memory.tag == fact.tag)
        && memory.metadata.get("event_at").and_then(Value::as_str) == fact.llm_event_at.as_deref()
        // An independent source must not silently lose its provenance/graph acknowledgement.
        && memory.metadata.get("source").and_then(|s| s.get("root_source_id").or_else(|| s.get("source_id")))
            .and_then(Value::as_str).map_or(true, |id| id == fact.root_source_id)
}

fn confirmed_metadata(metadata: &Value, owner: &str, tag: &str) -> crate::Result<Value> {
    let mut meta = crate::household::with_ownership(metadata, &[owner.to_string()])?;
    if !tag.is_empty() {
        meta["tag"] = json!(tag);
    }
    Ok(meta)
}

fn fact_update(existing: &crate::types::MemoryRecord, fact: &Fact) -> crate::Result<MemoryUpdate> {
    let mut meta = confirmed_metadata(&existing.metadata, &fact.owner_id, &fact.tag)?;
    // The acknowledgement covers original source ingestion, not a future
    // revision. Reconcile revisions must restore the normal durable projection.
    meta["skip_graph"] = json!(false);
    // Merge-patch revision must explicitly clear stale event dates.
    meta["event_at"] = json!(fact.llm_event_at);
    meta["event_at_ms"] = json!(fact.event_at_ms);
    meta["graph_text"] = json!(fact.evidence);
    meta["source"] = fact_source(fact);
    meta["speaker_member_id"] = json!(fact.speaker_member_id);
    Ok(MemoryUpdate {
        content: Some(fact.content.clone()),
        summary: Some(fact.content.clone()),
        importance: fact.importance,
        metadata: Some(meta),
        reason: Some("reconcile".into()),
        ..Default::default()
    })
}
fn resolve_member_id(token: &str, members: &[HouseholdMember]) -> Option<String> {
    let canonical = crate::household::canonical_owner_id(token).ok()?;
    let token = canonical.as_str();
    if token == PUBLIC_MEMBER_ID { return Some(canonical); }
    if let Some(member) = members.iter().find(|member| member.member_id == token) {
        return Some(member.member_id.clone());
    }
    let matches: Vec<_> = members
        .iter()
        .filter(|member| {
            member.display_name == token || member.aliases.iter().any(|alias| alias == token)
        })
        .collect();
    (matches.len() == 1).then(|| matches[0].member_id.clone())
}

fn resolve_tag(raw: &str, tags: &[crate::household::MemoryTag], content: &str) -> String {
    // Keep exact catalog labels/custom tags before legacy domain conversion.
    let canonical = crate::household::resolve_tag_id(tags, raw);
    if let Some(tag) = tags.iter().find(|tag| tag.tag_id == canonical) {
        return tag.tag_id.clone();
    }
    // A model category outside the catalog is not an implicit new tag.
    let inferred = crate::household::infer_tag_from_text(content, &[]);
    tags.iter().find(|tag| tag.tag_id == inferred)
        .or_else(|| tags.iter().find(|tag| tag.tag_id == "其他"))
        .map(|tag| tag.tag_id.clone()).unwrap_or_default()
}
/// Persist only structural reference locations. Generated summaries and full
/// cited paragraphs remain ephemeral to avoid widening each memory's ACL.
fn context_references(context: &Value) -> Value {
    let citations: Vec<Value> = context
        .get("notes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|note| {
            note.get("citations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|citation| {
            Some(json!({"unit_id":citation.get("unit_id")?.as_u64()?,
            "start":citation.get("start")?.as_u64()?,"end":citation.get("end")?.as_u64()?}))
        })
        .collect();
    json!({"purpose":"disambiguation_reference_only","citations":citations})
}

fn fact_source(fact: &Fact) -> Value {
    json!({"kind": fact.source_kind, "source_id": fact.source_id, "root_source_id": fact.root_source_id,
        "evidence": fact.evidence, "message_at_ms": fact.message_at_ms,
        "original_span": fact.source_span, "chunk_context": fact.context_provenance,
        "name": fact.origin.source_name, "locator": fact.origin.source_locator,
        "file_path": fact.origin.file_path, "session_id": fact.origin.session_id,
        "message_id": fact.origin.message_id, "utterance": fact.origin.utterance})
}

/// Group facts by the block they came from. Only facts that reached storage count,
/// so a block with no stored fact keeps no original text.
fn chunk_writes_for(
    facts: &[Fact],
    stored: &std::collections::HashMap<String, (String, &'static str)>,
    part_text: &std::collections::HashMap<String, String>,
    namespace: Option<&str>,
) -> (Vec<crate::chunks::ChunkWrite>, std::collections::HashMap<String, usize>) {
    let mut writes: Vec<crate::chunks::ChunkWrite> = Vec::new();
    let mut by_part: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut chunk_of_fact = std::collections::HashMap::new();
    for fact in facts {
        if !fact.origin.preserve_original {
            continue;
        }
        let key = fact_key(fact);
        let (Some((memory_id, _)), Some(text)) = (stored.get(&key), part_text.get(&fact.source_id)) else {
            continue;
        };
        // Same defense as fact writes: a secret must not persist through the excerpt.
        if !crate::write_filter::check_write(text, None).ok() {
            continue;
        }
        let index = *by_part.entry(fact.source_id.clone()).or_insert_with(|| {
            writes.push(crate::chunks::ChunkWrite {
                namespace: namespace.unwrap_or_default().to_string(),
                content: text.clone(),
                source_kind: fact.source_kind.clone(),
                source_name: fact.origin.source_name.clone(),
                source_path: fact.origin.file_path.clone(),
                source_locator: fact.origin.source_locator.clone(),
                session_id: fact.origin.session_id.clone(),
                message_id: fact.origin.message_id.clone(),
                start_char: fact.source_span["start"].as_i64().unwrap_or(0),
                end_char: fact.source_span["end"].as_i64().unwrap_or(0),
                ..Default::default()
            });
            writes.len() - 1
        });
        let write = &mut writes[index];
        write.memory_ids.push(memory_id.clone());
        if !write.subject_ids.contains(&fact.owner_id) {
            write.subject_ids.push(fact.owner_id.clone());
        }
        if !fact.tag.is_empty() && !write.tags.contains(&fact.tag) {
            write.tags.push(fact.tag.clone());
        }
        chunk_of_fact.insert(key, index);
    }
    (writes, chunk_of_fact)
}

fn fact_provenance(fact: &Fact, memory_id: &str, op: &str, chunk_id: Option<&String>) -> Value {
    json!({
        "memory_id": memory_id, "op": op, "owner": fact.owner_id, "tag": fact.tag,
        "content": fact.content, "chunk_id": chunk_id, "source": fact_source(fact),
    })
}

async fn remember_validated_fact(
    engine: &MemoryEngine,
    fact: &Fact,
    namespace: Option<&str>,
    session_id: Option<&str>,
) -> crate::Result<String> {
    let mut input = RememberInput::episodic(fact.content.clone());
    input.member_id = Some(fact.owner_id.clone());
    input.namespace = namespace.map(str::to_string);
    input.session_id = session_id.map(str::to_string);
    if let Some(importance) = fact.importance {
        input.importance = importance;
    }
    input.metadata = confirmed_metadata(
        &json!({
            "source": fact_source(fact), "graph_text": fact.evidence,
        "speaker_member_id": fact.speaker_member_id,
            "event_at": fact.llm_event_at, "event_at_ms": fact.event_at_ms,
            "skip_graph": fact.graph_already_ingested,
        }),
        &fact.owner_id,
        &fact.tag,
    )?;
    let id = engine.remember_deferred_graph(input).await?;
    Ok(id.to_string())
}

#[cfg(test)]
async fn remember_fact(
    engine: &MemoryEngine,
    content: &str,
    owner: &str,
    tag: &str,
    llm_event_at: Option<&str>,
    event_at_ms: Option<i64>,
    importance: Option<f64>,
    namespace: Option<&str>,
    session_id: Option<&str>,
    message_at_ms: i64,
    graph_already_ingested: bool,
) -> crate::Result<String> {
    remember_validated_fact(
        engine,
        &Fact {
            owner_id: owner.into(),
            content: content.into(),
            tag: tag.into(),
            llm_event_at: llm_event_at.map(str::to_string),
            event_at_ms,
            importance,
            evidence: content.into(),
            speaker_member_id: None,
            context_provenance: Value::Null,
            source_span: Value::Null,
            root_source_id: "user-text".into(),
            source_id: "user-text".into(),
            source_kind: "user_text".into(),
            message_at_ms,
            graph_already_ingested,
            origin: FactOrigin::default(),
        },
        namespace,
        session_id,
    )
    .await
}

#[cfg(test)]
mod document_regressions {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountExtractor(AtomicUsize);
    #[async_trait]
    impl crate::graph::GraphExtractor for CountExtractor {
        fn extractor_id(&self) -> &str {
            "count"
        }
        async fn extract(
            &self,
            _: crate::graph::GraphExtractInput,
        ) -> crate::Result<crate::graph::GraphExtractOutput> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(crate::graph::GraphExtractOutput::default())
        }
    }

    #[test]
    fn reconcile_rejects_unbounded_input_before_model_calls() {
        let mut req: ReconcileRequest =
            serde_json::from_value(json!({"query": "甲".repeat(28_000)})).unwrap();
        assert!(evidence_tasks(&req).is_ok());
        req.query = "甲".repeat(RECONCILE_MAX_SOURCE_CHARS + 1);
        assert!(evidence_tasks(&req).is_err());
        req.query.clear();
        req.evidence = serde_json::from_value(json!([
            {"id":"a", "text":"甲".repeat(61_000)},
            {"id":"b", "text":"乙".repeat(61_000)}
        ]))
        .unwrap();
        assert!(evidence_tasks(&req).is_err());
    }

    #[test]
    fn planning_receives_whole_original_sources_without_whitespace_loss() {
        let original = format!("  {}\n", "甲😀乙。".repeat(20_000));
        let req: ReconcileRequest = serde_json::from_value(json!({"query":original})).unwrap();
        let sources = evidence_tasks(&req).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].text, req.query);
        let invalid: ReconcileRequest = serde_json::from_value(
            json!({"query":"", "evidence":[{"id":"a","text":"甲"},{"id":"a","text":"乙"}]}),
        )
        .unwrap();
        assert!(evidence_tasks(&invalid).is_err());
    }
    #[tokio::test]
    async fn bounded_scheduler_is_concurrent_ordered_and_drains_every_item() {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let results = bounded_map((0..12).collect(), 4, |index| {
            let active = active.clone();
            let maximum = maximum.clone();
            async move {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(if index == 0 { 30 } else { 2 })).await;
                active.fetch_sub(1, Ordering::SeqCst);
                index
            }
        })
        .await
        .unwrap();
        assert_eq!(results, (0..12).collect::<Vec<_>>());
        assert_eq!(maximum.load(Ordering::SeqCst), 4);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn source_acknowledgement_defaults_false_and_preserves_original() {
        let source: EvidenceSource =
            serde_json::from_value(json!({"id":"attachment","kind":"document","text":"source"}))
                .unwrap();
        assert!(!source.graph_already_ingested);
        let req: ReconcileRequest = serde_json::from_value(json!({"query":"", "evidence":[{"id":"attachment","text":"甲".repeat(28_000),"graph_already_ingested":true}]})).unwrap();
        let sources = evidence_tasks(&req).unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources[0].graph_already_ingested);
        assert_eq!(sources[0].text, req.evidence[0].text);
    }
    #[tokio::test]
    async fn acknowledged_new_memory_skips_graph_but_unacknowledged_memory_projects() {
        let dir = tempfile::tempdir().unwrap();
        let extractor = Arc::new(CountExtractor(AtomicUsize::new(0)));
        let engine = MemoryEngine::builder()
            .db_path(dir.path().join("doc.db"))
            .graph_extractor(extractor.clone())
            .graph_projection_worker(false)
            .build()
            .await
            .unwrap();
        let id = remember_fact(
            &engine,
            "张三喜欢茶",
            "张三",
            "饮食",
            None,
            None,
            None,
            None,
            None,
            now_ms(),
            true,
        )
        .await
        .unwrap();
        let id = id.parse().unwrap();
        assert_eq!(
            engine.get(id).await.unwrap().unwrap().metadata["skip_graph"],
            true
        );
        assert_eq!(extractor.0.load(Ordering::SeqCst), 0);
        engine
            .revise(
                id,
                MemoryUpdate {
                    content: Some("张三喜欢咖啡".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(extractor.0.load(Ordering::SeqCst), 0);
        let row = engine.get(id).await.unwrap().unwrap();
        let correction = Fact {
            owner_id: "张三".into(),
            content: "张三喜欢红茶".into(),
            tag: "饮食".into(),
            llm_event_at: None,
            event_at_ms: None,
            importance: None,
            evidence: "张三喜欢红茶".into(),
            graph_already_ingested: false,
            context_provenance: Value::Null,
            source_span: Value::Null,
            root_source_id: "user-text".into(),
            source_id: "user-text".into(),
            source_kind: "user_text".into(),
            speaker_member_id: None,
            message_at_ms: now_ms(),
            origin: FactOrigin::default(),
        };
        engine
            .revise(id, fact_update(&row, &correction).unwrap())
            .await
            .unwrap();
        // Reconcile revisions also enqueue durable projection instead of running it inline.
        assert_eq!(extractor.0.load(Ordering::SeqCst), 0);
        assert_eq!(engine.get(id).await.unwrap().unwrap().metadata["skip_graph"], false);
        remember_fact(
            &engine,
            "李四喜欢咖啡",
            "李四",
            "饮食",
            None,
            None,
            None,
            None,
            None,
            now_ms(),
            false,
        )
        .await
        .unwrap();
        // Reconcile persists a projection job instead of waiting on another LLM.
        assert_eq!(extractor.0.load(Ordering::SeqCst), 0);
        engine.retry_graph_projections(8).await.unwrap();
        assert_eq!(extractor.0.load(Ordering::SeqCst), 2);
        engine.shutdown();
    }
}

#[cfg(test)]
mod text_regressions {
    use super::*;

    #[test]
    fn owners_are_canonical_without_a_roster_and_ambiguous_aliases_are_rejected() {
        for label in ["家庭", "公共", "全家", "共享", "PUBLIC"] {
            let item = json!({"owner":label,"subjects":[label]});
            assert_eq!(validated_owner(&item, &[]).as_deref(), Some(PUBLIC_MEMBER_ID));
        }
        let item = json!({"owner":"张三","subjects":["张三"]});
        assert_eq!(validated_owner(&item, &[]).as_deref(), Some("张三"));
        let invalid = json!({"owner":"bad|id","subjects":["bad|id"]});
        assert!(validated_owner(&invalid, &[]).is_none());
        for placeholder in ["用户", "我", "本人", "unknown", "USER"] {
            assert!(validated_owner(&json!({"owner":placeholder,"subjects":[placeholder]}), &[]).is_none());
        }
        let mut roster = members();
        roster[0].aliases.push("小张".into());
        roster[1].aliases.push("小张".into());
        let ambiguous = json!({"owner":"小张","subjects":["小张"]});
        assert!(validated_owner(&ambiguous, &roster).is_none());
        let meta = confirmed_metadata(&json!({"privacy":true,"source":{"id":"s"}}), "家庭", "生活习惯").unwrap();
        assert_eq!(meta["scope"], "public");
        assert_eq!(meta["subjects"][0]["member_id"], PUBLIC_MEMBER_ID);
        assert_eq!(meta["privacy"], true);
        assert_eq!(meta["source"]["id"], "s");
    }

    fn members() -> Vec<HouseholdMember> {
        [
            ("主驾", MemberRole::Self_),
            ("老公", MemberRole::Spouse),
            ("老婆", MemberRole::Spouse),
            ("爸爸", MemberRole::Parent),
            ("妈妈", MemberRole::Parent),
            (PUBLIC_MEMBER_ID, MemberRole::Other),
        ]
        .into_iter()
        .map(|(name, role)| HouseholdMember {
            member_id: name.into(),
            display_name: name.into(),
            role,
            aliases: vec![],
            age_band: None,
        })
        .collect()
    }
    fn fact(owner: &str) -> Fact {
        Fact {
            owner_id: owner.into(),
            content: "喜欢吃肉".into(),
            tag: "饮食".into(),
            llm_event_at: None,
            event_at_ms: None,
            importance: None,
            evidence: String::new(),
            graph_already_ingested: false,
            context_provenance: Value::Null,
            source_span: Value::Null,
            root_source_id: "user-text".into(),
            source_id: "user-text".into(),
            source_kind: "user_text".into(),
            speaker_member_id: None,
            message_at_ms: now_ms(),
            origin: FactOrigin::default(),
        }
    }

    #[test]
    fn duplicate_keys_and_ops_are_owner_scoped() {
        let a = json!({"owner":"老公", "tag":"饮食"});
        let b = json!({"owner":"老婆", "tag":"饮食"});
        assert_ne!(
            extracted_item_key("memories", "喜欢吃肉", &a),
            extracted_item_key("memories", "喜欢吃肉", &b)
        );
        let facts = vec![fact("老公"), fact("老婆")];
        assert!(matched_op_fact(&json!({"content":"喜欢吃肉"}), &facts, &members()).is_none());
        assert_eq!(
            matched_op_fact(
                &json!({"content":"喜欢吃肉", "subjects":["老婆"]}),
                &facts,
                &members()
            )
            .unwrap()
            .owner_id,
            "老婆"
        );
        for op in ["add", "update", "delete"] {
            assert!(matched_op_fact(
                &json!({"op":op, "owner":"老公", "content":"喜欢喝酒"}),
                &facts,
                &members()
            )
            .is_none());
        }
        assert!(matched_op_fact(
            &json!({"owner":"主驾", "content":"喜欢吃肉"}),
            &facts,
            &members()
        )
        .is_none());
    }

    #[test]
    fn llm_items_require_exact_provenance_and_roster_identity_not_keywords() {
        let item = json!({"owner":"老公", "subjects":["老公"], "content":"生日是11月6日",
            "tag":"生活习惯", "event_at":null, "source_id":"user-text:0",
            "evidence":"小航下个月6号就要过生日了"});
        assert!(valid_extraction_item("memories", &item));
        assert!(evidence_matches(
            &item,
            "user-text:0",
            "小航下个月6号就要过生日了，请问送什么礼物好呢"
        ));
        assert!(!evidence_matches(
            &item,
            "user-text",
            "小航下个月6号就要过生日了"
        ));
        assert_eq!(validated_owner(&item, &members()).as_deref(), Some("老公"));
        let mut wrong = item.clone();
        wrong["subjects"] = json!(["老婆"]);
        assert!(!valid_extraction_item("memories", &wrong));
        assert!(validated_owner(&wrong, &members()).is_none());
        // Content-derived owner not yet on roster is kept (host registers/merges).
        wrong = item.clone();
        wrong["owner"] = json!("小航");
        wrong["subjects"] = json!(["小航"]);
        assert_eq!(validated_owner(&wrong, &members()).as_deref(), Some("小航"));
        wrong = item.clone();
        wrong["event_at"] = json!("2026-02-30");
        assert!(!valid_extraction_item("memories", &wrong));
        // Host provenance intentionally does not reinterpret questions/quotes.
        assert!(evidence_matches(
            &json!({"source_id":"a", "evidence":"他可能喜欢什么"}),
            "a",
            "他可能喜欢什么？"
        ));
    }

    #[tokio::test]
    async fn remember_fact_locks_owner_and_revision_clears_event_metadata() {
        let db = std::env::temp_dir().join(format!("memora-text-{}.db", uuid::Uuid::now_v7()));
        let engine = MemoryEngine::builder().db_path(db).build().await.unwrap();
        let id = remember_fact(
            &engine,
            "喜欢吃肉",
            "老公",
            "饮食",
            None,
            None,
            None,
            None,
            None,
            now_ms(),
            false,
        )
        .await
        .unwrap();
        let id = id.parse().unwrap();
        let mut row = engine.get(id).await.unwrap().unwrap();
        assert_eq!(row.subject_ids, ["老公"]);
        assert_eq!(row.metadata["attribution"], "manual");
        assert!(same_fact_bucket(&row, &fact("老公")));
        assert!(!same_fact_bucket(&row, &fact("老婆")));
        row.metadata["event_at"] = json!("2020-01-01");
        row.metadata["event_at_ms"] = json!(1);
        row.metadata["skip_graph"] = json!(true);
        let update = fact_update(&row, &fact("老公")).unwrap();
        let metadata = update.metadata.unwrap();
        assert!(metadata["event_at"].is_null());
        assert!(metadata["event_at_ms"].is_null());
        assert_eq!(metadata["skip_graph"], false);
    }
}
#[cfg(test)]
mod staging_pure_regressions {
    use super::*;

    fn fixture_fact(root: &str, date: Option<&str>) -> Fact {
        let event = validated_event_date(date);
        Fact {
            owner_id: "Alice".into(),
            content: "likes tea".into(),
            tag: "diet".into(),
            llm_event_at: event.as_ref().map(|(date, _)| date.clone()),
            event_at_ms: event.map(|(_, timestamp)| timestamp),
            importance: None,
            evidence: "Alice likes tea".into(),
            graph_already_ingested: false,
            context_provenance: json!({"purpose":"disambiguation_reference_only","citations":[]}),
            source_span: json!({"start":0,"end":15,"unit":"unicode_character"}),
            root_source_id: root.into(),
            source_id: format!("{root}@0-15"),
            source_kind: "document".into(),
            speaker_member_id: None,
            message_at_ms: 1_700_000_000_000,
            origin: FactOrigin::default(),
        }
    }

    fn fixture_memory(fact: &Fact) -> crate::types::MemoryRecord {
        serde_json::from_value(json!({
            "id":"00000000-0000-0000-0000-000000000001", "kind":"episodic", "state":"active",
            "content":fact.content, "summary":null, "importance":0.5, "confidence":0.8,
            "salience":0.0, "created_at_ms":0, "updated_at_ms":0, "last_accessed_at_ms":null,
            "next_review_at_ms":null, "review_count":0, "source_hash":null, "embedding_model":null,
            "embedding_version":0, "embedding_pending":false, "parent_id":null,
            "namespace":"test", "session_id":null, "entities":[],
            "metadata":{"event_at":fact.llm_event_at, "source":{"root_source_id":fact.root_source_id}},
            "tag":fact.tag, "subject_ids":[fact.owner_id]
        })).unwrap()
    }

    #[test]
    fn persisted_context_keeps_only_structural_reference_locations() {
        let input = json!({
            "root_source_id":"source", "purpose":"model-context", "secret":"unrelated root data",
            "notes":[{"summary":"unrelated private summary", "refs":[1], "citations":[
                {"unit_id":1,"start":3,"end":8,"text":"private raw paragraph","secret":"private"},
                {"unit_id":2,"start":9,"text":"incomplete citation"},
                {"unit_id":"3","start":0,"end":1,"text":"invalid ID"}
            ]}]
        });
        let output = context_references(&input);
        assert_eq!(
            output,
            json!({"purpose":"disambiguation_reference_only","citations":[{"unit_id":1,"start":3,"end":8}]})
        );
        let mut fact = fixture_fact("source", None);
        fact.context_provenance = output;
        let persisted = fact_source(&fact);
        assert_eq!(persisted["root_source_id"], "source");
        assert_eq!(persisted["source_id"], "source@0-15");
        assert_eq!(persisted["evidence"], "Alice likes tea");
        assert_eq!(persisted["original_span"]["start"], 0);
        let wire = persisted.to_string();
        assert!(!wire.contains("private"));
        assert!(!wire.contains("summary"));
        assert!(!wire.contains("unrelated"));
    }

    #[test]
    fn distinct_dates_and_independent_roots_survive_fact_deduplication() {
        let a = fixture_fact("root-a", Some("2026-10-27"));
        let b = fixture_fact("root-a", Some("2026-10-28"));
        let c = fixture_fact("root-b", Some("2026-10-27"));
        assert_ne!(fact_key(&a), fact_key(&b));
        assert_ne!(fact_key(&a), fact_key(&c));
        let mut duplicate = a.clone();
        duplicate.content = "likes  tea".into();
        duplicate.source_id = "root-a@15-30".into();
        assert_eq!(fact_key(&a), fact_key(&duplicate));
        assert_eq!(
            HashSet::from([
                fact_key(&a),
                fact_key(&b),
                fact_key(&c),
                fact_key(&duplicate)
            ])
            .len(),
            3
        );
    }

    #[test]
    fn overlapping_window_records_deduplicate_without_erasing_dates_or_register_fields() {
        let item = json!({"owner":"Alice","subjects":["Alice"],"content":"likes tea",
            "tag":"diet","event_at":"2026-10-27","source_id":"root@w0:2400-2500",
            "evidence":"Alice likes tea"});
        let mut overlap = item.clone();
        overlap["source_id"] = json!("root@w1:2400-2500");
        assert_eq!(
            extracted_item_key("memories", "likes tea", &item),
            extracted_item_key("memories", "likes tea", &overlap)
        );
        overlap["event_at"] = json!("2026-10-28");
        assert_ne!(
            extracted_item_key("memories", "likes tea", &item),
            extracted_item_key("memories", "likes tea", &overlap)
        );
        let register =
            json!({"display_name":"Alice","role":"other","aliases":["A"],"evidence":"Alice"});
        let mut changed = register.clone();
        changed["aliases"] = json!(["A", "Alice Li"]);
        assert_ne!(
            extracted_item_key("register", "Alice", &register),
            extracted_item_key("register", "Alice", &changed)
        );
        changed = register.clone();
        changed["role"] = json!("child");
        assert_ne!(
            extracted_item_key("register", "Alice", &register),
            extracted_item_key("register", "Alice", &changed)
        );
    }

    fn overlap_checkpoint(complete: bool) -> ReconcileCheckpoint {
        let text = format!(
            "{}Alice likes tea{}",
            "甲".repeat(2400),
            "乙".repeat(2401 - 15)
        );
        let root = EvidenceSource {
            id: "overlap".into(),
            kind: "user_text".into(),
            name: "original".into(),
            text,
            locator: None,
            truncated: false,
            graph_already_ingested: false,
            preserve_original: false,
            file_path: None,
        };
        let parts = [(0, 2500), (2400, 4801)].into_iter().enumerate().map(|(index,(start,end))| {
            let mut source = root.clone();
            source.id = format!("overlap@w{index}:{start}-{end}");
            source.text = root.text.chars().skip(start).take(end-start).collect();
            let raw = format!("{{\"o\":\"Alice\",\"c\":\"likes tea\",\"t\":\"diet\",\"e\":\"Alice likes tea\"}}{}",
                if complete || index==0 { "\n{\"done\":true}" } else { "" });
            let parsed = crate::reconcile_protocol::parse_extraction(&raw, &source.id, &source.text, false);
            crate::long_extract::StagedPart { source, parsed, raw:Some(raw), start, end,
                context:json!({"root_source_id":"overlap","hard_window":{"index":index,"start":start,"end":end,"overlap_chars":if index==0 {0}else{100}}}), error:None }
        }).collect();
        ReconcileCheckpoint {
            request: serde_json::from_value(json!({"query":"","evidence":[root]})).unwrap(),
            context: crate::long_extract::ExtractContext {
                roster: "[]".into(),
                tags: "[]".into(),
                speaker: String::new(),
                transcript: String::new(),
                message_at_ms: 0,
            },
            sources: vec![root],
            staged: vec![crate::long_extract::StagedSource {
                parts,
                ..Default::default()
            }],
        }
    }

    #[test]
    fn overlapping_wire_copies_count_once_but_dates_and_roots_remain_distinct() {
        let mut cp = overlap_checkpoint(true);
        let progress = staged_progress(&cp);
        assert!(!progress.partial);
        assert_eq!(progress.chunks_completed, 2);
        assert_eq!(progress.valid_records, 1);
        cp.staged[0].parts[1].parsed.memories[0]["event_at"] = json!("2026-11-01");
        assert_eq!(staged_progress(&cp).valid_records, 2);
        let mut other = cp.sources[0].clone();
        other.id = "independent".into();
        let mut stage = cp.staged[0].clone();
        for part in &mut stage.parts {
            let index = part.context["hard_window"]["index"].as_u64().unwrap();
            part.source.id = format!("independent@w{index}:{}-{}", part.start, part.end);
            part.context["root_source_id"] = json!(other.id);
            for memory in &mut part.parsed.memories {
                memory["source_id"] = json!(part.source.id);
            }
        }
        cp.sources.push(other);
        cp.staged.push(stage);
        assert_eq!(staged_progress(&cp).valid_records, 4);
    }

    #[test]
    fn every_hard_window_must_complete_even_when_overlap_contains_usable_records() {
        let cp = overlap_checkpoint(false);
        let progress = staged_progress(&cp);
        assert!(progress.partial);
        assert_eq!(progress.chunks_completed, 1);
        assert_eq!(progress.valid_records, 1);
        assert_eq!(progress.remaining_ranges[0].start, 2400);
        let mut cp = overlap_checkpoint(true);
        cp.staged[0].parts.pop();
        let progress = staged_progress(&cp);
        assert!(progress.partial);
        assert_eq!(progress.remaining_ranges[0].reason, "coverage_invalid");
        cp.staged.clear();
        assert!(staged_progress(&cp).partial);
    }

    #[test]
    fn candidate_bucket_requires_matching_date_and_known_original_source() {
        let fact = fixture_fact("root-a", Some("2026-10-27"));
        let mut memory = fixture_memory(&fact);
        assert!(same_fact_bucket(&memory, &fact));
        memory.metadata["event_at"] = json!("2026-10-28");
        assert!(!same_fact_bucket(&memory, &fact));
        memory.metadata["event_at"] = Value::Null;
        assert!(!same_fact_bucket(&memory, &fact));
        memory.metadata["event_at"] = json!(fact.llm_event_at);
        memory.metadata["source"]["root_source_id"] = json!("root-b");
        assert!(!same_fact_bucket(&memory, &fact));
        memory.metadata["source"] = json!({"source_id":"root-a"});
        assert!(same_fact_bucket(&memory, &fact));
        memory.metadata["source"] = json!({"source_id":"root-b"});
        assert!(!same_fact_bucket(&memory, &fact));
        memory.metadata["source"] = Value::Null;
        assert!(
            same_fact_bucket(&memory, &fact),
            "legacy absent provenance preserves compatibility"
        );
        memory.subject_ids = vec!["Bob".into()];
        assert!(!same_fact_bucket(&memory, &fact));
    }

    #[test]
    fn ops_date_disambiguates_but_independent_sources_remain_ambiguous() {
        let members = vec![HouseholdMember {
            member_id: "Alice".into(),
            display_name: "Alice".into(),
            role: MemberRole::Other,
            aliases: vec![],
            age_band: None,
        }];
        let a = fixture_fact("root-a", Some("2026-10-27"));
        let b = fixture_fact("root-a", Some("2026-10-28"));
        let facts = vec![a.clone(), b];
        assert!(matched_op_fact(
            &json!({"owner":"Alice","content":"likes tea"}),
            &facts,
            &members
        )
        .is_none());
        let op =
            json!({"owner":"Alice","content":"likes tea","tag":"diet","event_at":"2026-10-27"});
        assert_eq!(
            matched_op_fact(&op, &facts, &members)
                .unwrap()
                .llm_event_at
                .as_deref(),
            Some("2026-10-27")
        );
        let c = fixture_fact("root-b", Some("2026-10-27"));
        assert!(matched_op_fact(&op, &[a, c], &members).is_none());
    }

    #[test]
    fn revision_clears_dates_and_retains_only_sanitized_provenance() {
        let mut fact = fixture_fact("root-a", None);
        fact.speaker_member_id = Some("explicit-speaker".into());
        let mut memory = fixture_memory(&fact);
        memory.metadata["event_at"] = json!("2020-01-01");
        memory.metadata["event_at_ms"] = json!(1);
        memory.metadata["skip_graph"] = json!(true);
        let meta = fact_update(&memory, &fact).unwrap().metadata.unwrap();
        assert!(meta["event_at"].is_null());
        assert!(meta["event_at_ms"].is_null());
        assert_eq!(meta["skip_graph"], false);
        assert_eq!(meta["source"]["root_source_id"], "root-a");
        assert_eq!(meta["source"]["chunk_context"], fact.context_provenance);
        assert_eq!(meta["speaker_member_id"], "explicit-speaker");
    }
}


// Keep an explicit birth year when the model paraphrases a full birth date as a birthday.
// Only the single matching date in this fact's grounded evidence can supply that year.
fn preserve_birth_precision(content: &str, evidence: &str) -> String {
    use std::sync::OnceLock;
    use regex::Regex;
    static FULL: OnceLock<Regex> = OnceLock::new();
    static MONTH_DAY: OnceLock<Regex> = OnceLock::new();
    static YEAR: OnceLock<Regex> = OnceLock::new();
    if !(content.contains("生日") || content.contains("出生")) ||
        !(evidence.contains("出生") || evidence.contains("生于") || evidence.contains("生日")) {
        return content.to_string();
    }
    let year = YEAR.get_or_init(|| Regex::new(r"(?:18|19|20)\d{2}").unwrap());
    if year.is_match(content) { return content.to_string(); }
    let full = FULL.get_or_init(|| Regex::new(r"(?P<y>(?:18|19|20)\d{2})[年/-]\s*(?P<m>\d{1,2})[月/-]\s*(?P<d>\d{1,2})[日号]?").unwrap());
    let dates: Vec<_> = full.captures_iter(evidence).collect();
    if dates.len() != 1 { return content.to_string(); }
    let date=&dates[0];
    let (y,m,d)=(date["y"].parse::<i32>().unwrap(),date["m"].parse::<u32>().unwrap(),date["d"].parse::<u32>().unwrap());
    if chrono::NaiveDate::from_ymd_opt(y,m,d).is_none() { return content.to_string(); }
    let md=MONTH_DAY.get_or_init(|| Regex::new(r"(?P<m>\d{1,2})月\s*(?P<d>\d{1,2})[日号]").unwrap());
    let Some(short)=md.captures(content) else { return content.to_string(); };
    if short["m"].parse::<u32>().ok()!=Some(m) || short["d"].parse::<u32>().ok()!=Some(d) { return content.to_string(); }
    let span=short.get(0).unwrap();
    format!("{}{}年{}月{}日{}",&content[..span.start()],y,m,d,&content[span.end()..])
}

#[cfg(test)]
mod birth_precision_tests {
    use super::preserve_birth_precision;
    #[test]
    fn birth_precision_keeps_explicit_year_without_inventing_one() {
        assert_eq!(preserve_birth_precision("赵明远的生日是4月17日","赵明远出生于1991年4月17日。"),"赵明远的生日是1991年4月17日");
        assert_eq!(preserve_birth_precision("生日是04月17号","出生日期：1991-04-17"),"生日是1991年4月17日");
        for evidence in ["下个月17日生日","生日4月17日","出生日期：1991年2月30日","甲出生1991年4月17日，乙出生1992年4月17日"] {
            assert_eq!(preserve_birth_precision("生日是4月17日",evidence),"生日是4月17日");
        }
        assert_eq!(preserve_birth_precision("生日是4月18日","出生于1991年4月17日"),"生日是4月18日");
        assert_eq!(preserve_birth_precision("复诊日是4月17日","出生于1991年4月17日"),"复诊日是4月17日");
    }
}

#[cfg(test)]
mod extraction_tag_regressions {
    use super::*;
    fn catalog() -> Vec<crate::household::MemoryTag> {
        crate::household::DEFAULT_TAGS.iter().map(|(id, aliases)|crate::household::MemoryTag {
            tag_id:(*id).into(),display_name:(*id).into(),aliases:aliases.iter().map(|a|(*a).into()).collect()
        }).collect()
    }
    #[test]
    fn extracted_categories_use_existing_domain_mapping_and_content_fallback() {
        let tags=catalog();
        assert_eq!(resolve_tag("教育",&tags,"就读四年级"),"工作学习");
        assert_eq!(resolve_tag("职业",&tags,"软件工程师"),"工作学习");
        assert_eq!(resolve_tag("收入",&tags,"每月税后7000元"),"金融");
        assert_eq!(resolve_tag("消费",&tags,"食品花费90元"),"金融");
        assert_eq!(resolve_tag("未定义标签",&tags,"每月工资7000元"),"金融");
        assert_eq!(resolve_tag("未定义标签",&tags,"喜欢喝茶"),"饮食");
        assert_eq!(resolve_tag("未定义标签",&tags,"出生于三月"),"其他");
    }
    #[test]
    fn explicit_custom_labels_and_valid_model_tags_keep_priority() {
        let mut tags=catalog();
        tags.push(crate::household::MemoryTag{tag_id:"收入".into(),display_name:"我的账本".into(),aliases:vec!["收支分类".into()]});
        assert_eq!(resolve_tag("收入",&tags,"每月工资7000元"),"收入");
        assert_eq!(resolve_tag("收支分类",&tags,"每月工资7000元"),"收入");
        assert_eq!(resolve_tag("医疗健康",&tags,"去医院后旅游"),"医疗健康");
        assert_eq!(resolve_tag("其他",&tags,"学过数学"),"其他");
    }
    #[test]
    fn domain_conversion_never_creates_a_tag_missing_from_catalog() {
        let tags=vec![crate::household::MemoryTag{tag_id:"其他".into(),display_name:"其他".into(),aliases:vec![]}];
        assert_eq!(resolve_tag("收入",&tags,"每月税后7000元"),"其他");
    }
}

fn non_person_source_label(label: &str, text: &str) -> bool {
    if label.is_empty() {return false;}
    let non_person=["分类","类别","费用类别","项目","商户","店名","开票单位","学校","医院","地点","地址","金额","出生日期","日期","发票号","编号","单号","票号","ID","主键"];
    // An explicit person-name field wins over a coincidentally unusual name.
    for line in text.lines() {
        if let Some((key,value))=line.split_once('：').or_else(||line.split_once(':')) {
            if ["姓名","患者","学生","人物"].contains(&key.trim()) && value.trim()==label {return false;}
        }
    }
    let record_id=label.chars().any(|c|c.is_ascii_alphabetic()) && label.chars().any(|c|c.is_ascii_digit())
        && label.chars().all(|c|c.is_ascii_alphanumeric() || matches!(c,'-'|'_'));
    let mut headers:Vec<&str>=Vec::new();let mut previous:Vec<&str>=Vec::new();
    for line in text.lines() {
        let line=line.trim();
        if let Some((key,value))=line.split_once('：').or_else(||line.split_once(':')) {
            if non_person.contains(&key.trim()) && value.trim()==label {return true;}
        }
        if !line.starts_with('|') {headers.clear();previous.clear();continue;}
        let cells:Vec<&str>=line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len()>1 && cells.iter().all(|c|!c.is_empty() && c.chars().all(|v|matches!(v,'-'|':'|' '))) {headers=previous.clone();continue;}
        if headers.len()==cells.len() {
            for (header,value) in headers.iter().zip(&cells) {
                let typed=non_person.contains(header) || header.starts_with("金额")
                    || (record_id && (header.contains("ID") || header.contains("编号") || header.contains("主键")));
                if typed && value.split("<br>").any(|token|token.trim()==label) {return true;}
            }
        }
        previous=cells;
    }
    false
}
#[cfg(test)]
mod document_owner_type_regressions {
    use super::*;
    #[test]
    fn business_ids_categories_and_named_fields_are_not_implicit_people() {
        let doc="| ID与持有人 | 金额 元 |\n| --- | --- |\n| ASSET102<br>爸爸 | 900 |\n";
        assert!(non_person_source_label("ASSET102",doc));
        assert!(!non_person_source_label("爸爸",doc));
        assert!(non_person_source_label("城市通勤","| 分类 | 金额 |\n| --- | --- |\n| 城市通勤 | 80 |"));
        assert!(non_person_source_label("山河超市","商户：山河超市"));
        assert!(!non_person_source_label("城市通勤","姓名：城市通勤\n商户：城市通勤"));
        assert!(validated_owner(&json!({"owner":"ASSET102","subjects":["ASSET102"],"_host_non_person_label":true}),&[]).is_none());
        let member=HouseholdMember{member_id:"ASSET102".into(),display_name:"王明".into(),role:MemberRole::Other,aliases:vec![],age_band:None};
        assert_eq!(validated_owner(&json!({"owner":"ASSET102","subjects":["ASSET102"],"_host_non_person_label":true}),&[member]),Some("ASSET102".into()));
    }
}

#[cfg(test)]
mod grounded_rule_integration_tests {
    use super::*;
    use crate::{ListQuery,QianfanConfig};
    async fn fixture()->(tempfile::TempDir,MemoryEngine) {
        let d=tempfile::tempdir().unwrap();let e=MemoryEngine::builder().db_path(d.path().join("rules.db")).build().await.unwrap();
        for (id,name) in [("child","梁子航"),("mom","宋娜娜")] {e.upsert_member(None,HouseholdMember{member_id:id.into(),display_name:name.into(),role:MemberRole::Other,aliases:vec![],age_band:None}).await.unwrap();}
        (d,e)
    }
    async fn apply(e:&MemoryEngine,text:&str,speaker:Option<&str>,evidence:Option<&str>,partial:bool)->ReconcileReport {
        let req:ReconcileRequest=serde_json::from_value(json!({"query":text,"speaker_id":speaker})).unwrap();
        let source:EvidenceSource=serde_json::from_value(json!({"id":"test","kind":"user_text","text":text})).unwrap();
        let mut parsed=crate::reconcile_protocol::parse_extraction("{\"done\":true}","test",text,false);
        if let Some(evidence)=evidence {parsed.memories.push(json!({"owner":"mom","subjects":["mom"],"content":"宋娜娜喜欢吃酸菜鱼","tag":"医疗健康","event_at":null,"source_id":"test","evidence":evidence}));}
        let part=crate::long_extract::StagedPart{source:source.clone(),parsed,raw:None,start:0,end:text.chars().count(),context:json!({}),error:None};
        let stage=crate::long_extract::StagedSource{parts:vec![part],..Default::default()};
        let members=e.list_members(None).await.unwrap();let tags=e.list_tags(None).await.unwrap();
        let context=crate::long_extract::ExtractContext{roster:roster_prompt(&members),tags:tags_prompt(&tags),speaker:speaker.unwrap_or("").into(),transcript:String::new(),message_at_ms:1794570000000};
        let mut config=QianfanConfig::for_embed("unit".into(),None,None,None);config.chat_url="http://127.0.0.1:1/never-called".into();
        let chat=QianfanChat::new(config).unwrap();
        apply_staged(e,&chat,req,context,vec![source],vec![stage],ExtractionProgress{partial,..Default::default()}).await.unwrap()
    }
    #[tokio::test]
    async fn invented_facts_from_hypotheses_quotes_and_unknown_voices_never_reach_sqlite() {
        for (text,evidence,speaker) in [
            ("如果我喜欢吃酸菜鱼就好了","我喜欢吃酸菜鱼",Some("child")),
            ("我喜欢吃酸菜鱼吗？","喜欢吃酸菜鱼",Some("child")),
            ("她说‘我喜欢吃酸菜鱼’","我喜欢吃酸菜鱼",Some("child")),
            ("我喜欢吃酸菜鱼","我喜欢吃酸菜鱼",None),
            ("我喜欢吃酸菜鱼","喜欢吃酸菜鱼",None),
        ] {
            let (_d,e)=fixture().await;let report=apply(&e,text,speaker,Some(evidence),false).await;
            assert!(report.written_ids.is_empty(),"{text}");assert_eq!(e.list(ListQuery::default()).await.unwrap().total,0,"{text}");
        }
    }
    #[tokio::test]
    async fn model_owner_and_tag_are_corrected_from_known_speaker_and_exact_evidence() {
        let (_d,e)=fixture().await;
        let report=apply(&e,"我喜欢吃酸菜鱼，今天风很大",Some("child"),Some("我喜欢吃酸菜鱼"),false).await;
        assert_eq!(report.written_ids.len(),1);
        let list=e.list(ListQuery::default()).await.unwrap();let f=&list.items[0];
        assert_eq!(f.content,"梁子航喜欢吃酸菜鱼");assert_eq!(f.tag,"饮食");assert_eq!(f.subject_ids,vec!["child"]);
    }
    #[tokio::test]
    async fn omitted_safe_fact_is_supplemented_without_hiding_partial_status() {
        let (_d,e)=fixture().await;
        let report=apply(&e,"我喜欢吃酸菜鱼，今天风很大",Some("child"),None,true).await;
        assert_eq!(report.written_ids.len(),1);assert!(report.extraction.partial);assert!(report.fallback_used);
        assert_eq!(e.list(ListQuery::default()).await.unwrap().items[0].tag,"饮食");
    }
    #[tokio::test]
    async fn eight_categories_store_through_fast_path_without_any_model() {
        let (_d,e)=fixture().await;let members=e.list_members(None).await.unwrap();
        for (text,tag) in [("我喜欢吃酸菜鱼","饮食"),("我爱好摄影","娱乐"),("我正在学习Rust","工作学习"),("我对花粉过敏","医疗健康"),("我明天坐高铁去南京","出行"),("我每天晚上九点睡觉","生活习惯"),("我每月零花钱300元","金融"),("我的生日是6月18日","其他")] {
            let req:ReconcileRequest=serde_json::from_value(json!({"query":text,"speaker_id":"child"})).unwrap();
            let source:EvidenceSource=serde_json::from_value(json!({"id":"unit","kind":"user_text","text":text})).unwrap();
            let report=crate::simple_preferences::remember(&e,&req,&members,&[source]).await.unwrap().unwrap();
            assert_eq!(report.facts[0]["tag"],tag);
        }
        let list=e.list(ListQuery{limit:Some(100),..Default::default()}).await.unwrap();assert_eq!(list.total,8);
        for f in list.items {assert_eq!(f.subject_ids,vec!["child"]);assert_eq!(f.metadata["rule"],"grounded_fact_rules_v2");}
    }
}
