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
                    "aliases": m.aliases,
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

const RECONCILE_MAX_SOURCE_CHARS: usize = 120_000;
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
    let members = engine.list_members(req.namespace.as_deref()).await?;
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
            fallback_used: false, facts: vec![],
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
    let bundle = prompt_latest();
    let mut registers = Vec::new();
    let mut memories = Vec::new();
    let mut retractions = Vec::new();
    let mut seen = HashSet::new();
    let mut extract_raw_last: Option<String> = None;
    // Count validated records after exact source-local deduplication; overlapping
    // hard windows must not inflate the report with the same wire record.
    let memories_extracted = extraction.valid_records;
    let mut warnings = Vec::new();
    for (root_source, staged_source) in sources.into_iter().zip(staged) {
        warnings.extend(staged_source.warnings);
        for staged_part in staged_source.parts {
            let part = staged_part.source;
            let parsed = staged_part.parsed;
            extract_raw_last = staged_part.raw.or(extract_raw_last);
            let collect = |key: &str,
                           items: &[Value],
                           out: &mut Vec<Value>,
                           warnings: &mut Vec<String>,
                           seen: &mut HashSet<String>| {
                for item in items {
                    if !evidence_matches(item, &part.id, &part.text) {
                        warnings.push(format!("extract drop invalid provenance: {key}"));
                        continue;
                    }
                    if matches!(key, "register" | "retractions") && part.kind != "user_text" {
                        continue;
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
                        accepted["_host_graph_already_ingested"] =
                            json!(part.graph_already_ingested);
                        accepted["_host_source_id"] = json!(part.id);
                        accepted["_host_source_kind"] = json!(part.kind);
                        accepted["_host_root_source_id"] = json!(root_source.id);
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
        let member_id = name.to_string();
        let _ = engine
            .upsert_member(
                ns,
                HouseholdMember {
                    member_id,
                    display_name: name.into(),
                    role,
                    aliases,
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
        let tag = resolve_tag(item.get("tag").and_then(Value::as_str).unwrap_or(""), &tags);
        let event = validated_event_date(item.get("event_at").and_then(Value::as_str));
        let llm_event_at = event.as_ref().map(|(date, _)| date.clone());
        let event_at_ms = event.map(|(_, ms)| ms);
        let fact = Fact {
            owner_id,
            content: content.to_string(),
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
        let hits = engine
            .recall(RecallQuery {
                text: topic.chars().take(240).collect(),
                namespace: req.namespace.clone(),
                top_k: Some(6),
                fts_only: true,
                skip_graph: true,
                ..Default::default()
            })
            .await?;
        let Some(owner) = validated_owner(item, &members) else {
            warnings.push("ignored retraction without a unique roster owner".into());
            continue;
        };
        let candidates: Vec<_> = hits
            .hits
            .into_iter()
            .filter(|h| normalize_memory_text(&h.memory.content) == normalize_memory_text(topic))
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
    let fallback_used = false;
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
    for batch in facts.chunks(RECONCILE_OPS_MAX_FACTS) {
        let candidates: Vec<_> = hits
            .iter()
            .filter(|hit| batch.iter().any(|fact| same_fact_bucket(hit, fact)))
            .take(RECONCILE_OPS_MAX_HITS)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let facts_json: Value = batch.iter().map(|f| json!({"owner": f.owner_id, "content": f.content, "tag": f.tag, "event_at": f.llm_event_at, "subjects": [f.owner_id.clone()]})).collect();
        let hits_json: Value = candidates.iter().map(|h| json!({"id": h.id, "content": h.content, "tag": h.tag, "subjects": h.subject_ids})).collect();
        let prompt = format!(
            "{}\n新事实：{}\n旧记忆（按成员和标签过滤的候选）：{}\n",
            bundle.ops_rules, facts_json, hits_json
        );
        if prompt.chars().count() + bundle.ops_system.chars().count()
            > RECONCILE_OPS_MAX_PROMPT_CHARS
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
                let target = hit_by_id.get(id).filter(|h| same_fact_bucket(h, fact));
                match op.as_str() {
                    "noop" | "keep" | "skip" => {
                        // The LLM decides semantic equivalence for an authorized
                        // candidate; host checks identity/bucket, not wording.
                        if target.is_some()
                            || hits.iter().any(|h| {
                                same_fact_bucket(h, fact)
                                    && memory_text_similar(&h.content, &fact.content)
                            })
                        {
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
                                    written_keys.insert(key);
                                }
                                Err(e) => warnings.push(format!("ops revise failed: {e}")),
                            }
                        }
                        // Cross-owner/cross-tag/missing targets fall back to add.
                    }
                    "add" | "remember" => {
                        if hits.iter().any(|h| {
                            same_fact_bucket(h, fact)
                                && memory_text_similar(&h.content, &fact.content)
                        }) {
                            written_keys.insert(key);
                            continue;
                        }
                        match remember_validated_fact(engine, fact, ns, req.session_id.as_deref())
                            .await
                        {
                            Ok(id) => {
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
                    written_keys.insert(key);
                }
                Err(e) => warnings.push(format!("fallback revise failed: {e}")),
            }
        } else {
            match remember_validated_fact(engine, fact, ns, req.session_id.as_deref()).await {
                Ok(id) => {
                    written_ids.push(id);
                    written_keys.insert(key);
                }
                Err(e) => warnings.push(format!("fallback remember failed: {e}")),
            }
        }
    }
    performance.mark("write");
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

fn resolve_tag(raw: &str, tags: &[crate::household::MemoryTag]) -> String {
    let raw = raw.trim();
    if let Some(t) = tags
        .iter()
        .find(|t| t.tag_id == raw || t.display_name == raw || t.aliases.iter().any(|a| a == raw))
    {
        return t.tag_id.clone();
    }
    tags.iter()
        .find(|tag| tag.tag_id == "其他")
        .map(|tag| tag.tag_id.clone())
        .unwrap_or_default()
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
        "original_span": fact.source_span, "chunk_context": fact.context_provenance})
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
