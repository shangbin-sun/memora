//! Side-effect-free, source-scoped extraction and semantic recovery.
//!
//! Offsets are Unicode scalar-value offsets into the unmodified original source.
//! A model plan is a boundary/context proposal, never evidence. The caller MUST
//! stage the entire request and apply nothing until every source is complete.
//! Complete siblings retain controls for a later explicit resume; incomplete
//! replies have their controls suppressed by the strict wire-protocol parser.

use crate::extraction_plan::{self, ContextNote, PlannedChunk, TextUnit};
use crate::qianfan::{ExtractionCompletion, QianfanChat, EXTRACTION_TIMEOUT_SECS};
use crate::reconcile::EvidenceSource;
use crate::reconcile_protocol::{parse_extraction, ParsedExtraction};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::Instant;

// Route each original source independently. Exactly 300 Unicode characters
// stays on direct extraction; transcript/roster context does not trigger planning.
const SHORT_SOURCE_CHARS: usize = 300;
const HARD_WINDOW_CHARS: usize = 2_500;
const HARD_WINDOW_OVERLAP: usize = 100;
const TARGET_CHUNK_CHARS: usize = 1_200;

/// Character-count hard cuts, including the overlap inside the 2500-char cap.
/// The tail always follows planning even when it contains <=300 characters.
fn hard_windows(chars: usize) -> Vec<(usize, usize)> {
    let mut windows = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + HARD_WINDOW_CHARS).min(chars);
        windows.push((start, end));
        if end == chars {
            break;
        }
        start = end - HARD_WINDOW_OVERLAP;
    }
    windows
}

/// Use shared global IDs even for overlapping windows. Split structural units
/// at every hard-window boundary so neither planning nor evidence loses a char.
fn source_units(text: &str) -> Vec<TextUnit> {
    let chars: Vec<char> = text.chars().collect();
    let mut boundaries: Vec<_> = hard_windows(chars.len())
        .into_iter()
        .flat_map(|(start, end)| [start, end])
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut result = Vec::new();
    for unit in extraction_plan::text_units(text) {
        let mut start = unit.start;
        for end in boundaries
            .iter()
            .copied()
            .filter(|end| *end > unit.start && *end < unit.end)
            .chain(std::iter::once(unit.end))
        {
            result.push(TextUnit {
                id: result.len(),
                start,
                end,
                text: chars[start..end].iter().collect(),
            });
            start = end;
        }
    }
    // Only compact when fragmentation would force >24 units in a hard window.
    // Keep normal paragraphs as-is, and never cross a hard-window boundary.
    let needs_compaction = hard_windows(chars.len()).iter().any(|(start, end)| {
        result
            .iter()
            .filter(|unit| unit.start >= *start && unit.end <= *end)
            .count()
            > 24
    });
    if needs_compaction {
        let mut compact: Vec<TextUnit> = Vec::new();
        for unit in result {
            if let Some(last) = compact.last_mut() {
                if !boundaries.contains(&unit.start)
                    && last.end - last.start < 250
                    && unit.end - last.start <= 500
                {
                    last.end = unit.end;
                    last.text.push_str(&unit.text);
                    continue;
                }
            }
            compact.push(unit);
        }
        for (id, unit) in compact.iter_mut().enumerate() {
            unit.id = id;
        }
        return compact;
    }
    result
}

fn part_window(part: &StagedPart) -> Option<(usize, usize)> {
    Some((
        usize::try_from(part.context["hard_window"]["start"].as_u64()?).ok()?,
        usize::try_from(part.context["hard_window"]["end"].as_u64()?).ok()?,
    ))
}

fn sort_parts(parts: &mut [StagedPart]) {
    parts.sort_by_key(|part| (part_window(part).map_or(part.start, |w| w.0), part.start));
}

/// Full coverage is checked inside each hard window, not by assuming global
/// non-overlap. Only the deliberate 100-char overlap is allowed across windows.
pub(crate) fn valid_staged_source(source: &EvidenceSource, staged: &StagedSource) -> bool {
    let mut parts: Vec<_> = staged.parts.iter().collect();
    parts.sort_by_key(|part| (part_window(part).map_or(part.start, |w| w.0), part.start));
    valid_coverage(source, &parts)
}

fn valid_coverage(source: &EvidenceSource, parts: &[&StagedPart]) -> bool {
    let mut index = 0;
    for (start, end) in hard_windows(source.text.chars().count()) {
        let mut cursor = start;
        let first = index;
        while let Some(part) = parts
            .get(index)
            .filter(|part| part_window(part) == Some((start, end)))
        {
            if part.start != cursor
                || part.end < part.start
                || part.end > end
                || (part.end == part.start && end != 0)
                || exact_slice(&source.text, part.start, part.end) != part.source.text
                || part.context["hard_window"]["index"].as_u64()
                    != Some((start / (HARD_WINDOW_CHARS - HARD_WINDOW_OVERLAP)) as u64)
                || part.context["hard_window"]["overlap_chars"].as_u64()
                    != Some(if start == 0 {
                        0
                    } else {
                        HARD_WINDOW_OVERLAP as u64
                    })
                || part.source.id
                    != format!(
                        "{}@w{}:{}-{}",
                        source.id,
                        start / (HARD_WINDOW_CHARS - HARD_WINDOW_OVERLAP),
                        part.start,
                        part.end
                    )
                || part.source.kind != source.kind
                || part.source.name != source.name
                || part.source.locator != source.locator
                || part.source.truncated != source.truncated
                || part.source.graph_already_ingested != source.graph_already_ingested
                || part.context["root_source_id"].as_str() != Some(source.id.as_str())
            {
                return false;
            }
            cursor = part.end;
            index += 1;
        }
        if index == first || cursor != end {
            return false;
        }
    }
    index == parts.len()
}
const MAX_PROMPT_CHARS: usize = 24_000;
const MAX_MODEL_CALLS: usize = 96;
const MAX_CHUNK_CALLS: usize = 7;
const MAX_SUBDIVISION_DEPTH: usize = 3;
const RECOVERY_OUTPUT_TOKENS: u32 = 12_288;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExtractContext {
    pub roster: String,
    pub tags: String,
    pub speaker: String,
    pub transcript: String,
    pub message_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StagedPart {
    pub source: EvidenceSource,
    pub parsed: ParsedExtraction,
    pub raw: Option<String>,
    pub start: usize,
    pub end: usize,
    pub context: Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct StagedSource {
    pub parts: Vec<StagedPart>,
    pub warnings: Vec<String>,
    pub planning_calls: usize,
    pub extraction_calls: usize,
    pub recovered_chunks: usize,
    pub truncated_attempts: usize,
    pub invalid_attempt_records: usize,
}

/// An owned request makes scripted/no-network implementations straightforward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CallKind {
    Planning,
    Extraction,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelRequest {
    pub kind: CallKind,
    pub system: String,
    pub prompt: String,
    pub max_tokens: u32,
    /// Remaining SOURCE deadline, including waiting for a provider permit.
    pub timeout: Duration,
}

#[async_trait::async_trait]
pub(crate) trait ExtractionModel: Send + Sync {
    fn edge_budget(&self) -> bool { false }
    async fn call(&self, request: ModelRequest) -> Result<ExtractionCompletion, String>;
}

#[async_trait::async_trait]
impl ExtractionModel for QianfanChat {
    fn edge_budget(&self) -> bool { self.edge_profile() }
    async fn call(&self, request: ModelRequest) -> Result<ExtractionCompletion, String> {
        self.extract_with_timeout(
            Some(&request.system),
            &request.prompt,
            Some(if self.edge_profile() { request.max_tokens.min(512) } else { request.max_tokens }),
            Some(if self.edge_profile() { 0.0 } else { 0.2 }),
            request.timeout,
        )
        .await
        .map(|mut completion| {
            if self.edge_profile() {
                completion.text = edge_records_to_lines(&completion.text);
            }
            completion
        })
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StageLimits {
    pub timeout: Duration,
    pub model_calls: usize,
    pub chunk_calls: usize,
    pub subdivision_depth: usize,
    /// Merge planned blocks up to this size; 0 keeps the model's blocks as they are.
    pub merge_chars: usize,
    /// Split structurally when no usable plan comes back, instead of losing the window.
    pub structural_fallback: bool,
}

impl Default for StageLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(EXTRACTION_TIMEOUT_SECS),
            model_calls: MAX_MODEL_CALLS,
            chunk_calls: MAX_CHUNK_CALLS,
            subdivision_depth: MAX_SUBDIVISION_DEPTH,
            merge_chars: 0,
            structural_fallback: false,
        }
    }
}

/// What a host actually runs. Tests opt in to these separately.
fn production_limits() -> StageLimits {
    StageLimits { merge_chars: TARGET_CHUNK_CHARS, structural_fallback: true, ..StageLimits::default() }
}

pub(crate) async fn stage_source(
    chat: QianfanChat,
    source: EvidenceSource,
    context: ExtractContext,
) -> StagedSource {
    stage_source_with_model(&chat, source, context, production_limits()).await
}

/// Explicit user-triggered resume starts one new bounded source deadline. Only
/// unresolved terminal ranges are retried; complete siblings and their controls
/// remain staged, not applied. A host-truncated original cannot be repaired here.
pub(crate) async fn resume_source(
    chat: QianfanChat,
    source: EvidenceSource,
    context: ExtractContext,
    previous: StagedSource,
) -> StagedSource {
    resume_source_with_model(&chat, source, context, previous, production_limits()).await
}

pub(crate) async fn stage_source_with_model<M: ExtractionModel>(
    model: &M,
    source: EvidenceSource,
    context: ExtractContext,
    limits: StageLimits,
) -> StagedSource {
    let mut run = Run::new(model, source, context, limits);
    let count = run.root.text.chars().count();
    // A clear user's memory command is not an affirmative fact. Treat the
    // original source as authoritative even when a model could quote only its tail.
    if run.root.kind == "user_text" {
        if let Some(intent)=crate::reconcile::user_memory_intent(&run.root.text) {
            let node=Node::new(0,count);
            let mut part=run.empty_part(&node,None);
            part.parsed.complete=!run.root.truncated;
            if let crate::reconcile::UserMemoryIntent::ForgetOwn(topic)=intent {
                if !run.context.speaker.is_empty() && !run.root.truncated {
                    let roster:Value=serde_json::from_str(&run.context.roster).unwrap_or(json!([]));
                    let display=roster.as_array().into_iter().flatten()
                        .find(|member|member["member_id"].as_str()==Some(run.context.speaker.as_str()))
                        .and_then(|member|member["display_name"].as_str()).unwrap_or(&run.context.speaker);
                    let raw=format!("{}\n{{\"done\":true}}",json!({"k":"f","o":run.context.speaker,"c":format!("{display}{topic}"),"e":run.root.text}));
                    part.parsed=parse_extraction(&raw,&part.source.id,&part.source.text,false);
                    part.raw=Some(raw);
                }
            }
            run.output.parts.push(part);
            run.output.warnings.push("explicit user memory command handled without inventing facts".into());
            return run.output;
        }
    }

    if count == 0 {
        let mut part = run.empty_part(&Node::new(0, 0), None);
        part.parsed.complete = !run.root.truncated;
        run.output.parts.push(part);
    } else if model.edge_budget() {
        // Use the same validated hard-window geometry for edge extraction and
        // resume. Each complete row stays local; short unrelated paragraphs no
        // longer consume one 512-token answer budget together.
        for (window_start, window_end) in hard_windows(count) {
            run.hard_window = (window_start, window_end);
            let window = exact_slice(&run.root.text, window_start, window_end);
            for (start, end) in edge_line_ranges(&window, 320) {
                let mut budget = 0;
                let parts = run.extract(Node::new(window_start+start, window_start+end), 0, &mut budget).await;
                run.output.parts.extend(parts);
            }
        }
    } else if count <= SHORT_SOURCE_CHARS {
        let mut budget = 0;
        let parts = run.extract(Node::new(0, count), 0, &mut budget).await;
        run.output.parts.extend(parts);
    } else {
        // Finish each hard window's planning + extraction before submitting the
        // next. Windows overlap by exactly 100 original Unicode characters.
        // Grounded carry is additional disambiguation, never replacement text.
        let mut carry = Vec::new();
        let mut planning_blocked = false;
        for (start, end) in hard_windows(count) {
            run.hard_window = (start, end);
            let unit_window = run.units_for(&Node::new(start, end));
            for window in extraction_plan::planning_windows(&unit_window) {
                if planning_blocked {
                    let node = window_node(&window, carry.clone());
                    let mut part = run.empty_part(
                        &node,
                        Some("waiting for preceding window's validated context".into()),
                    );
                    part.context["planning_required"] = json!(true);
                    run.output.parts.push(part);
                    continue;
                }
                let planned = match run.plan(&window, &carry, false, None).await {
                    Ok((chunks, next)) if run.limits.merge_chars > 0 => Ok((
                        extraction_plan::merge_small_chunks(chunks, &window, run.limits.merge_chars),
                        next,
                    )),
                    Err(error) if run.limits.structural_fallback && run.stopped.is_none() => {
                        run.output.warnings.push(format!("来源 {} 规划不可用，已按结构分块：{error}", run.root.id));
                        Ok((extraction_plan::structural_chunks(&window, TARGET_CHUNK_CHARS), carry.clone()))
                    }
                    other => other,
                };
                match planned {
                    Ok((chunks, next)) => {
                        carry = next;
                        for chunk in chunks {
                            let node = run.node_from_plan(chunk);
                            let mut budget = 0;
                            let parts = run.extract(node, 0, &mut budget).await;
                            run.output.parts.extend(parts);
                        }
                    }
                    Err(error) => {
                        planning_blocked = true;
                        let node = window_node(&window, carry.clone());
                        let mut part = run.empty_part(&node, Some(error));
                        part.context["planning_required"] = json!(true);
                        run.output.parts.push(part);
                    }
                }
            }
        }
    }
    run.finish()
}

pub(crate) async fn resume_source_with_model<M: ExtractionModel>(
    model: &M,
    source: EvidenceSource,
    context: ExtractContext,
    mut previous: StagedSource,
    limits: StageLimits,
) -> StagedSource {
    if source.truncated {
        for part in &mut previous.parts {
            part.parsed.complete = false;
            part.parsed.registers.clear();
            part.parsed.retractions.clear();
            part.error =
                Some("original source is truncated; resupply the complete original".into());
        }
        previous
            .warnings
            .push("原始来源已截断，续跑不能补回缺失原文；请重新提供完整来源。".into());
        return previous;
    }
    // Checkpoints are not permission to replace or mutate evidence. On mismatch,
    // fail closed instead of guessing how old spans map onto a changed document.
    sort_parts(&mut previous.parts);
    let valid = valid_staged_source(&source, &previous);
    if !valid {
        let mut run = Run::new(model, source, context, limits);
        for (start, end) in hard_windows(run.root.text.chars().count()) {
            run.hard_window = (start, end);
            let node = Node::new(start, end);
            let mut part = run.empty_part(
                &node,
                Some("resume checkpoint does not match the original source".into()),
            );
            part.context["planning_required"] =
                json!(run.root.text.chars().count() > SHORT_SOURCE_CHARS);
            run.output.parts.push(part);
        }
        return run.finish();
    }
    let mut run = Run::new(model, source, context, limits);
    // Historical attempts are audit counters, not final failed-record counts.
    run.output.planning_calls = previous.planning_calls;
    run.output.extraction_calls = previous.extraction_calls;
    run.output.recovered_chunks = previous.recovered_chunks;
    run.output.truncated_attempts = previous.truncated_attempts;
    run.output.invalid_attempt_records = previous.invalid_attempt_records;
    let mut resumed_carry: Option<Vec<ContextNote>> = None;
    let mut planning_blocked = false;
    for mut old in previous.parts {
        run.hard_window = part_window(&old).expect("validated hard-window coverage");
        if old.parsed.complete {
            run.output.parts.push(old);
            continue;
        }
        let mut node = Node {
            start: old.start,
            end: old.end,
            notes: notes_from_context(&old.context, &run.units),
            estimated_facts: old.context["estimated_facts"].as_u64().unwrap_or(0) as usize,
        };
        let mut budget = 0;
        let result = if old.context["planning_required"] == true {
            if planning_blocked {
                old.error = Some("waiting for preceding window's validated context".into());
                run.output.parts.push(old);
                continue;
            }
            if let Some(carry) = &resumed_carry {
                node.notes = carry.clone();
            }
            let window = run.units_for(&node);
            match run.plan(&window, &node.notes, false, None).await {
                Ok((chunks, next)) => {
                    resumed_carry = Some(next);
                    let mut parts = Vec::new();
                    for chunk in chunks {
                        let child = run.node_from_plan(chunk);
                        let mut child_budget = 0;
                        parts.extend(run.extract(child, 0, &mut child_budget).await);
                    }
                    parts
                }
                Err(error) => {
                    planning_blocked = true;
                    old.error = Some(error);
                    old.context = run.context_json(&node);
                    old.context["planning_required"] = json!(true);
                    vec![old.clone()]
                }
            }
        } else {
            run.extract(node.clone(), 0, &mut budget).await
        };
        if result.iter().all(|part| part.parsed.complete) {
            run.output.recovered_chunks += 1;
            run.output.parts.extend(result);
        } else {
            let parts = run.retain_unresolved_salvage(&node, Some(old), result);
            run.output.parts.extend(parts);
        }
    }
    run.finish()
}

#[derive(Clone)]
struct Node {
    start: usize,
    end: usize,
    notes: Vec<ContextNote>,
    estimated_facts: usize,
}

impl Node {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            notes: Vec::new(),
            estimated_facts: 0,
        }
    }
}

struct Run<'a, M> {
    model: &'a M,
    root: EvidenceSource,
    context: ExtractContext,
    units: Vec<TextUnit>,
    limits: StageLimits,
    deadline: Instant,
    calls: usize,
    stopped: Option<String>,
    output: StagedSource,
    hard_window: (usize, usize),
}

impl<'a, M: ExtractionModel> Run<'a, M> {
    fn new(
        model: &'a M,
        root: EvidenceSource,
        context: ExtractContext,
        mut limits: StageLimits,
    ) -> Self {
        // Test hooks may make limits stricter, never relax production safeguards.
        limits.timeout = limits
            .timeout
            .min(Duration::from_secs(EXTRACTION_TIMEOUT_SECS));
        limits.model_calls = limits.model_calls.min(MAX_MODEL_CALLS);
        limits.chunk_calls = limits.chunk_calls.min(MAX_CHUNK_CALLS);
        limits.subdivision_depth = limits.subdivision_depth.min(MAX_SUBDIVISION_DEPTH);
        Self {
            model,
            units: source_units(&root.text),
            hard_window: (0, root.text.chars().count().min(HARD_WINDOW_CHARS)),
            root,
            context,
            limits,
            deadline: Instant::now() + limits.timeout,
            calls: 0,
            stopped: None,
            output: StagedSource::default(),
        }
    }

    fn finish(mut self) -> StagedSource {
        sort_parts(&mut self.output.parts);
        if self.root.truncated {
            for part in &mut self.output.parts {
                part.parsed.complete = false;
                part.parsed.registers.clear();
                part.parsed.retractions.clear();
                part.error
                    .get_or_insert_with(|| "original source is truncated".into());
            }
            self.output.warnings.push(format!(
                "来源 {} 的原始输入已截断，不能视为完整提取。",
                self.root.id
            ));
        }
        let incomplete = self
            .output
            .parts
            .iter()
            .filter(|part| !part.parsed.complete)
            .count();
        if incomplete > 0 {
            self.output.warnings.push(format!(
                "来源 {} 有 {incomplete} 个范围未完成；整个请求仅暂存，成员注册和撤回等控制操作均不得执行。",
                self.root.id
            ));
        }
        for part in &self.output.parts {
            if let Some(error) = &part.error {
                self.output.warnings.push(format!(
                    "{} [{}..{}]: {error}",
                    self.root.id, part.start, part.end
                ));
            }
        }
        self.output
    }

    fn node_from_plan(&self, chunk: PlannedChunk) -> Node {
        // parse_plan has already validated IDs, coverage and bounds.
        let first = self
            .units
            .iter()
            .find(|unit| unit.id == chunk.first)
            .expect("validated first unit");
        let last = self
            .units
            .iter()
            .find(|unit| unit.id == chunk.last)
            .expect("validated last unit");
        Node {
            start: first.start,
            end: last.end,
            notes: chunk.context,
            estimated_facts: chunk.estimated_facts,
        }
    }

    fn units_for(&self, node: &Node) -> Vec<TextUnit> {
        self.units
            .iter()
            .filter(|unit| unit.start >= node.start && unit.end <= node.end)
            .cloned()
            .collect()
    }

    fn part_source(&self, node: &Node) -> EvidenceSource {
        EvidenceSource {
            id: format!(
                "{}@w{}:{}-{}",
                self.root.id,
                self.hard_window.0 / (HARD_WINDOW_CHARS - HARD_WINDOW_OVERLAP),
                node.start,
                node.end
            ),
            text: exact_slice(&self.root.text, node.start, node.end),
            ..self.root.clone()
        }
    }

    fn context_json(&self, node: &Node) -> Value {
        let notes: Vec<_> = node
            .notes
            .iter()
            .map(|note| {
                let citations: Vec<_> = note
                    .refs
                    .iter()
                    .filter_map(|id| {
                        self.units.iter().find(|unit| unit.id == *id).map(|unit| json!({
                    "unit_id": unit.id, "start": unit.start, "end": unit.end, "text": unit.text
                }))
                    })
                    .collect();
                json!({"summary": note.summary, "refs": note.refs, "citations": citations})
            })
            .collect();
        json!({
            "root_source_id": self.root.id,
            "hard_window": {"index":self.hard_window.0 / (HARD_WINDOW_CHARS - HARD_WINDOW_OVERLAP),
                "start":self.hard_window.0,"end":self.hard_window.1,
                "overlap_chars":if self.hard_window.0 == 0 { 0 } else { HARD_WINDOW_OVERLAP }},
            "span": {"start": node.start, "end": node.end, "unit": "unicode_scalar"},
            "purpose": "disambiguation_only_not_evidence_or_default_owner",
            "notes": notes,
            "estimated_facts": node.estimated_facts,
        })
    }

    fn empty_part(&self, node: &Node, error: Option<String>) -> StagedPart {
        StagedPart {
            source: self.part_source(node),
            parsed: ParsedExtraction::default(),
            raw: None,
            start: node.start,
            end: node.end,
            context: self.context_json(node),
            error,
        }
    }

    async fn call(
        &mut self,
        kind: CallKind,
        system: &str,
        prompt: String,
        max_tokens: u32,
        local: Option<&mut usize>,
    ) -> Result<ExtractionCompletion, String> {
        if let Some(error) = &self.stopped {
            return Err(error.clone());
        }
        if Instant::now() >= self.deadline {
            self.stopped = Some("source extraction deadline exhausted".into());
            return Err(self.stopped.clone().unwrap());
        }
        if self.calls >= self.limits.model_calls {
            return Err("source model-call budget exhausted".into());
        }
        if system.chars().count() + prompt.chars().count() > if self.model.edge_budget() { 2400 } else { MAX_PROMPT_CHARS } {
            return Err("local prompt context limit exceeded".into());
        }
        if let Some(local) = local {
            if *local >= self.limits.chunk_calls {
                return Err("original chunk model-call budget exhausted".into());
            }
            *local += 1;
        }
        self.calls += 1;
        match kind {
            CallKind::Planning => self.output.planning_calls += 1,
            CallKind::Extraction => self.output.extraction_calls += 1,
        }
        let request = ModelRequest {
            kind,
            system: system.into(),
            prompt,
            max_tokens,
            timeout: self.deadline.saturating_duration_since(Instant::now()),
        };
        let result = tokio::time::timeout_at(self.deadline, self.model.call(request)).await;
        let result = match result {
            Ok(result) => result,
            Err(_) => Err("source extraction deadline exhausted".into()),
        };
        if let Ok(completion) = &result {
            self.output.truncated_attempts += usize::from(completion.truncated);
        }
        if let Err(error) = &result {
            let lower = error.to_ascii_lowercase();
            if lower.contains("timeout")
                || lower.contains("timed out")
                || lower.contains("deadline")
            {
                // No retry after a transport timeout, including other queued
                // ranges of this original source. An explicit resume is separate.
                self.stopped = Some(error.clone());
            }
        }
        result
    }

    async fn plan(
        &mut self,
        window: &[TextUnit],
        carry: &[ContextNote],
        recovery: bool,
        local: Option<&mut usize>,
    ) -> Result<(Vec<PlannedChunk>, Vec<ContextNote>), String> {
        let mut prompt = extraction_plan::plan_prompt(&self.root.id, window, carry, &self.units);
        if recovery {
            // The normal prior selector excludes future/same-window references.
            // A previously validated parent context can legitimately cite those;
            // show it separately and inherit it, without widening parse authority.
            let inherited = self.context_json(&window_node(window, carry.to_vec()));
            prompt.push_str(&format!("\n{}\n恢复规划：上次输出或输入超限。请按语义拆成至少两个更小、连续且完整覆盖当前窗口的块，目标每块不超过600字；保留消歧引用。inherited_chunk_context仅消歧且由宿主继承，无需在context重复输出其外部引用。不得省略单元、改写原文或用机械二分代替语义规划。若无法安全拆分，返回原块。\n", json!({"inherited_chunk_context": inherited})));
        }
        let completion = self
            .call(
                CallKind::Planning,
                extraction_plan::SYSTEM,
                prompt,
                4_096,
                local,
            )
            .await?;
        if completion.truncated {
            return Err(
                "semantic planning response truncated; original window remains unresolved".into(),
            );
        }
        let result = extraction_plan::parse_plan(&completion.text, window, carry, &self.units)
            .map_err(|error| format!("semantic planning invalid: {error}"))?;
        if recovery && result.0.len() < 2 {
            return Err("semantic planner could not safely subdivide the original chunk".into());
        }
        Ok(result)
    }

    fn extract<'b>(
        &'b mut self,
        node: Node,
        depth: usize,
        local: &'b mut usize,
    ) -> Pin<Box<dyn Future<Output = Vec<StagedPart>> + Send + 'b>> {
        Box::pin(async move {
            let source = self.part_source(&node);
            // A whitespace-only span has no facts; preserve coverage without
            // asking a model to invent content for it.
            if source.text.trim().is_empty() || (self.model.edge_budget() && edge_markup_only(&source.text)) {
                let mut part = self.empty_part(&node, None);
                part.parsed.complete = !self.root.truncated;
                return vec![part];
            }
            if self.model.edge_budget() && source.kind != "user_text" {
                if let Some(record)=crate::edge_table::transaction_record(&self.root.text,node.start,&source.text,&self.context.roster,&self.context.tags) {
                    let mut part=self.empty_part(&node,None);
                    let raw=format!("{}\n{{\"done\":true}}",record);
                    part.parsed=parse_extraction(&raw,&source.id,&source.text,false);
                    part.raw=Some(raw);
                    part.context["format_parser"]=json!("typed-transaction-row-v1");
                    return vec![part];
                }
            }
            let metadata = self.context_json(&node);
            let edge=self.model.edge_budget();
            let base_system=if edge {edge_extraction_prompt(&source,&self.context,&self.root.text,node.start)} else {crate::prompt_latest().extract_system.to_owned()};
            let base_prompt=if edge {source.text.clone()} else {extraction_prompt(&source,&self.context,&metadata)};
            let mut salvage: Option<StagedPart> = None;
            let mut cause = String::new();
            let mut can_split = false;
            // One whole-chunk retry replaces, rather than appends to, a complete
            // recovery. Increased output budget is bounded independently of text.
            for attempt in 0..2 {
                let mut prompt = base_prompt.clone();
                let mut system = base_system.clone();
                if attempt == 1 {
                    let instructions=if edge {&mut system}else{&mut prompt};
                    instructions.push_str("\n上次回复不完整或存在无效记录。本次从头完整提取当前块，逐行严格满足原协议并以{\"done\":true}收尾；不要接续、不要引用上次回复、不补造事实。\n");
                }
                let tokens = if attempt == 0 {
                    crate::output_budget(source.text.chars().count()).min(8_192) as u32
                } else {
                    RECOVERY_OUTPUT_TOKENS
                };
                match self
                    .call(
                        CallKind::Extraction,
                        &system,
                        prompt,
                        tokens,
                        Some(local),
                    )
                    .await
                {
                    Ok(completion) => {
                        let mut parsed = parse_extraction(
                            &completion.text,
                            &source.id,
                            &source.text,
                            completion.truncated,
                        );
                        if self.model.edge_budget() {
                            parsed.memories.retain(|item|!edge_name_only(item,&self.context.roster));
                        }
                        if self.model.edge_budget() && source.kind != "user_text" {
                            if let Some((subject,display))=edge_table_subject(&self.root.text,node.start,&source.text,&self.context.roster) {
                                for item in &mut parsed.memories {
                                    let evidence=item["evidence"].as_str().unwrap_or("").to_owned();
                                    item["owner"]=json!(subject);item["subjects"]=json!([subject]);
                                    // The declared table identity is authoritative. Preserve
                                    // exactly the selected cell/row, not a model's inferred
                                    // family relation or omitted qualifier.
                                    item["content"]=json!(if evidence.starts_with(&display) {evidence}else{format!("{display}：{evidence}")});
                                }
                            }
                        }
                        self.output.invalid_attempt_records += parsed.invalid_records;
                        // Retrying a finished reply again will not fix a stray line. Keep the
                        // valid records; bad lines stay dropped and counted.
                        if attempt == 1 && parsed.marker_ok && !completion.truncated {
                            parsed.complete = true;
                        }
                        let complete = parsed.complete;
                        let mut part = StagedPart {
                            source: source.clone(),
                            parsed,
                            raw: Some(completion.text),
                            start: node.start,
                            end: node.end,
                            context: metadata.clone(),
                            error: None,
                        };
                        if complete {
                            if attempt > 0 && !self.root.truncated {
                                self.output.recovered_chunks += 1;
                            }
                            if self.root.truncated {
                                part.parsed.complete = false;
                                part.parsed.registers.clear();
                                part.parsed.retractions.clear();
                                part.error = Some("original source is truncated".into());
                            }
                            return vec![part];
                        }
                        cause = if completion.truncated {
                            "extraction output truncated".into()
                        } else {
                            "extraction protocol incomplete or contains invalid records".into()
                        };
                        part.error = Some(cause.clone());
                        if let Some(older) = salvage.take() {
                            append_unique_memories(
                                &mut part.parsed.memories,
                                older.parsed.memories,
                            );
                        }
                        salvage = Some(part);
                        can_split = true;
                    }
                    Err(error) => {
                        // A provider may reject the larger retry budget. The
                        // previous partial response still justifies semantic
                        // subdivision; do not strand it behind that rejection.
                        // call() marks timeouts stopped, so they cannot trigger
                        // another network call here.
                        can_split = salvage.is_some()
                            || error
                                .to_ascii_lowercase()
                                .contains("context limit exceeded");
                        cause = error;
                        break;
                    }
                }
            }
            if can_split && depth < self.limits.subdivision_depth {
                if self.model.edge_budget() && node.end-node.start>40 {
                    let ranges=edge_line_ranges(&source.text,((node.end-node.start)/2).max(40));
                    if ranges.len()>1 {
                        let mut children=Vec::new();
                        for (start,end) in ranges {
                            children.extend(self.extract(Node::new(node.start+start,node.start+end),depth+1,local).await);
                        }
                        if children.iter().all(|part|part.parsed.complete) {
                            self.output.recovered_chunks+=1;return children;
                        }
                        return self.retain_unresolved_salvage(&node,salvage,children);
                    }
                }
                let window = self.units_for(&node);
                if window.len() > 1 {
                    match self.plan(&window, &node.notes, true, Some(local)).await {
                        Ok((chunks, _)) => {
                            let mut children = Vec::new();
                            for chunk in chunks {
                                let mut child = self.node_from_plan(chunk);
                                // Parent context was already grounded and bounded;
                                // retain it even if the recovery planner omits it.
                                child.notes = merge_notes(&node.notes, &child.notes, &self.units);
                                children.extend(self.extract(child, depth + 1, local).await);
                            }
                            if children.iter().all(|part| part.parsed.complete) {
                                self.output.recovered_chunks += 1;
                                return children;
                            }
                            return self.retain_unresolved_salvage(&node, salvage, children);
                        }
                        Err(error) => cause = format!("{cause}; {error}"),
                    }
                } else {
                    cause.push_str("; no smaller validated semantic unit boundary");
                }
            }
            let mut part = salvage.unwrap_or_else(|| self.empty_part(&node, None));
            part.error = Some(cause);
            vec![part]
        })
    }

    fn retain_unresolved_salvage(
        &self,
        node: &Node,
        ancestor: Option<StagedPart>,
        mut children: Vec<StagedPart>,
    ) -> Vec<StagedPart> {
        let Some(mut ancestor) = ancestor else {
            return children;
        };
        if ancestor.parsed.memories.is_empty() {
            return children;
        }
        // Prefer complete child results; never retain parent successes whose
        // evidence belongs entirely to an already recovered child.
        ancestor.parsed.memories.retain(|memory| {
            let evidence = memory["evidence"].as_str().unwrap_or("");
            !children
                .iter()
                .any(|part| part.parsed.complete && part.source.text.contains(evidence))
        });
        if ancestor.parsed.memories.is_empty() {
            return children;
        }
        // Place salvage only in unresolved leaves when its evidence fits exactly.
        // Cross-child evidence cannot be squeezed into a false provenance span:
        // collapse this unresolved subtree to its original, non-overlapping span.
        let fits = ancestor.parsed.memories.iter().all(|memory| {
            let evidence = memory["evidence"].as_str().unwrap_or("");
            children
                .iter()
                .any(|part| !part.parsed.complete && part.source.text.contains(evidence))
        });
        if fits {
            for mut memory in ancestor.parsed.memories {
                let evidence = memory["evidence"].as_str().unwrap_or("").to_owned();
                if let Some(part) = children
                    .iter_mut()
                    .find(|part| !part.parsed.complete && part.source.text.contains(&evidence))
                {
                    memory["source_id"] = json!(part.source.id);
                    append_unique_memories(&mut part.parsed.memories, vec![memory]);
                }
            }
            return children;
        }
        let mut combined = self.empty_part(
            node,
            Some("semantic recovery remains incomplete; ancestor evidence retained once".into()),
        );
        combined.raw = ancestor.raw;
        combined.context["recovery_parts"] = json!(children
            .iter()
            .map(|part| json!({
                "start": part.start, "end": part.end, "complete": part.parsed.complete,
                "chunk_context": part.context,
            }))
            .collect::<Vec<_>>());
        for part in children {
            combined.parsed.invalid_records += part.parsed.invalid_records;
            for mut memory in part.parsed.memories {
                memory["source_id"] = json!(combined.source.id);
                append_unique_memories(&mut combined.parsed.memories, vec![memory]);
            }
        }
        for mut memory in ancestor.parsed.memories {
            memory["source_id"] = json!(combined.source.id);
            append_unique_memories(&mut combined.parsed.memories, vec![memory]);
        }
        vec![combined]
    }
}

fn append_unique_memories(target: &mut Vec<Value>, incoming: Vec<Value>) {
    for memory in incoming {
        if !target.contains(&memory) {
            target.push(memory);
        }
    }
}

fn window_node(window: &[TextUnit], notes: Vec<ContextNote>) -> Node {
    Node {
        start: window.first().map_or(0, |unit| unit.start),
        end: window.last().map_or(0, |unit| unit.end),
        notes,
        estimated_facts: 0,
    }
}

fn notes_from_context(context: &Value, units: &[TextUnit]) -> Vec<ContextNote> {
    context["notes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|note| {
            let summary = note["summary"].as_str()?.to_owned();
            let refs: Option<Vec<usize>> = note["refs"]
                .as_array()?
                .iter()
                .map(|id| {
                    let id = usize::try_from(id.as_u64()?).ok()?;
                    units.iter().any(|unit| unit.id == id).then_some(id)
                })
                .collect();
            let refs = refs?;
            (!refs.is_empty()).then_some(ContextNote { summary, refs })
        })
        .collect()
}

fn merge_notes(
    inherited: &[ContextNote],
    proposed: &[ContextNote],
    units: &[TextUnit],
) -> Vec<ContextNote> {
    let mut selected = Vec::new();
    let mut citation_chars = 0;
    for note in inherited.iter().chain(proposed) {
        if selected.len() == 6 {
            break;
        }
        if selected.contains(note)
            || note.summary.trim().is_empty()
            || note.summary.chars().count() > 240
            || note.refs.is_empty()
            || note.refs.len() > 4
        {
            continue;
        }
        let cited: Option<Vec<_>> = note
            .refs
            .iter()
            .map(|id| units.iter().find(|unit| unit.id == *id))
            .collect();
        let Some(cited) = cited else {
            continue;
        };
        let size: usize = cited.iter().map(|unit| unit.text.chars().count()).sum();
        if citation_chars + size <= 1_800 {
            citation_chars += size;
            selected.push(note.clone());
        }
    }
    selected
}

fn exact_slice(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// Preserve the versioned extraction body and source envelope. Context remains
/// a separate JSON field; only exact substrings of CURRENT source.text may be e.
pub(crate) fn extraction_prompt(
    source: &EvidenceSource,
    ctx: &ExtractContext,
    chunk_context: &Value,
) -> String {
    let bundle = crate::prompt_latest();
    let shanghai = chrono::FixedOffset::east_opt(8 * 3600).expect("UTC+8");
    let when = chrono::DateTime::from_timestamp_millis(ctx.message_at_ms)
        .unwrap_or_else(|| {
            // Same wall-clock fallback without requiring chrono's platform clock
            // feature; isolated offline tests need only chrono/std.
            let millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .min(i64::MAX as u128) as i64;
            chrono::DateTime::from_timestamp_millis(millis).expect("current timestamp")
        })
        .with_timezone(&shanghai);
    let transcript_snip: String = ctx
        .transcript
        .chars()
        .rev()
        .take(1_200)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let speaker_line = if ctx.speaker.is_empty() {
        "宿主未指定说话人（仅第一人称用于消歧，不编造默认说话人）".into()
    } else {
        format!(
            "宿主说话人 speaker_id：{}（仅用于第一人称消歧，不是默认归属）",
            ctx.speaker
        )
    };
    let name = if source.name.trim().is_empty() {
        "证据来源"
    } else {
        source.name.trim()
    };
    format!(
        "{rules}\n{speaker_line}\n当前名册：{roster}\n当前标签：{tags}\n本轮消息时间：中国时区 {when}\n最近会话（仅消歧）：{transcript}\n分块上下文（JSON数据，仅消歧，不是新事实、不是证据、不是默认归属；e必须逐字取自本批source.text，不得取自摘要、引用或会话）：\n{chunk_context}\n本批证据来源（JSON数据）：\n{source}",
        rules = bundle.extract_rules,
        roster = ctx.roster,
        tags = ctx.tags,
        transcript = serde_json::to_string(&transcript_snip).unwrap_or_else(|_| "\"\"".into()),
        chunk_context = json!({"chunk_context": chunk_context}),
        source = json!({
            "source_id": source.id, "kind": source.kind, "name": name,
            "text": source.text, "truncated": source.truncated,
            "locator": source.locator, "graph_already_ingested": source.graph_already_ingested,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    enum Reply {
        Text(String, bool),
        Error(&'static str),
        Delay(Duration, String, bool),
    }

    #[derive(Default)]
    struct Script {
        replies: Mutex<VecDeque<Reply>>,
        requests: Mutex<Vec<ModelRequest>>,
    }

    impl Script {
        fn new(replies: Vec<Reply>) -> Self {
            Self {
                replies: Mutex::new(replies.into()),
                ..Self::default()
            }
        }
        fn requests(&self) -> Vec<ModelRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl ExtractionModel for Script {
        async fn call(&self, request: ModelRequest) -> Result<ExtractionCompletion, String> {
            self.requests.lock().unwrap().push(request);
            let reply = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model call");
            let (text, truncated) = match reply {
                Reply::Text(text, truncated) => (text, truncated),
                Reply::Error(error) => return Err(error.into()),
                Reply::Delay(delay, text, truncated) => {
                    tokio::time::sleep(delay).await;
                    (text, truncated)
                }
            };
            Ok(ExtractionCompletion { text, truncated })
        }
    }

    fn source(text: &str) -> EvidenceSource {
        EvidenceSource {
            id: "root".into(),
            kind: "document".into(),
            name: "原文".into(),
            text: text.into(),
            locator: Some("page:7".into()),
            truncated: false,
            graph_already_ingested: true,
            preserve_original: false,
            file_path: None,
        }
    }
    fn ctx() -> ExtractContext {
        ExtractContext {
            roster: "[]".into(),
            tags: "[]".into(),
            speaker: String::new(),
            transcript: String::new(),
            message_at_ms: 0,
        }
    }
    fn memory(evidence: &str, content: &str) -> String {
        json!({"o":"self","t":"profile","c":content,"e":evidence}).to_string()
    }
    fn done(evidence: &str, content: &str) -> Reply {
        Reply::Text(
            format!("{}\n{{\"done\":true}}", memory(evidence, content)),
            false,
        )
    }
    fn partial(evidence: &str, content: &str) -> Reply {
        Reply::Text(format!("{}\n{{", memory(evidence, content)), true)
    }
    fn plan(chunks: &[(usize, usize)], notes: Vec<Vec<ContextNote>>) -> Reply {
        Reply::Text(
            json!({
                "chunks": chunks.iter().enumerate().map(|(i,(first,last))| json!({
                    "first":first,"last":last,"context":notes.get(i).cloned().unwrap_or_default(),
                    "estimated_facts":20
                })).collect::<Vec<_>>(), "carry": [], "complete": true,
            })
            .to_string(),
            false,
        )
    }
    fn request_source(request: &ModelRequest) -> Value {
        serde_json::from_str(
            request
                .prompt
                .split("本批证据来源（JSON数据）：\n")
                .last()
                .unwrap()
                .split("\n上次回复")
                .next()
                .unwrap(),
        )
        .unwrap()
    }
    fn assert_partition(staged: &StagedSource, original: &str) {
        assert!(
            valid_staged_source(&source(original), staged),
            "staged ranges must cover every hard window"
        );
        for part in &staged.parts {
            assert_eq!(
                part.source.text,
                exact_slice(original, part.start, part.end)
            );
            for memory in &part.parsed.memories {
                assert!(part
                    .source
                    .text
                    .contains(memory["evidence"].as_str().unwrap()));
                assert_eq!(memory["source_id"], part.source.id);
            }
        }
    }

    #[tokio::test]
    async fn sources_up_to_300_unicode_chars_extract_directly_without_planning() {
        for (unit, count) in [("甲", 299), ("甲", 300), ("😀", 300), ("a", 300)] {
            let original = unit.repeat(count);
            let model = Script::new(vec![done(unit, "有效事实")]);
            // Long auxiliary context is not a reason to preprocess short input.
            let mut context = ctx();
            context.transcript = "会话上下文".repeat(1_000);
            context.roster = json!([{"display_name":"名册上下文".repeat(100)}]).to_string();
            let staged =
                stage_source_with_model(&model, source(&original), context, StageLimits::default())
                    .await;
            assert_eq!(
                staged.planning_calls, 0,
                "{count} Unicode characters must skip preprocessing"
            );
            assert_eq!(staged.extraction_calls, 1);
            let requests = model.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].kind, CallKind::Extraction);
            assert_eq!(request_source(&requests[0])["text"], original);
            assert!(staged.parts[0].parsed.complete);
            assert_partition(&staged, &original);
        }
    }

    #[tokio::test]
    async fn source_at_301_unicode_chars_plans_before_extraction() {
        for unit in ["甲", "😀", "a"] {
            let original = unit.repeat(301);
            let model = Script::new(vec![plan(&[(0, 0)], vec![]), done(unit, "有效事实")]);
            let staged =
                stage_source_with_model(&model, source(&original), ctx(), StageLimits::default())
                    .await;
            assert_eq!(staged.planning_calls, 1);
            assert_eq!(staged.extraction_calls, 1);
            let requests = model.requests();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].kind, CallKind::Planning);
            assert_eq!(requests[1].kind, CallKind::Extraction);
            assert!(staged.parts[0].parsed.complete);
            assert_partition(&staged, &original);
        }
    }

    #[tokio::test]
    async fn multiple_short_sources_do_not_trigger_planning_by_aggregate_length() {
        let model = Script::new(vec![done("甲", "第一来源"), done("乙", "第二来源")]);
        for text in ["甲".repeat(300), "乙".repeat(300)] {
            let staged =
                stage_source_with_model(&model, source(&text), ctx(), StageLimits::default()).await;
            assert_eq!(staged.planning_calls, 0);
            assert_eq!(staged.extraction_calls, 1);
        }
        assert!(model
            .requests()
            .iter()
            .all(|request| request.kind == CallKind::Extraction));
    }

    #[tokio::test]
    async fn dense_truncation_retries_whole_chunk_with_larger_budget_and_replaces_parent() {
        let model = Script::new(vec![
            partial("甲", "旧的部分结果"),
            done("甲", "恢复的完整结果"),
        ]);
        let staged =
            stage_source_with_model(&model, source("甲甲甲"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.planning_calls, 0);
        assert_eq!(staged.extraction_calls, 2);
        assert_eq!(staged.recovered_chunks, 1);
        assert_eq!(staged.truncated_attempts, 1);
        assert_eq!(staged.invalid_attempt_records, 1);
        assert!(staged.parts[0].parsed.complete);
        assert_eq!(staged.parts[0].parsed.invalid_records, 0);
        assert_eq!(staged.parts[0].parsed.memories.len(), 1);
        assert_eq!(
            staged.parts[0].parsed.memories[0]["content"],
            "恢复的完整结果"
        );
        let requests = model.requests();
        assert!(requests[1].max_tokens > requests[0].max_tokens);
        assert_eq!(requests[1].max_tokens, 12_288);
        assert!(requests[1].timeout <= requests[0].timeout);
    }

    #[tokio::test]
    async fn repeated_truncation_uses_model_semantic_boundaries_and_never_double_applies() {
        let original = "甲的第一段\n乙的第二段";
        let model = Script::new(vec![
            partial("甲", "父结果"),
            partial("乙", "父结果2"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "完整甲"),
            done("乙", "完整乙"),
        ]);
        let staged =
            stage_source_with_model(&model, source(original), ctx(), StageLimits::default()).await;
        assert_eq!((staged.planning_calls, staged.extraction_calls), (1, 4));
        assert_eq!(staged.truncated_attempts, 2);
        assert_eq!(staged.parts.len(), 2);
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        assert_eq!(
            staged
                .parts
                .iter()
                .map(|part| part.parsed.memories.len())
                .sum::<usize>(),
            2
        );
        assert_partition(&staged, original);
        assert!(model.requests()[2].prompt.contains("至少两个"));
    }

    #[tokio::test]
    async fn rejected_larger_output_budget_still_recovers_with_semantic_children() {
        let original = "甲的第一段\n乙的第二段";
        let model = Script::new(vec![
            partial("甲", "暂存父结果"),
            Reply::Error("max_tokens exceeds provider output maximum"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "完整甲"),
            done("乙", "完整乙"),
        ]);
        let staged =
            stage_source_with_model(&model, source(original), ctx(), StageLimits::default()).await;
        assert_eq!((staged.planning_calls, staged.extraction_calls), (1, 4));
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        assert_partition(&staged, original);
    }

    #[tokio::test]
    async fn invalid_non_tail_record_gets_one_strict_full_retry() {
        let model = Script::new(vec![
            Reply::Text(
                format!(
                    "{{\"o\":1}}\n{}\n{{\"done\":true}}",
                    memory("甲", "partial")
                ),
                false,
            ),
            done("甲", "complete"),
        ]);
        let staged =
            stage_source_with_model(&model, source("甲"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.extraction_calls, 2);
        assert_eq!(staged.invalid_attempt_records, 1);
        assert_eq!(staged.truncated_attempts, 0);
        assert!(staged.parts[0].parsed.complete);
        assert_eq!(staged.parts[0].parsed.memories[0]["content"], "complete");
    }

    #[tokio::test]
    async fn provider_context_overflow_plans_smaller_chunks_not_blind_halves() {
        let original = "甲\n乙的特别长第二段";
        let model = Script::new(vec![
            Reply::Error("chat context limit exceeded"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "甲"),
            done("乙", "乙"),
        ]);
        let staged =
            stage_source_with_model(&model, source(original), ctx(), StageLimits::default()).await;
        assert_eq!(staged.parts[0].end, 2);
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        assert_eq!(model.requests()[1].kind, CallKind::Planning);
        assert_partition(&staged, original);
    }

    #[tokio::test]
    async fn unsafe_semantic_split_fails_closed_and_salvages_valid_memory_once() {
        let model = Script::new(vec![
            partial("甲", "有用"),
            partial("甲", "有用"),
            plan(&[(0, 1)], vec![]),
        ]);
        let staged =
            stage_source_with_model(&model, source("甲\n乙"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.parts.len(), 1);
        assert!(!staged.parts[0].parsed.complete);
        assert_eq!(staged.parts[0].parsed.memories.len(), 1);
        assert!(staged.parts[0]
            .error
            .as_ref()
            .unwrap()
            .contains("safely subdivide"));
        assert_eq!(model.requests().len(), 3);
    }

    #[tokio::test]
    async fn unknown_or_context_evidence_is_not_accepted_from_outside_current_chunk() {
        let source = source("此处没有姓名");
        let context =
            json!({"notes":[{"summary":"外部猜测属于张三", "citations":[{"text":"张三有车"}]}]});
        let prompt = extraction_prompt(&source, &ctx(), &context);
        assert!(prompt.contains("\"chunk_context\""));
        assert!(prompt.contains("不是默认归属"));
        assert!(prompt.contains("e必须逐字取自本批source.text"));
        assert!(prompt.contains("宿主未指定说话人"));
        let model = Script::new(vec![
            done("张三有车", "张三有车"),
            done("张三有车", "张三有车"),
        ]);
        let staged = stage_source_with_model(&model, source, ctx(), StageLimits::default()).await;
        // Evidence outside the block never gets in, however often the model repeats it.
        // The block itself is finished: a third identical answer would change nothing.
        assert!(staged.parts[0].parsed.memories.is_empty());
        assert!(staged.parts[0].parsed.complete);
        assert_eq!(staged.invalid_attempt_records, 2);
    }

    #[tokio::test]
    async fn unicode_source_is_unchanged_and_metadata_and_citations_are_exact() {
        let original = format!("{}\r\n{} 尾\u{301} ", "😀".repeat(700), "乙".repeat(700));
        let src = source(&original);
        let units = extraction_plan::text_units(&original);
        assert_eq!(units.len(), 4);
        let note = ContextNote {
            summary: "第二段承接第一段，仅供消歧".into(),
            refs: vec![0],
        };
        let model = Script::new(vec![
            plan(&[(0, 1), (2, 3)], vec![vec![], vec![note.clone()]]),
            done("😀", "第一段"),
            done("乙", "第二段"),
        ]);
        let staged =
            stage_source_with_model(&model, src.clone(), ctx(), StageLimits::default()).await;
        assert_partition(&staged, &original);
        assert_eq!(src.text, original);
        assert_eq!(staged.parts[1].context["notes"][0]["summary"], note.summary);
        assert_eq!(
            staged.parts[1].context["notes"][0]["citations"][0]["text"],
            units[0].text
        );
        assert_eq!(staged.parts[1].context["span"]["unit"], "unicode_scalar");
        let requests = model.requests();
        for (request, part) in requests
            .iter()
            .filter(|r| r.kind == CallKind::Extraction)
            .zip(&staged.parts)
        {
            let data = request_source(request);
            assert_eq!(data["text"], part.source.text);
            assert_eq!(data["locator"], "page:7");
            assert_eq!(data["graph_already_ingested"], true);
        }
    }

    #[tokio::test]
    async fn planning_failure_does_not_extract_unplanned_window() {
        let model = Script::new(vec![Reply::Text("{\"complete\":false}".into(), false)]);
        let original = "甲".repeat(1_501);
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert_eq!(staged.extraction_calls, 0);
        assert_eq!(staged.parts[0].source.text, original);
        assert_eq!(staged.parts[0].context["planning_required"], true);
        assert!(!staged.parts[0].parsed.complete);
    }

    #[tokio::test]
    async fn deadline_is_shared_and_transport_timeout_is_not_retried() {
        let model = Script::new(vec![Reply::Delay(
            Duration::from_millis(50),
            "{".into(),
            true,
        )]);
        let limits = StageLimits {
            timeout: Duration::from_millis(5),
            ..StageLimits::default()
        };
        let staged = stage_source_with_model(&model, source("甲\n乙"), ctx(), limits).await;
        assert_eq!(staged.extraction_calls, 1);
        assert!(staged.parts[0].error.as_ref().unwrap().contains("deadline"));
        let model = Script::new(vec![Reply::Error("chat request timeout (total deadline)")]);
        let staged =
            stage_source_with_model(&model, source("甲\n乙"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.extraction_calls, 1);
        assert_eq!(staged.planning_calls, 0);
    }

    #[tokio::test]
    async fn retries_receive_remaining_deadline_not_a_new_timeout() {
        let model = Script::new(vec![
            Reply::Delay(Duration::from_millis(10), "{".into(), true),
            done("甲", "完成"),
        ]);
        let limits = StageLimits {
            timeout: Duration::from_secs(1),
            ..StageLimits::default()
        };
        let staged = stage_source_with_model(&model, source("甲"), ctx(), limits).await;
        assert!(staged.parts[0].parsed.complete);
        let requests = model.requests();
        assert!(requests[1].timeout + Duration::from_millis(5) < requests[0].timeout);
    }

    #[tokio::test]
    async fn source_and_original_chunk_call_caps_cover_planning_and_all_children() {
        for limits in [
            StageLimits {
                model_calls: 3,
                ..StageLimits::default()
            },
            StageLimits {
                chunk_calls: 3,
                ..StageLimits::default()
            },
        ] {
            let model = Script::new(vec![
                partial("甲", "父"),
                partial("甲", "父"),
                plan(&[(0, 0), (1, 1)], vec![]),
            ]);
            let staged = stage_source_with_model(&model, source("甲\n乙"), ctx(), limits).await;
            assert_eq!(model.requests().len(), 3);
            assert_eq!(staged.planning_calls + staged.extraction_calls, 3);
            assert!(staged.parts.iter().all(|part| !part.parsed.complete));
            assert_partition(&staged, "甲\n乙");
        }
        let model = Script::default();
        let staged = stage_source_with_model(
            &model,
            source("甲"),
            ctx(),
            StageLimits {
                model_calls: 0,
                ..StageLimits::default()
            },
        )
        .await;
        assert!(model.requests().is_empty());
        assert!(!staged.parts[0].parsed.complete);
    }

    #[tokio::test]
    async fn truncated_original_never_becomes_complete_and_does_not_retry_on_resume() {
        let mut original = source("甲");
        original.truncated = true;
        let model = Script::new(vec![done("甲", "完整回复但原文不完整")]);
        let staged =
            stage_source_with_model(&model, original.clone(), ctx(), StageLimits::default()).await;
        assert!(!staged.parts[0].parsed.complete);
        assert_eq!(staged.extraction_calls, 1);
        assert!(staged.warnings.iter().any(|w| w.contains("原始输入已截断")));
        let resumed =
            resume_source_with_model(&model, original, ctx(), staged, StageLimits::default()).await;
        assert!(!resumed.parts[0].parsed.complete);
        assert_eq!(model.requests().len(), 1);
    }

    #[tokio::test]
    async fn incomplete_source_is_staged_only_complete_sibling_controls_survive_explicit_resume() {
        let original = format!("{}\n{}", "甲".repeat(700), "乙".repeat(700));
        let controls = Reply::Text(
            format!(
                "{}\n{}\n{{\"done\":true}}",
                json!({"k":"r","n":"甲","r":"other","e":"甲"}),
                json!({"k":"f","o":"self","c":"撤回甲","e":"甲"})
            ),
            false,
        );
        let note = ContextNote {
            summary: "第一段是甲，第二段不是默认归甲".into(),
            refs: vec![0],
        };
        let model = Script::new(vec![
            plan(&[(0, 1), (2, 3)], vec![vec![], vec![note]]),
            controls,
            Reply::Error("chat http 500"),
        ]);
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert!(staged.parts[0].parsed.complete);
        assert!(!staged.parts[1].parsed.complete);
        assert_eq!(staged.parts[0].parsed.registers.len(), 1);
        assert_eq!(staged.parts[0].parsed.retractions.len(), 1);
        assert!(staged.warnings.iter().any(|w| w.contains("均不得执行")));
        let resume = Script::new(vec![done("乙", "补全乙")]);
        let expected_context = staged.parts[1].context.clone();
        let staged = resume_source_with_model(
            &resume,
            source(&original),
            ctx(),
            staged,
            StageLimits::default(),
        )
        .await;
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        assert_eq!(resume.requests().len(), 1);
        assert_eq!(staged.parts[0].parsed.registers.len(), 1);
        assert_eq!(staged.parts[0].parsed.retractions.len(), 1);
        assert_eq!(staged.parts[1].context, expected_context);
        assert_partition(&staged, &original);
    }

    #[tokio::test]
    async fn incomplete_children_retain_parent_salvage_only_in_unresolved_spans() {
        let model = Script::new(vec![
            partial("甲", "不用的父甲"),
            partial("乙", "可用的父乙"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "完整甲"),
            Reply::Error("chat http 500"),
        ]);
        let staged =
            stage_source_with_model(&model, source("甲\n乙"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.parts.len(), 2);
        assert!(staged.parts[0].parsed.complete);
        assert_eq!(staged.parts[0].parsed.memories[0]["content"], "完整甲");
        assert!(!staged.parts[1].parsed.complete);
        assert_eq!(staged.parts[1].parsed.memories.len(), 1);
        assert_eq!(staged.parts[1].parsed.memories[0]["content"], "可用的父乙");
        assert_partition(&staged, "甲\n乙");
    }

    #[tokio::test]
    async fn cross_boundary_salvage_collapses_unresolved_subtree_without_overlapping_parts() {
        let model = Script::new(vec![
            partial("甲\n乙", "跨界证据"),
            partial("甲\n乙", "跨界证据"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "完整甲"),
            Reply::Error("chat http 500"),
        ]);
        let staged =
            stage_source_with_model(&model, source("甲\n乙"), ctx(), StageLimits::default()).await;
        assert_eq!(staged.parts.len(), 1);
        assert!(!staged.parts[0].parsed.complete);
        assert_eq!(staged.parts[0].parsed.memories.len(), 2);
        assert_eq!(
            staged.parts[0].context["recovery_parts"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_partition(&staged, "甲\n乙");
    }

    #[tokio::test]
    async fn resume_rejects_modified_original_and_checkpoint_roundtrips() {
        let model = Script::new(vec![Reply::Error("network failure")]);
        let staged =
            stage_source_with_model(&model, source("原始甲"), ctx(), StageLimits::default()).await;
        let serialized = serde_json::to_string(&staged).unwrap();
        let previous: StagedSource = serde_json::from_str(&serialized).unwrap();
        let model = Script::default();
        let staged = resume_source_with_model(
            &model,
            source("变更乙"),
            ctx(),
            previous,
            StageLimits::default(),
        )
        .await;
        assert!(model.requests().is_empty());
        assert!(!staged.parts[0].parsed.complete);
        assert!(staged.parts[0]
            .error
            .as_ref()
            .unwrap()
            .contains("does not match"));
    }

    #[tokio::test]
    async fn recovery_inherits_grounded_context_even_when_planner_omits_future_ref() {
        let original = format!("{}\n{}", "甲".repeat(700), "乙".repeat(700));
        let inherited = ContextNote {
            summary: "后文明确说明身份，不能把所有事实归默认人物".into(),
            refs: vec![2],
        };
        let model = Script::new(vec![
            plan(&[(0, 1), (2, 3)], vec![vec![inherited.clone()], vec![]]),
            partial("甲", "部分"),
            partial("甲", "部分"),
            plan(&[(0, 0), (1, 1)], vec![]),
            done("甲", "恢复甲一"),
            done("甲", "恢复甲二"),
            done("乙", "乙"),
        ]);
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        assert_eq!(staged.parts.len(), 3);
        assert_eq!(
            staged.parts[0].context["notes"][0]["summary"],
            inherited.summary
        );
        assert_eq!(staged.parts[1].context["notes"][0]["refs"], json!([2]));
        assert!(model.requests()[3]
            .prompt
            .contains("inherited_chunk_context"));
        assert_partition(&staged, &original);
    }

    fn window_units(original: &str) -> Vec<Vec<TextUnit>> {
        let units = source_units(original);
        hard_windows(original.chars().count())
            .into_iter()
            .map(|(start, end)| {
                units
                    .iter()
                    .filter(|unit| unit.start >= start && unit.end <= end)
                    .cloned()
                    .collect()
            })
            .collect()
    }

    fn full_window_plan(
        window: &[TextUnit],
        context: Vec<ContextNote>,
        carry: Vec<ContextNote>,
    ) -> Reply {
        Reply::Text(
            json!({"chunks":[{"first":window[0].id,"last":window.last().unwrap().id,
            "context":context,"estimated_facts":5}],"carry":carry,"complete":true})
            .to_string(),
            false,
        )
    }

    #[test]
    fn hard_windows_and_unicode_units_preserve_exact_2500_100_geometry() {
        assert_eq!(hard_windows(0), vec![(0, 0)]);
        assert_eq!(hard_windows(300), vec![(0, 300)]);
        assert_eq!(hard_windows(2500), vec![(0, 2500)]);
        assert_eq!(hard_windows(2501), vec![(0, 2500), (2400, 2501)]);
        assert_eq!(hard_windows(4900), vec![(0, 2500), (2400, 4900)]);
        assert_eq!(
            hard_windows(4901),
            vec![(0, 2500), (2400, 4900), (4800, 4901)]
        );
        for original in [
            "😀甲\r\n".repeat(1300),
            "甲\n".repeat(60_000),
            "x".repeat(120_000),
        ] {
            let units = source_units(&original);
            assert_eq!(
                units
                    .iter()
                    .map(|unit| unit.text.as_str())
                    .collect::<String>(),
                original
            );
            for (index, unit) in units.iter().enumerate() {
                assert_eq!(unit.id, index);
                assert!(unit.end - unit.start <= 500);
                assert_eq!(unit.text, exact_slice(&original, unit.start, unit.end));
            }
            for window in window_units(&original) {
                assert!(
                    window.len() <= 24,
                    "fragmentation must not add planning calls"
                );
                let raw = match full_window_plan(&window, vec![], vec![]) {
                    Reply::Text(raw, _) => raw,
                    _ => unreachable!(),
                };
                assert!(extraction_plan::parse_plan(&raw, &window, &[], &units).is_ok());
            }
        }
    }

    #[tokio::test]
    async fn window_planning_and_extraction_are_interleaved_with_exact_overlap_and_carry() {
        let original = "甲\n".repeat(3000);
        let windows = window_units(&original);
        let mut replies = Vec::new();
        for window in &windows {
            replies.push(full_window_plan(
                window,
                vec![],
                vec![ContextNote {
                    summary: "仅供消歧的上一窗口原文".into(),
                    refs: vec![window[0].id],
                }],
            ));
            replies.push(done("甲", "甲"));
        }
        let model = Script::new(replies);
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert!(staged.parts.iter().all(|part| part.parsed.complete));
        let requests = model.requests();
        assert_eq!(requests.len(), windows.len() * 2);
        for (index, window) in windows.iter().enumerate() {
            assert_eq!(requests[2 * index].kind, CallKind::Planning);
            assert_eq!(requests[2 * index + 1].kind, CallKind::Extraction);
            let data: Value = serde_json::from_str(&requests[2 * index].prompt).unwrap();
            assert_eq!(data["window"][0]["id"], window[0].id);
            let text: String = data["window"]
                .as_array()
                .unwrap()
                .iter()
                .map(|u| u["text"].as_str().unwrap())
                .collect();
            let (start, end) = hard_windows(original.chars().count())[index];
            assert_eq!(text, exact_slice(&original, start, end));
            assert_eq!(request_source(&requests[2 * index + 1])["text"], text);
            if index > 0 {
                assert_eq!(
                    data["prior_context"][0]["refs"][0],
                    windows[index - 1][0].id
                );
                let left = &staged.parts[index - 1].source.text;
                assert_eq!(
                    left.chars()
                        .rev()
                        .take(100)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect::<String>(),
                    text.chars().take(100).collect::<String>()
                );
            }
            assert_eq!(staged.parts[index].context["hard_window"]["index"], index);
        }
        assert_partition(&staged, &original);
    }

    #[tokio::test]
    async fn exactly_2500_plans_once_and_2501_forces_planning_for_short_tail() {
        for count in [2499, 2500, 2501, 4900, 4901] {
            let original = "😀".repeat(count);
            let windows = window_units(&original);
            let replies = windows
                .iter()
                .flat_map(|w| vec![full_window_plan(w, vec![], vec![]), done("😀", "事实")])
                .collect();
            let model = Script::new(replies);
            let staged =
                stage_source_with_model(&model, source(&original), ctx(), StageLimits::default())
                    .await;
            assert_eq!(staged.planning_calls, hard_windows(count).len());
            assert_eq!(staged.extraction_calls, hard_windows(count).len());
            assert!(staged.parts.iter().all(|part| part.parsed.complete));
            assert_partition(&staged, &original);
        }
    }

    #[tokio::test]
    async fn failed_window_blocks_carry_chain_and_repeated_resume_preserves_new_exact_refs() {
        let original = "甲".repeat(6000);
        let units = source_units(&original);
        let windows = window_units(&original);
        assert_eq!(windows.len(), 3);
        let note = |index: usize| ContextNote {
            summary: format!("窗口{index}已验证消歧"),
            refs: vec![windows[index][0].id],
        };
        let first = Script::new(vec![
            full_window_plan(&windows[0], vec![], vec![note(0)]),
            done("甲", "首窗口结果"),
            Reply::Text("{\"complete\":false}".into(), false),
        ]);
        let staged =
            stage_source_with_model(&first, source(&original), ctx(), StageLimits::default()).await;
        assert_eq!(staged.parts.len(), 3);
        assert!(staged.parts[0].parsed.complete);
        assert!(staged.parts[1..]
            .iter()
            .all(|p| !p.parsed.complete && p.context["planning_required"] == true));
        assert_eq!((staged.planning_calls, staged.extraction_calls), (2, 1));
        let first_saved = serde_json::to_value(&staged.parts[0]).unwrap();
        let resume = Script::new(vec![
            full_window_plan(&windows[1], vec![note(0)], vec![note(1)]),
            done("甲", "第二窗口结果"),
            Reply::Text("{\"complete\":false}".into(), false),
        ]);
        let staged = resume_source_with_model(
            &resume,
            source(&original),
            ctx(),
            staged,
            StageLimits::default(),
        )
        .await;
        assert!(staged.parts[0].parsed.complete && staged.parts[1].parsed.complete);
        assert!(!staged.parts[2].parsed.complete);
        assert_eq!(serde_json::to_value(&staged.parts[0]).unwrap(), first_saved);
        assert_eq!(
            staged.parts[2].context["notes"][0]["refs"],
            json!(note(1).refs)
        );
        let citation = &staged.parts[2].context["notes"][0]["citations"][0];
        let unit = &units[windows[1][0].id];
        assert_eq!(citation["text"], unit.text);
        assert_eq!(citation["start"], unit.start);
        assert_eq!(citation["end"], unit.end);
        let second_saved = serde_json::to_value(&staged.parts[1]).unwrap();
        let checkpoint: StagedSource =
            serde_json::from_str(&serde_json::to_string(&staged).unwrap()).unwrap();
        let last = Script::new(vec![
            full_window_plan(&windows[2], vec![note(1)], vec![note(2)]),
            done("甲", "第三窗口结果"),
        ]);
        let complete = resume_source_with_model(
            &last,
            source(&original),
            ctx(),
            checkpoint,
            StageLimits::default(),
        )
        .await;
        assert!(complete.parts.iter().all(|p| p.parsed.complete));
        assert_eq!(
            serde_json::to_value(&complete.parts[0]).unwrap(),
            first_saved
        );
        assert_eq!(
            serde_json::to_value(&complete.parts[1]).unwrap(),
            second_saved
        );
        assert_eq!(last.requests().len(), 2);
        let prompt: Value = serde_json::from_str(&last.requests()[0].prompt).unwrap();
        assert_eq!(prompt["prior_context"][0]["refs"], json!(note(1).refs));
        assert_partition(&complete, &original);
    }

    #[tokio::test]
    async fn extraction_failure_resumes_only_failed_hard_window_without_replanning() {
        let original = "甲".repeat(2501);
        let windows = window_units(&original);
        let initial = Script::new(vec![
            full_window_plan(&windows[0], vec![], vec![]),
            done("甲", "完整首窗"),
            full_window_plan(&windows[1], vec![], vec![]),
            Reply::Error("network failure"),
        ]);
        let staged =
            stage_source_with_model(&initial, source(&original), ctx(), StageLimits::default())
                .await;
        assert!(staged.parts[0].parsed.complete);
        assert!(!staged.parts[1].parsed.complete);
        let first = serde_json::to_value(&staged.parts[0]).unwrap();
        let resume = Script::new(vec![done("甲", "补全尾窗")]);
        let complete = resume_source_with_model(
            &resume,
            source(&original),
            ctx(),
            staged,
            StageLimits::default(),
        )
        .await;
        assert_eq!(resume.requests().len(), 1);
        assert_eq!(resume.requests()[0].kind, CallKind::Extraction);
        assert_eq!(
            request_source(&resume.requests()[0])["source_id"],
            "root@w1:2400-2501"
        );
        assert_eq!(serde_json::to_value(&complete.parts[0]).unwrap(), first);
        assert_partition(&complete, &original);
    }

    #[tokio::test]
    async fn coverage_rejects_missing_duplicate_or_forged_window_but_allows_exact_overlap() {
        let original = "甲".repeat(2501);
        let windows = window_units(&original);
        let model = Script::new(
            windows
                .iter()
                .flat_map(|w| vec![full_window_plan(w, vec![], vec![]), done("甲", "事实")])
                .collect(),
        );
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert!(valid_staged_source(&source(&original), &staged));
        let mut reversed = staged.clone();
        reversed.parts.reverse();
        assert!(valid_staged_source(&source(&original), &reversed));
        let mut bad = staged.clone();
        bad.parts.pop();
        assert!(!valid_staged_source(&source(&original), &bad));
        let mut bad = staged.clone();
        bad.parts.push(bad.parts[0].clone());
        assert!(!valid_staged_source(&source(&original), &bad));
        let mut bad = staged.clone();
        bad.parts[1].context["hard_window"]["index"] = json!(0);
        assert!(!valid_staged_source(&source(&original), &bad));
        let mut bad = staged.clone();
        bad.parts[1].source.text = "伪造".into();
        assert!(!valid_staged_source(&source(&original), &bad));
        let mut bad = staged.clone();
        bad.parts[1].source.id = bad.parts[0].source.id.clone();
        assert!(!valid_staged_source(&source(&original), &bad));
        for mutation in 0..8 {
            let mut bad = staged.clone();
            let part = &mut bad.parts[1];
            match mutation {
                0 => part.context["hard_window"]["overlap_chars"] = json!(99),
                1 => part.source.kind = "user_text".into(),
                2 => part.source.graph_already_ingested = !part.source.graph_already_ingested,
                3 => part.source.truncated = !part.source.truncated,
                4 => part.source.locator = Some("forged-page".into()),
                5 => part.source.name = "forged-name".into(),
                6 => part.context["root_source_id"] = json!("different-source"),
                _ => part.context["hard_window"]["end"] = json!(2500),
            }
            assert!(
                !valid_staged_source(&source(&original), &bad),
                "tampered checkpoint case {mutation}"
            );
        }
        assert!(!valid_staged_source(
            &source(&original),
            &StagedSource::default()
        ));
    }

    #[tokio::test]
    async fn production_96_call_cap_counts_all_initial_plans_and_extractions() {
        let original = "甲".repeat(120_000);
        let windows = window_units(&original);
        let model = Script::new(
            windows
                .iter()
                .take(48)
                .flat_map(|w| vec![full_window_plan(w, vec![], vec![]), done("甲", "事实")])
                .collect(),
        );
        let staged =
            stage_source_with_model(&model, source(&original), ctx(), StageLimits::default()).await;
        assert_eq!(model.requests().len(), 96);
        assert_eq!(staged.planning_calls + staged.extraction_calls, 96);
        assert!(staged.parts.iter().any(|p| !p.parsed.complete));
        assert_partition(&staged, &original);
    }

    #[tokio::test]
    async fn prompt_limit_fails_closed_without_sending_oversized_context() {
        let mut context = ctx();
        context.roster = "甲".repeat(MAX_PROMPT_CHARS);
        let model = Script::default();
        let staged =
            stage_source_with_model(&model, source("甲"), context, StageLimits::default()).await;
        assert!(model.requests().is_empty());
        assert!(!staged.parts[0].parsed.complete);
        assert!(staged.parts[0]
            .error
            .as_ref()
            .unwrap()
            .contains("context limit exceeded"));
    }

    #[tokio::test]
    async fn zero_deadline_does_not_start_a_model_call() {
        let model = Script::default();
        let staged = stage_source_with_model(
            &model,
            source("甲"),
            ctx(),
            StageLimits {
                timeout: Duration::ZERO,
                ..StageLimits::default()
            },
        )
        .await;
        assert!(model.requests().is_empty());
        assert!(!staged.parts[0].parsed.complete);
    }

    #[tokio::test]
    async fn whitespace_sources_never_call_a_model() {
        let model = Script::default();
        let root = source("  \n\t ");
        let staged = stage_source_with_model(&model, root.clone(), ctx(), StageLimits::default()).await;
        assert!(model.requests().is_empty());
        assert!(valid_staged_source(&root, &staged));
    }

    #[test]
    fn prompt_preserves_versioned_contract_context_tail_date_and_speaker() {
        let mut context = ctx();
        context.transcript = format!("REMOVE_PREFIX{}", "尾".repeat(1_200));
        context.speaker = "member-7".into();
        let prompt = extraction_prompt(&source("  甲\r\n🙂  "), &context, &json!({}));
        assert!(prompt.starts_with(crate::prompt_latest().extract_rules));
        assert!(prompt.contains("1970-01-01 08:00:00 +08:00"));
        assert!(!prompt.contains("REMOVE_PREFIX"));
        assert!(prompt.contains("speaker_id：member-7"));
        assert!(prompt.contains("不是默认归属"));
        assert!(prompt.contains("  甲\\r\\n🙂  "));
    }
    #[tokio::test]
    async fn user_no_store_and_own_forgetting_do_not_call_the_model() {
        let model=Script::new(vec![]);
        let mut src=source("不要记住这句话，我今天有点难过");src.kind="user_text".into();
        let staged=stage_source_with_model(&model,src.clone(),ctx(),StageLimits::default()).await;
        assert!(valid_staged_source(&src,&staged));
        assert!(staged.parts[0].parsed.complete);assert!(staged.parts[0].parsed.memories.is_empty());
        let mut src=source("忘掉我喜欢游泳这件事");src.kind="user_text".into();
        let mut context=ctx();context.speaker="member-1".into();context.roster=json!([{"member_id":"member-1","display_name":"王明"}]).to_string();
        let staged=stage_source_with_model(&model,src.clone(),context,StageLimits::default()).await;
        assert!(valid_staged_source(&src,&staged));assert!(staged.parts[0].parsed.complete);
        assert!(staged.parts[0].parsed.memories.is_empty());assert_eq!(staged.parts[0].parsed.retractions.len(),1);
        assert_eq!(staged.parts[0].parsed.retractions[0]["owner"],"member-1");
        assert_eq!(staged.parts[0].parsed.retractions[0]["content"],"王明喜欢游泳");
        assert!(model.requests().is_empty());
    }
    #[tokio::test]
    async fn unknown_identity_or_truncated_source_cannot_retract_personal_facts() {
        let model=Script::new(vec![]);
        let mut src=source("忘掉我喜欢游泳这件事");src.kind="user_text".into();
        let staged=stage_source_with_model(&model,src.clone(),ctx(),StageLimits::default()).await;
        assert!(staged.parts[0].parsed.retractions.is_empty());
        src.truncated=true;let mut context=ctx();context.speaker="王明".into();
        let staged=stage_source_with_model(&model,src,context,StageLimits::default()).await;
        assert!(!staged.parts[0].parsed.complete);assert!(staged.parts[0].parsed.retractions.is_empty());
        assert!(model.requests().is_empty());
    }
    #[tokio::test]
    async fn document_commands_are_data_and_do_not_execute_host_intents() {
        let model=Script::new(vec![done("例句","这是一份说明")]);
        let src=source("不要记住这句话。这里是一个例句。");
        let staged=stage_source_with_model(&model,src,ctx(),StageLimits::default()).await;
        assert_eq!(model.requests().len(),1);assert_eq!(staged.parts[0].parsed.memories.len(),1);
    }

}

fn edge_markup_only(text: &str) -> bool {
    let trimmed=text.trim();
    // Spatial ASCII diagrams do not establish a typed family relation. A prose
    // statement remains eligible in its own range, without guessing from layout.
    if trimmed.starts_with("```") && trimmed.ends_with("```") && text.chars().any(|c|matches!(c,'─'|'│'|'└'|'┌')) {return true;}
    text.lines().all(|line| {
        let line=line.trim();
        line.is_empty() || line.starts_with('#') || line.starts_with("```")
            || (line.starts_with('|') && line.chars().all(|c|matches!(c,'|'|'-'|':'|' ')))
    })
}
fn edge_name_only(item:&Value,roster:&str)->bool {
    let normalize=|value:&str|value.chars().filter(|c|!c.is_whitespace() && !matches!(c,'。'|'.'|'：'|':'|'；'|';'|'，'|','|'“'|'”'|'"')).collect::<String>();
    let content=normalize(item["content"].as_str().unwrap_or(""));
    let owner=item["owner"].as_str().unwrap_or("");
    if content.is_empty() || content==normalize(owner) {return true;}
    let members:Value=serde_json::from_str(roster).unwrap_or(json!([]));
    members.as_array().into_iter().flatten().filter(|m|m["member_id"]==owner || m["display_name"]==owner
        || m["aliases"].as_array().is_some_and(|a|a.iter().any(|v|v==owner))).any(|m| {
        m["display_name"].as_str().is_some_and(|name|content==normalize(name))
            || m["aliases"].as_array().is_some_and(|a|a.iter().filter_map(Value::as_str).any(|name|content==normalize(name)))
    })
}
fn edge_line_ranges(text: &str, cap: usize) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let cap=cap.max(1);
    let mut ranges = Vec::new();
    let push = |mut start: usize, end: usize, out: &mut Vec<(usize,usize)>| {
        while end.saturating_sub(start) > cap {
            let limit=start+cap;
            let cut=(start+cap/2..limit).rev().find(|&i|matches!(chars[i],'。'|'；'|'\n'))
                .map(|i|i+1).unwrap_or(limit);
            out.push((start,cut));start=cut;
        }
        if end>start {out.push((start,end));}
    };
    let mut cursor=0;let mut fence_start=0;let mut in_fence=false;
    for line in text.split_inclusive('\n') {
        let start=cursor;cursor+=line.chars().count();
        if line.trim_start().starts_with("```") {
            if !in_fence {fence_start=start;in_fence=true;continue;}
            in_fence=false;push(fence_start,cursor,&mut ranges);continue;
        }
        if !in_fence {
            if line.trim_start().starts_with('|') || line.trim_start().starts_with('#') {
                push(start,cursor,&mut ranges);
            } else {
                let mut sentence=start;
                for i in start..cursor {
                    if chars[i]=='。' {push(sentence,i+1,&mut ranges);sentence=i+1;}
                }
                if sentence<cursor {push(sentence,cursor,&mut ranges);}
            }
        }
    }
    if in_fence {push(fence_start,chars.len(),&mut ranges);}
    ranges
}
fn edge_table_header(root: &str, start: usize) -> String {
    let prefix: String=root.chars().take(start).collect();
    let mut previous=String::new();let mut header=String::new();
    for line in prefix.lines() {
        let t=line.trim();
        if !t.starts_with('|') {previous.clear();header.clear();continue;}
        let cells: Vec<&str>=t.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len()>1 && cells.iter().all(|c|!c.is_empty() && c.chars().all(|v|matches!(v,'-'|':'|' '))) {
            header=previous.clone();
        }
        previous=t.to_string();
    }
    header.chars().take(240).collect()
}
fn edge_table_subject(root: &str, start: usize, text: &str, roster: &str) -> Option<(String,String)> {
    let header=edge_table_header(root,start);
    let heads:Vec<_>=header.trim().trim_matches('|').split('|').map(str::trim).collect();
    let row=text.lines().find(|line|line.trim_start().starts_with('|'))?;
    if text.lines().filter(|line|line.trim_start().starts_with('|')).count()!=1 {return None;}
    let cells:Vec<_>=row.trim().trim_matches('|').split('|').map(str::trim).collect();
    if heads.len()!=cells.len() {return None;}
    let col=heads.iter().position(|h|["姓名","患者","患者姓名","学生","学生姓名","人物","成员","成员姓名","当前人物基准","持有人","ID与持有人"].contains(h))?;
    let cell=cells[col].replace("<br />","<br>").replace("<br/>","<br>");
    let person=if heads[col].contains("持有人") {cell.split("<br>").last()?}else{cell.as_str()};
    let name=person.split(['；',';','，',',','（','(']).next()?.trim();
    if let Some(owner)=crate::edge_table::resolve_subject(root,name,roster) {return Some(owner);}
    if heads[col].contains("持有人") {return None;}
    if name.is_empty() || name.chars().count()>24 || name.chars().any(|c|c.is_ascii_digit()) {return None;}
    Some((name.into(),name.into()))
}
fn edge_name_context(root: &str) -> String {
    root.lines().filter(|line| {
        let t=line.trim();
        (t.starts_with('#') || (!t.starts_with('|') && (t.contains("称呼") || t.contains("别名") || ["姓名：","姓名:","患者：","患者:","学生：","学生:"].iter().any(|prefix|t.starts_with(prefix)))))
            && !t.chars().any(|c|c.is_ascii_digit())
            && !t.contains('元') && !t.contains("如果") && !t.contains("假设")
    }).take(3).collect::<Vec<_>>().join("\n").chars().take(240).collect()
}
fn edge_reference_history(text: &str, ctx: &ExtractContext) -> String {
    let text=text.trim();
    let needs=["那", "这个", "那个", "他", "她", "它", "改到", "改成", "换成", "也一样"]
        .iter().any(|prefix|text.starts_with(prefix)) || text.contains("也一样");
    if !needs {return String::new();}
    ctx.transcript.lines().rev().filter_map(|line| {
        let t=line.trim();
        ["用户：", "用户:", "user:", "User:"].iter().find_map(|prefix|t.strip_prefix(prefix))
    }).map(str::trim).find(|line|!line.is_empty() && *line!=text)
        .map(|line|line.chars().take(80).collect()).unwrap_or_default()
}
fn edge_extraction_prompt(source: &EvidenceSource, ctx: &ExtractContext, root: &str, start: usize) -> String {
    let is_chat=source.kind=="user_text";
    let roster: Value=serde_json::from_str(&ctx.roster).unwrap_or(json!([]));
    let person_context=edge_person_context(root,start);
    let context_people=crate::edge_table::referenced_ids(root,&person_context,&ctx.roster);
    let document_subject=if let Some(name)=person_context.strip_prefix("姓名：") {
        crate::edge_table::resolve_subject(root,name,&ctx.roster).map(|(id,name)|json!({"id":id,"name":name}))
            .unwrap_or_else(||json!({"id":name,"name":name}))
    } else if context_people.len()==1 {
        roster.as_array().into_iter().flatten().find(|m|m["member_id"].as_str()==Some(context_people[0].as_str()))
            .map(|m|json!({"id":m["member_id"],"name":m["display_name"]})).unwrap_or(Value::Null)
    } else {Value::Null};
    let mut referenced=crate::edge_table::referenced_ids(root,&source.text,&ctx.roster);
    referenced.extend(crate::edge_table::referenced_ids(root,&person_context,&ctx.roster));
    referenced.sort();referenced.dedup();
    let mut members: Vec<Value>=roster.as_array().into_iter().flatten().filter(|r| {
        is_chat || r["member_id"].as_str().is_some_and(|id|referenced.iter().any(|value|value==id))
    }).map(|r|json!({"id":r["member_id"],"name":r["display_name"],"role":r["role"],"aliases":r["aliases"]})).collect();
    members.sort_by(|a,b|a["id"].as_str().cmp(&b["id"].as_str()));
    let catalog: Value=serde_json::from_str(&ctx.tags).unwrap_or(json!([]));
    let mut tags:Vec<&str>=catalog.as_array().into_iter().flatten().filter_map(|t|t["tag_id"].as_str()).collect();tags.sort();
    let rules=r#"只提取用户消息原文中的事实与确定计划。只输出JSON：{"records":[{"o":"归属者","t":"标签","c":"完整事实","e":"逐字原文"}],"done":true}。
o只能是人物或公共，不能用药品、机构、商户、物品、字段名或编号造人物。已明确的新人物可用姓名或有说话人锚点的亲属称呼。user_text中我指speaker，未知的我不归任何人；全家共同事项归公共。亲属相对说话人，不能把本人当自己的妈妈。文档不属于上传者；疾病和用药归患者，共同车辆资料归公共，明确某人时归本人。正文省略主语可用document_subject，明确其他人时以正文为准。
t只用tags中的标签：食物饮品口味（包括不吃不喝）用饮食；兴趣影视玩具用娱乐；职业学校课程学习用工作学习；疾病过敏检查药物用医疗健康；收入消费账单投资预算用金融；通勤交通旅行驾驶证车辆用出行；作息运动家务用生活习惯；关系生日及其余用其他。
c保留具体地点、数值、日期、否定和状态。同一事件的时间地点等细节保留在一条，不同兴趣或独立事项分条。文档的课程、交易、病例、比较资料、待办和计划都可记；报价不能说成已付款，候选不能说成已报名，计划不能说成已完成。表头和section可解释本行，不能漏本行中间字段。e逐字取当前用户原文连续片段，不引示例、系统参考或历史；参考仅消歧。
仅标题姓名、闲聊、假设、问句请求本身不记，请求中明确的事实可提取。密码验证码密钥不记，wifi名称可单独记。
示例原文：赵颖是陈凯的女儿。陈凯每月收入9000元。
示例输出：{"records":[{"o":"赵颖","t":"其他","c":"赵颖是陈凯的女儿","e":"赵颖是陈凯的女儿"},{"o":"陈凯","t":"金融","c":"陈凯每月收入9000元","e":"陈凯每月收入9000元"}],"done":true}
夫妻照抄夫妻，不推性别或改父子方向。无事实输出{"records":[],"done":true}。仅用户明确要求遗忘可用k=f，不能把命令写成新增事实。"#;
    let speaker_hint = if is_chat && !ctx.speaker.is_empty() {
        format!("当前声纹已识别，说话者ID为{}。用户说的我/对我/我的均指{}，不要把食物或活动名当人，也不要归公共。例：原文‘我休息时常听音乐’，输出{{\"records\":[{{\"o\":\"{}\",\"t\":\"娱乐\",\"c\":\"休息时常听音乐\",\"e\":\"我休息时常听音乐\"}}],\"done\":true}}。示例不入库，只抽当前用户原文。\n", ctx.speaker, ctx.speaker, ctx.speaker)
    } else { String::new() };
    format!("{}{}\n参考信息（不是待抽取原文）={}",speaker_hint,rules,json!({
        "kind":source.kind,"source_id":source.id,"members":members,"tags":tags,
        "speaker":if is_chat {json!(ctx.speaker)} else {Value::Null},
        "message_at_ms":if is_chat {json!(ctx.message_at_ms)} else {Value::Null},
        "history_only_for_names":if is_chat {edge_reference_history(&source.text,ctx)}else{String::new()},
        "document_name_context_only":if is_chat {String::new()}else{person_context},
        "document_subject":if is_chat {Value::Null}else{document_subject},
        "document_aliases_only":if is_chat {json!({})}else{crate::edge_table::span_aliases(root,&source.text,&ctx.roster)},
        "section_title_only":root.chars().take(start).collect::<String>().lines().rev().find(|line|line.trim_start().starts_with('#')).unwrap_or("").chars().take(100).collect::<String>(),
        "table_header_only":edge_table_header(root,start)
    }))
}

fn edge_person_context(root:&str,start:usize)->String {
    let prefix:String=root.chars().take(start).collect();
    let mut headings:Vec<(usize,String)>=Vec::new();
    let mut person=None;
    for line in prefix.lines() {
        let text=line.trim();
        if text.starts_with('#') {
            let level=text.chars().take_while(|c|*c=='#').count();
            while headings.last().is_some_and(|(depth,_)|*depth>=level) {headings.pop();}
            headings.push((level,text.to_owned()));
        } else if let Some((key,value))=text.split_once('：').or_else(||text.split_once(':')) {
            if ["姓名","患者","学生","人物"].contains(&key.trim()) {
                let name=value.split(['；',';','，',',','（','(',' ','\t']).next().unwrap_or("").trim();
                if !name.is_empty() && name.chars().count()<=24 {person=Some(format!("姓名：{name}"));}
            }
        }
    }
    person.unwrap_or_else(||headings.into_iter().map(|(_,heading)|heading)
        .filter(|heading|!heading.chars().any(|c|c.is_ascii_digit()) && !heading.contains('元'))
        .collect::<Vec<_>>().join("\n")).chars().take(180).collect()
}

/// Typed envelope rejects duplicate records/done keys instead of silently
/// discarding all but the last array. Invalid replies follow normal recovery.
fn edge_records_to_lines(text: &str) -> String {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Envelope { records: Vec<Value>, done: bool }
    match serde_json::from_str::<Envelope>(text) {
        Ok(value) => {
            let mut lines:Vec<String>=value.records.iter().map(Value::to_string).collect();
            if value.done { lines.push(json!({"done":true}).to_string()); }
            lines.join("\n")
        }
        Err(_) => text.to_owned(),
    }
}

#[cfg(test)]
mod edge_protocol_regressions {
    use super::*;
    #[test]
    fn duplicate_envelopes_are_not_accepted_as_complete() {
        let duplicate=r#"{"records":[{"o":"甲","t":"其他","c":"甲怕冷","e":"甲怕冷"}],"records":[],"done":true}"#;
        assert!(!parse_extraction(&edge_records_to_lines(duplicate),"s","甲怕冷",false).complete);
        let complete=r#"{"records":[{"o":"甲","t":"其他","c":"甲怕冷","e":"甲怕冷"}],"done":true}"#;
        assert!(parse_extraction(&edge_records_to_lines(complete),"s","甲怕冷",false).complete);
    }
}
#[cfg(test)]
mod edge_line_regressions {
    use super::*;
    #[test]
    fn short_multiline_profiles_keep_subject_and_values_together() {
        let text="姓名：王明\n出生日期：1990年4月17日\n职业：软件工程师\n";
        assert_eq!(edge_line_ranges(text,320).iter().map(|(a,b)|exact_slice(text,*a,*b)).collect::<String>(),text);
        assert!(edge_name_context(text).contains("姓名："));
        assert!(edge_name_context(text).contains("姓名：王明"));
        assert!(!edge_name_context(text).contains("1990"));
    }
    #[test]
    fn ranges_preserve_complete_rows_and_every_unicode_character() {
        let text="# 资料\n| 姓名 | 收入 |\n| --- | --- |\n| 王明 | 每月7000元；软件工程师 |\n\n王明是王禾的父亲，也是王木的祖父。\n🙂尾";
        let chars:Vec<char>=text.chars().collect();let ranges=edge_line_ranges(text,320);
        let rebuilt:String=ranges.iter().map(|(s,e)|chars[*s..*e].iter().collect::<String>()).collect();
        assert_eq!(rebuilt,text);assert_eq!(ranges.first().unwrap().0,0);assert_eq!(ranges.last().unwrap().1,chars.len());
        for pair in ranges.windows(2){assert_eq!(pair[0].1,pair[1].0);}
        assert!(ranges.iter().any(|(s,e)|chars[*s..*e].iter().collect::<String>().contains("王明是王禾的父亲，也是王木的祖父")));
    }
    #[test]
    fn long_and_unclosed_code_blocks_stay_bounded_without_losing_coverage() {
        let text=format!("```text\n{}\n", "甲🙂".repeat(500));let chars:Vec<char>=text.chars().collect();
        let ranges=edge_line_ranges(&text,320);assert!(ranges.iter().all(|(s,e)|e-s<=320));
        assert_eq!(ranges.iter().map(|(s,e)|chars[*s..*e].iter().collect::<String>()).collect::<String>(),text);
    }
    #[test]
    fn header_is_local_and_name_context_excludes_other_amounts() {
        let root="# 家庭收支\n本文件称呼：爸爸是王明。\n上一期收入99999元。\n| 项目 | 金额 元 |\n| --- | --- |\n| 餐饮 | 80 |\n\n尾段";
        let row_start=root[..root.find("| 餐饮").unwrap()].chars().count();
        assert_eq!(edge_table_header(root,row_start),"| 项目 | 金额 元 |");
        assert!(!edge_name_context(root).contains("99999"));assert!(edge_name_context(root).contains("爸爸是王明"));
        assert!(edge_table_header(root,root.chars().count()).is_empty());
    }

    #[test]
    fn complete_current_claims_never_receive_unrelated_speaker_history() {
        let ctx=ExtractContext {roster:"[]".into(),tags:"[]".into(),speaker:"王明".into(),message_at_ms:0,
            transcript:"用户：我喜欢喝拿铁\n用户：王明周三去上海\n助手：好的。\n用户：那改到周四".into()};
        assert!(edge_reference_history("我对某种药物过敏",&ctx).is_empty());
        assert!(edge_reference_history("我血压偏高，需要复诊",&ctx).is_empty());
        assert_eq!(edge_reference_history("那改到周四",&ctx),"王明周三去上海");
        assert!(!edge_reference_history("那改到周四",&ctx).contains("拿铁"));
    }
}

#[cfg(test)]
mod edge_window_resume_regressions {
    use super::*;
    struct EmptyEdge;
    #[async_trait::async_trait]
    impl ExtractionModel for EmptyEdge {
        fn edge_budget(&self)->bool {true}
        async fn call(&self,_request:ModelRequest)->Result<ExtractionCompletion,String>{
            Ok(ExtractionCompletion{text:"{\"done\":true}".into(),truncated:false})
        }
    }
    #[tokio::test]
    async fn long_edge_document_checkpoints_keep_valid_coverage_and_resume() {
        let source=EvidenceSource{id:"document".into(),kind:"document".into(),name:"long".into(),text:format!("{}。\n","这是一行独立资料".repeat(12)).repeat(50),locator:None,truncated:false,graph_already_ingested:false,preserve_original:false,file_path:None};
        let context=ExtractContext{roster:"[]".into(),tags:"[]".into(),speaker:String::new(),transcript:String::new(),message_at_ms:0};
        let staged=stage_source_with_model(&EmptyEdge,source.clone(),context.clone(),StageLimits::default()).await;
        assert!(valid_staged_source(&source,&staged));
        assert!(staged.parts.iter().all(|part|part.parsed.complete));
        let resumed=resume_source_with_model(&EmptyEdge,source.clone(),context,staged,StageLimits::default()).await;
        assert!(valid_staged_source(&source,&resumed));
        assert!(resumed.parts.iter().all(|part|part.parsed.complete));
    }
    #[test]
    fn dense_paragraphs_do_not_share_one_short_answer_budget() {
        let text=(0..8).map(|n|format!("第{n}项是独立资料，需保留自己的完整日期、金额和归属。\n")).collect::<String>();
        let ranges=edge_line_ranges(&text,320);
        assert!(ranges.len()>1);
        assert_eq!(ranges.iter().map(|(a,b)|exact_slice(&text,*a,*b)).collect::<String>(),text);
    }
    #[tokio::test]
    async fn dense_short_sentences_preserve_coverage_when_source_call_budget_is_exhausted() {
        let source=EvidenceSource{id:"dense".into(),kind:"document".into(),name:"dense".into(),text:"独立的一句。".repeat(180),locator:None,truncated:false,graph_already_ingested:false,preserve_original:false,file_path:None};
        let context=ExtractContext{roster:"[]".into(),tags:"[]".into(),speaker:String::new(),transcript:String::new(),message_at_ms:0};
        let staged=stage_source_with_model(&EmptyEdge,source.clone(),context,StageLimits::default()).await;
        assert!(valid_staged_source(&source,&staged));
        assert!(staged.extraction_calls<=MAX_MODEL_CALLS);
        assert!(staged.parts.iter().any(|part|!part.parsed.complete));
    }
}

#[cfg(test)]
mod table_identity_regressions {
    use super::*;
    #[test]
    fn explicit_person_column_is_not_replaced_by_a_relative_in_description() {
        let root="| 当前人物基准 | 说明 |\n| --- | --- |\n| 王明；爷爷 | 王禾的父亲、王木的祖父 |\n";
        let start=root.find("| 王明").unwrap();
        assert_eq!(edge_table_subject(root,root[..start].chars().count(),&root[start..],"[]"),Some(("王明".into(),"王明".into())));
        let finance="| 付款人 | 受益人 | 金额 |\n| --- | --- |\n| 王明 | 王木 | 90 |";
        let start=finance.find("| 王明").unwrap();
        assert_eq!(edge_table_subject(finance,finance[..start].chars().count(),&finance[start..],"[]"),None);
    }
}

#[cfg(test)]
mod edge_grounding_regressions {
    use super::*;
    use std::sync::Mutex;
    #[derive(Default)]
    struct Capture(Mutex<Vec<ModelRequest>>);
    #[async_trait::async_trait]
    impl ExtractionModel for Capture {
        fn edge_budget(&self)->bool {true}
        async fn call(&self, request:ModelRequest)->Result<ExtractionCompletion,String> {
            self.0.lock().unwrap().push(request);
            Ok(ExtractionCompletion{text:"{\"done\":true}".into(),truncated:false})
        }
    }
    fn source(text:&str)->EvidenceSource {
        EvidenceSource{id:"doc".into(),kind:"document".into(),name:"资料".into(),text:text.into(),locator:None,truncated:false,graph_already_ingested:false,preserve_original:true,file_path:None}
    }
    fn context()->ExtractContext {
        ExtractContext{roster:json!([{"member_id":"王明","display_name":"王明","aliases":[]}]).to_string(),tags:json!([{"tag_id":"金融","display_name":"金融","aliases":[]}]).to_string(),speaker:String::new(),transcript:String::new(),message_at_ms:0}
    }
    #[tokio::test]
    async fn reference_data_is_not_part_of_user_evidence() {
        let model=Capture::default();let src=source("王明每月收入7000元。");
        stage_source_with_model(&model,src.clone(),context(),StageLimits::default()).await;
        let calls=model.0.lock().unwrap();
        assert_eq!(calls[0].prompt,src.text);
        assert!(calls[0].system.contains("参考信息"));
        assert!(!calls[0].prompt.contains("member_id"));
    }
    #[tokio::test]
    async fn transaction_rows_preserve_fields_and_skip_generation_for_that_row() {
        let row="| TX950 | 10-04 | 爸爸 → 全家 | 日用品<br>松桥商店 | 86.70 |";
        let src=source(&format!("本文件的称呼：爸爸是王明。\n| ID | 日期 | 付款与受益 | 内容与商户 | 金额 元 |\n| --- | --- | --- | --- | --- |\n{row}\n"));
        let model=Capture::default();
        let staged=stage_source_with_model(&model,src.clone(),context(),StageLimits::default()).await;
        assert!(valid_staged_source(&src,&staged));
        let memories:Vec<_>=staged.parts.iter().flat_map(|p|p.parsed.memories.iter()).collect();
        assert_eq!(memories.len(),1);
        assert_eq!(memories[0]["owner"],"公共");
        assert_eq!(memories[0]["evidence"],row);
        assert!(memories[0]["content"].as_str().unwrap().contains("松桥商店"));
        assert!(model.0.lock().unwrap().iter().all(|call|!call.prompt.contains("TX950")));
    }
    #[test]
    fn holder_column_uses_declared_aliases_without_registering_the_asset_id() {
        let root="本文件的称呼：爸爸是王明，妈妈是苏禾。\n| ID与持有人 | 产品 | 本金 元 |\n| --- | --- | --- |\n| INV900<br>爸爸 | 基金样例 | 9000 |\n| INV901<br>妈妈苏禾 | 定期存款 | 8000 |\n";
        let members=json!([{"member_id":"person-a","display_name":"王明","aliases":[]},{"member_id":"person-b","display_name":"苏禾","aliases":[]}]).to_string();
        for (line,expected) in [("| INV900<br>爸爸 | 基金样例 | 9000 |",("person-a","王明")),("| INV901<br>妈妈苏禾 | 定期存款 | 8000 |",("person-b","苏禾"))] {
            let offset=root[..root.find(line).unwrap()].chars().count();
            assert_eq!(edge_table_subject(root,offset,line,&members),Some((expected.0.into(),expected.1.into())));
        }
    }
    #[tokio::test]
    async fn document_reference_roster_excludes_unrelated_people() {
        let mut ctx=context();
        ctx.roster=json!([{"member_id":"王明","display_name":"王明","aliases":[]},{"member_id":"苏禾","display_name":"苏禾","aliases":[]}]).to_string();
        let src=source("# 财务资料\n本文件的称呼：爸爸是王明，妈妈是苏禾。\n爸爸每月收入7000元。");
        let model=Capture::default();
        stage_source_with_model(&model,src,ctx,StageLimits::default()).await;
        let calls=model.0.lock().unwrap();
        let request=calls.iter().find(|r|r.prompt.contains("7000")).unwrap();
        assert!(request.system.contains("王明"));
        assert!(!request.system.contains("苏禾"));
        assert!(!request.system.contains("7000"));
    }
    #[test]
    fn repeated_person_names_are_not_facts_but_real_predicates_are_kept() {
        let roster=json!([{"member_id":"person-a","display_name":"王明","aliases":["老王"]}]).to_string();
        assert!(edge_name_only(&json!({"owner":"王明","content":"王明。"}),&roster));
        assert!(edge_name_only(&json!({"owner":"person-a","content":"王明"}),&roster));
        assert!(edge_name_only(&json!({"owner":"王明","content":"老王"}),&roster));
        assert!(!edge_name_only(&json!({"owner":"王明","content":"王明对花生过敏"}),&roster));
        assert!(!edge_name_only(&json!({"owner":"王明","content":"王明；爸爸"}),&roster));
    }
}
