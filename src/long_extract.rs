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
    async fn call(&self, request: ModelRequest) -> Result<ExtractionCompletion, String>;
}

#[async_trait::async_trait]
impl ExtractionModel for QianfanChat {
    async fn call(&self, request: ModelRequest) -> Result<ExtractionCompletion, String> {
        self.extract_with_timeout(
            Some(&request.system),
            &request.prompt,
            Some(request.max_tokens),
            Some(0.2),
            request.timeout,
        )
        .await
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
    if count == 0 {
        let mut part = run.empty_part(&Node::new(0, 0), None);
        part.parsed.complete = !run.root.truncated;
        run.output.parts.push(part);
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
        if system.chars().count() + prompt.chars().count() > MAX_PROMPT_CHARS {
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
            let metadata = self.context_json(&node);
            let base_prompt = extraction_prompt(&source, &self.context, &metadata);
            let mut salvage: Option<StagedPart> = None;
            let mut cause = String::new();
            let mut can_split = false;
            // One whole-chunk retry replaces, rather than appends to, a complete
            // recovery. Increased output budget is bounded independently of text.
            for attempt in 0..2 {
                let mut prompt = base_prompt.clone();
                if attempt == 1 {
                    prompt.push_str("\n上次回复不完整或存在无效记录。本次从头完整提取当前块，逐行严格满足原协议并以{\"done\":true}收尾；不要接续、不要引用上次回复、不补造事实。\n");
                }
                let tokens = if attempt == 0 {
                    crate::output_budget(source.text.chars().count()).min(8_192) as u32
                } else {
                    RECOVERY_OUTPUT_TOKENS
                };
                match self
                    .call(
                        CallKind::Extraction,
                        crate::prompt_latest().extract_system,
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
}
