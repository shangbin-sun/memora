//! Engine completion regressions: public API only, except deterministic SQLite fixtures.
//! No remote LLM, cargo invocation, or production-source inclusion is required.

use async_trait::async_trait;
use memora::graph::{
    EntityType, ExtractedEntity, ExtractedTriple, FactPolarity, FactStatus, GrantInput,
    GraphExtractInput, GraphExtractOutput, GraphExtractor, PrivacyLevel,
};
use memora::{
    ContextRole, DreamOptions, ForgetReason, HouseholdMember, MemberRole, MemoryEngine, MemoryId,
    MemoryKind, MemoryService, MemoryState, MemoryUpdate, PersonGraphQuery, RecallQuery,
    RecallResult, RememberInput, SummarizeInput, SummarizeOutput, Summarizer,
};
use rusqlite::{params, Connection};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

const DB_NAME: &str = "engine-completion.db";

/// The speaker determines the subject, making ownership reprojection observable.
/// Objects differ by source text, so unrelated fixtures never merge accidentally.
struct RecordingExtractor {
    privacy: PrivacyLevel,
    failures_left: AtomicUsize,
    inputs: Mutex<Vec<GraphExtractInput>>,
}

impl RecordingExtractor {
    fn new(privacy: PrivacyLevel, failures: usize) -> Self {
        Self {
            privacy,
            failures_left: AtomicUsize::new(failures),
            inputs: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl GraphExtractor for RecordingExtractor {
    fn extractor_id(&self) -> &str {
        "completion-recording-extractor"
    }

    async fn extract(&self, input: GraphExtractInput) -> memora::Result<GraphExtractOutput> {
        self.inputs.lock().unwrap().push(input.clone());
        if self
            .failures_left
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
        {
            return Err(memora::MemoraError::Other(
                "deliberate projection failure".into(),
            ));
        }
        let owner = input
            .speaker_member_id
            .clone()
            .unwrap_or_else(|| "alpha".into());
        let name = input
            .roster
            .iter()
            .find(|member| member.member_id == owner)
            .map(|member| member.display_name.clone())
            .unwrap_or_else(|| format!("Person {owner}"));
        Ok(GraphExtractOutput {
            entities: vec![
                ExtractedEntity {
                    temp_id: "subject".into(),
                    name,
                    entity_type: EntityType::Person,
                    aliases: vec![],
                    member_id: Some(owner),
                },
                ExtractedEntity {
                    temp_id: "object".into(),
                    name: format!("Artifact {}", input.text),
                    entity_type: EntityType::Concept,
                    aliases: vec![],
                    member_id: None,
                },
            ],
            triples: vec![ExtractedTriple {
                subject: "subject".into(),
                relation: "likes".into(),
                object: "object".into(),
                object_type: Some(EntityType::Concept),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["medical".into()],
                privacy: Some(self.privacy),
                condition_text: None,
                evidence_text: input.text,
                confidence: 0.95,
                attrs: json!({}),
            }],
            extractor_id: self.extractor_id().into(),
            ..Default::default()
        })
    }
}

async fn build(
    dir: &tempfile::TempDir,
    namespace: &str,
    extractor: Option<Arc<RecordingExtractor>>,
) -> MemoryEngine {
    let mut builder = MemoryEngine::builder()
        .db_path(dir.path().join(DB_NAME))
        .namespace(namespace)
        .graph_projection_worker(false);
    if let Some(extractor) = extractor {
        builder = builder.graph_extractor(extractor);
    }
    builder.build().await.unwrap()
}

async fn roster(engine: &MemoryEngine, namespace: &str, id: &str, name: &str) {
    engine
        .upsert_member(
            Some(namespace),
            HouseholdMember {
                member_id: id.into(),
                display_name: name.into(),
                role: MemberRole::Self_,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
}

fn source(namespace: &str, owner: &str, content: &str, tag: &str, private: bool) -> RememberInput {
    let mut input = RememberInput::episodic(content);
    input.namespace = Some(namespace.into());
    input.member_id = Some(owner.into());
    input.skip_consolidation = true;
    input.importance = 0.9;
    input.confidence = 1.0;
    input.metadata = json!({
        "attribution": "manual",
        "subjects": [{"member_id": owner, "confidence": 1.0}],
        "scope": "members",
        "tag": tag,
        "privacy": private
    });
    input
}

fn local_query(namespace: &str, text: &str) -> RecallQuery {
    RecallQuery {
        namespace: Some(namespace.into()),
        text: text.into(),
        fts_only: true,
        skip_graph: true,
        ..Default::default()
    }
}

fn graph_query(namespace: &str, owner: &str, viewer: &str) -> RecallQuery {
    RecallQuery {
        namespace: Some(namespace.into()),
        // Source text deliberately does not contain this relation keyword.
        text: "likes".into(),
        member_id: Some(owner.into()),
        viewer_member_id: Some(viewer.into()),
        fts_only: true,
        top_k: Some(20),
        ..Default::default()
    }
}

fn has_graph_source(result: &RecallResult, id: MemoryId) -> bool {
    result
        .graph_hits
        .iter()
        .any(|hit| hit.fact.source_memory_id == Some(id))
}

fn sqlite(dir: &tempfile::TempDir) -> Connection {
    let conn = Connection::open(dir.path().join(DB_NAME)).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn
}

fn make_projection_due(dir: &tempfile::TempDir, id: MemoryId) {
    let changed = sqlite(dir)
        .execute(
            "UPDATE graph_projection_jobs SET next_attempt_at_ms = 0 WHERE memory_id = ?1",
            [id.to_string()],
        )
        .unwrap();
    assert_eq!(
        changed, 1,
        "failed extraction must leave a durable retry job"
    );
}

#[tokio::test]
async fn identical_member_ids_are_isolated_across_namespaces_on_remember_revise_forget() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 0));
    let engine = build(&dir, "home", Some(extractor)).await;
    roster(&engine, "A", "same-member", "Alice A").await;
    roster(&engine, "B", "same-member", "Bob B").await;
    let a = engine
        .remember(source(
            "A",
            "same-member",
            "namespace original",
            "其他",
            false,
        ))
        .await
        .unwrap();
    let b = engine
        .remember(source(
            "B",
            "same-member",
            "namespace original",
            "其他",
            false,
        ))
        .await
        .unwrap();
    engine.retry_graph_projections(32).await.unwrap();
    assert_ne!(a, b);
    for (namespace, expected, excluded, name) in [("A", a, b, "Alice A"), ("B", b, a, "Bob B")] {
        let recalled = engine
            .recall(local_query(namespace, "namespace"))
            .await
            .unwrap();
        assert!(recalled.hits.iter().any(|hit| hit.memory.id == expected));
        assert!(recalled.hits.iter().all(|hit| hit.memory.id != excluded));
        let graph = engine
            .query_person_graph(PersonGraphQuery {
                namespace: Some(namespace.into()),
                member_id: "same-member".into(),
                viewer_member_id: Some("same-member".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(graph.person.unwrap().canonical_name, name);
        assert!(graph
            .facts
            .iter()
            .any(|fact| fact.source_memory_id == Some(expected)));
        assert!(graph
            .facts
            .iter()
            .all(|fact| fact.source_memory_id != Some(excluded)));
    }
    engine
        .revise(
            a,
            MemoryUpdate {
                content: Some("namespace revised".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    engine.retry_graph_projections(32).await.unwrap();
    assert_eq!(
        engine.get(b).await.unwrap().unwrap().content,
        "namespace original"
    );
    assert!(has_graph_source(
        &engine
            .recall(graph_query("B", "same-member", "same-member"))
            .await
            .unwrap(),
        b
    ));
    assert!(has_graph_source(
        &engine
            .recall(graph_query("A", "same-member", "same-member"))
            .await
            .unwrap(),
        a
    ));
    engine
        .forget(
            a,
            ForgetReason::User {
                reason: "namespace A only".into(),
            },
        )
        .await
        .unwrap();
    let a_after = engine
        .recall(graph_query("A", "same-member", "same-member"))
        .await
        .unwrap();
    assert!(!has_graph_source(&a_after, a));
    let b_after = engine
        .recall(graph_query("B", "same-member", "same-member"))
        .await
        .unwrap();
    assert!(has_graph_source(&b_after, b));
    assert_eq!(
        engine.get(b).await.unwrap().unwrap().state,
        MemoryState::Active
    );
    let b_memory = engine.recall(local_query("B", "namespace")).await.unwrap();
    assert!(b_memory.hits.iter().any(|hit| hit.memory.id == b));
    assert!(engine
        .recall(local_query("A", "namespace"))
        .await
        .unwrap()
        .hits
        .is_empty());
    engine.shutdown();
}

#[tokio::test]
async fn confirmed_medical_tag_and_domain_survive_keyword_free_content_revision() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, "tags", None).await;
    let id = engine
        .remember(source(
            "tags",
            "alpha",
            "每周三下午两点固定安排",
            "医疗健康",
            false,
        ))
        .await
        .unwrap();
    for content in [None, Some("每周五上午九点固定安排")] {
        if let Some(content) = content {
            engine
                .revise(
                    id,
                    MemoryUpdate {
                        content: Some(content.into()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
        }
        let row = engine.get(id).await.unwrap().unwrap();
        assert_eq!(row.tag, "医疗健康");
        assert_eq!(row.metadata["tag"], "医疗健康");
        assert!(row.metadata["domains"]
            .as_array()
            .unwrap()
            .iter()
            .any(|domain| domain["id"] == "medical"));
        let result = engine
            .recall(RecallQuery {
                domains: vec!["medical".into()],
                viewer_member_id: Some("alpha".into()),
                ..local_query("tags", &row.content)
            })
            .await
            .unwrap();
        assert!(result.hits.iter().any(|hit| hit.memory.id == id));
    }
    engine.shutdown();
}

#[tokio::test]
async fn content_only_revise_removes_inline_summary_from_storage_fts_and_cached_context() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, "summary", None).await;
    let mut input = source("summary", "alpha", "originaldetail", "其他", false);
    input.summary = Some("obsoletesummary stale inline description".into());
    let id = engine.remember(input).await.unwrap();
    let warm = engine
        .recall(local_query("summary", "originaldetail"))
        .await
        .unwrap();
    assert!(warm
        .context
        .items
        .iter()
        .any(|item| item.text.contains("obsoletesummary")));
    engine
        .revise(
            id,
            MemoryUpdate {
                content: Some("replacementdetail current content".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let row = engine.get(id).await.unwrap().unwrap();
    assert!(
        row.summary.is_none(),
        "a content-only revise must invalidate the old inline summary"
    );
    let current = engine
        .recall(local_query("summary", "replacementdetail"))
        .await
        .unwrap();
    assert!(current.hits.iter().any(|hit| hit.memory.id == id));
    assert!(current
        .context
        .items
        .iter()
        .any(|item| item.memory_id == id && item.text.contains("replacementdetail")));
    assert!(current.context.items.iter().all(
        |item| !item.text.contains("obsoletesummary") && !item.text.contains("originaldetail")
    ));
    for obsolete in ["obsoletesummary", "originaldetail"] {
        let result = engine
            .recall(local_query("summary", obsolete))
            .await
            .unwrap();
        assert!(result.hits.iter().all(|hit| hit.memory.id != id));
        assert!(result
            .context
            .items
            .iter()
            .all(|item| !item.text.contains("obsoletesummary")));
    }
    engine.shutdown();
}

#[tokio::test]
async fn graph_recall_honours_source_tag_kind_time_member_and_viewer_filters() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 0));
    let engine = build(&dir, "filters", Some(extractor)).await;
    roster(&engine, "filters", "alpha", "Alice").await;
    roster(&engine, "filters", "beta", "Bob").await;
    let id = engine
        .remember(source(
            "filters",
            "alpha",
            "graphseed alpha private",
            "医疗健康",
            true,
        ))
        .await
        .unwrap();
    let other = engine
        .remember(source(
            "filters",
            "beta",
            "graphseed beta shared",
            "医疗健康",
            false,
        ))
        .await
        .unwrap();
    engine.retry_graph_projections(32).await.unwrap();
    let timestamp = engine.get(id).await.unwrap().unwrap().created_at_ms;
    let allowed = graph_query("filters", "alpha", "alpha");
    let baseline = engine.recall(allowed.clone()).await.unwrap();
    assert!(
        baseline.hits.is_empty(),
        "fixture must exercise the graph path alone"
    );
    assert!(has_graph_source(&baseline, id));
    assert!(!has_graph_source(&baseline, other));
    let matching = engine
        .recall(RecallQuery {
            tag: Some("医疗健康".into()),
            kinds: vec![MemoryKind::Episodic],
            time_from_ms: Some(timestamp),
            time_to_ms: Some(timestamp),
            ..allowed.clone()
        })
        .await
        .unwrap();
    assert!(
        has_graph_source(&matching, id),
        "inclusive source timestamp bounds must retain the fact"
    );
    for rejected in [
        RecallQuery {
            tag: Some("饮食".into()),
            ..allowed.clone()
        },
        RecallQuery {
            kinds: vec![MemoryKind::Semantic],
            ..allowed.clone()
        },
        RecallQuery {
            time_from_ms: Some(timestamp + 1),
            ..allowed.clone()
        },
        RecallQuery {
            time_to_ms: Some(timestamp - 1),
            ..allowed.clone()
        },
        RecallQuery {
            viewer_member_id: Some("beta".into()),
            ..allowed.clone()
        },
        RecallQuery {
            member_id: Some("unknown-member".into()),
            ..allowed.clone()
        },
    ] {
        let result = engine.recall(rejected).await.unwrap();
        assert!(
            !has_graph_source(&result, id),
            "excluded source must not leak through graph recall"
        );
        assert!(result.context.items.iter().all(|item| item.memory_id != id));
    }
    engine.shutdown();
}

#[tokio::test]
async fn metadata_only_subject_revision_reprojects_graph_ownership_and_visibility() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 0));
    let engine = build(&dir, "subjects", Some(extractor.clone())).await;
    roster(&engine, "subjects", "alpha", "Alice").await;
    roster(&engine, "subjects", "beta", "Bob").await;
    let id = engine
        .remember(source(
            "subjects",
            "alpha",
            "subjectless fixed arrangement",
            "其他",
            true,
        ))
        .await
        .unwrap();
    engine.retry_graph_projections(32).await.unwrap();
    assert!(has_graph_source(
        &engine
            .recall(graph_query("subjects", "alpha", "alpha"))
            .await
            .unwrap(),
        id
    ));
    engine
        .revise(id, MemoryUpdate {
            metadata: Some(json!({"attribution":"manual", "subjects":[{"member_id":"beta", "confidence":1.0}]})),
            ..Default::default()
        })
        .await
        .unwrap();
    engine.retry_graph_projections(32).await.unwrap();
    let row = engine.get(id).await.unwrap().unwrap();
    assert_eq!(row.content, "subjectless fixed arrangement");
    assert_eq!(row.subject_ids, vec!["beta"]);
    assert!(!has_graph_source(
        &engine
            .recall(graph_query("subjects", "alpha", "alpha"))
            .await
            .unwrap(),
        id
    ));
    let new_owner = engine
        .recall(graph_query("subjects", "beta", "beta"))
        .await
        .unwrap();
    assert!(has_graph_source(&new_owner, id));
    assert!(new_owner
        .graph_hits
        .iter()
        .filter(|hit| hit.fact.source_memory_id == Some(id))
        .all(|hit| hit.fact.owner_member_id.as_deref() == Some("beta")));
    let old_graph = engine
        .query_person_graph(PersonGraphQuery {
            namespace: Some("subjects".into()),
            member_id: "alpha".into(),
            viewer_member_id: Some("beta".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(old_graph
        .facts
        .iter()
        .all(|fact| fact.source_memory_id != Some(id)));
    assert_eq!(
        extractor
            .inputs
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .speaker_member_id
            .as_deref(),
        Some("beta")
    );
    engine.shutdown();
}

#[tokio::test]
async fn viewer_filter_precedes_fts_limit_when_ninety_private_rows_outrank_shared_row() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, "fts", None).await;
    for n in 0..90 {
        engine
            .remember(source("fts", "alpha", &format!("needle {n}"), "其他", true))
            .await
            .unwrap();
    }
    let shared = engine
        .remember(source(
            "fts",
            "alpha",
            &format!("needle {}", "filler ".repeat(150)),
            "其他",
            false,
        ))
        .await
        .unwrap();
    // Prove this is quota starvation, not merely a privacy-result assertion.
    let conn = sqlite(&dir);
    let mut statement = conn.prepare(
        "SELECT memory_id FROM memory_fts WHERE memory_fts MATCH 'needle' ORDER BY bm25(memory_fts) ASC"
    ).unwrap();
    let ranked: Vec<String> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(|row| row.unwrap())
        .collect();
    assert_eq!(
        ranked.iter().position(|id| id == &shared.to_string()),
        Some(90)
    );
    drop(statement);
    drop(conn);
    let result = engine
        .recall(RecallQuery {
            viewer_member_id: Some("other-viewer".into()),
            ..local_query("fts", "needle")
        })
        .await
        .unwrap();
    assert_eq!(
        result.hits.len(),
        1,
        "private rows must be filtered before the default FTS top-80 quota"
    );
    assert_eq!(result.hits[0].memory.id, shared);
    assert!(result.stm_hits.iter().all(|row| row.id == shared));
    assert!(result
        .context
        .items
        .iter()
        .all(|item| item.memory_id == shared));
    engine.shutdown();
}

#[tokio::test]
async fn fail_once_projection_keeps_text_and_public_retry_processes_due_job() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 1));
    let engine = build(&dir, "retry", Some(extractor.clone())).await;
    roster(&engine, "retry", "alpha", "Alice").await;
    let id = engine
        .remember(source(
            "retry",
            "alpha",
            "durable text stays searchable",
            "其他",
            false,
        ))
        .await
        .unwrap();
    assert_eq!(extractor.inputs.lock().unwrap().len(), 0);
    assert_eq!(engine.retry_graph_projections(1).await.unwrap(), 1);
    assert_eq!(extractor.inputs.lock().unwrap().len(), 1);
    let text = engine
        .recall(local_query("retry", "durable"))
        .await
        .unwrap();
    assert!(text.hits.iter().any(|hit| hit.memory.id == id));
    assert!(!has_graph_source(
        &engine
            .recall(graph_query("retry", "alpha", "alpha"))
            .await
            .unwrap(),
        id
    ));
    make_projection_due(&dir, id);
    assert_eq!(engine.retry_graph_projections(1).await.unwrap(), 1);
    assert_eq!(extractor.inputs.lock().unwrap().len(), 2);
    assert!(has_graph_source(
        &engine
            .recall(graph_query("retry", "alpha", "alpha"))
            .await
            .unwrap(),
        id
    ));
    let pending: i64 = sqlite(&dir)
        .query_row(
            "SELECT count(*) FROM graph_projection_jobs WHERE memory_id = ?1",
            [id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        pending, 0,
        "successful projection must acknowledge its persisted job"
    );
    engine.shutdown();
}

#[tokio::test]
async fn projection_retry_after_restart_uses_persisted_attachment_text_and_speaker() {
    let dir = tempfile::tempdir().unwrap();
    let failed = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 1));
    let first = build(&dir, "restart", Some(failed)).await;
    roster(&first, "restart", "alpha", "Alice").await;
    roster(&first, "restart", "beta", "Bob").await;
    // Speaker differs from confirmed memory owner; do not reconstruct it from subjects.
    let mut input = source(
        "restart",
        "alpha",
        "short attachment description",
        "其他",
        false,
    );
    input.member_id = Some("beta".into());
    let full_text = "persisted full attachment text with unique evidence";
    input.metadata["graph_text"] = json!(full_text);
    let id = first.remember(input).await.unwrap();
    first.retry_graph_projections(1).await.unwrap();
    first.shutdown();
    drop(first);
    let recovered = Arc::new(RecordingExtractor::new(PrivacyLevel::FamilyShared, 0));
    let second = build(&dir, "restart", Some(recovered.clone())).await;
    make_projection_due(&dir, id);
    // tick is the supported maintenance entry point and must invoke durable retry.
    second.tick().await.unwrap();
    let inputs = recovered.inputs.lock().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].text, full_text);
    assert_eq!(inputs[0].speaker_member_id.as_deref(), Some("beta"));
    assert!(inputs[0]
        .roster
        .iter()
        .any(|member| member.member_id == "beta"));
    drop(inputs);
    let result = second
        .recall(RecallQuery {
            member_id: None,
            viewer_member_id: Some("alpha".into()),
            ..graph_query("restart", "beta", "beta")
        })
        .await
        .unwrap();
    assert!(has_graph_source(&result, id));
    assert!(result
        .graph_hits
        .iter()
        .filter(|hit| hit.fact.source_memory_id == Some(id))
        .all(|hit| hit.fact.owner_member_id.as_deref() == Some("beta")));
    assert_eq!(
        second.get(id).await.unwrap().unwrap().content,
        "short attachment description"
    );
    second.shutdown();
}

#[tokio::test]
async fn cached_graph_only_grant_disappears_when_its_validity_window_naturally_expires() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(RecordingExtractor::new(PrivacyLevel::OwnerPrivate, 0));
    let engine = build(&dir, "expiry", Some(extractor)).await;
    roster(&engine, "expiry", "alpha", "Alice").await;
    let report = engine
        .ingest_graph_text(
            "graphonly private assertion",
            Some("alpha".into()),
            Some("expiry".into()),
        )
        .await
        .unwrap();
    assert_eq!(report.facts_written, 1);
    let query = graph_query("expiry", "alpha", "beta");
    assert!(engine
        .recall(query.clone())
        .await
        .unwrap()
        .graph_hits
        .is_empty());
    let now = chrono::Utc::now().timestamp_millis();
    let grant = engine
        .grant_access(GrantInput {
            namespace: Some("expiry".into()),
            grantor_member_id: "alpha".into(),
            grantee_member_id: "beta".into(),
            scope_kind: "all".into(),
            operations: vec!["view".into()],
            valid_from_ms: Some(now - 1_000),
            valid_to_ms: Some(now + 60_000),
            ..Default::default()
        })
        .await
        .unwrap();
    let warm = engine.recall(query.clone()).await.unwrap();
    assert_eq!(warm.graph_hits.len(), 1);
    assert!(warm.hits.is_empty() && warm.stm_hits.is_empty());
    assert!(
        warm.graph_hits[0].fact.source_memory_id.is_none(),
        "grant test needs graph-only provenance, not a private text source"
    );
    // Arm a short future deadline directly, without invoking cache invalidation.
    // The repeated query then observes expiry only because wall-clock time passes.
    let expires_at = chrono::Utc::now().timestamp_millis() + 60;
    sqlite(&dir)
        .execute(
            "UPDATE kg_grants SET valid_to_ms = ?1 WHERE namespace = ?2 AND id = ?3",
            params![expires_at, "expiry", grant.id],
        )
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let expired = engine.recall(query).await.unwrap();
    assert!(
        expired.graph_hits.is_empty(),
        "a warm graph result must not outlive its grant"
    );
    assert!(expired
        .context
        .items
        .iter()
        .all(|item| item.role != ContextRole::GraphFact));
    engine.shutdown();
}

struct OverviewSummarizer;

#[async_trait]
impl Summarizer for OverviewSummarizer {
    fn summarizer_id(&self) -> &str {
        "completion-overview"
    }
    fn prompt_version(&self) -> &str {
        "1"
    }
    async fn summarize(&self, _input: SummarizeInput) -> memora::Result<SummarizeOutput> {
        // Deliberately omit each unique source detail: FTS cannot rescue them via this summary.
        Ok(SummarizeOutput {
            summary: "overview 固定安排已汇总".into(),
        })
    }
}

#[tokio::test]
async fn explicit_archive_preserves_direct_recall_of_protected_source_details() {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join(DB_NAME))
        .namespace("archive")
        .summarizer(Arc::new(OverviewSummarizer))
        .build()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for text in [
        "detailalpha 每周三下午固定安排",
        "detailbeta 每周五上午固定安排",
    ] {
        let mut input = source("archive", "alpha", text, "医疗健康", true);
        input.entities = vec!["固定安排".into()];
        ids.push(engine.remember(input).await.unwrap());
    }
    let ordinary_archived = engine
        .remember(source(
            "archive",
            "alpha",
            "unprotectedarchived unrelated source",
            "其他",
            false,
        ))
        .await
        .unwrap();
    sqlite(&dir)
        .execute(
            "UPDATE memories SET state = 'archived' WHERE id = ?1",
            [ordinary_archived.to_string()],
        )
        .unwrap();
    let report = engine
        .dream(DreamOptions {
            archive_sources: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(report.clusters, 1);
    assert_eq!(report.purged, 0);
    assert_eq!(report.output_ids.len(), 1);
    let summary = engine.get(report.output_ids[0]).await.unwrap().unwrap();
    assert!(!summary.content.contains("detailalpha") && !summary.content.contains("detailbeta"));
    for (id, detail) in ids.into_iter().zip(["detailalpha", "detailbeta"]) {
        let source = engine.get(id).await.unwrap().unwrap();
        assert_eq!(
            source.state,
            MemoryState::Archived,
            "explicit archival must not require sources to remain active"
        );
        let found = engine
            .recall(RecallQuery {
                viewer_member_id: Some("alpha".into()),
                tag: Some("医疗健康".into()),
                ..local_query("archive", detail)
            })
            .await
            .unwrap();
        assert!(
            found.hits.iter().any(|hit| hit.memory.id == id),
            "archived source details must remain directly retrievable"
        );
        assert!(found
            .context
            .items
            .iter()
            .any(|item| item.memory_id == id && item.text.contains(detail)));
        let hidden = engine
            .recall(RecallQuery {
                viewer_member_id: Some("beta".into()),
                ..local_query("archive", detail)
            })
            .await
            .unwrap();
        assert!(hidden.hits.iter().all(|hit| hit.memory.id != id));
        assert!(hidden.context.items.iter().all(|item| item.memory_id != id));
    }
    let ordinary = engine
        .recall(local_query("archive", "unprotectedarchived"))
        .await
        .unwrap();
    assert!(
        ordinary
            .hits
            .iter()
            .all(|hit| hit.memory.id != ordinary_archived),
        "archived recall exception is limited to protected summarized sources"
    );
    engine.shutdown();
}

#[tokio::test]
async fn expanded_sources_obey_viewer_tag_and_both_time_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, "expansion", None).await;
    let mut input = source(
        "expansion",
        "alpha",
        "overviewanchor summary root",
        "医疗健康",
        false,
    );
    input.kind = MemoryKind::Summary;
    let root = engine.remember(input).await.unwrap();
    let allowed = engine
        .remember(source(
            "expansion",
            "alpha",
            "alloweddetail inside range",
            "医疗健康",
            false,
        ))
        .await
        .unwrap();
    let private = engine
        .remember(source(
            "expansion",
            "alpha",
            "privatedetail inside range",
            "医疗健康",
            true,
        ))
        .await
        .unwrap();
    let wrong_tag = engine
        .remember(source(
            "expansion",
            "alpha",
            "dietdetail inside range",
            "饮食",
            false,
        ))
        .await
        .unwrap();
    let early = engine
        .remember(source(
            "expansion",
            "alpha",
            "earlydetail before range",
            "医疗健康",
            false,
        ))
        .await
        .unwrap();
    let late = engine
        .remember(source(
            "expansion",
            "alpha",
            "latedetail after range",
            "医疗健康",
            false,
        ))
        .await
        .unwrap();
    let conn = sqlite(&dir);
    for (id, timestamp) in [
        (root, 1_500),
        (allowed, 1_500),
        (private, 1_500),
        (wrong_tag, 1_500),
        (early, 999),
        (late, 2_001),
    ] {
        conn.execute(
            "UPDATE memories SET created_at_ms = ?1 WHERE id = ?2",
            params![timestamp, id.to_string()],
        )
        .unwrap();
    }
    for child in [allowed, private, wrong_tag, early, late] {
        conn.execute(
            "INSERT INTO memory_relations(from_id, to_id, relation, weight, created_at_ms) VALUES (?1, ?2, 'derived_from', 1.0, 1500)",
            params![root.to_string(), child.to_string()],
        ).unwrap();
    }
    drop(conn);
    let baseline = engine
        .recall(RecallQuery {
            viewer_member_id: Some("beta".into()),
            ..local_query("expansion", "overviewanchor")
        })
        .await
        .unwrap();
    assert_eq!(baseline.hits.len(), 1);
    assert_eq!(baseline.hits[0].memory.id, root);
    for visible in [allowed, wrong_tag, early, late] {
        assert!(
            baseline
                .context
                .items
                .iter()
                .any(|item| item.role == ContextRole::ExpandedSource && item.memory_id == visible),
            "fixture must genuinely expand its visible provenance sources"
        );
    }
    assert!(
        baseline
            .context
            .items
            .iter()
            .all(|item| item.memory_id != private),
        "viewer restriction applies during source expansion too"
    );
    let filtered = engine
        .recall(RecallQuery {
            viewer_member_id: Some("beta".into()),
            tag: Some("医疗健康".into()),
            time_from_ms: Some(1_000),
            time_to_ms: Some(2_000),
            ..local_query("expansion", "overviewanchor")
        })
        .await
        .unwrap();
    assert!(filtered.hits.iter().any(|hit| hit.memory.id == root));
    assert!(filtered
        .context
        .items
        .iter()
        .any(|item| item.role == ContextRole::ExpandedSource && item.memory_id == allowed));
    for excluded in [private, wrong_tag, early, late] {
        assert!(
            filtered
                .context
                .items
                .iter()
                .all(|item| item.memory_id != excluded),
            "expanded source must independently satisfy the originating recall filters"
        );
    }
    engine.shutdown();
}
