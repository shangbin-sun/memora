//! Foreground graph deferral: no network calls, controllable extractor futures.
use async_trait::async_trait;
use memora::graph::{
    EntityType, ExtractedEntity, ExtractedTriple, FactPolarity, FactStatus, GraphExtractInput,
    GraphExtractOutput, GraphExtractor, PrivacyLevel,
};
use memora::{
    HouseholdMember, MemberRole, MemoryEngine, MemoryId, MemoryService, MemoryUpdate,
    PersonGraphQuery, RememberInput,
};
use rusqlite::{params, Connection};
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

struct ControlledExtractor {
    blocked: AtomicBool,
    calls: AtomicUsize,
    inputs: Mutex<Vec<GraphExtractInput>>,
}

impl ControlledExtractor {
    fn new(blocked: bool) -> Self {
        Self {
            blocked: AtomicBool::new(blocked),
            calls: AtomicUsize::new(0),
            inputs: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl GraphExtractor for ControlledExtractor {
    fn extractor_id(&self) -> &str {
        "deferred-regression"
    }

    async fn extract(&self, input: GraphExtractInput) -> memora::Result<GraphExtractOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inputs.lock().unwrap().push(input.clone());
        if self.blocked.load(Ordering::SeqCst) {
            // An accidentally eager write cannot succeed merely by being fast.
            std::future::pending::<()>().await;
        }
        Ok(GraphExtractOutput {
            entities: vec![
                ExtractedEntity {
                    temp_id: "owner".into(),
                    name: "Alice".into(),
                    entity_type: EntityType::Person,
                    aliases: vec![],
                    member_id: Some("alpha".into()),
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
                subject: "owner".into(),
                relation: "likes".into(),
                object: "object".into(),
                object_type: Some(EntityType::Concept),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["household".into()],
                privacy: Some(PrivacyLevel::FamilyShared),
                condition_text: None,
                evidence_text: input.text,
                confidence: 1.0,
                attrs: json!({}),
            }],
            extractor_id: self.extractor_id().into(),
            ..Default::default()
        })
    }
}

async fn build(dir: &tempfile::TempDir, extractor: Arc<ControlledExtractor>) -> MemoryEngine {
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("deferred.db"))
        .namespace("deferred")
        .graph_extractor(extractor)
        .graph_projection_worker(false)
        .build()
        .await
        .unwrap();
    engine
        .upsert_member(
            Some("deferred"),
            HouseholdMember {
                member_id: "alpha".into(),
                display_name: "Alice".into(),
                role: MemberRole::Self_,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
    engine
}

fn input(text: &str) -> RememberInput {
    let mut input = RememberInput::episodic(text);
    input.member_id = Some("alpha".into());
    input.skip_consolidation = true;
    input.metadata = json!({
        "attribution": "manual",
        "subjects": [{"member_id":"alpha", "confidence":1.0}],
        "tag":"其他",
    });
    input
}

fn sqlite(dir: &tempfile::TempDir) -> Connection {
    Connection::open(dir.path().join("deferred.db")).unwrap()
}

fn job_count(dir: &tempfile::TempDir, id: MemoryId) -> i64 {
    sqlite(dir)
        .query_row(
            "SELECT count(*) FROM graph_projection_jobs WHERE memory_id = ?1",
            [id.to_string()],
            |row| row.get(0),
        )
        .unwrap()
}

fn make_due(dir: &tempfile::TempDir, id: MemoryId) {
    assert_eq!(
        sqlite(dir)
            .execute(
                "UPDATE graph_projection_jobs SET next_attempt_at_ms = 0 WHERE memory_id = ?1",
                [id.to_string()],
            )
            .unwrap(),
        1
    );
}

async fn graph_source_count(engine: &MemoryEngine, id: MemoryId) -> usize {
    engine
        .query_person_graph(PersonGraphQuery {
            member_id: "alpha".into(),
            namespace: Some("deferred".into()),
            viewer_member_id: Some("alpha".into()),
            ..Default::default()
        })
        .await
        .unwrap()
        .facts
        .iter()
        .filter(|fact| fact.source_memory_id == Some(id))
        .count()
}

#[tokio::test]
async fn deferred_remember_returns_without_extractor_and_retries_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    let blocked = Arc::new(ControlledExtractor::new(true));
    let engine = build(&dir, blocked.clone()).await;
    let mut source = input("short description");
    let full_text = "full attachment evidence for delayed projection";
    source.metadata["graph_text"] = json!(full_text);
    let id = tokio::time::timeout(
        Duration::from_secs(3),
        engine.remember_deferred_graph(source),
    )
    .await
    .expect("a deferred write must not wait for the blocked graph extractor")
    .unwrap();
    assert_eq!(blocked.calls.load(Ordering::SeqCst), 0);
    assert_eq!(job_count(&dir, id), 1);
    let record = engine.get(id).await.unwrap().unwrap();
    assert_eq!(record.content, "short description");
    assert!(record.metadata.get("defer_graph_projection").is_none());
    assert!(record.metadata.get("skip_graph").is_none());
    // Another mutation must not remain blocked on graph work from this write.
    let second = tokio::time::timeout(
        Duration::from_secs(3),
        engine.remember_deferred_graph(input("second local write")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(job_count(&dir, second), 1);
    engine.shutdown();
    drop(engine);

    let ready = Arc::new(ControlledExtractor::new(false));
    let recovered = build(&dir, ready.clone()).await;
    make_due(&dir, id);
    recovered.retry_graph_projections(8).await.unwrap();
    assert_eq!(job_count(&dir, id), 0);
    assert!(ready.inputs.lock().unwrap().iter().any(|input| {
        input.text == full_text && input.speaker_member_id.as_deref() == Some("alpha")
    }));
    assert!(graph_source_count(&recovered, id).await > 0);
    recovered.shutdown();
}

#[tokio::test]
async fn deferred_revise_withdraws_old_graph_and_queues_current_revision() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(ControlledExtractor::new(false));
    let engine = build(&dir, extractor.clone()).await;
    let id = engine.remember(input("original source")).await.unwrap();
    assert_eq!(engine.retry_graph_projections(8).await.unwrap(), 1);
    assert!(graph_source_count(&engine, id).await > 0);
    let before = engine.get(id).await.unwrap().unwrap();
    extractor.blocked.store(true, Ordering::SeqCst);
    tokio::time::timeout(
        Duration::from_secs(3),
        engine.revise_deferred_graph(
            id,
            MemoryUpdate {
                content: Some("revised source".into()),
                expected_revision: Some(before.revision),
                ..Default::default()
            },
        ),
    )
    .await
    .expect("a deferred revision must not wait for graph extraction")
    .unwrap();
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(graph_source_count(&engine, id).await, 0);
    let revised = engine.get(id).await.unwrap().unwrap();
    assert!(revised.revision > before.revision);
    assert_eq!(revised.content, "revised source");
    let queued_revision: i64 = sqlite(&dir)
        .query_row(
            "SELECT revision FROM graph_projection_jobs WHERE memory_id = ?1",
            [id.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(queued_revision, revised.revision);
    assert!(revised.metadata.get("defer_graph_projection").is_none());
    extractor.blocked.store(false, Ordering::SeqCst);
    make_due(&dir, id);
    engine.retry_graph_projections(1).await.unwrap();
    assert_eq!(job_count(&dir, id), 0);
    assert!(graph_source_count(&engine, id).await > 0);
    assert_eq!(
        extractor.inputs.lock().unwrap().last().unwrap().text,
        "revised source"
    );
    engine.shutdown();
}

#[tokio::test]
async fn skip_graph_still_disables_jobs_in_both_deferred_writes() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(ControlledExtractor::new(true));
    let engine = build(&dir, extractor.clone()).await;
    let mut source = input("memory only source");
    source.metadata["skip_graph"] = json!(true);
    let id = engine.remember_deferred_graph(source).await.unwrap();
    engine
        .revise_deferred_graph(
            id,
            MemoryUpdate {
                content: Some("memory only revised".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(job_count(&dir, id), 0);
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        engine.get(id).await.unwrap().unwrap().metadata["skip_graph"],
        true
    );
    engine.shutdown();
}

#[tokio::test]
async fn ordinary_and_compatibility_writes_both_defer_without_persistent_flags() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(ControlledExtractor::new(false));
    let engine = build(&dir, extractor.clone()).await;
    let deferred = engine
        .remember_deferred_graph(input("deferred source"))
        .await
        .unwrap();
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(job_count(&dir, deferred), 1);
    engine
        .revise(
            deferred,
            MemoryUpdate {
                content: Some("eager revised source".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(job_count(&dir, deferred), 1);
    let eager = engine
        .remember(input("ordinary created source"))
        .await
        .unwrap();
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(job_count(&dir, eager), 1);
    assert_eq!(engine.retry_graph_projections(8).await.unwrap(), 2);
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 2);
    assert_eq!(job_count(&dir, deferred), 0);
    assert_eq!(job_count(&dir, eager), 0);
    let count: i64 = sqlite(&dir)
        .query_row(
            "SELECT count(*) FROM memories WHERE id IN (?1, ?2)",
            params![deferred.to_string(), eager.to_string()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 2);
    engine.shutdown();
}
