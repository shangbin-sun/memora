//! Ordinary writes never wait on graph LLM; revision fences survive races.
use async_trait::async_trait;
use memora::graph::{
    EntityType, ExtractedEntity, ExtractedTriple, FactPolarity, FactStatus, GraphExtractInput,
    GraphExtractOutput, GraphExtractor, PrivacyLevel,
};
use memora::{
    ForgetReason, HouseholdMember, MemberRole, MemoryEngine, MemoryId, MemoryService, MemoryUpdate,
    PersonGraphQuery, RecallQuery, RememberInput,
};
use rusqlite::Connection;
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::Semaphore;

const DB: &str = "foreground-graph.db";
const FAST: Duration = Duration::from_secs(2);
struct Gated {
    calls: AtomicUsize,
    gate: Semaphore,
}
impl Gated {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            gate: Semaphore::new(0),
        }
    }
}
#[async_trait]
impl GraphExtractor for Gated {
    fn extractor_id(&self) -> &str {
        "foreground-gated"
    }
    async fn extract(&self, input: GraphExtractInput) -> memora::Result<GraphExtractOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.gate
            .acquire()
            .await
            .expect("gate remains open")
            .forget();
        Ok(GraphExtractOutput {
            entities: vec![
                ExtractedEntity {
                    temp_id: "p".into(),
                    name: "Alice".into(),
                    entity_type: EntityType::Person,
                    member_id: Some("alpha".into()),
                    aliases: vec![],
                },
                ExtractedEntity {
                    temp_id: "o".into(),
                    name: format!("Artifact {}", input.text),
                    entity_type: EntityType::Concept,
                    member_id: None,
                    aliases: vec![],
                },
            ],
            triples: vec![ExtractedTriple {
                subject: "p".into(),
                relation: "likes".into(),
                object: "o".into(),
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
async fn build(dir: &tempfile::TempDir, graph: Arc<Gated>, worker: bool) -> MemoryEngine {
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join(DB))
        .namespace("foreground")
        .graph_extractor(graph)
        .graph_projection_worker(worker)
        .build()
        .await
        .unwrap();
    engine
        .upsert_member(
            Some("foreground"),
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
    input.namespace = Some("foreground".into());
    input.member_id = Some("alpha".into());
    input.skip_consolidation = true;
    input.metadata = json!({"attribution":"manual","scope":"members","subjects":[{"member_id":"alpha","confidence":1.0}]});
    input
}
async fn wait_calls(graph: &Gated, n: usize) {
    tokio::time::timeout(FAST, async {
        while graph.calls.load(Ordering::SeqCst) < n {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("drain must enter the graph extractor");
}
fn pending(dir: &tempfile::TempDir, id: MemoryId) -> Option<i64> {
    use rusqlite::OptionalExtension;
    Connection::open(dir.path().join(DB))
        .unwrap()
        .query_row(
            "SELECT revision FROM graph_projection_jobs WHERE memory_id=?1",
            [id.to_string()],
            |row| row.get(0),
        )
        .optional()
        .unwrap()
}
async fn source_facts(engine: &MemoryEngine, id: MemoryId) -> usize {
    engine
        .query_person_graph(PersonGraphQuery {
            namespace: Some("foreground".into()),
            member_id: "alpha".into(),
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
async fn ordinary_writes_and_recall_return_while_background_graph_is_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(Gated::new());
    let engine = build(&dir, graph.clone(), true).await;
    let id = tokio::time::timeout(FAST, engine.remember(input("needle original")))
        .await
        .expect("ordinary remember must not await graph LLM")
        .unwrap();
    wait_calls(&graph, 1).await;
    tokio::time::timeout(
        FAST,
        engine.revise(
            id,
            MemoryUpdate {
                content: Some("needle current".into()),
                ..Default::default()
            },
        ),
    )
    .await
    .expect("ordinary revise must not wait on graph lock")
    .unwrap();
    let recalled = tokio::time::timeout(
        FAST,
        engine.recall(RecallQuery {
            text: "needle".into(),
            namespace: Some("foreground".into()),
            fts_only: true,
            skip_graph: true,
            ..Default::default()
        }),
    )
    .await
    .expect("recall must not wait on graph LLM")
    .unwrap();
    assert!(recalled
        .hits
        .iter()
        .any(|hit| hit.memory.id == id && hit.memory.content == "needle current"));
    let second = tokio::time::timeout(FAST, engine.remember(input("second local write")))
        .await
        .unwrap()
        .unwrap();
    assert!(pending(&dir, second).is_some());
    assert_eq!(
        pending(&dir, id),
        Some(engine.get(id).await.unwrap().unwrap().revision)
    );
    engine.shutdown();
}
#[tokio::test]
async fn stale_revision_cannot_publish_and_metadata_only_revise_preserves_new_job() {
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(Gated::new());
    let engine = build(&dir, graph.clone(), false).await;
    let id = engine.remember(input("old source")).await.unwrap();
    let drain_engine = engine.clone();
    let drain = tokio::spawn(async move { drain_engine.retry_graph_projections(1).await.unwrap() });
    wait_calls(&graph, 1).await;
    tokio::time::timeout(
        FAST,
        engine.revise(
            id,
            MemoryUpdate {
                content: Some("new source".into()),
                ..Default::default()
            },
        ),
    )
    .await
    .unwrap()
    .unwrap();
    let current = engine.get(id).await.unwrap().unwrap();
    assert_eq!(pending(&dir, id), Some(current.revision));
    graph.gate.add_permits(1);
    assert_eq!(drain.await.unwrap(), 1);
    assert_eq!(source_facts(&engine, id).await, 0);
    assert_eq!(pending(&dir, id), Some(current.revision));
    engine
        .revise(
            id,
            MemoryUpdate {
                importance: Some(0.9),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let newest = engine.get(id).await.unwrap().unwrap();
    assert!(newest.revision > current.revision);
    assert_eq!(pending(&dir, id), Some(newest.revision));
    graph.gate.add_permits(1);
    assert_eq!(engine.retry_graph_projections(1).await.unwrap(), 1);
    assert!(source_facts(&engine, id).await > 0);
    assert_eq!(pending(&dir, id), None);
    engine.shutdown();
}
#[tokio::test]
async fn foreground_revise_withdraws_old_support_in_commit_before_worker_can_publish_new() {
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(Gated::new());
    let engine = build(&dir, graph.clone(), false).await;
    let id = engine.remember(input("old source")).await.unwrap();
    graph.gate.add_permits(1);
    engine.retry_graph_projections(1).await.unwrap();
    assert!(source_facts(&engine, id).await > 0);
    tokio::time::timeout(
        FAST,
        engine.revise(
            id,
            MemoryUpdate {
                content: Some("new source".into()),
                ..Default::default()
            },
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(source_facts(&engine, id).await, 0);
    graph.gate.add_permits(1);
    engine.retry_graph_projections(1).await.unwrap();
    assert!(source_facts(&engine, id).await > 0);
    assert_eq!(pending(&dir, id), None);
    engine.shutdown();
}
#[tokio::test]
async fn metadata_only_revise_rebases_job_without_replacing_host_speaker_with_owner() {
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(Gated::new());
    let engine = build(&dir, graph, false).await;
    engine
        .upsert_member(
            Some("foreground"),
            HouseholdMember {
                member_id: "beta".into(),
                display_name: "Bob".into(),
                role: MemberRole::Other,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
    let mut source = input("owner alpha fact");
    source.member_id = Some("beta".into());
    source.metadata["graph_text"] = json!("original evidence spoken by beta");
    let id = engine.remember(source).await.unwrap();
    engine
        .revise(
            id,
            MemoryUpdate {
                importance: Some(0.9),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (speaker, source_text, revision): (String, String, i64) = Connection::open(dir.path().join(DB)).unwrap()
        .query_row("SELECT speaker_member_id, source_text, revision FROM graph_projection_jobs WHERE memory_id=?1",
            [id.to_string()], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
    assert_eq!(speaker, "beta");
    assert_eq!(source_text, "original evidence spoken by beta");
    assert_eq!(revision, engine.get(id).await.unwrap().unwrap().revision);
    engine.shutdown();
}

#[tokio::test]
async fn forgotten_memory_cannot_be_resurrected_by_inflight_graph_result() {
    let dir = tempfile::tempdir().unwrap();
    let graph = Arc::new(Gated::new());
    let engine = build(&dir, graph.clone(), false).await;
    let id = engine.remember(input("old source")).await.unwrap();
    let draining = engine.clone();
    let task = tokio::spawn(async move { draining.retry_graph_projections(1).await.unwrap() });
    wait_calls(&graph, 1).await;
    tokio::time::timeout(
        FAST,
        engine.forget(
            id,
            ForgetReason::User {
                reason: "forget explicitly".into(),
            },
        ),
    )
    .await
    .expect("forget remains unblocked")
    .unwrap();
    assert_eq!(pending(&dir, id), None);
    graph.gate.add_permits(1);
    task.await.unwrap();
    assert_eq!(source_facts(&engine, id).await, 0);
    assert_eq!(pending(&dir, id), None);
    engine.shutdown();
}
