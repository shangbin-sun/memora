//! Stage 1: durable facts, revocation, privacy and background task safety.
use async_trait::async_trait;
use memora::{
    ClassifyItem, ClassifyOutput, Config, DreamOptions, Embedder, ExportRequest, ForgetReason,
    HashEmbedder, MemoryClassifier, MemoryEngine, MemoryKind, MemoryService, MemoryState,
    MemoryUpdate, RecallQuery, RelationKind, RememberInput, SummarizeInput, SummarizeOutput,
    Summarizer,
};
use std::io::{Cursor, Read};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::{Notify, Semaphore};

async fn plain() -> (tempfile::TempDir, MemoryEngine) {
    let dir = tempfile::tempdir().unwrap();
    let e = MemoryEngine::builder()
        .db_path(dir.path().join("memory.db"))
        .build()
        .await
        .unwrap();
    (dir, e)
}
fn query(text: &str) -> RecallQuery {
    RecallQuery {
        text: text.into(),
        ..Default::default()
    }
}
fn fact(content: &str, member: &str, _tag: &str) -> RememberInput {
    let mut i = RememberInput::episodic(content);
    i.skip_consolidation = true;
    // Tag is inferred inside remember; content should carry domain cues.
    i.metadata = serde_json::json!({"attribution":"manual", "subjects":
        [{"member_id":member,"confidence":1.0}], "scope":"members"});
    i
}
async fn sources(e: &MemoryEngine) -> Vec<memora::MemoryId> {
    let mut ids = Vec::new();
    for text in ["上午向医生确认药物甲剂量", "下午向医生确认药物乙剂量"] {
        let mut i = fact(text, "张三", "医疗健康");
        i.entities = vec!["医生".into()];
        ids.push(e.remember(i).await.unwrap());
    }
    ids
}
async fn dream(e: &MemoryEngine, archive_sources: bool) -> memora::DreamReport {
    e.dream(DreamOptions {
        archive_sources,
        ..Default::default()
    })
    .await
    .unwrap()
}
fn archive_text(bytes: Vec<u8>, name: &str) -> String {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
    let mut text = String::new();
    zip.by_name(name)
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    text
}
#[tokio::test]
async fn committed_write_invalidates_empty_recall() {
    let (_d, e) = plain().await;
    assert!(e.recall(query("医院")).await.unwrap().hits.is_empty());
    let id = e.remember(RememberInput::episodic("医院")).await.unwrap();
    assert!(e
        .recall(query("医院"))
        .await
        .unwrap()
        .hits
        .iter()
        .any(|h| h.memory.id == id));
}
#[tokio::test]
async fn forget_invalidates_cached_recall_and_stm() {
    let (_d, e) = plain().await;
    let mut i = RememberInput::episodic("医院");
    i.pin_to_stm = true;
    let id = e.remember(i).await.unwrap();
    assert!(!e.recall(query("医院")).await.unwrap().stm_hits.is_empty());
    e.forget(
        id,
        ForgetReason::User {
            reason: "withdraw".into(),
        },
    )
    .await
    .unwrap();
    let r = e.recall(query("医院")).await.unwrap();
    assert!(r.hits.is_empty() && r.stm_hits.is_empty() && r.context.items.is_empty());
    assert!(!e.get(id).await.unwrap().unwrap().embedding_pending);
}
#[tokio::test]
async fn revise_invalidates_old_text_and_stm() {
    let (_d, e) = plain().await;
    let mut i = RememberInput::episodic("医院旧安排");
    i.pin_to_stm = true;
    let id = e.remember(i).await.unwrap();
    e.recall(query("医院")).await.unwrap();
    e.revise(
        id,
        MemoryUpdate {
            content: Some("医院新安排".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let r = e.recall(query("医院")).await.unwrap();
    assert!(r.hits.iter().all(|h| h.memory.content == "医院新安排"));
    assert!(r.stm_hits.iter().all(|h| h.content == "医院新安排"));
    assert_eq!(e.get(id).await.unwrap().unwrap().revision, 2);
}
#[tokio::test]
async fn cache_key_includes_time_window() {
    let (_d, e) = plain().await;
    e.remember(RememberInput::episodic("医院")).await.unwrap();
    assert!(!e.recall(query("医院")).await.unwrap().hits.is_empty());
    let mut q = query("医院");
    q.time_to_ms = Some(0);
    assert!(e.recall(q).await.unwrap().hits.is_empty());
}
#[tokio::test]
async fn dream_never_deduplicates_different_members_or_tags() {
    let (_d, e) = plain().await;
    let mut ids = Vec::new();
    // Tag is inferred from content; keep member/tag buckets distinct via wording.
    for (who, content) in [
        ("张三", "明天去医院复查"),
        ("小航", "明天去医院复查"),
        ("张三", "明天去超市买菜"),
    ] {
        ids.push(e.remember(fact(content, who, "")).await.unwrap());
    }
    let r = dream(&e, true).await;
    assert_eq!(r.merged, 0);
    for id in ids {
        assert_eq!(e.get(id).await.unwrap().unwrap().state, MemoryState::Active);
    }
}
#[tokio::test]
async fn same_bucket_duplicates_preserve_records_and_relation() {
    let (_d, e) = plain().await;
    let a = e.remember(fact("重复事实", "张三", "其他")).await.unwrap();
    let b = e.remember(fact("重复事实", "张三", "其他")).await.unwrap();
    let r = dream(&e, true).await;
    assert_eq!(r.merged, 1);
    assert_eq!(r.purged, 0);
    let records = [
        e.get(a).await.unwrap().unwrap(),
        e.get(b).await.unwrap().unwrap(),
    ];
    let survivor = records
        .iter()
        .find(|r| r.state == MemoryState::Active)
        .unwrap();
    assert!(e
        .explain(survivor.id)
        .await
        .unwrap()
        .relations
        .iter()
        .any(|r| r.relation == RelationKind::Supersedes));
    assert_eq!(dream(&e, true).await.merged, 0);
}
#[tokio::test]
async fn dream_preserves_active_sources_and_provenance_without_repeating() {
    let (_d, e) = plain().await;
    let ids = sources(&e).await;
    let r = dream(&e, false).await;
    assert_eq!(r.clusters, 1);
    assert_eq!(r.purged, 0);
    for id in &ids {
        assert_eq!(
            e.get(*id).await.unwrap().unwrap().state,
            MemoryState::Active
        );
    }
    let relations = e.explain(r.output_ids[0]).await.unwrap().relations;
    for id in ids {
        assert!(relations
            .iter()
            .any(|r| r.to_id == id && r.relation == RelationKind::DerivedFrom));
        assert!(relations
            .iter()
            .any(|r| r.to_id == id && r.relation == RelationKind::Summarizes));
    }
    assert_eq!(dream(&e, false).await.clusters, 0);
}
#[tokio::test]
async fn dream_archives_but_never_purges_sources() {
    let (_d, e) = plain().await;
    let ids = sources(&e).await;
    let r = dream(&e, true).await;
    assert_eq!(r.clusters, 1);
    assert_eq!(r.purged, 0);
    for id in ids {
        assert_eq!(
            e.get(id).await.unwrap().unwrap().state,
            MemoryState::Archived
        );
    }
}
#[tokio::test]
async fn source_revocation_invalidates_summary_and_context() {
    let (_d, e) = plain().await;
    let ids = sources(&e).await;
    let r = dream(&e, false).await;
    e.recall(query("医生")).await.unwrap();
    e.forget(
        ids[0],
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        e.get(r.output_ids[0]).await.unwrap().unwrap().state,
        MemoryState::Forgotten
    );
    let r2 = e.recall(query("医生")).await.unwrap();
    assert!(r2
        .context
        .items
        .iter()
        .all(|i| i.memory_id != ids[0] && i.memory_id != r.output_ids[0]));
}
#[tokio::test]
async fn source_correction_invalidates_old_summary() {
    let (_d, e) = plain().await;
    let ids = sources(&e).await;
    let r = dream(&e, false).await;
    e.revise(
        ids[0],
        MemoryUpdate {
            content: Some("更正剂量".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        e.get(r.output_ids[0]).await.unwrap().unwrap().state,
        MemoryState::Forgotten
    );
}
#[tokio::test]
async fn stm_expiry_never_replays_committed_or_revoked_facts() {
    let d = tempfile::tempdir().unwrap();
    let c = Config {
        session_events_ttl: Duration::from_millis(20),
        working_facts_ttl: Duration::from_millis(20),
        ..Default::default()
    };
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .config(c)
        .build()
        .await
        .unwrap();
    let mut i = RememberInput::episodic("需忘掉的事实");
    i.pin_to_stm = true;
    let id = e.remember(i).await.unwrap();
    e.forget(
        id,
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(e.tick().await.unwrap().promoted, 0);
    assert!(e.recall(query("事实")).await.unwrap().hits.is_empty());
}

#[tokio::test]
async fn privacy_export_filters_all_side_tables_and_historical_snapshots() {
    let (_d, e) = plain().await;
    let secret = e
        .remember(fact("PRIVATE_DIAGNOSIS 医院诊断", "张三", "医疗健康"))
        .await
        .unwrap();
    let removed = e
        .remember(fact("REVOKED_FACT 日常琐事", "张三", "其他"))
        .await
        .unwrap();
    e.forget(
        removed,
        ForgetReason::User {
            reason: "REVOKED_REASON".into(),
        },
    )
    .await
    .unwrap();
    let safe = e
        .remember(fact("PUBLIC_FACT 日常琐事", "张三", "其他"))
        .await
        .unwrap();
    // Current projection becomes public; its historical private body must not leak.
    e.revise(
        secret,
        MemoryUpdate {
            content: Some("PUBLIC_REPLACEMENT".into()),
            metadata: Some(serde_json::json!({"tag":"其他","attribution":"manual","subjects":[]})),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bytes = e
        .export(ExportRequest {
            include_vectors: true,
            exclude_sensitive: true,
            ..Default::default()
        })
        .await
        .unwrap();
    for name in [
        "memories.jsonl",
        "versions.jsonl",
        "relations.jsonl",
        "audit.jsonl",
        "embeddings.jsonl",
    ] {
        let text = archive_text(bytes.clone(), name);
        assert!(!text.contains("PRIVATE_DIAGNOSIS"), "leaked in {name}");
        assert!(!text.contains("REVOKED_FACT"), "leaked in {name}");
        assert!(
            !text.contains(&removed.to_string()),
            "revoked id leaked in {name}"
        );
    }
    assert!(archive_text(bytes.clone(), "versions.jsonl").contains(&safe.to_string()));
    assert!(archive_text(bytes.clone(), "audit.jsonl").is_empty());
    assert!(archive_text(bytes, "manifest.json").contains("audit_filtered"));
}
#[tokio::test]
async fn private_summary_relations_do_not_escape_export() {
    let (_d, e) = plain().await;
    let ids = sources(&e).await;
    let r = dream(&e, false).await;
    let bytes = e
        .export(ExportRequest {
            include_vectors: true,
            exclude_sensitive: true,
            ..Default::default()
        })
        .await
        .unwrap();
    for name in [
        "memories.jsonl",
        "versions.jsonl",
        "relations.jsonl",
        "audit.jsonl",
        "embeddings.jsonl",
    ] {
        let text = archive_text(bytes.clone(), name);
        for id in ids.iter().chain(r.output_ids.iter()) {
            assert!(!text.contains(&id.to_string()));
        }
    }
}
struct DelayedClassifier {
    entered: Notify,
    gate: Semaphore,
}
#[async_trait]
impl MemoryClassifier for DelayedClassifier {
    fn classifier_id(&self) -> &str {
        "delayed-test"
    }
    async fn classify_batch(&self, items: &[ClassifyItem]) -> memora::Result<Vec<ClassifyOutput>> {
        self.entered.notify_one();
        self.gate.acquire().await.unwrap().forget();
        Ok(items
            .iter()
            .map(|i| ClassifyOutput {
                id: i.id,
                kind: MemoryKind::Episodic,
                importance: 0.5,
                confidence: 0.8,
                archive: false,
                entities: None,
                subjects: None,
                domains: None,
                patterns: vec![],
            })
            .collect())
    }
}
#[tokio::test]
async fn concurrent_classification_never_resurrects_forgotten_record() {
    let d = tempfile::tempdir().unwrap();
    let c = Arc::new(DelayedClassifier {
        entered: Notify::new(),
        gate: Semaphore::new(0),
    });
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .classifier(c.clone())
        .build()
        .await
        .unwrap();
    let id = e.remember(RememberInput::episodic("医院")).await.unwrap();
    let task = {
        let e = e.clone();
        tokio::spawn(async move { dream(&e, true).await })
    };
    c.entered.notified().await;
    e.forget(
        id,
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    c.gate.add_permits(1);
    let r = task.await.unwrap();
    assert_eq!(r.consolidated, 0);
    assert_eq!(
        e.get(id).await.unwrap().unwrap().state,
        MemoryState::Forgotten
    );
}
struct DelayedEmbedder {
    calls: AtomicUsize,
    entered: Notify,
    first: Semaphore,
    next: Semaphore,
}
#[async_trait]
impl Embedder for DelayedEmbedder {
    fn model_id(&self) -> &str {
        "delayed-test"
    }
    fn dimension(&self) -> usize {
        2
    }
    async fn embed(&self, _: &[String]) -> memora::Result<Vec<Vec<f32>>> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_one();
        if n == 0 {
            self.first.acquire().await.unwrap().forget();
        } else {
            self.next.acquire().await.unwrap().forget();
        }
        Ok(vec![vec![1.0, 0.0]])
    }
}
fn delayed_embedder() -> Arc<DelayedEmbedder> {
    Arc::new(DelayedEmbedder {
        calls: AtomicUsize::new(0),
        entered: Notify::new(),
        first: Semaphore::new(0),
        next: Semaphore::new(0),
    })
}
#[tokio::test]
async fn old_embedding_completion_cannot_rollback_revision() {
    let d = tempfile::tempdir().unwrap();
    let embed = delayed_embedder();
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .embedder(embed.clone())
        .build()
        .await
        .unwrap();
    let id = e.remember(RememberInput::episodic("old")).await.unwrap();
    embed.entered.notified().await;
    e.revise(
        id,
        MemoryUpdate {
            content: Some("new".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    embed.first.add_permits(1);
    embed.entered.notified().await;
    let rec = e.get(id).await.unwrap().unwrap();
    assert_eq!(rec.revision, 2);
    assert_eq!(rec.embedding_version, 2);
    assert!(rec.embedding_pending);
    e.shutdown();
}
#[tokio::test]
async fn forgotten_embedding_is_not_published() {
    let d = tempfile::tempdir().unwrap();
    let embed = delayed_embedder();
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .embedder(embed.clone())
        .build()
        .await
        .unwrap();
    let id = e.remember(RememberInput::episodic("old")).await.unwrap();
    embed.entered.notified().await;
    e.forget(
        id,
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    // Use a second work item as a deterministic barrier after the first completion.
    e.remember(RememberInput::episodic("barrier"))
        .await
        .unwrap();
    embed.first.add_permits(1);
    embed.entered.notified().await;
    let conn = rusqlite::Connection::open(d.path().join("memory.db")).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE memory_id=?1",
            [id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0);
    e.shutdown();
}
#[tokio::test]
async fn dropping_idle_engine_releases_embedding_worker() {
    let d = tempfile::tempdir().unwrap();
    let embed = Arc::new(HashEmbedder::new(32));
    let weak = Arc::downgrade(&embed);
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .embedder(embed)
        .build()
        .await
        .unwrap();
    drop(e);
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(weak.upgrade().is_none());
}
struct DelayedSummarizer {
    entered: Notify,
    gate: Semaphore,
}
#[async_trait]
impl Summarizer for DelayedSummarizer {
    fn summarizer_id(&self) -> &str {
        "delayed-test"
    }
    fn prompt_version(&self) -> &str {
        "1"
    }
    async fn summarize(&self, i: SummarizeInput) -> memora::Result<SummarizeOutput> {
        self.entered.notify_one();
        self.gate.acquire().await.unwrap().forget();
        Ok(SummarizeOutput {
            summary: i.texts.join("; "),
        })
    }
}
#[tokio::test]
async fn source_changed_during_summary_wait_discards_output() {
    let d = tempfile::tempdir().unwrap();
    let s = Arc::new(DelayedSummarizer {
        entered: Notify::new(),
        gate: Semaphore::new(0),
    });
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .summarizer(s.clone())
        .build()
        .await
        .unwrap();
    let ids = sources(&e).await;
    let task = {
        let e = e.clone();
        tokio::spawn(async move { dream(&e, false).await })
    };
    s.entered.notified().await;
    e.forget(
        ids[0],
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    s.gate.add_permits(1);
    let r = task.await.unwrap();
    assert_eq!(r.clusters, 0);
    assert!(r.output_ids.is_empty());
    assert_eq!(
        e.get(ids[0]).await.unwrap().unwrap().state,
        MemoryState::Forgotten
    );
}

#[tokio::test]
async fn event_dates_are_distinct_deduplication_buckets() {
    let (_d, e) = plain().await;
    let mut ids = Vec::new();
    for date in ["2026-10-22", "2026-10-23"] {
        let mut i = fact("去医院", "张三", "医疗健康");
        i.metadata["event_at"] = serde_json::json!(date);
        ids.push(e.remember(i).await.unwrap());
    }
    let r = dream(&e, true).await;
    assert_eq!(r.merged, 0);
    for id in ids {
        assert_eq!(e.get(id).await.unwrap().unwrap().state, MemoryState::Active);
    }
}
#[tokio::test]
async fn concurrent_classification_does_not_overwrite_corrected_content() {
    let d = tempfile::tempdir().unwrap();
    let c = Arc::new(DelayedClassifier {
        entered: Notify::new(),
        gate: Semaphore::new(0),
    });
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .classifier(c.clone())
        .build()
        .await
        .unwrap();
    let id = e.remember(RememberInput::episodic("old")).await.unwrap();
    let task = {
        let e = e.clone();
        tokio::spawn(async move { dream(&e, true).await })
    };
    c.entered.notified().await;
    e.revise(
        id,
        MemoryUpdate {
            content: Some("new".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    // First stale classification is discarded; a later pass may classify new content.
    c.gate.add_permits(10);
    task.await.unwrap();
    assert_eq!(e.get(id).await.unwrap().unwrap().content, "new");
}
#[tokio::test]
async fn shutdown_cancels_in_flight_embedding_request() {
    let d = tempfile::tempdir().unwrap();
    let embed = delayed_embedder();
    let weak = Arc::downgrade(&embed);
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .embedder(embed.clone())
        .build()
        .await
        .unwrap();
    e.remember(RememberInput::episodic("old")).await.unwrap();
    embed.entered.notified().await;
    drop(embed);
    e.shutdown();
    drop(e);
    tokio::time::timeout(Duration::from_secs(2), async {
        while weak.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

struct PatternClassifier;
#[async_trait]
impl MemoryClassifier for PatternClassifier {
    fn classifier_id(&self) -> &str {
        "pattern-test"
    }
    async fn classify_batch(&self, items: &[ClassifyItem]) -> memora::Result<Vec<ClassifyOutput>> {
        Ok(items
            .iter()
            .map(|i| ClassifyOutput {
                id: i.id,
                kind: MemoryKind::Episodic,
                importance: 0.5,
                confidence: 0.8,
                archive: false,
                entities: None,
                subjects: None,
                domains: None,
                patterns: vec!["用户经常游泳".into()],
            })
            .collect())
    }
}
#[tokio::test]
async fn classifier_pattern_is_revocable_through_its_source() {
    let d = tempfile::tempdir().unwrap();
    let e = MemoryEngine::builder()
        .db_path(d.path().join("memory.db"))
        .classifier(Arc::new(PatternClassifier))
        .build()
        .await
        .unwrap();
    let id = e
        .remember(RememberInput::episodic("用户周二游泳"))
        .await
        .unwrap();
    dream(&e, false).await;
    let page = e
        .list(memora::ListQuery {
            kind: Some(MemoryKind::Pattern),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(page.items.len(), 1);
    let pattern = page.items[0].id;
    assert!(e
        .explain(pattern)
        .await
        .unwrap()
        .relations
        .iter()
        .any(|r| r.to_id == id && r.relation == RelationKind::DerivedFrom));
    e.forget(
        id,
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        e.get(pattern).await.unwrap().unwrap().state,
        MemoryState::Forgotten
    );
}
#[tokio::test]
async fn actual_import_invalidates_previous_empty_cache() {
    let (_d, source) = plain().await;
    let id = source
        .remember(fact("医院导入事实", "张三", "医疗健康"))
        .await
        .unwrap();
    let bytes = source.export(ExportRequest::default()).await.unwrap();
    let (_d2, dest) = plain().await;
    assert!(dest.recall(query("医院")).await.unwrap().hits.is_empty());
    let report = dest
        .import(&bytes, memora::ImportMode::Merge)
        .await
        .unwrap();
    assert_eq!(report.inserted, 1);
    assert_eq!(dest.get(id).await.unwrap().unwrap().revision, 1);
    assert!(dest
        .recall(query("医院"))
        .await
        .unwrap()
        .hits
        .iter()
        .any(|h| h.memory.id == id));
}
#[tokio::test]
async fn explicit_restore_invalidates_previous_empty_cache() {
    let (_d, e) = plain().await;
    let id = e
        .remember(fact("医院事实", "张三", "医疗健康"))
        .await
        .unwrap();
    e.forget(
        id,
        ForgetReason::User {
            reason: "erase".into(),
        },
    )
    .await
    .unwrap();
    assert!(e.recall(query("医院")).await.unwrap().hits.is_empty());
    e.restore(id).await.unwrap();
    assert!(e
        .recall(query("医院"))
        .await
        .unwrap()
        .hits
        .iter()
        .any(|h| h.memory.id == id));
    assert!(!e.get(id).await.unwrap().unwrap().embedding_pending);
}
