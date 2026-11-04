//! Mounted as engine::candidate_recall_tests to exercise the private lightweight API.
//! Local fixtures only: no HTTP, LLM, dependency downloads, or toolchain changes.
use super::*;
use async_trait::async_trait;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct BatchEmbedder {
    calls: AtomicUsize,
    mode: AtomicUsize,
    batches: std::sync::Mutex<Vec<Vec<String>>>,
}
#[async_trait]
impl Embedder for BatchEmbedder {
    fn model_id(&self) -> &str {
        "candidate-test"
    }
    fn space_id(&self) -> String {
        "candidate-test:8".into()
    }
    fn dimension(&self) -> usize {
        8
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.batches.lock().unwrap().push(texts.to_vec());
        match self.mode.load(Ordering::SeqCst) {
            1 => return Err(MemoraError::Embedding("fixture offline".into())),
            2 => std::future::pending::<()>().await,
            _ => {}
        }
        Ok(texts
            .iter()
            .map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect())
    }
}
impl BatchEmbedder {
    fn reset(&self) {
        self.calls.store(0, Ordering::SeqCst);
        self.batches.lock().unwrap().clear();
    }
}
async fn build(dir: &tempfile::TempDir, embedder: Option<Arc<BatchEmbedder>>) -> MemoryEngine {
    let mut builder = MemoryEngine::builder()
        .db_path(dir.path().join("candidate.db"))
        .graph_projection_worker(false)
        .config(Config {
            recall_embed_timeout: Duration::from_millis(100),
            embedding_timeout: Duration::from_millis(300),
            query_cache_ttl: Duration::from_secs(300),
            ..Default::default()
        });
    if let Some(embedder) = embedder {
        builder = builder.embedder(embedder);
    }
    builder.build().await.unwrap()
}
fn source(namespace: &str, owner: &str, text: &str, tag: &str, private: bool) -> RememberInput {
    let mut input = RememberInput::episodic(text);
    input.namespace = Some(namespace.into());
    input.member_id = Some(owner.into());
    input.skip_consolidation = true;
    input.importance = 0.9;
    input.confidence = 1.0;
    input.metadata = json!({"attribution":"manual", "scope":"members", "skip_graph":true,
        "subjects":[{"member_id":owner,"confidence":1.0}], "tag":tag,"privacy":private});
    input
}
fn query(namespace: &str, text: &str) -> RecallQuery {
    RecallQuery {
        namespace: Some(namespace.into()),
        text: text.into(),
        skip_graph: true,
        top_k: Some(20),
        ..Default::default()
    }
}
async fn settled(engine: &MemoryEngine, id: MemoryId) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if !engine.get(id).await.unwrap().unwrap().embedding_pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
fn ids(records: &[MemoryRecord]) -> Vec<MemoryId> {
    records.iter().map(|r| r.id).collect()
}

#[tokio::test]
async fn candidates_batch_embeds_unique_queries_once_preserves_order_and_has_no_recall_side_effects(
) {
    let dir = tempfile::tempdir().unwrap();
    let provider = Arc::new(BatchEmbedder::default());
    let engine = build(&dir, Some(provider.clone())).await;
    let hospital = engine
        .remember(source("A", "alpha", "hospital visit", "其他", false))
        .await
        .unwrap();
    let school = engine
        .remember(source("A", "beta", "school course", "其他", false))
        .await
        .unwrap();
    settled(&engine, hospital).await;
    settled(&engine, school).await;
    provider.reset();
    let before_hospital = engine.get(hospital).await.unwrap().unwrap();
    let before_school = engine.get(school).await.unwrap().unwrap();
    let mut q1 = query("A", "hospital");
    q1.member_id = Some("alpha".into());
    let mut q2 = query("A", "school");
    q2.member_id = Some("beta".into());
    let output = engine
        .recall_candidates_batch(vec![q1.clone(), q2.clone(), q1])
        .await
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *provider.batches.lock().unwrap(),
        vec![vec!["hospital".to_string(), "school".to_string()]]
    );
    assert_eq!(
        output.iter().map(|r| ids(r)).collect::<Vec<_>>(),
        vec![vec![hospital], vec![school], vec![hospital]]
    );
    for (id, before) in [(hospital, before_hospital), (school, before_school)] {
        let after = engine.get(id).await.unwrap().unwrap();
        assert_eq!(after.last_accessed_at_ms, before.last_accessed_at_ms);
        assert_eq!(after.review_count, before.review_count);
        assert_eq!(after.updated_at_ms, before.updated_at_ms);
        assert!(engine.inner.last_scores.get(&id).await.is_none());
        assert_eq!(
            serde_json::to_value(&after).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
    }
    engine.inner.stm.query_cache.run_pending_tasks().await;
    assert_eq!(
        engine.inner.stm.query_cache.entry_count(),
        0,
        "candidates must not construct/cache context packs"
    );
    engine.shutdown();
}

#[tokio::test]
async fn query_vector_cache_survives_memory_mutation_and_does_not_cache_old_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let provider = Arc::new(BatchEmbedder::default());
    let engine = build(&dir, Some(provider.clone())).await;
    let first = engine
        .remember(source("A", "alpha", "hospital first", "其他", false))
        .await
        .unwrap();
    settled(&engine, first).await;
    provider.reset();
    let q = query("A", "hospital");
    assert_eq!(
        ids(&engine
            .recall_candidates_batch(vec![q.clone()])
            .await
            .unwrap()[0]),
        vec![first]
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let second = engine
        .remember(source("A", "alpha", "hospital second", "其他", false))
        .await
        .unwrap();
    settled(&engine, second).await;
    provider.reset();
    let current = engine
        .recall_candidates_batch(vec![q.clone()])
        .await
        .unwrap();
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        0,
        "query vectors survive result invalidation"
    );
    assert!(ids(&current[0]).contains(&first) && ids(&current[0]).contains(&second));
    engine
        .forget(
            first,
            ForgetReason::User {
                reason: "fixture".into(),
            },
        )
        .await
        .unwrap();
    let current = engine.recall_candidates_batch(vec![q]).await.unwrap();
    assert_eq!(ids(&current[0]), vec![second]);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    engine.shutdown();
}

#[tokio::test]
async fn mixed_namespaces_and_duplicate_queries_preserve_input_order_without_record_leakage() {
    let dir = tempfile::tempdir().unwrap();
    let provider = Arc::new(BatchEmbedder::default());
    let engine = build(&dir, Some(provider.clone())).await;
    let a = engine
        .remember(source("A", "alpha", "hospital household A", "其他", false))
        .await
        .unwrap();
    let b = engine
        .remember(source("B", "alpha", "hospital household B", "其他", false))
        .await
        .unwrap();
    settled(&engine, a).await;
    settled(&engine, b).await;
    provider.reset();
    let output = engine
        .recall_candidates_batch(vec![
            query("A", "hospital"),
            query("A", "hospital"),
            query("B", "hospital"),
            query("A", "hospital"),
        ])
        .await
        .unwrap();
    assert_eq!(
        output.iter().map(|r| ids(r)).collect::<Vec<_>>(),
        vec![vec![a], vec![a], vec![b], vec![a]]
    );
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        2,
        "same text is independently cached by namespace"
    );
    assert!(engine
        .recall_candidates_batch(vec![])
        .await
        .unwrap()
        .is_empty());
    engine.shutdown();
}

#[tokio::test]
async fn candidates_apply_namespace_subject_tag_viewer_time_kind_and_lifecycle_filters() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, None).await;
    let allowed = engine
        .remember(source("A", "alpha", "needle permitted", "饮食", false))
        .await
        .unwrap();
    let other_owner = engine
        .remember(source("A", "beta", "needle wrong owner", "饮食", false))
        .await
        .unwrap();
    let wrong_tag = engine
        .remember(source("A", "alpha", "needle wrong tag", "其他", false))
        .await
        .unwrap();
    let private = engine
        .remember(source("A", "alpha", "needle private", "饮食", true))
        .await
        .unwrap();
    let mut semantic = source("A", "alpha", "needle semantic", "饮食", false);
    semantic.kind = MemoryKind::Semantic;
    let wrong_kind = engine.remember(semantic).await.unwrap();
    let old = engine
        .remember(source("A", "alpha", "needle old", "饮食", false))
        .await
        .unwrap();
    let deleted = engine
        .remember(source("A", "alpha", "needle deleted", "饮食", false))
        .await
        .unwrap();
    let other_ns = engine
        .remember(source(
            "B",
            "alpha",
            "needle other namespace",
            "饮食",
            false,
        ))
        .await
        .unwrap();
    engine
        .forget(
            deleted,
            ForgetReason::User {
                reason: "fixture".into(),
            },
        )
        .await
        .unwrap();
    {
        let conn = engine.inner.pool.get().unwrap();
        conn.execute(
            "UPDATE memories SET created_at_ms=1000 WHERE id!=?1",
            [old.to_string()],
        )
        .unwrap();
        conn.execute(
            "UPDATE memories SET created_at_ms=100 WHERE id=?1",
            [old.to_string()],
        )
        .unwrap();
    }
    let mut q = query("A", "needle");
    q.fts_only = true;
    q.member_id = Some("alpha".into());
    q.tag = Some("饮食".into());
    q.domains = vec!["diet".into()];
    q.viewer_member_id = Some("outsider".into());
    q.time_from_ms = Some(500);
    q.time_to_ms = Some(1500);
    q.kinds = vec![MemoryKind::Episodic];
    let found = engine
        .recall_candidates_batch(vec![q.clone()])
        .await
        .unwrap();
    assert_eq!(ids(&found[0]), vec![allowed]);
    assert!(ids(&found[0]).iter().all(|id| ![
        other_owner,
        wrong_tag,
        private,
        wrong_kind,
        old,
        deleted,
        other_ns
    ]
    .contains(id)));
    q.viewer_member_id = Some("alpha".into());
    let owner = engine.recall_candidates_batch(vec![q]).await.unwrap();
    assert!(ids(&owner[0]).contains(&private));
    assert!(ids(&owner[0]).contains(&allowed));
    engine.shutdown();
}

#[tokio::test]
async fn failed_or_slow_batch_embedding_returns_fts_and_does_not_store_failure_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let provider = Arc::new(BatchEmbedder::default());
    let engine = build(&dir, Some(provider.clone())).await;
    let hospital = engine
        .remember(source("A", "alpha", "hospital clinic", "其他", false))
        .await
        .unwrap();
    settled(&engine, hospital).await;
    provider.reset();
    provider.mode.store(1, Ordering::SeqCst);
    let q = query("A", "hospital");
    let first = engine
        .recall_candidates_batch(vec![q.clone()])
        .await
        .unwrap();
    assert_eq!(ids(&first[0]), vec![hospital]);
    let second = engine.recall_candidates_batch(vec![q]).await.unwrap();
    assert_eq!(ids(&second[0]), vec![hospital]);
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        2,
        "failed embeddings cannot become a cached success"
    );
    provider.mode.store(2, Ordering::SeqCst);
    let output = tokio::time::timeout(
        Duration::from_secs(2),
        engine.recall_candidates_batch(vec![query("A", "clinic")]),
    )
    .await
    .expect("slow semantic branch must not block FTS indefinitely")
    .unwrap();
    assert_eq!(ids(&output[0]), vec![hospital]);
    engine.shutdown();
}

#[tokio::test]
async fn lightweight_fts_ranking_matches_public_recall_before_any_access_touch() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, None).await;
    let mut all = Vec::new();
    for (name, importance) in [("low", 0.2), ("medium", 0.5), ("high", 0.9)] {
        let mut input = source("A", "alpha", &format!("hospital {name}"), "其他", false);
        input.importance = importance;
        all.push(engine.remember(input).await.unwrap());
    }
    {
        let conn = engine.inner.pool.get().unwrap();
        let now = crate::core::now_ms();
        conn.execute(
            "UPDATE memories SET created_at_ms=?1,last_accessed_at_ms=?1,review_count=0",
            [now],
        )
        .unwrap();
    }
    let mut q = query("A", "hospital");
    q.fts_only = true;
    let candidates = engine
        .recall_candidates_batch(vec![q.clone()])
        .await
        .unwrap();
    let public = engine.recall(q).await.unwrap();
    assert_eq!(
        ids(&candidates[0]),
        public.hits.iter().map(|h| h.memory.id).collect::<Vec<_>>()
    );
    assert_eq!(ids(&candidates[0]), vec![all[2], all[1], all[0]]);
    engine.shutdown();
}
