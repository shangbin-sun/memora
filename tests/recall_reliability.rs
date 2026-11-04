//! Stage 2: local fallback, filtering and vector-space isolation.
use async_trait::async_trait;
use memora::{
    Config, Embedder, ForgetReason, MemoryEngine, MemoryKind, MemoryService, RecallQuery,
    RememberInput,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
struct TestEmbedder {
    space: String,
    dim: usize,
    mode: u8,
    calls: AtomicUsize,
}
#[async_trait]
impl Embedder for TestEmbedder {
    fn model_id(&self) -> &str {
        "same-model-name"
    }
    fn space_id(&self) -> String {
        self.space.clone()
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> memora::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.mode == 1 {
            return Err(memora::MemoraError::Embedding("offline".into()));
        }
        if self.mode == 2 {
            std::future::pending::<()>().await;
        }
        if self.mode == 3 {
            return Ok(vec![vec![f32::NAN; self.dim]; texts.len()]);
        }
        Ok(texts
            .iter()
            .map(|_| {
                let mut v = vec![0.0; self.dim];
                v[0] = 2.0;
                v
            })
            .collect())
    }
}
fn embed(space: &str, dim: usize, mode: u8) -> Arc<TestEmbedder> {
    Arc::new(TestEmbedder {
        space: space.into(),
        dim,
        mode,
        calls: AtomicUsize::new(0),
    })
}
fn fact(text: &str, owner: &str, _tag: &str) -> RememberInput {
    let mut i = RememberInput::episodic(text);
    i.skip_consolidation = true;
    i.metadata = serde_json::json!({"attribution":"manual","subjects":[{"member_id":owner,"confidence":1.0}]});
    i
}
async fn engine(d: &tempfile::TempDir, e: Arc<TestEmbedder>) -> MemoryEngine {
    MemoryEngine::builder()
        .db_path(d.path().join("m.db"))
        .embedder(e)
        .config(Config {
            recall_embed_timeout: Duration::from_millis(40),
            ..Default::default()
        })
        .build()
        .await
        .unwrap()
}
async fn settled(e: &MemoryEngine, id: memora::MemoryId) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if !e.get(id).await.unwrap().unwrap().embedding_pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn embedding_failure_returns_local_fts() {
    let d = tempfile::tempdir().unwrap();
    let e = engine(&d, embed("a", 8, 1)).await;
    let id = e
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    let r = e
        .recall(RecallQuery {
            text: "hospital".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(r.hits.iter().any(|h| h.memory.id == id));
    e.shutdown();
}
#[tokio::test]
async fn restart_with_same_space_reuses_compatible_vectors() {
    let d = tempfile::tempdir().unwrap();
    let old = engine(&d, embed("a", 8, 0)).await;
    let id = old
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&old, id).await;
    old.shutdown();
    drop(old);
    let provider = embed("a", 8, 0);
    let new = engine(&d, provider.clone()).await;
    assert!(!new.get(id).await.unwrap().unwrap().embedding_pending);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    new.shutdown();
}

#[tokio::test]
async fn imported_vectors_from_different_space_are_reembedded() {
    let a = tempfile::tempdir().unwrap();
    let source = engine(&a, embed("a", 8, 0)).await;
    let id = source
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&source, id).await;
    let bytes = source
        .export(memora::ExportRequest {
            include_vectors: true,
            ..Default::default()
        })
        .await
        .unwrap();
    let b = tempfile::tempdir().unwrap();
    let provider = embed("b", 8, 0);
    let target = engine(&b, provider.clone()).await;
    target
        .import(&bytes, memora::ImportMode::Merge)
        .await
        .unwrap();
    settled(&target, id).await;
    let conn = rusqlite::Connection::open(b.path().join("m.db")).unwrap();
    let space: String = conn
        .query_row(
            "SELECT space_id FROM embeddings WHERE memory_id=?1",
            [id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(space, "b");
    assert!(provider.calls.load(Ordering::SeqCst) > 0);
    source.shutdown();
    target.shutdown();
}

#[tokio::test]
async fn slow_embedding_times_out_and_returns_local_fts() {
    let d = tempfile::tempdir().unwrap();
    let e = engine(&d, embed("a", 8, 2)).await;
    let id = e
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    let r = tokio::time::timeout(
        Duration::from_secs(2),
        e.recall(RecallQuery {
            text: "hospital".into(),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(r.hits.iter().any(|h| h.memory.id == id));
    e.shutdown();
}
#[tokio::test]
async fn fts_only_never_calls_embedding_service() {
    let d = tempfile::tempdir().unwrap();
    let remote = embed("a", 8, 0);
    let e = engine(&d, remote.clone()).await;
    let id = e
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&e, id).await;
    let calls = remote.calls.load(Ordering::SeqCst);
    let r = e
        .recall(RecallQuery {
            text: "hospital".into(),
            fts_only: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(remote.calls.load(Ordering::SeqCst), calls);
    assert!(!r.hits.is_empty());
    e.shutdown();
}
#[tokio::test]
async fn member_and_tag_filter_precedes_fts_top80_and_rrf_top20() {
    let d = tempfile::tempdir().unwrap();
    let e = MemoryEngine::builder()
        .db_path(d.path().join("m.db"))
        .build()
        .await
        .unwrap();
    for n in 0..90 {
        e.remember(fact("hospital", &format!("other{n}"), "娱乐"))
            .await
            .unwrap();
    }
    let id = e
        .remember(fact(
            "医院 hospital additional words lowering rank",
            "target",
            "医疗健康",
        ))
        .await
        .unwrap();
    let r = e
        .recall(RecallQuery {
            text: "hospital".into(),
            member_id: Some("target".into()),
            tag: Some("医疗健康".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].memory.id, id);
}
#[tokio::test]
async fn stm_respects_kind_and_time_filters() {
    let d = tempfile::tempdir().unwrap();
    let e = MemoryEngine::builder()
        .db_path(d.path().join("m.db"))
        .build()
        .await
        .unwrap();
    let mut i = fact("医院 hospital", "target", "医疗健康");
    i.pin_to_stm = true;
    e.remember(i).await.unwrap();
    for q in [
        RecallQuery {
            text: "hospital".into(),
            kinds: vec![MemoryKind::Semantic],
            ..Default::default()
        },
        RecallQuery {
            text: "hospital".into(),
            time_to_ms: Some(0),
            ..Default::default()
        },
    ] {
        let r = e.recall(q).await.unwrap();
        assert!(r.hits.is_empty());
        assert!(r.stm_hits.is_empty());
    }
}
#[tokio::test]
async fn endpoint_change_reembeds_even_with_same_model_and_dim() {
    let d = tempfile::tempdir().unwrap();
    let old = engine(&d, embed("endpoint-a", 8, 0)).await;
    let id = old
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&old, id).await;
    old.shutdown();
    drop(old);
    let remote = embed("endpoint-b", 8, 0);
    let new = engine(&d, remote.clone()).await;
    settled(&new, id).await;
    assert!(remote.calls.load(Ordering::SeqCst) > 0);
    let c = rusqlite::Connection::open(d.path().join("m.db")).unwrap();
    let space: String = c
        .query_row(
            "SELECT space_id FROM embeddings WHERE memory_id=?1",
            [id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(space, "endpoint-b");
    new.shutdown();
}
#[tokio::test]
async fn changing_vector_dimensions_never_mixes_hnsw_spaces() {
    let d = tempfile::tempdir().unwrap();
    let old = engine(&d, embed("same-space", 8, 0)).await;
    let id = old
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&old, id).await;
    old.shutdown();
    drop(old);
    let new = engine(&d, embed("same-space", 16, 0)).await;
    settled(&new, id).await;
    let r = new
        .recall(RecallQuery {
            text: "vector-only-query".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(r.hits.iter().any(|h| h.memory.id == id));
    new.shutdown();
}
#[tokio::test]
async fn no_embedder_can_browse_recall_and_store_existing_facts() {
    let d = tempfile::tempdir().unwrap();
    let old = engine(&d, embed("a", 8, 0)).await;
    let id = old
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    settled(&old, id).await;
    old.shutdown();
    drop(old);
    let local = MemoryEngine::builder()
        .db_path(d.path().join("m.db"))
        .build()
        .await
        .unwrap();
    assert!(!local
        .recall(RecallQuery {
            text: "hospital".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .hits
        .is_empty());
    assert!(local
        .recall(RecallQuery {
            text: "unrelated query".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .hits
        .is_empty());
    local
        .remember(fact("new local fact", "target", "其他"))
        .await
        .unwrap();
}
#[tokio::test]
async fn invalid_query_or_provider_vectors_fall_back_without_index_publication() {
    let d = tempfile::tempdir().unwrap();
    let e = engine(&d, embed("a", 8, 3)).await;
    let id = e
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    let r = e
        .recall(RecallQuery {
            text: "hospital".into(),
            query_embedding: Some(vec![f32::NAN; 8]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!r.hits.is_empty());
    assert!(e.get(id).await.unwrap().unwrap().embedding_pending);
    e.shutdown();
}
#[tokio::test]
async fn shutdown_fences_foreground_requests_holding_old_handles() {
    let d = tempfile::tempdir().unwrap();
    let e = engine(&d, embed("a", 8, 0)).await;
    let id = e
        .remember(fact("hospital", "target", "医疗健康"))
        .await
        .unwrap();
    e.shutdown();
    assert!(e.remember(fact("late", "target", "其他")).await.is_err());
    assert!(e
        .recall(RecallQuery {
            text: "hospital".into(),
            ..Default::default()
        })
        .await
        .is_err());
    assert!(e
        .forget(
            id,
            ForgetReason::User {
                reason: "late".into()
            }
        )
        .await
        .is_err());
}
