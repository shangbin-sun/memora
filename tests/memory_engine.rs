use std::sync::Arc;

use memora::{
    DreamOptions, ExportRequest, ForgetReason, HashEmbedder, ImportMode, MemoryEngine,
    MemoryKind, MemoryService, MemoryState, MemoryUpdate, RecallQuery, RememberInput,
};

async fn engine() -> (tempfile::TempDir, MemoryEngine) {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("memora.db"))
        .index_path(dir.path().join("memora.hnsw"))
        .namespace("test")
        .embedder(Arc::new(HashEmbedder::new(32)))
        .build()
        .await
        .unwrap();
    (dir, engine)
}

#[tokio::test]
async fn remember_recall_explain() {
    let (_dir, engine) = engine().await;
    let mut input = RememberInput::episodic("用户偏好 SQLite 作为本地存储");
    input.importance = 0.9;
    input.entities = vec!["SQLite".into()];
    input.session_id = Some("s1".into());
    input.pin_to_stm = true;
    let id = engine.remember(input).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let result = engine
        .recall(RecallQuery {
            text: "SQLite 本地".into(),
            session_id: Some("s1".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!result.hits.is_empty() || !result.stm_hits.is_empty());
    assert!(result.context.estimated_tokens <= result.context.budget);

    let explained = engine.explain(id).await.unwrap();
    assert_eq!(explained.memory.id, id);
    assert!(!explained.versions.is_empty());
    assert!(!explained.recent_audit.is_empty());
}

#[tokio::test]
async fn revise_forget_restore() {
    let (_dir, engine) = engine().await;
    let id = engine
        .remember(RememberInput::episodic("旧内容"))
        .await
        .unwrap();
    engine
        .revise(
            id,
            MemoryUpdate {
                content: Some("新内容".into()),
                reason: Some("更正".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let got = engine.get(id).await.unwrap().unwrap();
    assert_eq!(got.content, "新内容");

    engine
        .forget(
            id,
            ForgetReason::User {
                reason: "测试删除".into(),
            },
        )
        .await
        .unwrap();
    let forgotten = engine.get(id).await.unwrap().unwrap();
    assert_eq!(forgotten.state, MemoryState::Forgotten);

    engine.restore(id).await.unwrap();
    let restored = engine.get(id).await.unwrap().unwrap();
    assert_eq!(restored.state, MemoryState::Active);
}

#[tokio::test]
async fn export_import_roundtrip() {
    let (_dir, engine) = engine().await;
    engine
        .remember(RememberInput::episodic("导入导出测试记忆"))
        .await
        .unwrap();
    let bytes = engine
        .export(ExportRequest {
            namespace: Some("test".into()),
            include_forgotten: false,
            include_vectors: true,
            exclude_sensitive: false,
        })
        .await
        .unwrap();
    assert!(!bytes.is_empty());
    let report = engine.import(&bytes, ImportMode::DryRun).await.unwrap();
    assert!(report.inserted >= 1);
}

#[tokio::test]
async fn dream_and_tick() {
    let (_dir, engine) = engine().await;
    for i in 0..3 {
        let mut input = RememberInput::episodic(format!("工作日处理日报 {i}。SQLite 本地存储。"));
        input.importance = 0.8;
        input.entities = vec!["日报".into(), "SQLite".into()];
        engine.remember(input).await.unwrap();
    }
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let dream = engine
        .dream(DreamOptions {
            lookback_ms: 24 * 3600 * 1000,
            candidate_limit: Some(20),
            archive_sources: false,
            ..Default::default()
        })
        .await
        .unwrap();
    let _ = dream.clusters;
    let tick = engine.tick().await.unwrap();
    assert!(tick.reviewed < 1000);
}

#[tokio::test]
async fn kind_semantic_roundtrip() {
    let (_dir, engine) = engine().await;
    let mut input = RememberInput::episodic("稳定事实：宿主注入 Embedder");
    input.kind = MemoryKind::Semantic;
    let id = engine.remember(input).await.unwrap();
    let got = engine.get(id).await.unwrap().unwrap();
    assert_eq!(got.kind, MemoryKind::Semantic);
}
