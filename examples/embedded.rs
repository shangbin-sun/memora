//! Host integration sample: remember → recall → dream → export.

use std::sync::Arc;
use std::time::Duration;

use memora::{
    DreamOptions, ExportRequest, ForgetReason, HashEmbedder, ImportMode, MemoryEngine,
    MemoryService, QianfanClassifier, QianfanConfig, QianfanEmbedder, QianfanSummarizer,
    RecallQuery, RememberInput,
};

#[tokio::main]
async fn main() -> memora::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = std::env::temp_dir().join("memora-example");
    let _ = std::fs::create_dir_all(&dir);

    let mut builder = MemoryEngine::builder()
        .db_path(dir.join("memora.db"))
        .index_path(dir.join("memora.hnsw"))
        .namespace("demo");
    if let Ok(cfg) = QianfanConfig::from_env() {
        let archive = memora::Config::default().consolidation_archive_importance;
        builder = builder
            .embedder(Arc::new(QianfanEmbedder::new(cfg.clone())?))
            .summarizer(Arc::new(QianfanSummarizer::new(cfg.clone())?))
            .classifier(Arc::new(QianfanClassifier::new(cfg, archive)?));
        println!("using Qianfan embedder/summarizer/classifier");
    } else {
        builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
        println!("using HashEmbedder (set LLM_API_KEY to enable Qianfan)");
    }
    let engine = builder.build().await?;

    let mut a = RememberInput::episodic("用户偏好 SQLite 作为本地存储，不想依赖云数据库。");
    a.importance = 0.85;
    a.session_id = Some("s1".into());
    a.pin_to_stm = true;
    a.entities = vec!["SQLite".into(), "本地存储".into()];
    a.namespace = Some("demo".into());
    let id1 = engine.remember(a).await?;

    let mut b = RememberInput::episodic("工作日 09:00 用户经常处理日报和项目状态。");
    b.importance = 0.7;
    b.session_id = Some("s1".into());
    b.entities = vec!["日报".into()];
    b.namespace = Some("demo".into());
    let id2 = engine.remember(b).await?;

    let mut c = RememberInput::episodic("再次确认：本地优先，SQLite 加 FTS5 做关键词检索。");
    c.importance = 0.8;
    c.session_id = Some("s1".into());
    c.entities = vec!["SQLite".into(), "FTS5".into()];
    c.namespace = Some("demo".into());
    let id3 = engine.remember(c).await?;

    tokio::time::sleep(Duration::from_millis(300)).await;

    let recalled = engine
        .recall(RecallQuery {
            text: "本地 SQLite 存储".into(),
            namespace: Some("demo".into()),
            session_id: Some("s1".into()),
            ..Default::default()
        })
        .await?;
    println!(
        "recall hits={}, stm={}, tokens={}",
        recalled.hits.len(),
        recalled.stm_hits.len(),
        recalled.context.estimated_tokens
    );
    for hit in &recalled.hits {
        println!(
            "  {} score={:.4} rrf={:.4} {}",
            hit.memory.id, hit.scores.final_score, hit.scores.rrf, hit.memory.content
        );
    }

    let explained = engine.explain(id1).await?;
    println!(
        "explain versions={} audit={}",
        explained.versions.len(),
        explained.recent_audit.len()
    );

    if let Some(got) = engine.get(id2).await? {
        println!("get {} kind={}", got.id, got.kind.as_str());
    }

    let dream = engine
        .dream(DreamOptions {
            lookback_ms: 7 * 24 * 3600 * 1000,
            candidate_limit: Some(50),
            archive_sources: false,
            ..Default::default()
        })
        .await?;
    println!(
        "dream clusters={} patterns={} consolidated={} outputs={}",
        dream.clusters,
        dream.patterns,
        dream.consolidated,
        dream.output_ids.len()
    );

    let tick = engine.tick().await?;
    println!(
        "tick promoted={} reviewed={} archived={}",
        tick.promoted, tick.reviewed, tick.archived
    );

    let bytes = engine
        .export(ExportRequest {
            namespace: Some("demo".into()),
            include_forgotten: false,
            include_vectors: true,
            exclude_sensitive: false,
        })
        .await?;
    println!("exported {} bytes (ids {id1}, {id2}, {id3})", bytes.len());

    let report = engine.import(&bytes, ImportMode::DryRun).await?;
    println!("dry_run inserted estimate={}", report.inserted);

    engine
        .forget(
            id3,
            ForgetReason::User {
                reason: "example cleanup".into(),
            },
        )
        .await?;
    engine.restore(id3).await?;
    println!("forgot+restored {id3}");
    Ok(())
}
