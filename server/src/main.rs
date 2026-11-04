mod api;
mod assets;
mod reconcile;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use chrono::{FixedOffset, Utc};
use clap::Parser;
use memora::{
    HashEmbedder, HouseholdMember, MemberRole, MemoryEngine, QianfanChat, QianfanClassifier,
    QianfanConfig, QianfanEmbedder, QianfanGraphExtractor, QianfanSummarizer,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::EnvFilter;

use crate::api::AppState;

/// Asia/Shanghai（UTC+8，无夏令时）
struct ShanghaiTime;

impl FormatTime for ShanghaiTime {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        let shanghai = FixedOffset::east_opt(8 * 3600).expect("UTC+8");
        write!(
            w,
            "{}",
            Utc::now()
                .with_timezone(&shanghai)
                .format("%Y-%m-%d %H:%M:%S%.3f")
        )
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "memora-server",
    about = "Memora HTTP 服务与 Web 测试台（独立于 memora 库）"
)]
struct Args {
    /// 监听地址
    #[arg(long, env = "MEMORA_BIND", default_value = "127.0.0.1:8787")]
    bind: String,

    /// SQLite 路径
    #[arg(long, env = "MEMORA_DB", default_value = "data/memora.db")]
    db: PathBuf,

    /// 默认命名空间
    #[arg(long, env = "MEMORA_NAMESPACE", default_value = "default")]
    namespace: String,

    /// 嵌入后端：auto | hash | llm | qianfan
    #[arg(long, env = "MEMORA_EMBEDDER", default_value = "auto")]
    embedder: String,

    /// 强制使用 HashEmbedder（等同于 --embedder hash）
    #[arg(long)]
    hash_embedder: bool,

    /// 后台 tick 间隔（秒）；默认 600；设为 0 则仅手动触发
    #[arg(long, env = "MEMORA_TICK_INTERVAL", default_value = "600")]
    tick_interval: u64,

    /// 后台 dream 间隔（秒）；默认 180；设为 0 则仅手动触发
    #[arg(long, env = "MEMORA_DREAM_INTERVAL", default_value = "180")]
    dream_interval: u64,
}

fn load_dotenv() {
    for path in [".env", "../.env"] {
        if dotenvy::from_filename(path).is_ok() {
            break;
        }
    }
}

#[tokio::main]
async fn main() -> memora::Result<()> {
    load_dotenv();
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_timer(ShanghaiTime)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    if let Some(parent) = args.db.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut builder = MemoryEngine::builder()
        .db_path(&args.db)
        .namespace(&args.namespace);

    let mode = if args.hash_embedder {
        "hash"
    } else {
        args.embedder.trim()
    };

    let qianfan_cfg = QianfanConfig::from_env().ok();

    let embedder_name = match mode {
        "hash" => {
            builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
            tracing::info!("embedder=HashEmbedder");
            "hash-ngram-v1".to_string()
        }
        "llm" | "qianfan" => {
            let cfg = qianfan_cfg.clone().ok_or_else(|| {
                memora::MemoraError::Other(format!("embedder={mode} 需要 LLM_API_KEY"))
            })?;
            let archive = memora::Config::default().consolidation_archive_importance;
            builder = builder
                .embedder(Arc::new(QianfanEmbedder::new(cfg.clone())?))
                .summarizer(Arc::new(QianfanSummarizer::new(cfg.clone())?))
                .classifier(Arc::new(QianfanClassifier::new(cfg, archive)?));
            tracing::info!("embedder=Qianfan classifier=Qianfan");
            "qianfan".to_string()
        }
        "auto" => {
            if let Some(cfg) = qianfan_cfg.clone() {
                let archive = memora::Config::default().consolidation_archive_importance;
                builder = builder
                    .embedder(Arc::new(QianfanEmbedder::new(cfg.clone())?))
                    .summarizer(Arc::new(QianfanSummarizer::new(cfg.clone())?))
                    .classifier(Arc::new(QianfanClassifier::new(cfg, archive)?));
                tracing::info!("embedder=Qianfan classifier=Qianfan (auto)");
                "qianfan".to_string()
            } else {
                builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
                tracing::info!("embedder=HashEmbedder (auto; set LLM_API_KEY for LLM embeddings)");
                "hash-ngram-v1".to_string()
            }
        }
        other => {
            return Err(memora::MemoraError::Other(format!(
                "未知 embedder={other}，可选: auto | hash | llm | qianfan"
            )));
        }
    };

    if let Some(cfg) = qianfan_cfg.clone() {
        match QianfanGraphExtractor::new(cfg) {
            Ok(ex) => {
                builder = builder.graph_extractor(Arc::new(ex));
                tracing::info!("graph_extractor=QianfanGraphExtractor");
            }
            Err(e) => tracing::warn!("QianfanGraphExtractor unavailable: {e}"),
        }
    } else {
        tracing::info!("graph_extractor=EmptyGraphExtractor (set LLM_API_KEY for LLM KG extract)");
    }

    let engine = builder.build().await?;
    let shutdown_engine = engine.clone();
    bootstrap_default_driver(&engine).await?;
    tracing::info!(
        enabled = engine.graph_projection_worker_enabled(),
        "durable graph projection worker (write-triggered; independent of tick interval)"
    );

    let tick_interval_secs = (args.tick_interval > 0).then_some(args.tick_interval);
    let maintenance_job = tick_interval_secs.map(|secs| {
        tracing::info!(secs, "background tick enabled");
        engine.spawn_maintenance(Duration::from_secs(secs))
    });
    let dream_interval_secs = (args.dream_interval > 0).then_some(args.dream_interval);
    let dream_job = dream_interval_secs.map(|secs| {
        tracing::info!(secs, "background dream enabled");
        engine.spawn_dream(Duration::from_secs(secs))
    });

    let chat = match qianfan_cfg {
        Some(cfg) => match QianfanChat::new(cfg) {
            Ok(c) => {
                tracing::info!(model = %c.model(), "reconcile chat=QianfanChat");
                Some(c)
            }
            Err(e) => {
                tracing::warn!("QianfanChat unavailable: {e}");
                None
            }
        },
        None => {
            tracing::info!(
                "reconcile chat disabled (set LLM_API_KEY to enable Android-aligned path)"
            );
            None
        }
    };

    let reconcile_jobs = api::ReconcileJobs::open(&args.db)?;
    let state = AppState {
        reconcile_jobs,
        engine,
        chat,
        bind: args.bind.clone(),
        db_path: args.db.display().to_string(),
        embedder_fallback: embedder_name,
        tick_interval_secs,
        dream_interval_secs,
    };

    let app = Router::new()
        .nest("/api", api::router(state))
        .fallback(assets::static_handler)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = args
        .bind
        .parse()
        .map_err(|e| memora::MemoraError::Other(format!("invalid bind address: {e}")))?;
    tracing::info!("listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    if let Some(job) = maintenance_job {
        job.abort();
    }
    if let Some(job) = dream_job {
        job.abort();
    }
    shutdown_engine.shutdown();
    served.map_err(|e| memora::MemoraError::Other(e.to_string()))?;
    Ok(())
}

async fn bootstrap_default_driver(engine: &MemoryEngine) -> memora::Result<()> {
    const DRIVER: &str = "主驾";
    let members = engine.list_members(None).await?;
    let has_driver = members.iter().any(|m| {
        m.member_id == DRIVER
            || m.display_name == DRIVER
            || m.aliases.iter().any(|a| a == DRIVER || a == "主驾驶")
            || matches!(m.role, MemberRole::Self_)
    });
    if !has_driver && members.is_empty() {
        engine
            .upsert_member(
                None,
                HouseholdMember {
                    member_id: DRIVER.into(),
                    display_name: DRIVER.into(),
                    role: MemberRole::Other,
                    aliases: vec!["主驾驶".into(), "司机".into()],
                    age_band: None,
                },
            )
            .await?;
        tracing::info!("bootstrap household member={DRIVER}");
    }
    let members = engine.list_members(None).await?;
    let has_public = members.iter().any(|m| {
        m.member_id == memora::PUBLIC_MEMBER_ID
            || m.display_name == memora::PUBLIC_MEMBER_ID
            || m.aliases
                .iter()
                .any(|a| matches!(a.as_str(), "大家" | "全家" | "家庭" | "公共"))
    });
    if !has_public {
        engine
            .upsert_member(
                None,
                HouseholdMember {
                    member_id: memora::PUBLIC_MEMBER_ID.into(),
                    display_name: memora::PUBLIC_MEMBER_ID.into(),
                    role: MemberRole::Other,
                    aliases: vec!["大家".into(), "全家".into(), "家庭".into()],
                    age_band: None,
                },
            )
            .await?;
        tracing::info!("bootstrap household member={}", memora::PUBLIC_MEMBER_ID);
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
