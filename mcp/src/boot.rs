//! Bootstrap [`MemoryEngine`] from CLI / env (same knobs as memora-server).

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use memora::{
    HashEmbedder, MemoryEngine, QianfanClassifier, QianfanConfig, QianfanEmbedder,
    QianfanGraphExtractor, QianfanSummarizer,
};

#[derive(Parser, Debug)]
#[command(
    name = "memora-mcp",
    about = "Memora MCP stdio server — remember / recall / dream for agents"
)]
pub struct Args {
    /// SQLite path
    #[arg(long, env = "MEMORA_DB", default_value = "data/memora.db")]
    pub db: PathBuf,

    /// Default namespace
    #[arg(long, env = "MEMORA_NAMESPACE", default_value = "default")]
    pub namespace: String,

    /// Embedder: auto | hash | llm | qianfan
    #[arg(long, env = "MEMORA_EMBEDDER", default_value = "auto")]
    pub embedder: String,

    /// Force HashEmbedder (same as --embedder hash)
    #[arg(long)]
    pub hash_embedder: bool,

    /// LLM API key (embed + chat). MCP `env.LLM_API_KEY` overrides `.env`.
    #[arg(long, env = "LLM_API_KEY")]
    pub llm_api_key: Option<String>,

    /// Embedding endpoint
    #[arg(long, env = "LLM_EMBED_URL")]
    pub llm_embed_url: Option<String>,

    /// Embedding model id
    #[arg(long, env = "LLM_EMBED_MODEL")]
    pub llm_embed_model: Option<String>,

    /// Embedding vector dimension
    #[arg(long, env = "LLM_EMBED_DIM")]
    pub llm_embed_dim: Option<usize>,

    /// Chat endpoint (summarizer / classifier)
    #[arg(long, env = "LLM_CHAT_URL")]
    pub llm_chat_url: Option<String>,

    /// Chat model id
    #[arg(long, env = "LLM_CHAT_MODEL")]
    pub llm_chat_model: Option<String>,

    /// One-shot command. Absent means the stdio MCP server.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// One-shot commands for hooks. Print a single JSON line to stdout and exit.
/// Text/content comes from the flag, or from stdin when the flag is omitted.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Semantic recall. Prints `{"ok":true,"hits":[...]}`.
    Recall {
        /// Query text. Omit to read stdin.
        #[arg(long)]
        text: Option<String>,
        /// Max hits (default 5)
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Queue one memory and return without waiting. Prints `{"ok":true,"accepted":true}`.
    Remember {
        /// Memory body. Omit to read stdin.
        #[arg(long)]
        content: Option<String>,
    },
}

/// Write CLI / MCP env values into the process environment so [`QianfanConfig::from_env`] sees them.
/// Non-empty values override whatever `.env` already loaded.
fn apply_llm_env(args: &Args) {
    let set = |key: &str, value: &str| {
        if !value.is_empty() {
            std::env::set_var(key, value);
        }
    };
    if let Some(v) = &args.llm_api_key {
        set("LLM_API_KEY", v);
    }
    if let Some(v) = &args.llm_embed_url {
        set("LLM_EMBED_URL", v);
    }
    if let Some(v) = &args.llm_embed_model {
        set("LLM_EMBED_MODEL", v);
    }
    if let Some(v) = args.llm_embed_dim {
        std::env::set_var("LLM_EMBED_DIM", v.to_string());
    }
    if let Some(v) = &args.llm_chat_url {
        set("LLM_CHAT_URL", v);
    }
    if let Some(v) = &args.llm_chat_model {
        set("LLM_CHAT_MODEL", v);
    }
}

pub fn load_dotenv() {
    for path in [".env", "../.env"] {
        if dotenvy::from_filename(path).is_ok() {
            break;
        }
    }
}

pub async fn build_engine(args: &Args) -> memora::Result<(MemoryEngine, String)> {
    apply_llm_env(args);

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

    let embedder_name = match mode {
        "hash" => {
            builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
            tracing::info!("embedder=HashEmbedder");
            "hash-ngram-v1".to_string()
        }
        "llm" | "qianfan" => {
            let cfg = QianfanConfig::from_env().map_err(|e| {
                memora::MemoraError::Other(format!("embedder={mode} 需要 LLM_API_KEY: {e}"))
            })?;
            let name = format!("qianfan:{}", cfg.embed_model);
            tracing::info!(
                model = %cfg.embed_model,
                dim = cfg.embed_dim,
                url = %cfg.embed_url,
                "embedder=Qianfan"
            );
            let archive = memora::Config::default().consolidation_archive_importance;
            builder = builder
                .embedder(Arc::new(QianfanEmbedder::new(cfg.clone())?))
                .summarizer(Arc::new(QianfanSummarizer::new(cfg.clone())?))
                .classifier(Arc::new(QianfanClassifier::new(cfg.clone(), archive)?));
            match QianfanGraphExtractor::new(cfg) {
                Ok(ex) => {
                    builder = builder.graph_extractor(Arc::new(ex));
                    tracing::info!("graph_extractor=QianfanGraphExtractor");
                }
                Err(e) => tracing::warn!("QianfanGraphExtractor unavailable: {e}"),
            }
            name
        }
        "auto" => {
            if let Ok(cfg) = QianfanConfig::from_env() {
                let name = format!("qianfan:{}", cfg.embed_model);
                tracing::info!(
                    model = %cfg.embed_model,
                    dim = cfg.embed_dim,
                    url = %cfg.embed_url,
                    "embedder=Qianfan (auto)"
                );
                let archive = memora::Config::default().consolidation_archive_importance;
                builder = builder
                    .embedder(Arc::new(QianfanEmbedder::new(cfg.clone())?))
                    .summarizer(Arc::new(QianfanSummarizer::new(cfg.clone())?))
                    .classifier(Arc::new(QianfanClassifier::new(cfg.clone(), archive)?));
                match QianfanGraphExtractor::new(cfg) {
                    Ok(ex) => {
                        builder = builder.graph_extractor(Arc::new(ex));
                        tracing::info!("graph_extractor=QianfanGraphExtractor (auto)");
                    }
                    Err(e) => tracing::warn!("QianfanGraphExtractor unavailable: {e}"),
                }
                name
            } else {
                builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
                tracing::info!("embedder=HashEmbedder (auto)");
                "hash-ngram-v1".to_string()
            }
        }
        other => {
            return Err(memora::MemoraError::Other(format!(
                "未知 embedder={other}，可选: auto | hash | llm | qianfan"
            )));
        }
    };

    let engine = builder.build().await?;
    Ok((engine, embedder_name))
}
