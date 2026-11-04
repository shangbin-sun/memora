mod boot;
mod oneshot;
mod tools;

use clap::Parser;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use crate::boot::{build_engine, load_dotenv, Args};
use crate::tools::MemoraMcp;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // MCP uses stdout for the protocol — log only to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_ansi(false)
        .init();

    load_dotenv();
    let args = Args::parse();
    let (engine, embedder_name) = build_engine(&args).await?;
    if let Some(command) = args.command {
        return oneshot::run(engine, command).await;
    }
    tracing::info!(
        db = %args.db.display(),
        namespace = %args.namespace,
        embedder = %embedder_name,
        "memora-mcp ready (stdio)"
    );

    let service = MemoraMcp::new(engine, embedder_name)
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!(?e, "serve failed"))?;
    service.waiting().await?;
    Ok(())
}
