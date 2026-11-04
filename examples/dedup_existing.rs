//! Offline, model-free exact duplicate maintenance. Always back up the DB first.
use memora::MemoryEngine;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let db = args.next().ok_or("usage: dedup_existing <database> [namespace]")?;
    let ns = args.next().unwrap_or_else(|| "default".into());
    if args.next().is_some() || !std::path::Path::new(&db).is_file() {
        return Err("expected an existing database path and optional namespace".into());
    }
    // No embedder, classifier or network extractor is configured or invoked.
    let engine = MemoryEngine::builder().db_path(&db).namespace(ns).build().await?;
    let changed = engine.deduplicate_exact_memories().await?;
    let repeated = engine.deduplicate_exact_memories().await?;
    if repeated != 0 { return Err("deduplication was not idempotent".into()); }
    engine.shutdown();
    drop(engine);
    let conn = rusqlite::Connection::open(&db)?;
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    println!("{}", serde_json::json!({"superseded": changed, "second_pass": repeated}));
    Ok(())
}
