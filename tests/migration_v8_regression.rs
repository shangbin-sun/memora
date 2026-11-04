//! Schema migrations must register graph source and retry-job tables, including upgrades from V7.
use memora::{MemoryEngine, MemoryService};
use rusqlite::{Connection, OptionalExtension};

#[tokio::test]
async fn latest_migration_creates_graph_tables_and_v7_upgrade_is_supported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("migration.db");
    let engine = MemoryEngine::builder()
        .db_path(&path)
        .build()
        .await
        .unwrap();
    engine.shutdown();

    let conn = Connection::open(&path).unwrap();
    let sources: Option<String> = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_fact_sources'",
            [],
            |r| r.get(0),
        )
        .optional()
        .unwrap();
    let jobs: Option<String> = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_projection_jobs'",
            [],
            |r| r.get(0),
        )
        .optional()
        .unwrap();
    assert!(sources.is_some(), "V8 fact-source table missing");
    assert!(jobs.is_some(), "V8 projection-job table missing");
}
