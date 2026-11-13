use jieba_rs::Jieba;
use rusqlite::{params, Connection, OptionalExtension};
use rusqlite_migration::{Migrations, M};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::core::now_ms;
use crate::core::tokenize_for_fts;
use crate::embedding::bytes_to_vector;
use crate::error::{MemoraError, Result};
use crate::types::{
    AuditEvent, MemoryId, MemoryKind, MemoryRecord, MemoryRelation, MemoryState, MemoryVersion,
    RelationKind,
};

pub type Pool = r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>;

const SCHEMA_V1: &str = r#"
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  state TEXT NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  importance REAL NOT NULL,
  confidence REAL NOT NULL,
  salience REAL NOT NULL,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  last_accessed_at_ms INTEGER,
  next_review_at_ms INTEGER,
  review_count INTEGER NOT NULL DEFAULT 0,
  source_hash TEXT,
  embedding_model TEXT,
  embedding_version INTEGER NOT NULL DEFAULT 1,
  embedding_pending INTEGER NOT NULL DEFAULT 0,
  parent_id TEXT,
  namespace TEXT NOT NULL DEFAULT 'default',
  session_id TEXT,
  entities_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_memories_state ON memories(state);
CREATE INDEX idx_memories_kind ON memories(kind);
CREATE INDEX idx_memories_namespace ON memories(namespace);
CREATE INDEX idx_memories_next_review ON memories(next_review_at_ms);
CREATE INDEX idx_memories_updated ON memories(updated_at_ms);

CREATE VIRTUAL TABLE memory_fts USING fts5(
  memory_id UNINDEXED,
  content,
  summary,
  entities,
  tokenize = 'unicode61'
);

CREATE TABLE memory_versions (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  operation TEXT NOT NULL,
  before_json TEXT,
  after_json TEXT NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT,
  created_at_ms INTEGER NOT NULL,
  UNIQUE(memory_id, version)
);
CREATE INDEX idx_versions_memory ON memory_versions(memory_id);

CREATE TABLE memory_relations (
  from_id TEXT NOT NULL,
  to_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  weight REAL NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (from_id, to_id, relation)
);

CREATE TABLE embeddings (
  memory_id TEXT PRIMARY KEY,
  label INTEGER NOT NULL UNIQUE,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  version INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE TABLE audit_events (
  id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL UNIQUE,
  memory_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  prev_hash TEXT,
  hash TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL
);

CREATE TABLE stm_promotions (
  id TEXT PRIMARY KEY,
  session_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  importance REAL NOT NULL,
  created_at_ms INTEGER NOT NULL,
  processed_at_ms INTEGER
);

CREATE TABLE dream_runs (
  id TEXT PRIMARY KEY,
  started_at_ms INTEGER NOT NULL,
  finished_at_ms INTEGER,
  status TEXT NOT NULL,
  input_memory_ids_json TEXT,
  output_memory_ids_json TEXT,
  summarizer_id TEXT,
  prompt_version TEXT,
  stats_json TEXT
);

CREATE TABLE pattern_observations (
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  timezone TEXT NOT NULL,
  weekday INTEGER NOT NULL,
  hour_bucket INTEGER NOT NULL,
  count INTEGER NOT NULL,
  last_observed_at_ms INTEGER NOT NULL,
  UNIQUE(namespace, entity_key, timezone, weekday, hour_bucket)
);
"#;

const SCHEMA_V2: &str = r#"
ALTER TABLE memories ADD COLUMN consolidation_pending INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_memories_consolidation ON memories(namespace, consolidation_pending);
"#;

fn configure_connection(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "busy_timeout", 15000)?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    Ok(())
}

pub fn open_pool(path: &std::path::Path) -> Result<Pool> {
    // Refuse a future database before configuring WAL or modifying its header.
    if path.exists() {
        let probe = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let disk: usize = probe.pragma_query_value(None, "user_version", |row| row.get(0))?;
        if disk > schema_version() {
            return Err(MemoraError::Migration(format!("database schema {disk} exceeds supported {}", schema_version())));
        }
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    // Migrate on a single connection before the pool warms up, so parallel
    // opens do not race on WAL init / schema changes.
    {
        let mut conn = Connection::open(path)?;
        configure_connection(&conn)?;
        migrate(&mut conn)?;
    }
    let manager =
        r2d2_sqlite::SqliteConnectionManager::file(path).with_init(|c| configure_connection(c));
    // Avoid r2d2's default min_idle=max_size prewarm (8 parallel connects),
    // which commonly logs "database is locked" on SQLite startup.
    let pool = Pool::builder()
        .max_size(8)
        .min_idle(Some(1))
        .build(manager)?;
    Ok(pool)
}

const SCHEMA_V3: &str = r#"
CREATE TABLE household_members (
  namespace TEXT NOT NULL,
  member_id TEXT NOT NULL,
  display_name TEXT NOT NULL,
  role TEXT NOT NULL,
  aliases_json TEXT NOT NULL DEFAULT '[]',
  age_band TEXT,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, member_id)
);
"#;

const SCHEMA_V4: &str = r#"
CREATE TABLE memory_tags (
  namespace TEXT NOT NULL,
  tag_id TEXT NOT NULL,
  display_name TEXT NOT NULL,
  aliases_json TEXT NOT NULL DEFAULT '[]',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, tag_id)
);

ALTER TABLE memories ADD COLUMN tag TEXT NOT NULL DEFAULT '';
ALTER TABLE memories ADD COLUMN subject_ids TEXT NOT NULL DEFAULT '[]';
CREATE INDEX idx_memories_tag ON memories(namespace, tag);
CREATE INDEX idx_memories_subject_ids ON memories(namespace, subject_ids);
"#;

const SCHEMA_V5: &str = r#"
ALTER TABLE memories ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;
-- Legacy promotions copy committed LTM rows without source ids. They cannot
-- be checked for user revocation, so never replay them after upgrade.
UPDATE stm_promotions SET processed_at_ms = CAST(strftime('%s', 'now') AS INTEGER) * 1000
WHERE processed_at_ms IS NULL;
"#;

const SCHEMA_V6: &str = r#"
ALTER TABLE embeddings ADD COLUMN space_id TEXT NOT NULL DEFAULT '';
"#;

const SCHEMA_V7: &str = r#"
CREATE TABLE kg_entities (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  subtype TEXT,
  canonical_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  owner_member_id TEXT,
  privacy TEXT NOT NULL DEFAULT 'family_shared',
  first_seen_at_ms INTEGER NOT NULL,
  last_confirmed_at_ms INTEGER NOT NULL,
  attrs_json TEXT NOT NULL DEFAULT '{}',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, id)
);
CREATE INDEX idx_kg_entities_type ON kg_entities(namespace, entity_type);
CREATE INDEX idx_kg_entities_owner ON kg_entities(namespace, owner_member_id);
CREATE INDEX idx_kg_entities_name ON kg_entities(namespace, canonical_name);

CREATE TABLE kg_aliases (
  namespace TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  scope_member_id TEXT NOT NULL DEFAULT '',
  confirmed INTEGER NOT NULL DEFAULT 1,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, entity_id, alias, scope_member_id)
);
CREATE INDEX idx_kg_aliases_alias ON kg_aliases(namespace, alias);

CREATE TABLE kg_facts (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  object_entity_id TEXT,
  object_value_type TEXT,
  object_value_json TEXT,
  owner_member_id TEXT,
  polarity TEXT NOT NULL DEFAULT 'affirmative',
  status TEXT NOT NULL DEFAULT 'active',
  valid_from_ms INTEGER,
  valid_to_ms INTEGER,
  condition_text TEXT,
  exception_text TEXT,
  reason_text TEXT,
  evidence_text TEXT,
  evidence_summary TEXT,
  source_memory_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.8,
  privacy TEXT NOT NULL DEFAULT 'family_shared',
  revision INTEGER NOT NULL DEFAULT 1,
  semantic_key TEXT NOT NULL,
  first_seen_at_ms INTEGER NOT NULL,
  last_confirmed_at_ms INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  attrs_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (namespace, id),
  CHECK (
    (object_entity_id IS NOT NULL AND object_value_json IS NULL)
    OR (object_entity_id IS NULL AND object_value_json IS NOT NULL)
  )
);
CREATE INDEX idx_kg_facts_subject ON kg_facts(namespace, subject_id, relation);
CREATE INDEX idx_kg_facts_object ON kg_facts(namespace, object_entity_id, relation);
CREATE INDEX idx_kg_facts_status ON kg_facts(namespace, status, valid_from_ms, valid_to_ms);
CREATE INDEX idx_kg_facts_owner ON kg_facts(namespace, owner_member_id);
CREATE INDEX idx_kg_facts_semantic ON kg_facts(namespace, semantic_key);
CREATE INDEX idx_kg_facts_source ON kg_facts(namespace, source_memory_id);

CREATE TABLE kg_fact_versions (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  operation TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, id),
  UNIQUE (namespace, fact_id, revision)
);

CREATE TABLE kg_change_events (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  fact_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, id)
);
CREATE INDEX idx_kg_change_fact ON kg_change_events(namespace, fact_id);

CREATE TABLE kg_resource_domains (
  namespace TEXT NOT NULL,
  resource_kind TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  domain TEXT NOT NULL,
  PRIMARY KEY (namespace, resource_kind, resource_id, domain)
);

CREATE TABLE kg_resource_owners (
  namespace TEXT NOT NULL,
  resource_kind TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  owner_member_id TEXT NOT NULL,
  PRIMARY KEY (namespace, resource_kind, resource_id, owner_member_id)
);

CREATE TABLE kg_grants (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  grantor_member_id TEXT NOT NULL,
  grantee_member_id TEXT NOT NULL,
  scope_json TEXT NOT NULL,
  operations_json TEXT NOT NULL,
  valid_from_ms INTEGER,
  valid_to_ms INTEGER,
  revoked INTEGER NOT NULL DEFAULT 0,
  evidence_text TEXT,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, id)
);
CREATE INDEX idx_kg_grants_grantee ON kg_grants(namespace, grantee_member_id, revoked);

CREATE TABLE kg_pending_items (
  namespace TEXT NOT NULL,
  id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, id)
);

CREATE TABLE kg_ingest_requests (
  namespace TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  source_memory_id TEXT,
  status TEXT NOT NULL,
  report_json TEXT,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  PRIMARY KEY (namespace, idempotency_key)
);
"#;

const SCHEMA_V8: &str = r#"
CREATE TABLE kg_fact_sources (
  namespace TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  source_memory_id TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY(namespace, fact_id, source_memory_id),
  FOREIGN KEY(namespace, fact_id) REFERENCES kg_facts(namespace, id) ON DELETE CASCADE
);
CREATE INDEX idx_kg_fact_sources_memory ON kg_fact_sources(namespace, source_memory_id);
INSERT OR IGNORE INTO kg_fact_sources(namespace, fact_id, source_memory_id, created_at_ms)
SELECT namespace, id, source_memory_id, created_at_ms FROM kg_facts WHERE source_memory_id IS NOT NULL;

-- Durable projection work: network failures must not silently lose graph facts.
CREATE TABLE graph_projection_jobs (
  memory_id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  revision INTEGER NOT NULL,
  source_text TEXT NOT NULL,
  speaker_member_id TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at_ms INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE INDEX idx_graph_projection_due ON graph_projection_jobs(next_attempt_at_ms);
"#;

const SCHEMA_V9: &str = r#"
ALTER TABLE kg_fact_sources ADD COLUMN evidence_text TEXT;
ALTER TABLE kg_fact_sources ADD COLUMN evidence_summary TEXT;
-- Historic evidence has only one known author. Never copy it to every source.
UPDATE kg_fact_sources SET
 evidence_text = (SELECT f.evidence_text FROM kg_facts f WHERE f.namespace = kg_fact_sources.namespace AND f.id = kg_fact_sources.fact_id),
 evidence_summary = (SELECT f.evidence_summary FROM kg_facts f WHERE f.namespace = kg_fact_sources.namespace AND f.id = kg_fact_sources.fact_id)
WHERE EXISTS (SELECT 1 FROM kg_facts f WHERE f.namespace = kg_fact_sources.namespace AND f.id = kg_fact_sources.fact_id
 AND COALESCE(f.source_memory_id, '') = kg_fact_sources.source_memory_id);
-- Preserve direct lookup of archived atoms replaced by a summary.
UPDATE memories SET metadata_json = json_set(metadata_json, '$.recall_archived', json('true'))
WHERE state = 'archived' AND EXISTS (SELECT 1 FROM memory_relations r
 WHERE r.to_id = memories.id AND r.relation IN ('summarizes', 'derived_from'));
"#;

// Compatibility with databases produced by the previously shipped Android library.
// Align FTS rowids with memories and retain its two lookup indexes. Never lower user_version.
const SCHEMA_V10: &str = r#"
CREATE TEMP TABLE fts_rekey AS
  SELECT m.rowid AS rid, f.memory_id, f.content, f.summary, f.entities
  FROM memory_fts f JOIN memories m ON m.id = f.memory_id;
DELETE FROM memory_fts;
INSERT INTO memory_fts(rowid, memory_id, content, summary, entities)
  SELECT rid, memory_id, content, summary, entities FROM fts_rekey;
DROP TABLE fts_rekey;
CREATE INDEX IF NOT EXISTS idx_memory_relations_to ON memory_relations(to_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_memory ON audit_events(memory_id);
"#;

// Third memory type: original txt/md excerpts behind attachment facts.
const SCHEMA_V11: &str = r#"
CREATE TABLE memory_chunks (
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  content TEXT NOT NULL,
  subject_ids TEXT NOT NULL DEFAULT '[]',
  tags TEXT NOT NULL DEFAULT '[]',
  source_kind TEXT NOT NULL,
  source_name TEXT,
  source_path TEXT,
  source_locator TEXT,
  session_id TEXT,
  message_id TEXT,
  start_char INTEGER NOT NULL,
  end_char INTEGER NOT NULL,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  UNIQUE(namespace, content_sha256)
);
CREATE VIRTUAL TABLE memory_chunk_fts USING fts5(
  chunk_id UNINDEXED,
  content,
  tokenize = 'unicode61'
);
CREATE TABLE memory_chunk_links (
  namespace TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  PRIMARY KEY (memory_id, chunk_id)
);
CREATE INDEX idx_memory_chunk_links_chunk ON memory_chunk_links(chunk_id);
"#;

// Original excerpts remain available by fact ID, with no searchable index.
const SCHEMA_V12: &str = r#"
DROP TABLE IF EXISTS memory_chunk_fts;
"#;

pub const MIGRATION_SQL: &[&str] = &[
    SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5,
    SCHEMA_V6, SCHEMA_V7, SCHEMA_V8, SCHEMA_V9, SCHEMA_V10,
    SCHEMA_V11, SCHEMA_V12,
];

pub fn schema_version() -> usize { MIGRATION_SQL.len() }

fn migrations() -> Migrations<'static> {
    Migrations::new(MIGRATION_SQL.iter().copied().map(M::up).collect())
}

pub fn migrate(conn: &mut Connection) -> Result<()> {
    migrations().to_latest(conn)?;
    Ok(())
}

pub fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let id_s: String = row.get("id")?;
    let parent: Option<String> = row.get("parent_id")?;
    let entities_json: String = row.get("entities_json")?;
    let metadata_json: String = row.get("metadata_json")?;
    let subject_ids_json: String = row.get("subject_ids").unwrap_or_else(|_| "[]".into());
    let pending: i64 = row.get("embedding_pending")?;
    let consolidation: i64 = row.get("consolidation_pending").unwrap_or(0);
    Ok(MemoryRecord {
        id: Uuid::parse_str(&id_s).unwrap_or(Uuid::nil()),
        revision: row.get("revision")?,
        kind: MemoryKind::parse(&row.get::<_, String>("kind")?).unwrap_or(MemoryKind::Episodic),
        state: MemoryState::parse(&row.get::<_, String>("state")?).unwrap_or(MemoryState::Active),
        content: row.get("content")?,
        summary: row.get("summary")?,
        importance: row.get("importance")?,
        confidence: row.get("confidence")?,
        salience: row.get("salience")?,
        created_at_ms: row.get("created_at_ms")?,
        updated_at_ms: row.get("updated_at_ms")?,
        last_accessed_at_ms: row.get("last_accessed_at_ms")?,
        next_review_at_ms: row.get("next_review_at_ms")?,
        review_count: row.get("review_count")?,
        source_hash: row.get("source_hash")?,
        embedding_model: row.get("embedding_model")?,
        embedding_version: row.get("embedding_version")?,
        embedding_pending: pending != 0,
        consolidation_pending: consolidation != 0,
        parent_id: parent.and_then(|s| Uuid::parse_str(&s).ok()),
        namespace: row.get("namespace")?,
        session_id: row.get("session_id")?,
        entities: serde_json::from_str(&entities_json).unwrap_or_default(),
        metadata: serde_json::from_str(&metadata_json).unwrap_or(serde_json::json!({})),
        tag: row.get::<_, String>("tag").unwrap_or_default(),
        subject_ids: serde_json::from_str(&subject_ids_json).unwrap_or_default(),
    })
}

pub fn insert_memory(conn: &Connection, rec: &MemoryRecord, jieba: &Jieba) -> Result<()> {
    conn.execute(
        r#"INSERT INTO memories (
            id, kind, state, content, summary, importance, confidence, salience,
            created_at_ms, updated_at_ms, last_accessed_at_ms, next_review_at_ms, review_count,
            source_hash, embedding_model, embedding_version, embedding_pending, consolidation_pending,
            parent_id, namespace, session_id, entities_json, metadata_json, tag, subject_ids, revision
        ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26)"#,
        params![
            rec.id.to_string(),
            rec.kind.as_str(),
            rec.state.as_str(),
            rec.content,
            rec.summary,
            rec.importance,
            rec.confidence,
            rec.salience,
            rec.created_at_ms,
            rec.updated_at_ms,
            rec.last_accessed_at_ms,
            rec.next_review_at_ms,
            rec.review_count,
            rec.source_hash,
            rec.embedding_model,
            rec.embedding_version,
            if rec.embedding_pending { 1 } else { 0 },
            if rec.consolidation_pending { 1 } else { 0 },
            rec.parent_id.map(|id| id.to_string()),
            rec.namespace,
            rec.session_id,
            serde_json::to_string(&rec.entities)?,
            rec.metadata.to_string(),
            rec.tag,
            serde_json::to_string(&rec.subject_ids)?,
            rec.revision,
        ],
    )?;
    upsert_fts(conn, rec, jieba)?;
    Ok(())
}

pub fn update_memory(conn: &Connection, rec: &mut MemoryRecord, jieba: &Jieba) -> Result<()> {
    let changed = conn.execute(
        r#"UPDATE memories SET
            kind=?2, state=?3, content=?4, summary=?5, importance=?6, confidence=?7, salience=?8,
            updated_at_ms=?9, last_accessed_at_ms=?10, next_review_at_ms=?11, review_count=?12,
            source_hash=?13, embedding_model=?14, embedding_version=?15, embedding_pending=?16,
            consolidation_pending=?17, parent_id=?18, namespace=?19, session_id=?20,
            entities_json=?21, metadata_json=?22, tag=?23, subject_ids=?24, revision=revision+1
          WHERE id=?1 AND revision=?25"#,
        params![
            rec.id.to_string(),
            rec.kind.as_str(),
            rec.state.as_str(),
            rec.content,
            rec.summary,
            rec.importance,
            rec.confidence,
            rec.salience,
            rec.updated_at_ms,
            rec.last_accessed_at_ms,
            rec.next_review_at_ms,
            rec.review_count,
            rec.source_hash,
            rec.embedding_model,
            rec.embedding_version,
            if rec.embedding_pending { 1 } else { 0 },
            if rec.consolidation_pending { 1 } else { 0 },
            rec.parent_id.map(|id| id.to_string()),
            rec.namespace,
            rec.session_id,
            serde_json::to_string(&rec.entities)?,
            rec.metadata.to_string(),
            rec.tag,
            serde_json::to_string(&rec.subject_ids)?,
            rec.revision,
        ],
    )?;
    if changed != 1 {
        return Err(MemoraError::InvalidState(format!(
            "stale memory revision: {}",
            rec.id
        )));
    }
    rec.revision += 1;
    upsert_fts(conn, rec, jieba)?;
    Ok(())
}

/// Archived atoms supporting a summary remain directly searchable. Ordinary
/// low-value archives, forgotten and superseded rows are still excluded.
pub fn memory_is_recallable(rec: &MemoryRecord) -> bool {
    // 兼容已落库的旧工具记录；仅展示事实仍可由list供UI读取。
    if rec.metadata["display_only"] == true || rec.metadata["record_type"] == "practice_results" {
        return false;
    }
    rec.state == MemoryState::Active
        || (rec.state == MemoryState::Archived
            && rec
                .metadata
                .get("recall_archived")
                .and_then(serde_json::Value::as_bool)
                == Some(true))
}

pub fn upsert_fts(conn: &Connection, rec: &MemoryRecord, jieba: &Jieba) -> Result<()> {
    conn.execute(
        "DELETE FROM memory_fts WHERE memory_id = ?1",
        params![rec.id.to_string()],
    )?;
    if !memory_is_recallable(rec) {
        return Ok(());
    }
    let content = tokenize_for_fts(jieba, &rec.content);
    let summary = tokenize_for_fts(jieba, rec.summary.as_deref().unwrap_or(""));
    // The owner is searchable even when the sentence itself never names them.
    let entities = tokenize_for_fts(jieba, &format!("{} {}", rec.entities.join(" "), rec.subject_ids.join(" ")));
    conn.execute(
        "INSERT INTO memory_fts(rowid, memory_id, content, summary, entities)
         VALUES ((SELECT rowid FROM memories WHERE id = ?1),?1,?2,?3,?4)",
        params![rec.id.to_string(), content, summary, entities],
    )?;
    Ok(())
}

pub fn remove_fts(conn: &Connection, id: MemoryId) -> Result<()> {
    conn.execute(
        "DELETE FROM memory_fts WHERE memory_id = ?1",
        params![id.to_string()],
    )?;
    Ok(())
}

pub fn get_memory(conn: &Connection, id: MemoryId) -> Result<Option<MemoryRecord>> {
    let mut stmt = conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
    let rec = stmt
        .query_row(params![id.to_string()], row_to_memory)
        .optional()?;
    Ok(rec)
}

/// Batch-load authorized candidates without preparing one SELECT per ID.
/// Namespace is enforced in SQL; lifecycle and reader filters remain caller-owned.
pub(crate) fn get_memories_by_ids(
    conn: &Connection,
    namespace: &str,
    ids: &[MemoryId],
) -> Result<Vec<MemoryRecord>> {
    let mut records = Vec::new();
    for chunk in ids.chunks(400) {
        let placeholders = std::iter::repeat("?")
            .take(chunk.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT * FROM memories WHERE namespace = ? AND id IN ({placeholders})");
        let mut values = vec![namespace.to_string()];
        values.extend(chunk.iter().map(ToString::to_string));
        let mut statement = conn.prepare(&sql)?;
        for row in statement.query_map(rusqlite::params_from_iter(values), row_to_memory)? {
            records.push(row?);
        }
    }
    Ok(records)
}

/// Invalidate every derived projection in the same transaction as a source
/// correction/revocation. A summary must not retain facts its source withdrew.
pub fn invalidate_derivatives(
    conn: &Connection,
    source: MemoryId,
    jieba: &Jieba,
) -> Result<Vec<MemoryId>> {
    let ids: Vec<MemoryId> = {
        let mut stmt = conn.prepare(
            "WITH RECURSIVE dependents(id) AS (
                SELECT from_id FROM memory_relations
                WHERE to_id = ?1 AND relation IN ('derived_from','summarizes','supersedes')
                UNION
                SELECT r.from_id FROM memory_relations r JOIN dependents d ON r.to_id = d.id
                WHERE r.relation IN ('derived_from','summarizes','supersedes')
             ) SELECT id FROM dependents WHERE id != ?1",
        )?;
        let rows = stmt.query_map(params![source.to_string()], |row| {
            Ok(Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or(Uuid::nil()))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for id in &ids {
        if let Some(mut rec) = get_memory(conn, *id)? {
            if rec.state == MemoryState::Forgotten {
                continue;
            }
            let before = rec.clone();
            rec.state = MemoryState::Forgotten;
            rec.embedding_pending = false;
            rec.consolidation_pending = false;
            rec.updated_at_ms = now_ms();
            update_memory(conn, &mut rec, jieba)?;
            delete_embedding(conn, *id)?;
            insert_version(
                conn,
                *id,
                "invalidate_source",
                Some(&before),
                &rec,
                "engine",
                None,
            )?;
            append_audit(
                conn,
                Some(*id),
                "invalidate_source",
                serde_json::json!({"source": source}),
            )?;
        }
    }
    Ok(ids)
}

pub fn delete_memory(conn: &Connection, id: MemoryId) -> Result<()> {
    crate::chunks::unlink_memory(conn, &id.to_string())?;
    remove_fts(conn, id)?;
    conn.execute(
        "DELETE FROM embeddings WHERE memory_id = ?1",
        params![id.to_string()],
    )?;
    conn.execute(
        "DELETE FROM memory_relations WHERE from_id = ?1 OR to_id = ?1",
        params![id.to_string()],
    )?;
    conn.execute(
        "DELETE FROM memories WHERE id = ?1",
        params![id.to_string()],
    )?;
    Ok(())
}

pub fn next_version(conn: &Connection, id: MemoryId) -> Result<i64> {
    let n: i64 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM memory_versions WHERE memory_id = ?1",
        params![id.to_string()],
        |r| r.get(0),
    )?;
    Ok(n + 1)
}

pub fn insert_version(
    conn: &Connection,
    memory_id: MemoryId,
    operation: &str,
    before: Option<&MemoryRecord>,
    after: &MemoryRecord,
    actor: &str,
    reason: Option<&str>,
) -> Result<()> {
    let version = next_version(conn, memory_id)?;
    conn.execute(
        r#"INSERT INTO memory_versions
            (id, memory_id, version, operation, before_json, after_json, actor, reason, created_at_ms)
           VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)"#,
        params![
            Uuid::now_v7().to_string(),
            memory_id.to_string(),
            version,
            operation,
            before.map(serde_json::to_string).transpose()?,
            serde_json::to_string(after)?,
            actor,
            reason,
            now_ms(),
        ],
    )?;
    Ok(())
}

pub fn list_versions(conn: &Connection, memory_id: MemoryId) -> Result<Vec<MemoryVersion>> {
    let mut stmt = conn.prepare(
        "SELECT id, memory_id, version, operation, before_json, after_json, actor, reason, created_at_ms
         FROM memory_versions WHERE memory_id = ?1 ORDER BY version ASC",
    )?;
    let rows = stmt.query_map(params![memory_id.to_string()], |r| {
        Ok(MemoryVersion {
            id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            memory_id: Uuid::parse_str(&r.get::<_, String>(1)?).unwrap_or(Uuid::nil()),
            version: r.get(2)?,
            operation: r.get(3)?,
            before_json: r.get(4)?,
            after_json: r.get(5)?,
            actor: r.get(6)?,
            reason: r.get(7)?,
            created_at_ms: r.get(8)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn insert_relation(
    conn: &Connection,
    from: MemoryId,
    to: MemoryId,
    relation: RelationKind,
    weight: f64,
) -> Result<()> {
    conn.execute(
        r#"INSERT INTO memory_relations(from_id, to_id, relation, weight, created_at_ms)
           VALUES (?1,?2,?3,?4,?5)
           ON CONFLICT(from_id, to_id, relation) DO UPDATE SET weight = excluded.weight"#,
        params![
            from.to_string(),
            to.to_string(),
            relation.as_str(),
            weight,
            now_ms(),
        ],
    )?;
    Ok(())
}

pub fn list_relations(conn: &Connection, id: MemoryId) -> Result<Vec<MemoryRelation>> {
    let mut stmt = conn.prepare(
        "SELECT from_id, to_id, relation, weight, created_at_ms FROM memory_relations
         WHERE from_id = ?1 OR to_id = ?1",
    )?;
    let rows = stmt.query_map(params![id.to_string()], |r| {
        Ok(MemoryRelation {
            from_id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            to_id: Uuid::parse_str(&r.get::<_, String>(1)?).unwrap_or(Uuid::nil()),
            relation: RelationKind::parse(&r.get::<_, String>(2)?)
                .unwrap_or(RelationKind::Supports),
            weight: r.get(3)?,
            created_at_ms: r.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn has_relations(conn: &Connection, id: MemoryId) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memory_relations WHERE from_id = ?1 OR to_id = ?1",
        params![id.to_string()],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

pub fn append_audit(
    conn: &Connection,
    memory_id: Option<MemoryId>,
    event_type: &str,
    payload: serde_json::Value,
) -> Result<AuditEvent> {
    let (seq, prev_hash): (i64, Option<String>) = conn
        .query_row(
            "SELECT seq, hash FROM audit_events ORDER BY seq DESC LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
        .unwrap_or((0, None));
    let seq = seq + 1;
    let id = Uuid::now_v7();
    let payload_json = payload.to_string();
    let created_at_ms = now_ms();
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.as_deref().unwrap_or("").as_bytes());
    hasher.update(payload_json.as_bytes());
    hasher.update(id.as_bytes());
    hasher.update(seq.to_le_bytes());
    hasher.update(event_type.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    conn.execute(
        r#"INSERT INTO audit_events(id, seq, memory_id, event_type, payload_json, prev_hash, hash, created_at_ms)
           VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"#,
        params![
            id.to_string(),
            seq,
            memory_id.map(|x| x.to_string()),
            event_type,
            payload_json,
            prev_hash,
            hash,
            created_at_ms,
        ],
    )?;
    Ok(AuditEvent {
        id,
        seq,
        memory_id,
        event_type: event_type.into(),
        payload_json: payload.to_string(),
        prev_hash: prev_hash.clone(),
        hash,
        created_at_ms,
    })
}

pub fn list_audit(conn: &Connection, memory_id: MemoryId, limit: i64) -> Result<Vec<AuditEvent>> {
    let mut stmt = conn.prepare(
        "SELECT id, seq, memory_id, event_type, payload_json, prev_hash, hash, created_at_ms
         FROM audit_events WHERE memory_id = ?1 ORDER BY seq DESC LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![memory_id.to_string(), limit], |r| {
        Ok(AuditEvent {
            id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            seq: r.get(1)?,
            memory_id: r
                .get::<_, Option<String>>(2)?
                .and_then(|s| Uuid::parse_str(&s).ok()),
            event_type: r.get(3)?,
            payload_json: r.get(4)?,
            prev_hash: r.get(5)?,
            hash: r.get(6)?,
            created_at_ms: r.get(7)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn fts_search(
    conn: &Connection,
    match_query: &str,
    namespace: &str,
    top_k: usize,
    query: &crate::types::RecallQuery,
) -> Result<Vec<(MemoryId, f64)>> {
    if match_query.trim().is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    // Hard filters precede ranking/limit. Legacy domain filters use the same
    // authoritative Rust predicate before counting a candidate.
    let prefix = "SELECT m.*, bm25(memory_fts) AS score FROM memory_fts f
        JOIN memories m ON m.id = f.memory_id
        WHERE memory_fts MATCH ? AND ";
    let mut sql = format!("{prefix}(m.state = 'active' OR (m.state = 'archived' AND json_extract(m.metadata_json, '$.recall_archived') = 1)) AND m.namespace = ?");
    use rusqlite::types::Value;
    let mut binds = vec![
        Value::Text(match_query.into()),
        Value::Text(namespace.into()),
    ];
    if let Some(mid) = query.member_id.as_deref().filter(|s| !s.trim().is_empty()) {
        let shared = crate::household::is_shared_member_id(mid);
        sql.push_str(&owner_filter_sql("m.", &format!("?{}", binds.len() + 1), shared));
        binds.push(Value::Text(if shared { serde_json::to_string(crate::household::SHARED_MEMBER_IDS)? } else { mid.trim().into() }));
    }
    if let Some(tag) = query.tag.as_deref().filter(|s| !s.trim().is_empty()) {
        sql.push_str(
            " AND COALESCE(NULLIF(m.tag,''), json_extract(m.metadata_json,'$.tag'),'') = ?",
        );
        binds.push(Value::Text(tag.trim().into()));
    }
    if !query.kinds.is_empty() {
        sql.push_str(" AND m.kind IN (");
        sql.push_str(&vec!["?"; query.kinds.len()].join(","));
        sql.push(')');
        for kind in &query.kinds {
            binds.push(Value::Text(kind.as_str().into()));
        }
    }
    if let Some(from) = query.time_from_ms {
        sql.push_str(" AND m.created_at_ms >= ?");
        binds.push(Value::Integer(from));
    }
    if let Some(to) = query.time_to_ms {
        sql.push_str(" AND m.created_at_ms <= ?");
        binds.push(Value::Integer(to));
    }
    sql.push_str(" ORDER BY score");
    let hits = filtered_fts_rows(conn, &sql, &binds, query, top_k)?;
    // Jieba may index 每周六 as 每周 + 六 while a 周六 query is one token.
    // On a short, exact CJK miss, reuse every namespace/owner/lifecycle/privacy
    // filter against literal text. Existing indexes need no migration.
    let literal = query.text.trim();
    if !hits.is_empty() || !(2..=16).contains(&literal.chars().count())
        || !literal.chars().all(|c| ('\u{3400}'..='\u{9fff}').contains(&c)) {
        return Ok(hits);
    }
    let fallback = sql.replacen(prefix,
        "SELECT m.*, 0.0 AS score FROM memories m WHERE (m.content LIKE ?1 OR m.summary LIKE ?1) AND ", 1);
    binds[0] = Value::Text(format!("%{literal}%"));
    filtered_fts_rows(conn, &fallback, &binds, query, top_k)
}

fn filtered_fts_rows(conn: &Connection, sql: &str, binds: &[rusqlite::types::Value],
    query: &crate::types::RecallQuery, top_k: usize) -> Result<Vec<(MemoryId, f64)>> {
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(binds.iter()), |r| {
        Ok((row_to_memory(r)?, r.get::<_, f64>("score")?))
    })?;
    let mut hits = Vec::new();
    for row in rows {
        let (rec, score) = row?;
        if crate::household::matches_member_and_tag(
            &rec.metadata,
            &rec.subject_ids,
            &rec.tag,
            query.member_id.as_deref(),
            query.tag.as_deref(),
            &query.domains,
        ) && crate::household::visible_to_viewer(
            &rec.metadata,
            &rec.subject_ids,
            query.viewer_member_id.as_deref(),
        ) && (!rec
            .metadata
            .get("graph_only")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false))
        {
            hits.push((rec.id, score));
            if hits.len() >= top_k {
                break;
            }
        }
    }
    Ok(hits)
}

/// Index tokens depend on jieba's word boundaries: the row says 车牌号, the question says
/// 车牌, and FTS finds nothing. A substring scan does not care. Rows that cover most of the
/// question come first. Local and bounded: a few milliseconds on thousands of rows.
pub(crate) fn keyword_like_search(
    conn: &Connection,
    namespace: &str,
    keywords: &[String],
    limit: usize,
) -> Result<Vec<MemoryId>> {
    let spellings = crate::chunks::keyword_spellings(keywords, 24);
    if spellings.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let clauses = vec!["content LIKE ?"; spellings.len()].join(" OR ");
    let sql = format!(
        "SELECT id, content, subject_ids FROM memories WHERE namespace = ? AND state = 'active' \
         AND COALESCE(json_extract(metadata_json, '$.graph_only'), 0) = 0 AND ({clauses})"
    );
    let mut binds = vec![namespace.to_string()];
    binds.extend(spellings.iter().map(|s| format!("%{}%", s.replace(['%', '_'], ""))));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(binds.iter()), |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
    })?;
    let mut scored: Vec<(f64, MemoryId)> = Vec::new();
    for row in rows {
        let (id, content, subjects) = row?;
        let Ok(id) = Uuid::parse_str(&id) else { continue };
        let (_, coverage) = crate::chunks::keyword_coverage(keywords, &format!("{content} {subjects}"));
        scored.push((coverage, id));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(limit).map(|(_, id)| id).collect())
}

pub fn next_embedding_label(conn: &Connection) -> Result<i64> {
    let n: i64 = conn.query_row("SELECT COALESCE(MAX(label), 0) FROM embeddings", [], |r| {
        r.get(0)
    })?;
    Ok(n + 1)
}

pub fn upsert_embedding(
    conn: &Connection,
    memory_id: MemoryId,
    label: i64,
    model: &str,
    dim: usize,
    version: i64,
    vector: &[u8],
) -> Result<()> {
    conn.execute(
        r#"INSERT INTO embeddings(memory_id, label, model, dim, version, vector, created_at_ms)
           VALUES (?1,?2,?3,?4,?5,?6,?7)
           ON CONFLICT(memory_id) DO UPDATE SET
             label=excluded.label, model=excluded.model, dim=excluded.dim,
             version=excluded.version, vector=excluded.vector, created_at_ms=excluded.created_at_ms"#,
        params![
            memory_id.to_string(),
            label,
            model,
            dim as i64,
            version,
            vector,
            now_ms(),
        ],
    )?;
    conn.execute(
        "UPDATE memories SET embedding_pending = 0, embedding_model = ?2, embedding_version = ?3 WHERE id = ?1",
        params![memory_id.to_string(), model, version],
    )?;
    Ok(())
}

pub fn get_embedding(conn: &Connection, memory_id: MemoryId) -> Result<Option<(i64, Vec<f32>)>> {
    let mut stmt = conn.prepare("SELECT label, vector FROM embeddings WHERE memory_id = ?1")?;
    let row = stmt
        .query_row(params![memory_id.to_string()], |r| {
            let label: i64 = r.get(0)?;
            let blob: Vec<u8> = r.get(1)?;
            Ok((label, bytes_to_vector(&blob)))
        })
        .optional()?;
    Ok(row)
}

/// Legacy vectors have no space identity and must be re-embedded.
pub fn prepare_vector_space(
    conn: &mut Connection,
    space: &str,
    model: &str,
    dim: usize,
) -> Result<()> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute("DELETE FROM embeddings WHERE space_id != ?1 OR model != ?3 OR (?2 > 0 AND dim != ?2)
        OR dim <= 0 OR length(vector) != dim * 4
        OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_id
            AND m.state = 'active' AND m.embedding_version = embeddings.version AND m.embedding_pending = 0)",
        params![space, dim as i64, model])?;
    tx.execute("UPDATE memories SET embedding_pending = 1, embedding_model = ?1
        WHERE state = 'active' AND tag != '错题集' AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.memory_id = memories.id)", params![model])?;
    tx.commit()?;
    Ok(())
}

pub fn load_all_embeddings(
    conn: &Connection,
    space: &str,
    dim: usize,
) -> Result<Vec<(MemoryId, i64, Vec<f32>)>> {
    let mut stmt = conn.prepare(
        "SELECT e.memory_id, e.label, e.vector FROM embeddings e JOIN memories m ON m.id = e.memory_id
         WHERE m.state = 'active' AND m.embedding_pending = 0 AND e.version = m.embedding_version
            AND e.space_id = ?1 AND (?2 = 0 OR e.dim = ?2) AND length(e.vector) = e.dim * 4",
    )?;
    let rows = stmt.query_map(params![space, dim as i64], |r| {
        let id = Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil());
        let label: i64 = r.get(1)?;
        let blob: Vec<u8> = r.get(2)?;
        Ok((id, label, bytes_to_vector(&blob)))
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn delete_embedding(conn: &Connection, memory_id: MemoryId) -> Result<()> {
    conn.execute(
        "DELETE FROM embeddings WHERE memory_id = ?1",
        params![memory_id.to_string()],
    )?;
    Ok(())
}

pub fn bump_pattern(
    conn: &Connection,
    namespace: &str,
    entity_key: &str,
    timezone: &str,
    weekday: i64,
    hour_bucket: i64,
) -> Result<()> {
    conn.execute(
        r#"INSERT INTO pattern_observations(id, namespace, entity_key, timezone, weekday, hour_bucket, count, last_observed_at_ms)
           VALUES (?1,?2,?3,?4,?5,?6,1,?7)
           ON CONFLICT(namespace, entity_key, timezone, weekday, hour_bucket)
           DO UPDATE SET count = count + 1, last_observed_at_ms = excluded.last_observed_at_ms"#,
        params![
            Uuid::now_v7().to_string(),
            namespace,
            entity_key,
            timezone,
            weekday,
            hour_bucket,
            now_ms(),
        ],
    )?;
    Ok(())
}

pub struct PatternRow {
    pub entity_key: String,
    pub timezone: String,
    pub weekday: i64,
    pub hour_bucket: i64,
    pub count: i64,
}

pub fn mature_patterns(
    conn: &Connection,
    namespace: &str,
    support: i64,
) -> Result<Vec<PatternRow>> {
    let mut stmt = conn.prepare(
        "SELECT entity_key, timezone, weekday, hour_bucket, count
         FROM pattern_observations WHERE namespace = ?1 AND count >= ?2",
    )?;
    let rows = stmt.query_map(params![namespace, support], |r| {
        Ok(PatternRow {
            entity_key: r.get(0)?,
            timezone: r.get(1)?,
            weekday: r.get(2)?,
            hour_bucket: r.get(3)?,
            count: r.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn insert_dream_run(
    conn: &Connection,
    started_at_ms: i64,
    finished_at_ms: i64,
    status: &str,
    input_ids: &[MemoryId],
    output_ids: &[MemoryId],
    summarizer_id: &str,
    prompt_version: &str,
    stats: serde_json::Value,
) -> Result<()> {
    conn.execute(
        r#"INSERT INTO dream_runs(
            id, started_at_ms, finished_at_ms, status,
            input_memory_ids_json, output_memory_ids_json,
            summarizer_id, prompt_version, stats_json
        ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)"#,
        params![
            Uuid::now_v7().to_string(),
            started_at_ms,
            finished_at_ms,
            status,
            serde_json::to_string(input_ids)?,
            serde_json::to_string(output_ids)?,
            summarizer_id,
            prompt_version,
            stats.to_string(),
        ],
    )?;
    Ok(())
}

// The same owner filter precedes both FTS ranking and list pagination/counts.
// Prefix and parameter are internal SQL fragments, never user-supplied strings.
fn owner_filter_sql(prefix: &str, parameter: &str, shared: bool) -> String {
    let ids = format!("{prefix}subject_ids");
    let meta = format!("{prefix}metadata_json");
    if shared {
        format!(" AND (EXISTS (SELECT 1 FROM json_each({ids}) WHERE lower(trim(value)) IN (SELECT value FROM json_each({parameter})))
            OR EXISTS (SELECT 1 FROM json_each({meta}, '$.subjects') WHERE lower(trim(json_extract(value, '$.member_id'))) IN (SELECT value FROM json_each({parameter})))
            OR (json_extract({meta}, '$.scope') = 'public' AND COALESCE(json_array_length({ids}),0) = 0 AND COALESCE(json_array_length({meta}, '$.subjects'),0) = 0))")
    } else {
        format!(" AND (EXISTS (SELECT 1 FROM json_each({ids}) WHERE value = {parameter})
            OR EXISTS (SELECT 1 FROM json_each({meta}, '$.subjects') WHERE json_extract(value, '$.member_id') = {parameter}))")
    }
}

pub fn list_memories_page(
    conn: &Connection,
    namespace: &str,
    state: Option<MemoryState>,
    kind: Option<MemoryKind>,
    keyword: Option<&str>,
    member_id: Option<&str>,
    tag: Option<&str>,
    limit: i64,
    offset: i64,
    facts_only: bool,
) -> Result<(Vec<MemoryRecord>, i64)> {
    let limit = limit.clamp(1, 200);
    let offset = offset.max(0);
    let keyword = keyword.map(str::trim).filter(|s| !s.is_empty());
    let member_id = member_id.map(str::trim).filter(|s| !s.is_empty());
    let tag = tag.map(str::trim).filter(|s| !s.is_empty());

    let mut where_sql = String::from("namespace = ?1");
    let mut binds: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(namespace.to_string())];
    let mut idx = 2u32;

    if facts_only {
        // Summary-derived atoms remain valid facts; ordinary archived/deleted facts do not.
        // A dream summary can be kind=semantic, so kind alone cannot identify it.
        where_sql.push_str(" AND (state = 'active' OR (state = 'archived' AND json_extract(metadata_json, '$.recall_archived') = 1)) AND kind NOT IN ('summary', 'pattern') AND NOT (COALESCE(json_extract(metadata_json, '$.dream'), 0) = 1 AND COALESCE(json_extract(metadata_json, '$.support_count'), 0) > 0)");
    } else if let Some(state) = state {
        where_sql.push_str(&format!(" AND state = ?{idx}"));
        binds.push(Box::new(state.as_str().to_string()));
        idx += 1;
    }
    if let Some(kind) = kind {
        where_sql.push_str(&format!(" AND kind = ?{idx}"));
        binds.push(Box::new(kind.as_str().to_string()));
        idx += 1;
    }
    if let Some(tag) = tag {
        where_sql.push_str(&format!(" AND tag = ?{idx}"));
        binds.push(Box::new(tag.to_string()));
        idx += 1;
    }
    if let Some(member_id) = member_id {
        let shared = crate::household::is_shared_member_id(member_id);
        where_sql.push_str(&owner_filter_sql("", &format!("?{idx}"), shared));
        binds.push(Box::new(if shared { serde_json::to_string(crate::household::SHARED_MEMBER_IDS)? } else { member_id.to_string() }));
        idx += 1;
    }
    if let Some(keyword) = keyword {
        where_sql.push_str(&format!(
            " AND (content LIKE ?{idx} ESCAPE '\\' OR IFNULL(summary, '') LIKE ?{idx} ESCAPE '\\')"
        ));
        let escaped = keyword
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        binds.push(Box::new(format!("%{escaped}%")));
        idx += 1;
    }

    let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {where_sql}");
    let bind_refs: Vec<&dyn rusqlite::types::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let total: i64 = conn.query_row(&count_sql, bind_refs.as_slice(), |r| r.get(0))?;

    let list_sql = format!(
        "SELECT * FROM memories WHERE {where_sql} ORDER BY updated_at_ms DESC, id DESC LIMIT ?{idx} OFFSET ?{}",
        idx + 1
    );
    binds.push(Box::new(limit));
    binds.push(Box::new(offset));
    let bind_refs: Vec<&dyn rusqlite::types::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&list_sql)?;
    let rows = stmt.query_map(bind_refs.as_slice(), row_to_memory)?;
    Ok((rows.collect::<std::result::Result<Vec<_>, _>>()?, total))
}

pub fn list_memories_for_export(
    conn: &Connection,
    namespace: &str,
    include_forgotten: bool,
) -> Result<Vec<MemoryRecord>> {
    let sql = if include_forgotten {
        "SELECT * FROM memories WHERE namespace = ?1"
    } else {
        "SELECT * FROM memories WHERE namespace = ?1 AND state != 'forgotten'"
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![namespace], row_to_memory)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn list_all_versions(conn: &Connection, namespace: &str) -> Result<Vec<MemoryVersion>> {
    let mut stmt = conn.prepare(
        r#"SELECT v.id, v.memory_id, v.version, v.operation, v.before_json, v.after_json, v.actor, v.reason, v.created_at_ms
           FROM memory_versions v
           JOIN memories m ON m.id = v.memory_id
           WHERE m.namespace = ?1
           ORDER BY v.created_at_ms"#,
    )?;
    let rows = stmt.query_map(params![namespace], |r| {
        Ok(MemoryVersion {
            id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            memory_id: Uuid::parse_str(&r.get::<_, String>(1)?).unwrap_or(Uuid::nil()),
            version: r.get(2)?,
            operation: r.get(3)?,
            before_json: r.get(4)?,
            after_json: r.get(5)?,
            actor: r.get(6)?,
            reason: r.get(7)?,
            created_at_ms: r.get(8)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn list_all_relations(conn: &Connection, namespace: &str) -> Result<Vec<MemoryRelation>> {
    let mut stmt = conn.prepare(
        r#"SELECT r.from_id, r.to_id, r.relation, r.weight, r.created_at_ms
           FROM memory_relations r
           JOIN memories m ON m.id = r.from_id
           WHERE m.namespace = ?1"#,
    )?;
    let rows = stmt.query_map(params![namespace], |r| {
        Ok(MemoryRelation {
            from_id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            to_id: Uuid::parse_str(&r.get::<_, String>(1)?).unwrap_or(Uuid::nil()),
            relation: RelationKind::parse(&r.get::<_, String>(2)?)
                .unwrap_or(RelationKind::Supports),
            weight: r.get(3)?,
            created_at_ms: r.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn list_all_audit(conn: &Connection, namespace: &str) -> Result<Vec<AuditEvent>> {
    let mut stmt = conn.prepare(
        r#"SELECT a.id, a.seq, a.memory_id, a.event_type, a.payload_json, a.prev_hash, a.hash, a.created_at_ms
           FROM audit_events a
           LEFT JOIN memories m ON m.id = a.memory_id
           WHERE a.memory_id IS NULL OR m.namespace = ?1
           ORDER BY a.seq"#,
    )?;
    let rows = stmt.query_map(params![namespace], |r| {
        Ok(AuditEvent {
            id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
            seq: r.get(1)?,
            memory_id: r
                .get::<_, Option<String>>(2)?
                .and_then(|s| Uuid::parse_str(&s).ok()),
            event_type: r.get(3)?,
            payload_json: r.get(4)?,
            prev_hash: r.get(5)?,
            hash: r.get(6)?,
            created_at_ms: r.get(7)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn forget_namespace(conn: &Connection, namespace: &str) -> Result<usize> {
    conn.execute(
        "DELETE FROM memory_fts WHERE memory_id IN (SELECT id FROM memories WHERE namespace = ?1)",
        params![namespace],
    )?;
    let n = conn.execute(
        "UPDATE memories SET state = 'forgotten', updated_at_ms = ?2 WHERE namespace = ?1",
        params![namespace, now_ms()],
    )?;
    Ok(n)
}

pub fn retry_busy<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let mut delay_ms = 20u64;
    for _ in 0..12 {
        match f() {
            Err(MemoraError::Sqlite(e))
                if matches!(
                    e.sqlite_error_code(),
                    Some(rusqlite::ErrorCode::DatabaseBusy)
                        | Some(rusqlite::ErrorCode::DatabaseLocked)
                ) =>
            {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                delay_ms = (delay_ms * 2).min(250);
            }
            other => return other,
        }
    }
    f()
}

pub fn find_pattern(
    conn: &Connection,
    namespace: &str,
    entity_key: &str,
    timezone: &str,
    weekday: i64,
    hour_bucket: i64,
) -> Result<Option<MemoryRecord>> {
    let mut stmt = conn.prepare(
        r#"SELECT * FROM memories
           WHERE namespace = ?1 AND kind = 'pattern' AND state = 'active'
             AND json_extract(metadata_json, '$.timezone') = ?2
             AND json_extract(metadata_json, '$.weekday_mask') = ?3
             AND json_extract(metadata_json, '$.time_bucket') = ?4
             AND entities_json LIKE ?5
           LIMIT 1"#,
    )?;
    let like = format!("%\"{entity_key}\"%");
    let rec = stmt
        .query_row(
            params![namespace, timezone, weekday, hour_bucket, like],
            row_to_memory,
        )
        .optional()?;
    Ok(rec)
}

pub fn touch_memories(conn: &Connection, ids: &[MemoryId], now: i64) -> Result<()> {
    for id in ids {
        conn.execute(
            "UPDATE memories SET last_accessed_at_ms = ?2 WHERE id = ?1 AND state = 'active'",
            params![id.to_string(), now],
        )?;
    }
    Ok(())
}

pub fn pending_embedding_ids(conn: &Connection) -> Result<Vec<MemoryId>> {
    let mut stmt =
        conn.prepare("SELECT id FROM memories WHERE embedding_pending = 1 AND state = 'active' AND tag != '错题集'")?;
    let rows = stmt.query_map([], |r| {
        Ok(Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()))
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn pending_consolidation(
    conn: &Connection,
    namespace: &str,
    limit: usize,
) -> Result<Vec<MemoryRecord>> {
    let mut stmt = conn.prepare(
        r#"SELECT * FROM memories
           WHERE state = 'active' AND namespace = ?1 AND consolidation_pending = 1
           ORDER BY created_at_ms ASC LIMIT ?2"#,
    )?;
    let rows = stmt.query_map(params![namespace, limit as i64], row_to_memory)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Exact-content duplicate groups (same `content` string), namespace-wide.
/// Active candidates only; the engine checks ownership buckets and preserves history.
pub fn find_exact_content_duplicate_groups(
    conn: &Connection,
    namespace: &str,
    max_groups: usize,
) -> Result<Vec<Vec<MemoryRecord>>> {
    let mut stmt = conn.prepare(
        r#"SELECT content FROM memories
           WHERE namespace = ?1
             AND kind IN ('episodic', 'semantic', 'pattern')
             AND state = 'active'
             AND consolidation_pending = 0
             AND length(trim(content)) > 0
           GROUP BY content
           HAVING COUNT(*) > 1
           ORDER BY COUNT(*) DESC
           LIMIT ?2"#,
    )?;
    let contents: Vec<String> = stmt
        .query_map(params![namespace, max_groups as i64], |r| r.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut groups = Vec::with_capacity(contents.len());
    for content in contents {
        let mut load = conn.prepare(
            r#"SELECT * FROM memories
               WHERE namespace = ?1
                 AND kind IN ('episodic', 'semantic', 'pattern')
                 AND state = 'active'
                 AND consolidation_pending = 0
                 AND content = ?2
               ORDER BY
                 CASE state WHEN 'active' THEN 0 ELSE 1 END,
                 updated_at_ms DESC"#,
        )?;
        let rows = load.query_map(params![namespace, content], row_to_memory)?;
        let members: Vec<MemoryRecord> = rows.filter_map(|r| r.ok()).collect();
        if members.len() >= 2 {
            groups.push(members);
        }
    }
    Ok(groups)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingExport {
    pub memory_id: MemoryId,
    #[serde(default)]
    pub space_id: String,
    pub model: String,
    pub dim: i64,
    pub version: i64,
    pub vector: Vec<f32>,
}

pub fn list_embeddings_for_namespace(
    conn: &Connection,
    namespace: &str,
) -> Result<Vec<EmbeddingExport>> {
    let mut stmt = conn.prepare(
        r#"SELECT e.memory_id, e.model, e.dim, e.version, e.vector, e.space_id
           FROM embeddings e
           JOIN memories m ON m.id = e.memory_id
           WHERE m.namespace = ?1"#,
    )?;
    let rows = stmt.query_map(params![namespace], |r| {
        let id = Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil());
        let blob: Vec<u8> = r.get(4)?;
        Ok(EmbeddingExport {
            memory_id: id,
            space_id: r.get(5)?,
            model: r.get(1)?,
            dim: r.get(2)?,
            version: r.get(3)?,
            vector: bytes_to_vector(&blob),
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

#[cfg(test)]
mod safety_tests {
    use super::*;

    #[test]
    fn fact_browse_keeps_archived_atoms_and_paginates_without_summaries() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let insert = |kind: &str, state: &str, content: &str, meta: &str| {
            conn.execute("INSERT INTO memories(id,kind,state,content,importance,confidence,salience,created_at_ms,updated_at_ms,namespace,metadata_json,tag,subject_ids) VALUES(?1,?2,?3,?4,0.5,0.8,0.4,1,1,'default',?5,'金融','[\"张三\"]')",
                params![Uuid::now_v7().to_string(),kind,state,content,meta]).unwrap();
        };
        for i in 0..1005 { insert("episodic", "active", &format!("事实{i}"), "{}"); }
        insert("episodic", "archived", "摘要来源事实", r#"{"recall_archived":true}"#);
        insert("semantic", "active", "合并摘要", r#"{"dream":true,"support_count":3}"#);
        insert("summary", "active", "摘要", "{}");
        insert("pattern", "active", "模式", "{}");
        insert("episodic", "archived", "普通归档", "{}");
        insert("episodic", "forgotten", "已删除事实", r#"{"recall_archived":true}"#);
        insert("episodic", "superseded", "已替换事实", "{}");
        let mut ids=std::collections::HashSet::new();
        for page in 0..11 {
            let (rows,total)=list_memories_page(&conn,"default",Some(MemoryState::Active),None,None,None,None,100,page*100,true).unwrap();
            assert_eq!(total,1006);
            for row in rows { assert!(ids.insert(row.id)); }
        }
        assert_eq!(ids.len(),1006);
        let (rows,total)=list_memories_page(&conn,"default",Some(MemoryState::Active),None,Some("摘要来源"),Some("张三"),Some("金融"),100,0,true).unwrap();
        assert_eq!(total,1); assert_eq!(rows[0].content,"摘要来源事实");
        // Existing callers retain their active-only behavior.
        let (_,total)=list_memories_page(&conn,"default",Some(MemoryState::Active),None,None,None,None,100,0,false).unwrap();
        assert_eq!(total,1008);
    }


    #[test]
    fn migration_v10_rekeys_fts_and_preserves_existing_memory() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrations().to_version(&mut conn, 9).unwrap();
        let id = Uuid::now_v7();
        conn.execute("INSERT INTO memories(id,kind,state,content,importance,confidence,salience,
            created_at_ms,updated_at_ms,namespace) VALUES(?1,'episodic','active','retained',0.5,0.8,0.4,1,1,'default')",
            params![id.to_string()]).unwrap();
        conn.execute("INSERT INTO memory_fts(rowid,memory_id,content,summary,entities) VALUES(999,?1,'retained','','')",
            params![id.to_string()]).unwrap();
        migrate(&mut conn).unwrap();
        assert_eq!(conn.pragma_query_value(None,"user_version",|r|r.get::<_,i64>(0)).unwrap(),schema_version() as i64);
        assert_eq!(get_memory(&conn,id).unwrap().unwrap().content,"retained");
        let aligned: i64 = conn.query_row("SELECT COUNT(*) FROM memory_fts f JOIN memories m ON f.memory_id=m.id AND f.rowid=m.rowid",[],|r|r.get(0)).unwrap();
        assert_eq!(aligned,1);
        let indexes: i64=conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name IN ('idx_memory_relations_to','idx_audit_events_memory')",[],|r|r.get(0)).unwrap();
        assert_eq!(indexes,2);
        // Reopening the v10 DB must be a no-op, including its persisted fact.
        migrate(&mut conn).unwrap();
        assert_eq!(get_memory(&conn,id).unwrap().unwrap().content,"retained");
        let record=get_memory(&conn,id).unwrap().unwrap();
        for _ in 0..3 { upsert_fts(&conn,&record,&Jieba::new()).unwrap(); }
        let aligned: i64 = conn.query_row("SELECT COUNT(*) FROM memory_fts f JOIN memories m ON f.memory_id=m.id AND f.rowid=m.rowid",[],|r|r.get(0)).unwrap();
        assert_eq!(aligned,1);
    }

    #[test]
    fn failed_upgrade_rolls_back_earlier_fts_changes_and_schema_version() {
        let mut conn=Connection::open_in_memory().unwrap();
        migrations().to_version(&mut conn,9).unwrap();
        let id=Uuid::now_v7().to_string();
        conn.execute("INSERT INTO memories(id,kind,state,content,importance,confidence,salience,created_at_ms,updated_at_ms) VALUES(?1,'episodic','active','preserve',0.5,0.8,0.4,1,1)",params![id]).unwrap();
        conn.execute("INSERT INTO memory_fts(rowid,memory_id,content,summary,entities) VALUES(999,?1,'preserve','','')",params![id]).unwrap();
        // Force the final v10 step to fail after its FTS rekey statements ran.
        conn.execute_batch("ALTER TABLE audit_events RENAME TO original_audit_events;").unwrap();
        assert!(migrate(&mut conn).is_err());
        assert_eq!(conn.pragma_query_value(None,"user_version",|r|r.get::<_,i64>(0)).unwrap(),9);
        assert_eq!(conn.query_row("SELECT rowid FROM memory_fts",[],|r|r.get::<_,i64>(0)).unwrap(),999);
        assert_eq!(conn.query_row("SELECT content FROM memories WHERE id=?1",params![id],|r|r.get::<_,String>(0)).unwrap(),"preserve");
    }

    #[test]
    fn future_pool_open_does_not_even_reconfigure_its_header() {
        let dir=tempfile::tempdir().unwrap();let path=dir.path().join("future.db");
        let conn=Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE sentinel(value TEXT); INSERT INTO sentinel VALUES('preserve'); PRAGMA user_version=99;").unwrap();
        drop(conn);
        let before=std::fs::read(&path).unwrap();
        assert!(open_pool(&path).is_err());
        assert_eq!(before,std::fs::read(&path).unwrap());
    }

    #[test]
    fn future_database_is_rejected_without_lowering_its_version() {
        let mut conn=Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        conn.pragma_update(None,"user_version",99).unwrap();
        assert!(migrate(&mut conn).is_err());
        assert_eq!(conn.pragma_query_value(None,"user_version",|r|r.get::<_,i64>(0)).unwrap(),99);
    }

    #[test]
    fn migration_preserves_v4_facts_and_discards_unsafe_promotions() {
        let mut conn = Connection::open_in_memory().unwrap();
        Migrations::new(vec![
            M::up(SCHEMA_V1),
            M::up(SCHEMA_V2),
            M::up(SCHEMA_V3),
            M::up(SCHEMA_V4),
        ])
        .to_latest(&mut conn)
        .unwrap();
        let id = Uuid::now_v7();
        conn.execute("INSERT INTO memories(id,kind,state,content,importance,confidence,salience,
            created_at_ms,updated_at_ms,namespace) VALUES(?1,'episodic','active','kept',0.5,0.8,0.4,1,1,'default')",
            params![id.to_string()]).unwrap();
        conn.execute(
            "INSERT INTO stm_promotions(id,session_key,payload_json,importance,created_at_ms)
            VALUES('legacy','default/session','{}',0.5,1)",
            [],
        )
        .unwrap();
        migrate(&mut conn).unwrap();
        let rec = get_memory(&conn, id).unwrap().unwrap();
        assert_eq!(rec.content, "kept");
        assert_eq!(rec.revision, 1);
        let pending: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM stm_promotions WHERE processed_at_ms IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pending, 0);
        migrate(&mut conn).unwrap(); // repeat startup is idempotent
    }

    #[test]
    fn migration_registers_v8_fact_sources_and_projection_jobs() {
        let mut conn = Connection::open_in_memory().unwrap();
        // Simulate a V7 production database, then upgrade through registered
        // SCHEMA_V8 / SCHEMA_V9 so graph writes can open kg_fact_sources.
        Migrations::new(vec![
            M::up(SCHEMA_V1),
            M::up(SCHEMA_V2),
            M::up(SCHEMA_V3),
            M::up(SCHEMA_V4),
            M::up(SCHEMA_V5),
            M::up(SCHEMA_V6),
            M::up(SCHEMA_V7),
        ])
        .to_latest(&mut conn)
        .unwrap();
        assert!(conn
            .prepare("SELECT 1 FROM kg_fact_sources LIMIT 1")
            .is_err());
        assert!(conn
            .prepare("SELECT 1 FROM graph_projection_jobs LIMIT 1")
            .is_err());
        migrate(&mut conn).unwrap();
        // Tables must exist and accept the V9 evidence columns.
        let cols: Vec<String> = {
            let mut stmt = conn.prepare("PRAGMA table_info(kg_fact_sources)").unwrap();
            stmt.query_map([], |r| r.get::<_, String>(1))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };
        assert!(cols.iter().any(|c| c == "evidence_text"));
        assert!(cols.iter().any(|c| c == "evidence_summary"));
        conn.execute(
            "INSERT INTO graph_projection_jobs(memory_id,namespace,revision,source_text,attempts,next_attempt_at_ms,created_at_ms,updated_at_ms)
             VALUES('m1','ns',1,'正文',0,0,1,1)",
            [],
        )
        .unwrap();
        let jobs: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_projection_jobs", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(jobs, 1);
        let sources: i64 = conn
            .query_row("SELECT COUNT(*) FROM kg_fact_sources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sources, 0);
        migrate(&mut conn).unwrap();
    }

    #[test]
    fn stale_revision_cannot_update_projection_or_fts() {
        let dir = tempfile::tempdir().unwrap();
        let pool = open_pool(&dir.path().join("m.db")).unwrap();
        let conn = pool.get().unwrap();
        let id = Uuid::now_v7();
        conn.execute("INSERT INTO memories(id,kind,state,content,importance,confidence,salience,
            created_at_ms,updated_at_ms,namespace) VALUES(?1,'episodic','active','old',0.5,0.8,0.4,1,1,'default')",
            params![id.to_string()]).unwrap();
        let mut a = get_memory(&conn, id).unwrap().unwrap();
        let mut stale = a.clone();
        let jieba = Jieba::new();
        a.content = "new".into();
        update_memory(&conn, &mut a, &jieba).unwrap();
        stale.content = "stale".into();
        assert!(update_memory(&conn, &mut stale, &jieba).is_err());
        assert_eq!(get_memory(&conn, id).unwrap().unwrap().content, "new");
        assert_eq!(a.revision, 2);
    }
}
