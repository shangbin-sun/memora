use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::ZipArchive;

use crate::config::Config;
use crate::embedding::vector_to_bytes;
use crate::error::{MemoraError, Result};
use crate::store::{
    append_audit, forget_namespace, get_memory, insert_memory, insert_relation, list_all_audit,
    list_all_relations, list_all_versions, list_embeddings_for_namespace, list_memories_for_export,
    next_embedding_label, update_memory, upsert_embedding, EmbeddingExport,
};
use crate::types::{
    AuditEvent, ExportRequest, ImportMode, ImportReport, MemoryRecord, MemoryRelation,
    MemoryVersion,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub exported_at_ms: i64,
    pub namespace: String,
    pub embedding_model: Option<String>,
    pub embedding_dim: Option<usize>,
    pub hashes: HashMap<String, String>,
    /// Exported audit is a subset of the original chain, not a complete chain.
    #[serde(default)]
    pub audit_filtered: bool,
}

fn sha256_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn jsonl<T: Serialize>(rows: &[T]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut buf, row)?;
        buf.push(b'\n');
    }
    Ok(buf)
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for (i, line) in bytes.split(|b| *b == b'\n').enumerate() {
        if line.iter().all(|c| c.is_ascii_whitespace()) {
            continue;
        }
        match serde_json::from_slice::<T>(line) {
            Ok(v) => out.push(v),
            Err(e) => {
                return Err(MemoraError::Import(format!("line {}: {e}", i + 1)));
            }
        }
    }
    Ok(out)
}

pub fn export_archive(
    conn: &rusqlite::Connection,
    req: &ExportRequest,
    default_ns: &str,
) -> Result<Vec<u8>> {
    // Body and side tables must come from one SQLite read snapshot.
    let tx = conn.unchecked_transaction()?;
    let conn = &tx;
    let ns = req.namespace.as_deref().unwrap_or(default_ns);
    let mut memories = list_memories_for_export(conn, ns, req.include_forgotten)?;
    if req.exclude_sensitive {
        memories.retain(|m| !crate::household::is_sensitive(&m.metadata) && m.tag != "医疗健康");
    }
    let allowed: std::collections::HashSet<_> = memories.iter().map(|m| m.id).collect();
    let versions: Vec<_> = list_all_versions(conn, ns)?
        .into_iter()
        .filter(|v| allowed.contains(&v.memory_id))
        .filter(|v| {
            !req.exclude_sensitive
                || (!sensitive_snapshot(v.before_json.as_deref())
                    && !sensitive_snapshot(Some(&v.after_json)))
        })
        .collect();
    let relations: Vec<_> = list_all_relations(conn, ns)?
        .into_iter()
        .filter(|r| allowed.contains(&r.from_id) && allowed.contains(&r.to_id))
        .collect();
    // Free-text audit may contain historical sensitive values. Omit it in
    // privacy exports, and never include namespace-less global events.
    let audit: Vec<_> = if req.exclude_sensitive {
        Vec::new()
    } else {
        list_all_audit(conn, ns)?
            .into_iter()
            .filter(|a| a.memory_id.is_some_and(|id| allowed.contains(&id)))
            .collect()
    };
    let mem_bytes = jsonl(&memories)?;
    let ver_bytes = jsonl(&versions)?;
    let rel_bytes = jsonl(&relations)?;
    let aud_bytes = jsonl(&audit)?;
    let embeddings = if req.include_vectors {
        list_embeddings_for_namespace(conn, ns)?
            .into_iter()
            .filter(|e| allowed.contains(&e.memory_id))
            .collect()
    } else {
        Vec::new()
    };
    let emb_bytes = jsonl(&embeddings)?;
    let mut hashes = HashMap::new();
    hashes.insert("memories.jsonl".into(), sha256_bytes(&mem_bytes));
    hashes.insert("versions.jsonl".into(), sha256_bytes(&ver_bytes));
    hashes.insert("relations.jsonl".into(), sha256_bytes(&rel_bytes));
    hashes.insert("audit.jsonl".into(), sha256_bytes(&aud_bytes));
    if req.include_vectors {
        hashes.insert("embeddings.jsonl".into(), sha256_bytes(&emb_bytes));
    }
    let first_emb = embeddings.first();
    let manifest = Manifest {
        schema_version: 1,
        exported_at_ms: crate::core::now_ms(),
        namespace: ns.to_string(),
        embedding_model: first_emb.map(|e| e.model.clone()),
        embedding_dim: first_emb.map(|e| e.dim as usize),
        hashes,
        audit_filtered: true,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;

    tx.commit()?;
    let cursor = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("manifest.json", opts)?;
    zip.write_all(&manifest_bytes)?;
    zip.start_file("memories.jsonl", opts)?;
    zip.write_all(&mem_bytes)?;
    zip.start_file("versions.jsonl", opts)?;
    zip.write_all(&ver_bytes)?;
    zip.start_file("relations.jsonl", opts)?;
    zip.write_all(&rel_bytes)?;
    zip.start_file("audit.jsonl", opts)?;
    zip.write_all(&aud_bytes)?;
    if req.include_vectors {
        zip.start_file("embeddings.jsonl", opts)?;
        zip.write_all(&emb_bytes)?;
    }
    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

fn sensitive_snapshot(json: Option<&str>) -> bool {
    json.is_some_and(|raw| match serde_json::from_str::<MemoryRecord>(raw) {
        Ok(record) => crate::household::is_sensitive(&record.metadata) || record.tag == "医疗健康",
        Err(_) => true, // fail closed for unrecognised historical records
    })
}

fn read_zip_file(archive: &mut ZipArchive<Cursor<&[u8]>>, name: &str, max: u64) -> Result<Vec<u8>> {
    if name.contains("..") || name.starts_with('/') {
        return Err(MemoraError::Import(format!("unsafe path: {name}")));
    }
    let mut file = archive.by_name(name)?;
    if file.size() > max {
        return Err(MemoraError::Import(format!(
            "file {name} exceeds size limit"
        )));
    }
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn import_archive(
    conn: &rusqlite::Connection,
    jieba: &jieba_rs::Jieba,
    archive: &[u8],
    mode: ImportMode,
    config: &Config,
    default_ns: &str,
) -> Result<ImportReport> {
    let mut zip = ZipArchive::new(Cursor::new(archive))?;
    let mut uncompressed = 0u64;
    for i in 0..zip.len() {
        let f = zip.by_index(i)?;
        let name = f.name().to_string();
        if name.contains("..") || name.starts_with('/') {
            return Err(MemoraError::Import(format!("unsafe path: {name}")));
        }
        uncompressed += f.size();
        if uncompressed > config.max_import_uncompressed {
            return Err(MemoraError::Import("uncompressed archive too large".into()));
        }
    }
    let manifest_bytes = read_zip_file(&mut zip, "manifest.json", config.max_import_uncompressed)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema_version != 1 {
        return Err(MemoraError::Import(format!(
            "unsupported schema_version {}",
            manifest.schema_version
        )));
    }
    let memories_b = read_zip_file(&mut zip, "memories.jsonl", config.max_import_uncompressed)?;
    let versions_b = read_zip_file(&mut zip, "versions.jsonl", config.max_import_uncompressed)?;
    let relations_b = read_zip_file(&mut zip, "relations.jsonl", config.max_import_uncompressed)?;
    let audit_b = read_zip_file(&mut zip, "audit.jsonl", config.max_import_uncompressed)?;
    let embeddings_b =
        match read_zip_file(&mut zip, "embeddings.jsonl", config.max_import_uncompressed) {
            Ok(b) => Some(b),
            Err(_) => None,
        };
    for (name, bytes) in [
        ("memories.jsonl", &memories_b),
        ("versions.jsonl", &versions_b),
        ("relations.jsonl", &relations_b),
        ("audit.jsonl", &audit_b),
    ] {
        if let Some(expected) = manifest.hashes.get(name) {
            if &sha256_bytes(bytes) != expected {
                return Err(MemoraError::Import(format!("hash mismatch: {name}")));
            }
        }
    }
    if let (Some(bytes), Some(expected)) = (&embeddings_b, manifest.hashes.get("embeddings.jsonl"))
    {
        if &sha256_bytes(bytes) != expected {
            return Err(MemoraError::Import(
                "hash mismatch: embeddings.jsonl".into(),
            ));
        }
    }
    let memories: Vec<MemoryRecord> = parse_jsonl(&memories_b)?;
    let versions: Vec<MemoryVersion> = parse_jsonl(&versions_b)?;
    let relations: Vec<MemoryRelation> = parse_jsonl(&relations_b)?;
    let _audit: Vec<AuditEvent> = parse_jsonl(&audit_b)?;

    let mut report = ImportReport::default();
    if mode == ImportMode::DryRun {
        report.inserted = memories.len();
        return Ok(report);
    }

    let tx = conn.unchecked_transaction()?;
    let conn = &tx;
    let ns = memories
        .first()
        .map(|m| m.namespace.clone())
        .unwrap_or_else(|| default_ns.to_string());

    if mode == ImportMode::ReplaceNamespace {
        forget_namespace(conn, &ns)?;
    }

    let reembed = mode == ImportMode::Reembed;
    for rec in &memories {
        match get_memory(conn, rec.id)? {
            Some(existing) => {
                if rec.updated_at_ms > existing.updated_at_ms {
                    let mut next = rec.clone();
                    next.revision = existing.revision;
                    if reembed {
                        next.embedding_pending = true;
                        report.requeued_embed += 1;
                    }
                    update_memory(conn, &mut next, jieba)?;
                    report.updated += 1;
                } else {
                    report.skipped += 1;
                }
            }
            None => {
                let mut next = rec.clone();
                if reembed {
                    next.embedding_pending = true;
                    report.requeued_embed += 1;
                }
                insert_memory(conn, &next, jieba)?;
                report.inserted += 1;
            }
        }
    }
    for rel in relations {
        let _ = insert_relation(conn, rel.from_id, rel.to_id, rel.relation, rel.weight);
    }
    for v in versions {
        if get_memory(conn, v.memory_id)?.is_some() {
            let _ = conn.execute(
                r#"INSERT OR IGNORE INTO memory_versions
                    (id, memory_id, version, operation, before_json, after_json, actor, reason, created_at_ms)
                   VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)"#,
                rusqlite::params![
                    v.id.to_string(),
                    v.memory_id.to_string(),
                    v.version,
                    v.operation,
                    v.before_json,
                    v.after_json,
                    v.actor,
                    v.reason,
                    v.created_at_ms,
                ],
            );
        }
    }
    if mode != ImportMode::Reembed {
        if let Some(bytes) = embeddings_b {
            let rows: Vec<EmbeddingExport> = parse_jsonl(&bytes)?;
            for row in rows {
                let Some(record) = get_memory(conn, row.memory_id)? else {
                    continue;
                };
                if record.state != crate::MemoryState::Active
                    || record.embedding_version != row.version
                {
                    continue;
                }
                if row.space_id.is_empty() {
                    conn.execute(
                        "UPDATE memories SET embedding_pending = 1 WHERE id = ?1",
                        rusqlite::params![row.memory_id.to_string()],
                    )?;
                    continue;
                }
                let vector = crate::embedding::normalize_vector(row.vector)?;
                if vector.len() != row.dim as usize {
                    return Err(MemoraError::Import("invalid embedding dimension".into()));
                }
                let label = next_embedding_label(conn)?;
                let blob = vector_to_bytes(&vector);
                upsert_embedding(
                    conn,
                    row.memory_id,
                    label,
                    &row.model,
                    row.dim as usize,
                    row.version,
                    &blob,
                )?;
                conn.execute(
                    "UPDATE embeddings SET space_id=?2 WHERE memory_id=?1",
                    rusqlite::params![row.memory_id.to_string(), row.space_id],
                )?;
            }
        }
    }
    append_audit(
        conn,
        None,
        "import",
        serde_json::json!({"mode": format!("{mode:?}"), "inserted": report.inserted, "updated": report.updated}),
    )?;
    tx.commit()?;
    Ok(report)
}
