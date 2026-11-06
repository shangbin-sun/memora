use rusqlite::params;

use crate::config::Config;
use crate::core::{days_between, next_interval_days, now_ms, retention, stability};
use crate::error::Result;
use crate::store::{
    append_audit, delete_memory, get_memory, has_relations, remove_fts, update_memory, Pool,
};
use crate::types::{MemoryId, MemoryState, TickReport};

pub fn apply_forgetting(
    pool: &Pool,
    config: &Config,
    jieba: &jieba_rs::Jieba,
) -> Result<TickReport> {
    let mut connection = pool.get()?;
    let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let conn = &tx;
    let now = now_ms();
    let mut report = TickReport::default();

    let mut due: Vec<MemoryId> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id FROM memories WHERE state = 'active' AND next_review_at_ms IS NOT NULL AND next_review_at_ms <= ?1 LIMIT 500",
        )?;
        let rows = stmt.query_map(params![now], |r| {
            Ok(uuid::Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(uuid::Uuid::nil()))
        })?;
        due.extend(rows.filter_map(|r| r.ok()));
    }
    for id in due {
        if let Some(mut rec) = get_memory(&conn, id)? {
            rec.review_count += 1;
            rec.last_accessed_at_ms = Some(now);
            rec.updated_at_ms = now;
            let s = stability(
                config.s0,
                config.alpha,
                config.beta,
                rec.importance,
                rec.review_count,
            );
            let interval = next_interval_days(s);
            rec.next_review_at_ms = Some(now + (interval * 86_400_000.0) as i64);
            rec.salience = rec.importance * rec.confidence * retention(0.0, s);
            update_memory(conn, &mut rec, jieba)?;
            append_audit(
                &conn,
                Some(id),
                "review",
                serde_json::json!({"review_count": rec.review_count, "next_review_at_ms": rec.next_review_at_ms}),
            )?;
            report.reviewed += 1;
        }
    }

    let mut candidates: Vec<MemoryId> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id FROM memories WHERE state = 'active' ORDER BY COALESCE(last_accessed_at_ms, created_at_ms) ASC LIMIT 1000",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(uuid::Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(uuid::Uuid::nil()))
        })?;
        candidates.extend(rows.filter_map(|r| r.ok()));
    }
    for id in candidates {
        if let Some(mut rec) = get_memory(&conn, id)? {
            let last = rec.last_accessed_at_ms.unwrap_or(rec.created_at_ms);
            let s = stability(
                config.s0,
                config.alpha,
                config.beta,
                rec.importance,
                rec.review_count,
            );
            let r = retention(days_between(last, now), s);
            rec.salience = rec.importance * rec.confidence * r;
            if r < config.r_archive {
                rec.state = MemoryState::Archived;
                rec.updated_at_ms = now;
                update_memory(conn, &mut rec, jieba)?;
                remove_fts(&conn, id)?;
                append_audit(
                    &conn,
                    Some(id),
                    "archive",
                    serde_json::json!({"retention": r}),
                )?;
                report.archived += 1;
            }
        }
    }

    let grace = now - config.forgotten_grace_days * 86_400_000;
    let mut archived: Vec<MemoryId> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id FROM memories WHERE state = 'archived' AND updated_at_ms <= ?1 LIMIT 500",
        )?;
        let rows = stmt.query_map(params![grace], |r| {
            Ok(uuid::Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(uuid::Uuid::nil()))
        })?;
        archived.extend(rows.filter_map(|r| r.ok()));
    }
    for id in archived {
        if has_relations(&conn, id)? {
            continue;
        }
        if let Some(mut rec) = get_memory(&conn, id)? {
            if rec.metadata.get("retain").and_then(|v| v.as_bool()) == Some(true) {
                continue;
            }
            rec.state = MemoryState::Forgotten;
            rec.updated_at_ms = now;
            update_memory(conn, &mut rec, jieba)?;
            remove_fts(&conn, id)?;
            crate::chunks::unlink_memory(&conn, &id.to_string())?;
            append_audit(&conn, Some(id), "forget_decay", serde_json::json!({}))?;
            report.forgotten += 1;
        }
    }

    // forgotten past grace → physical delete
    let mut to_purge: Vec<MemoryId> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id FROM memories WHERE state = 'forgotten' AND updated_at_ms <= ?1 LIMIT 500",
        )?;
        let rows = stmt.query_map(params![grace], |r| {
            Ok(uuid::Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(uuid::Uuid::nil()))
        })?;
        to_purge.extend(rows.filter_map(|r| r.ok()));
    }
    for id in to_purge {
        if let Some(rec) = get_memory(&conn, id)? {
            if rec.metadata.get("retain").and_then(|v| v.as_bool()) == Some(true) {
                continue;
            }
            append_audit(
                &conn,
                Some(id),
                "purge_decay",
                serde_json::json!({
                    "after_days": config.forgotten_grace_days,
                }),
            )?;
            delete_memory(&conn, id)?;
            report.purged += 1;
        }
    }

    tx.commit()?;
    Ok(report)
}
