//! SQLite persistence for the family knowledge graph.

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use uuid::Uuid;

use crate::core::now_ms;
use crate::error::{MemoraError, Result};

use super::ontology::{
    normalize_relation, EntityType, FactPolarity, FactStatus, PrivacyLevel, RelationType,
};
use super::types::{EntityId, FactObject, GrantInput, KgEntity, KgFact, KgGrant, RevokeGrantInput};

/// Persisted work uses the memory revision to prevent stale extraction results
/// from deleting or replacing work scheduled by a newer write.
#[derive(Debug, Clone)]
pub struct GraphProjectionJob {
    pub memory_id: Uuid,
    pub namespace: String,
    pub revision: i64,
    pub source_text: String,
    pub speaker_member_id: Option<String>,
    pub attempts: u32,
}

pub fn enqueue_graph_projection_job(
    conn: &Connection,
    memory_id: Uuid,
    namespace: &str,
    revision: i64,
    source_text: &str,
    speaker: Option<&str>,
) -> Result<()> {
    conn.execute(
        r#"INSERT INTO graph_projection_jobs(memory_id,namespace,revision,source_text,speaker_member_id,attempts,next_attempt_at_ms,created_at_ms,updated_at_ms)
           VALUES (?1,?2,?3,?4,?5,0,0,?6,?6)
           ON CONFLICT(memory_id) DO UPDATE SET
             namespace=excluded.namespace, revision=excluded.revision,
             source_text=excluded.source_text, speaker_member_id=excluded.speaker_member_id,
             attempts=CASE WHEN graph_projection_jobs.revision=excluded.revision THEN graph_projection_jobs.attempts ELSE 0 END,
             next_attempt_at_ms=CASE WHEN graph_projection_jobs.revision=excluded.revision THEN graph_projection_jobs.next_attempt_at_ms ELSE 0 END,
             last_error=CASE WHEN graph_projection_jobs.revision=excluded.revision THEN graph_projection_jobs.last_error ELSE NULL END,
             updated_at_ms=excluded.updated_at_ms
           WHERE excluded.revision>=graph_projection_jobs.revision"#,
        params![memory_id.to_string(), namespace, revision, source_text, speaker, now_ms()],
    )?;
    Ok(())
}

pub fn due_graph_projection_jobs(
    conn: &Connection,
    now: i64,
    limit: usize,
) -> Result<Vec<GraphProjectionJob>> {
    let mut stmt = conn.prepare(
        "SELECT memory_id,namespace,revision,source_text,speaker_member_id,attempts FROM graph_projection_jobs WHERE next_attempt_at_ms<=?1 ORDER BY next_attempt_at_ms,created_at_ms,memory_id LIMIT ?2"
    )?;
    let rows = stmt.query_map(params![now, limit.min(i64::MAX as usize) as i64], |row| {
        let id: String = row.get(0)?;
        let memory_id = Uuid::parse_str(&id).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
        Ok(GraphProjectionJob {
            memory_id,
            namespace: row.get(1)?,
            revision: row.get(2)?,
            source_text: row.get(3)?,
            speaker_member_id: row.get(4)?,
            attempts: row.get(5)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn delete_graph_projection_job(
    conn: &Connection,
    memory_id: Uuid,
    revision: i64,
) -> Result<usize> {
    Ok(conn.execute(
        "DELETE FROM graph_projection_jobs WHERE memory_id=?1 AND revision=?2",
        params![memory_id.to_string(), revision],
    )?)
}

/// Drop every pending projection job for a memory (any revision).
pub fn delete_graph_projection_job_for_memory(conn: &Connection, memory_id: Uuid) -> Result<usize> {
    Ok(conn.execute(
        "DELETE FROM graph_projection_jobs WHERE memory_id=?1",
        params![memory_id.to_string()],
    )?)
}

pub fn fail_graph_projection_job(
    conn: &Connection,
    memory_id: Uuid,
    revision: i64,
    error: &str,
) -> Result<()> {
    // Retry indefinitely with bounded exponential backoff; transient outages must
    // not become permanent data loss. Errors are bounded and never source text.
    let attempts: Option<u32> = conn
        .query_row(
            "SELECT attempts FROM graph_projection_jobs WHERE memory_id=?1 AND revision=?2",
            params![memory_id.to_string(), revision],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(attempts) = attempts {
        let delay = 1000_i64
            .saturating_mul(1_i64 << attempts.min(10))
            .min(900_000);
        let bounded_error: String = error.chars().take(512).collect();
        let now = now_ms();
        conn.execute("UPDATE graph_projection_jobs SET attempts=attempts+1,next_attempt_at_ms=?3,last_error=?4,updated_at_ms=?5 WHERE memory_id=?1 AND revision=?2",
            params![memory_id.to_string(),revision,now.saturating_add(delay),bounded_error,now])?;
    }
    Ok(())
}

pub fn ensure_person_entity(
    conn: &Connection,
    namespace: &str,
    member_id: &str,
    display_name: &str,
    aliases: &[String],
) -> Result<EntityId> {
    let id = person_entity_id(member_id);
    let now = now_ms();
    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM kg_entities WHERE namespace = ?1 AND id = ?2",
            params![namespace, id],
            |r| r.get(0),
        )
        .optional()?;
    if existing.is_some() {
        conn.execute(
            r#"UPDATE kg_entities SET
                 canonical_name = ?3,
                 last_confirmed_at_ms = ?4,
                 updated_at_ms = ?4
               WHERE namespace = ?1 AND id = ?2"#,
            params![namespace, id, display_name, now],
        )?;
        for alias in aliases {
            upsert_alias(conn, namespace, &id, alias, Some(member_id))?;
        }
        return Ok(id);
    }
    let entity = KgEntity {
        id: id.clone(),
        entity_type: EntityType::Person,
        subtype: Some("household_member".into()),
        canonical_name: display_name.to_string(),
        aliases: aliases.to_vec(),
        status: "active".into(),
        domains: vec!["household".into()],
        owner_member_id: Some(member_id.to_string()),
        privacy: PrivacyLevel::FamilyShared,
        first_seen_at_ms: now,
        last_confirmed_at_ms: now,
        attrs: Value::Object(Default::default()),
    };
    upsert_entity(conn, namespace, &entity)?;
    Ok(id)
}

pub fn person_entity_id(member_id: &str) -> EntityId {
    format!("person:{member_id}")
}

pub fn upsert_entity(conn: &Connection, namespace: &str, entity: &KgEntity) -> Result<()> {
    let now = now_ms();
    conn.execute(
        r#"INSERT INTO kg_entities(
             namespace, id, entity_type, subtype, canonical_name, status,
             owner_member_id, privacy, first_seen_at_ms, last_confirmed_at_ms,
             attrs_json, created_at_ms, updated_at_ms
           ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?12)
           ON CONFLICT(namespace, id) DO UPDATE SET
             entity_type = excluded.entity_type,
             subtype = excluded.subtype,
             canonical_name = excluded.canonical_name,
             status = excluded.status,
             owner_member_id = excluded.owner_member_id,
             privacy = excluded.privacy,
             last_confirmed_at_ms = excluded.last_confirmed_at_ms,
             attrs_json = excluded.attrs_json,
             updated_at_ms = excluded.updated_at_ms"#,
        params![
            namespace,
            entity.id,
            entity.entity_type.as_str(),
            entity.subtype,
            entity.canonical_name,
            entity.status,
            entity.owner_member_id,
            entity.privacy.as_str(),
            entity.first_seen_at_ms,
            entity.last_confirmed_at_ms.max(now),
            serde_json::to_string(&entity.attrs)?,
            now,
        ],
    )?;
    set_resource_domains(conn, namespace, "entity", &entity.id, &entity.domains)?;
    for alias in &entity.aliases {
        upsert_alias(
            conn,
            namespace,
            &entity.id,
            alias,
            entity.owner_member_id.as_deref(),
        )?;
    }
    Ok(())
}

pub fn upsert_alias(
    conn: &Connection,
    namespace: &str,
    entity_id: &str,
    alias: &str,
    scope_member_id: Option<&str>,
) -> Result<()> {
    let alias = alias.trim();
    if alias.is_empty() {
        return Ok(());
    }
    let now = now_ms();
    conn.execute(
        r#"INSERT INTO kg_aliases(namespace, entity_id, alias, scope_member_id, confirmed, created_at_ms)
           VALUES (?1,?2,?3,?4,1,?5)
           ON CONFLICT(namespace, entity_id, alias, scope_member_id) DO UPDATE SET
             confirmed = 1"#,
        params![
            namespace,
            entity_id,
            alias,
            scope_member_id.unwrap_or(""),
            now
        ],
    )?;
    Ok(())
}

pub fn find_entity_by_name(
    conn: &Connection,
    namespace: &str,
    entity_type: EntityType,
    name: &str,
) -> Result<Option<KgEntity>> {
    find_entity_by_name_owned(conn, namespace, entity_type, name, None)
}

pub fn find_entity_by_name_owned(
    conn: &Connection,
    namespace: &str,
    entity_type: EntityType,
    name: &str,
    owner_member_id: Option<&str>,
) -> Result<Option<KgEntity>> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    return find_named_candidate(conn, namespace, name, Some(entity_type), owner_member_id);
}

fn entity_matches_owner(entity: &KgEntity, owner: Option<&str>) -> bool {
    if RelationType::owner_scoped_entity(entity.entity_type)
        || entity.privacy == PrivacyLevel::OwnerPrivate
    {
        return owner.is_some() && entity.owner_member_id.as_deref() == owner;
    }
    if entity.entity_type == EntityType::Person && owner.is_some() {
        return entity.owner_member_id.as_deref() == owner
            || owner.is_some_and(|m| entity.id == person_entity_id(m));
    }
    true
}

fn find_named_candidate(
    conn: &Connection,
    namespace: &str,
    name: &str,
    entity_type: Option<EntityType>,
    owner_member_id: Option<&str>,
) -> Result<Option<KgEntity>> {
    let owner = owner_member_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|m| m.strip_prefix("person:").unwrap_or(m));
    // Never surface status=merged losers: merge moves aliases onto the keeper.
    // resolve_merged_entity remains a safety net for stale IDs in other paths.
    let mut stmt = conn.prepare(
        r#"SELECT e.id FROM kg_entities e WHERE e.namespace = ?1
           AND e.status != 'merged'
           AND (?3 IS NULL OR e.entity_type = ?3)
           AND (lower(e.canonical_name) = lower(?2) OR EXISTS (
             SELECT 1 FROM kg_aliases a WHERE a.namespace = e.namespace
               AND a.entity_id = e.id AND lower(a.alias) = lower(?2)
               AND (a.scope_member_id = '' OR ?4 IS NULL OR a.scope_member_id = ?4)))
           ORDER BY lower(e.canonical_name) != lower(?2), e.id"#,
    )?;
    let ids: Vec<String> = stmt
        .query_map(
            params![namespace, name, entity_type.map(EntityType::as_str), owner],
            |r| r.get(0),
        )?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut candidates = std::collections::BTreeMap::new();
    for id in ids {
        if let Some(entity) = resolve_merged_entity(conn, namespace, &id)? {
            if entity.status == "merged"
                || entity_type.is_some_and(|t| entity.entity_type != t)
                || !entity_matches_owner(&entity, owner)
            {
                continue;
            }
            candidates.insert(entity.id.clone(), entity);
        }
    }
    if candidates.len() == 1 {
        Ok(candidates.into_values().next())
    } else {
        Ok(None)
    }
}

/// Follow legacy references; malformed/cyclic chains never return merged nodes.
pub fn resolve_merged_entity(
    conn: &Connection,
    namespace: &str,
    id: &str,
) -> Result<Option<KgEntity>> {
    let mut id = id.to_string();
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..64 {
        if !seen.insert(id.clone()) {
            return Ok(None);
        }
        let Some(entity) = get_entity(conn, namespace, &id)? else {
            return Ok(None);
        };
        if entity.status != "merged" {
            return Ok(Some(entity));
        }
        let Some(target) = entity.attrs.get("merged_into").and_then(Value::as_str) else {
            return Ok(None);
        };
        id = target.to_string();
    }
    Ok(None)
}

pub fn get_entity(conn: &Connection, namespace: &str, id: &str) -> Result<Option<KgEntity>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, entity_type, subtype, canonical_name, status, owner_member_id, privacy,
                  first_seen_at_ms, last_confirmed_at_ms, attrs_json
           FROM kg_entities WHERE namespace = ?1 AND id = ?2"#,
    )?;
    let row = stmt
        .query_row(params![namespace, id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, i64>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
            ))
        })
        .optional()?;
    let Some((
        id,
        entity_type,
        subtype,
        canonical_name,
        status,
        owner_member_id,
        privacy,
        first_seen_at_ms,
        last_confirmed_at_ms,
        attrs_json,
    )) = row
    else {
        return Ok(None);
    };
    let aliases = list_aliases(conn, namespace, &id)?;
    let domains = list_resource_domains(conn, namespace, "entity", &id)?;
    Ok(Some(KgEntity {
        id,
        entity_type: EntityType::parse(&entity_type)?,
        subtype,
        canonical_name,
        aliases,
        status,
        domains,
        owner_member_id,
        privacy: PrivacyLevel::parse(&privacy)?,
        first_seen_at_ms,
        last_confirmed_at_ms,
        attrs: serde_json::from_str(&attrs_json).unwrap_or(Value::Object(Default::default())),
    }))
}

fn list_aliases(conn: &Connection, namespace: &str, entity_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT alias FROM kg_aliases WHERE namespace = ?1 AND entity_id = ?2 ORDER BY alias",
    )?;
    let rows = stmt.query_map(params![namespace, entity_id], |r| r.get::<_, String>(0))?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

fn set_resource_domains(
    conn: &Connection,
    namespace: &str,
    resource_kind: &str,
    resource_id: &str,
    domains: &[String],
) -> Result<()> {
    conn.execute(
        "DELETE FROM kg_resource_domains WHERE namespace = ?1 AND resource_kind = ?2 AND resource_id = ?3",
        params![namespace, resource_kind, resource_id],
    )?;
    for d in domains {
        let d = d.trim();
        if d.is_empty() {
            continue;
        }
        conn.execute(
            r#"INSERT OR IGNORE INTO kg_resource_domains(namespace, resource_kind, resource_id, domain)
               VALUES (?1,?2,?3,?4)"#,
            params![namespace, resource_kind, resource_id, d],
        )?;
    }
    Ok(())
}

fn list_resource_domains(
    conn: &Connection,
    namespace: &str,
    resource_kind: &str,
    resource_id: &str,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        r#"SELECT domain FROM kg_resource_domains
           WHERE namespace = ?1 AND resource_kind = ?2 AND resource_id = ?3
           ORDER BY domain"#,
    )?;
    let rows = stmt.query_map(params![namespace, resource_kind, resource_id], |r| {
        r.get::<_, String>(0)
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn resolve_or_create_entity(
    conn: &Connection,
    namespace: &str,
    entity_type: EntityType,
    name: &str,
    owner_member_id: Option<&str>,
    domains: &[String],
    privacy: PrivacyLevel,
) -> Result<EntityId> {
    if let Some(existing) =
        find_entity_by_name_owned(conn, namespace, entity_type, name, owner_member_id)?
    {
        return Ok(existing.id);
    }
    let now = now_ms();
    let id = format!("{}:{}", entity_type.as_str(), Uuid::now_v7().simple());
    let entity = KgEntity {
        id: id.clone(),
        entity_type,
        subtype: None,
        canonical_name: name.trim().to_string(),
        aliases: vec![name.trim().to_string()],
        status: "active".into(),
        domains: domains.to_vec(),
        owner_member_id: owner_member_id.map(str::to_string),
        privacy,
        first_seen_at_ms: now,
        last_confirmed_at_ms: now,
        attrs: Value::Object(Default::default()),
    };
    upsert_entity(conn, namespace, &entity)?;
    Ok(id)
}

/// Keep the fact, its provenance links and history in one savepoint, including
/// callers already running a transaction.
pub fn write_fact(conn: &Connection, namespace: &str, fact: &KgFact) -> Result<()> {
    graph_savepoint(conn, || {
        let mut normalized = fact.clone();
        normalized.relation = super::ontology::normalize_relation(&fact.relation);
        if let Some(entity) = resolve_merged_entity(conn, namespace, &fact.subject_id)? {
            normalized.subject_id = entity.id;
        }
        if let FactObject::Entity { entity_id } = &fact.object {
            if let Some(entity) = resolve_merged_entity(conn, namespace, entity_id)? {
                normalized.object = FactObject::entity(entity.id);
            }
        }
        write_fact_inner(conn, namespace, &normalized)
    })
}

fn graph_savepoint<T>(conn: &Connection, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    conn.execute_batch("SAVEPOINT graph_store_write")?;
    match operation() {
        Ok(value) => {
            conn.execute_batch("RELEASE graph_store_write")?;
            Ok(value)
        }
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK TO graph_store_write; RELEASE graph_store_write");
            Err(error)
        }
    }
}

fn write_fact_inner(conn: &Connection, namespace: &str, fact: &KgFact) -> Result<()> {
    let (object_entity_id, object_value_type, object_value_json) = match &fact.object {
        FactObject::Entity { entity_id } => (Some(entity_id.as_str()), None, None),
        FactObject::Value { value_type, value } => (
            None,
            Some(value_type.as_str()),
            Some(serde_json::to_string(value)?),
        ),
    };
    if object_entity_id.is_some() == object_value_json.is_some() {
        return Err(MemoraError::InvalidState(
            "fact object must be either entity or typed value".into(),
        ));
    }
    let semantic_key = semantic_fingerprint(fact);
    // Confirm existing active fact with same fingerprint.
    let mut existing: Option<(String, i64)> = conn
        .query_row(
            r#"SELECT id, revision FROM kg_facts
               WHERE namespace = ?1 AND semantic_key = ?2 AND status IN ('active','planned','paused')
               LIMIT 1"#,
            params![namespace, semantic_key],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    // Old databases used a less complete semantic_key. Re-evaluate candidates
    // instead of confirming facts with different dose, privacy or validity.
    if existing.is_none() {
        let mut stmt = conn.prepare(&format!(
            "SELECT {FACT_SELECT} FROM kg_facts f WHERE f.namespace = ?1 AND f.subject_id = ?2 AND f.status IN ('active','planned','paused') ORDER BY f.created_at_ms, f.id"
        ))?;
        let rows = stmt.query_map(params![namespace, fact.subject_id], |row| {
            row_to_fact(conn, namespace, row)
        })?;
        for row in rows {
            let prior = row?;
            if semantic_fingerprint(&prior) == semantic_key {
                existing = Some((prior.id.to_string(), prior.revision));
                break;
            }
        }
    }
    let now = now_ms();
    if let Some((id, revision)) = existing {
        // Backfill only facts with no link rows (legacy data); never mistake a
        // NULL compatibility pointer for new independent graph-only evidence.
        ensure_legacy_source_link(conn, namespace, &id)?;
        link_fact_source_with_evidence(
            conn,
            namespace,
            &id,
            fact.source_memory_id,
            fact.evidence_text.as_deref(),
            fact.evidence_summary.as_deref(),
        )?;
        let new_rev = revision + 1;
        let attrs_json = serde_json::to_string(&fact.attrs)?;
        conn.execute(
            r#"UPDATE kg_facts SET
                 last_confirmed_at_ms = ?3,
                 updated_at_ms = ?3,
                 revision = ?8,
                 evidence_text = CASE WHEN source_memory_id IS ?6 THEN COALESCE(?4, evidence_text) ELSE evidence_text END,
                 evidence_summary = CASE WHEN source_memory_id IS ?6 THEN COALESCE(?5, evidence_summary) ELSE evidence_summary END,
                 source_memory_id = CASE WHEN source_memory_id IS NULL AND NOT EXISTS (
                    SELECT 1 FROM kg_fact_sources s WHERE s.namespace=kg_facts.namespace
                      AND s.fact_id=kg_facts.id AND s.source_memory_id=''
                  ) THEN ?6 ELSE source_memory_id END,
                 confidence = MAX(confidence, ?7),
                 attrs_json = ?9,
                 semantic_key = ?10,
                 relation = ?11
               WHERE namespace = ?1 AND id = ?2"#,
            params![
                namespace,
                id,
                now,
                fact.evidence_text,
                fact.evidence_summary,
                fact.source_memory_id.map(|u| u.to_string()),
                fact.confidence,
                new_rev,
                attrs_json,
                semantic_key,
                fact.relation,
            ],
        )?;
        // Snapshot the actual persisted source pointer and evidence, not the candidate.
        if let Some(snapshot) = get_fact(conn, namespace, &id)? {
            insert_fact_version(conn, namespace, &id, new_rev, "confirm", &snapshot)?;
        }
        return Ok(());
    }
    conn.execute(
        r#"INSERT INTO kg_facts(
             namespace, id, subject_id, relation, object_entity_id, object_value_type, object_value_json,
             owner_member_id, polarity, status, valid_from_ms, valid_to_ms,
             condition_text, exception_text, reason_text, evidence_text, evidence_summary,
             source_memory_id, confidence, privacy, revision, semantic_key,
             first_seen_at_ms, last_confirmed_at_ms, created_at_ms, updated_at_ms, attrs_json
           ) VALUES (
             ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?23,?23,?23,?24
           )"#,
        params![
            namespace,
            fact.id.to_string(),
            fact.subject_id,
            fact.relation,
            object_entity_id,
            object_value_type,
            object_value_json,
            fact.owner_member_id,
            fact.polarity.as_str(),
            fact.status.as_str(),
            fact.valid_from_ms,
            fact.valid_to_ms,
            fact.condition_text,
            fact.exception_text,
            fact.reason_text,
            fact.evidence_text,
            fact.evidence_summary,
            fact.source_memory_id.map(|u| u.to_string()),
            fact.confidence,
            fact.privacy.as_str(),
            fact.revision.max(1),
            semantic_key,
            now,
            serde_json::to_string(&fact.attrs)?,
        ],
    )?;
    set_resource_domains(conn, namespace, "fact", &fact.id.to_string(), &fact.domains)?;
    link_fact_source_with_evidence(
        conn,
        namespace,
        &fact.id.to_string(),
        fact.source_memory_id,
        fact.evidence_text.as_deref(),
        fact.evidence_summary.as_deref(),
    )?;
    insert_fact_version(
        conn,
        namespace,
        &fact.id.to_string(),
        fact.revision.max(1),
        "create",
        fact,
    )?;
    conn.execute(
        r#"INSERT INTO kg_change_events(namespace, id, fact_id, event_type, payload_json, created_at_ms)
           VALUES (?1,?2,?3,'create',?4,?5)"#,
        params![
            namespace,
            Uuid::now_v7().to_string(),
            fact.id.to_string(),
            serde_json::to_string(&serde_json::json!({
                "relation": fact.relation,
                "subject_id": fact.subject_id,
            }))?,
            now,
        ],
    )?;
    Ok(())
}

/// Attach independent evidence. None means an explicit graph-only assertion;
/// it is not a fallback for a missing/deleted memory. Schema is migrated by store.
pub fn link_fact_source(
    conn: &Connection,
    namespace: &str,
    fact_id: &str,
    source_memory_id: Option<Uuid>,
) -> Result<()> {
    link_fact_source_with_evidence(conn, namespace, fact_id, source_memory_id, None, None)
}

/// Source-local evidence. Missing evidence never copies a different source's
/// private full text, including for the explicit graph-only ('') source.
pub fn link_fact_source_with_evidence(
    conn: &Connection,
    namespace: &str,
    fact_id: &str,
    source_memory_id: Option<Uuid>,
    evidence_text: Option<&str>,
    evidence_summary: Option<&str>,
) -> Result<()> {
    let source = source_memory_id
        .map(|id| id.to_string())
        .unwrap_or_default();
    conn.execute(
        r#"INSERT INTO kg_fact_sources(namespace, fact_id, source_memory_id, created_at_ms, evidence_text, evidence_summary)
           VALUES (?1,?2,?3,?4,?5,?6)
           ON CONFLICT(namespace, fact_id, source_memory_id) DO UPDATE SET
             evidence_text = COALESCE(excluded.evidence_text, evidence_text),
             evidence_summary = COALESCE(excluded.evidence_summary, evidence_summary)"#,
        params![namespace, fact_id, source, now_ms(), evidence_text, evidence_summary],
    )?;
    Ok(())
}

/// Read evidence from exactly one support. None selects graph-only evidence;
/// it does not mean any available memory source or the compatibility pointer.
pub fn fact_source_evidence(
    conn: &Connection,
    namespace: &str,
    fact_id: &str,
    source: Option<Uuid>,
) -> Result<(Option<String>, Option<String>)> {
    let source = source.map(|id| id.to_string()).unwrap_or_default();
    Ok(conn.query_row(
        "SELECT evidence_text, evidence_summary FROM kg_fact_sources WHERE namespace = ?1 AND fact_id = ?2 AND source_memory_id = ?3",
        params![namespace, fact_id, source], |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional()?.unwrap_or((None, None)))
}

fn ensure_legacy_source_link(conn: &Connection, namespace: &str, fact_id: &str) -> Result<()> {
    conn.execute(
        r#"INSERT OR IGNORE INTO kg_fact_sources(namespace, fact_id, source_memory_id, created_at_ms, evidence_text, evidence_summary)
           SELECT namespace, id, COALESCE(source_memory_id, ''), created_at_ms, evidence_text, evidence_summary FROM kg_facts f
           WHERE namespace = ?1 AND id = ?2
              AND NOT EXISTS (SELECT 1 FROM kg_change_events e WHERE e.namespace = f.namespace
                AND e.fact_id = f.id AND e.event_type = 'unlink_source')
              AND NOT EXISTS (
             SELECT 1 FROM kg_fact_sources s WHERE s.namespace = f.namespace AND s.fact_id = f.id
           )"#,
        params![namespace, fact_id],
    )?;
    Ok(())
}

pub fn list_fact_source_memory_ids(
    conn: &Connection,
    namespace: &str,
    fact_id: &str,
) -> Result<Vec<Uuid>> {
    let mut stmt = conn.prepare(
        "SELECT source_memory_id FROM kg_fact_sources WHERE namespace = ?1 AND fact_id = ?2 AND source_memory_id != '' ORDER BY created_at_ms, source_memory_id"
    )?;
    let rows = stmt.query_map(params![namespace, fact_id], |r| r.get::<_, String>(0))?;
    let mut ids = Vec::new();
    for row in rows {
        if let Ok(id) = Uuid::parse_str(&row?) {
            ids.push(id);
        }
    }
    Ok(ids)
}

pub fn has_graph_only_source(conn: &Connection, namespace: &str, fact_id: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM kg_fact_sources WHERE namespace = ?1 AND fact_id = ?2 AND source_memory_id = '')",
        params![namespace, fact_id], |r| r.get(0),
    )?)
}

/// Remove only this memory's support; withdraw current facts only when their
/// last support disappears. The returned count is facts actually withdrawn.
pub fn withdraw_facts_by_source_memory(
    conn: &Connection,
    namespace: &str,
    source_memory_id: Uuid,
) -> Result<usize> {
    graph_savepoint(conn, || {
        let source = source_memory_id.to_string();
        let mut stmt = conn.prepare(
            r#"SELECT f.id FROM kg_facts f WHERE f.namespace = ?1 AND (
                 f.source_memory_id = ?2 OR EXISTS (
                   SELECT 1 FROM kg_fact_sources s WHERE s.namespace = f.namespace
                     AND s.fact_id = f.id AND s.source_memory_id = ?2))"#,
        )?;
        let ids: Vec<String> = stmt
            .query_map(params![namespace, source], |r| r.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut unsupported = Vec::new();
        for id in ids {
            ensure_legacy_source_link(conn, namespace, &id)?;
            conn.execute(
                "DELETE FROM kg_fact_sources WHERE namespace = ?1 AND fact_id = ?2 AND source_memory_id = ?3",
                params![namespace, id, source],
            )?;
            let remaining: i64 = conn.query_row(
                "SELECT count(*) FROM kg_fact_sources WHERE namespace = ?1 AND fact_id = ?2",
                params![namespace, id],
                |r| r.get(0),
            )?;
            // The old scalar pointer is only a compatibility representative.
            conn.execute(
                r#"UPDATE kg_facts SET source_memory_id = (
                     SELECT NULLIF(source_memory_id, '') FROM kg_fact_sources
                     WHERE namespace = ?1 AND fact_id = ?2
                     ORDER BY source_memory_id = '', created_at_ms, source_memory_id LIMIT 1),
                     evidence_text = (SELECT evidence_text FROM kg_fact_sources
                        WHERE namespace = ?1 AND fact_id = ?2
                        ORDER BY source_memory_id = '', created_at_ms, source_memory_id LIMIT 1),
                      evidence_summary = (SELECT evidence_summary FROM kg_fact_sources
                        WHERE namespace = ?1 AND fact_id = ?2
                        ORDER BY source_memory_id = '', created_at_ms, source_memory_id LIMIT 1),
                      updated_at_ms = ?3, revision = revision + 1
                   WHERE namespace = ?1 AND id = ?2"#,
                params![namespace, id, now_ms()],
            )?;
            if let Some(snapshot) = get_fact(conn, namespace, &id)? {
                insert_fact_version(
                    conn,
                    namespace,
                    &id,
                    snapshot.revision,
                    "unlink_source",
                    &snapshot,
                )?;
            }
            conn.execute(
                "INSERT INTO kg_change_events(namespace,id,fact_id,event_type,payload_json,created_at_ms) VALUES (?1,?2,?3,'unlink_source',?4,?5)",
                params![namespace, Uuid::now_v7().to_string(), id,
                    serde_json::json!({"source_memory_id": source}).to_string(), now_ms()],
            )?;
            if remaining == 0 {
                unsupported.push(id);
            }
        }
        supersede_facts(
            conn,
            namespace,
            &unsupported,
            FactStatus::Withdrawn,
            now_ms(),
        )
    })
}

/// Load one fact by id (any status).
pub fn get_fact(conn: &Connection, namespace: &str, fact_id: &str) -> Result<Option<KgFact>> {
    let mut stmt = conn.prepare(&format!(
        r#"SELECT {FACT_SELECT} FROM kg_facts f
           WHERE f.namespace = ?1 AND f.id = ?2"#
    ))?;
    let row = stmt
        .query_row(params![namespace, fact_id], |row| {
            row_to_fact(conn, namespace, row)
        })
        .optional()?;
    Ok(row)
}

/// Mark matching active facts as corrected / ended / withdrawn.
pub fn supersede_facts(
    conn: &Connection,
    namespace: &str,
    fact_ids: &[String],
    status: FactStatus,
    now: i64,
) -> Result<usize> {
    if fact_ids.is_empty() {
        return Ok(0);
    }
    let mut n = 0usize;
    for id in fact_ids {
        let changed = conn.execute(
            r#"UPDATE kg_facts SET
                 status = ?3,
                 valid_to_ms = COALESCE(valid_to_ms, ?4),
                 updated_at_ms = ?4,
                 revision = revision + 1
               WHERE namespace = ?1 AND id = ?2
                 AND status IN ('active','planned','paused')"#,
            params![namespace, id, status.as_str(), now],
        )?;
        if changed > 0 {
            n += 1;
            if let Some(mut snap) = get_fact(conn, namespace, id)? {
                snap.status = status;
                snap.valid_to_ms = snap.valid_to_ms.or(Some(now));
                snap.updated_at_ms = now;
                insert_fact_version(conn, namespace, id, snap.revision, status.as_str(), &snap)?;
            }
            conn.execute(
                r#"INSERT INTO kg_change_events(namespace, id, fact_id, event_type, payload_json, created_at_ms)
                   VALUES (?1,?2,?3,?4,?5,?6)"#,
                params![
                    namespace,
                    Uuid::now_v7().to_string(),
                    id,
                    status.as_str(),
                    serde_json::to_string(&serde_json::json!({
                        "reason": "reconcile",
                        "status": status.as_str(),
                    }))?,
                    now,
                ],
            )?;
        }
    }
    Ok(n)
}

/// Active facts for subject + relation (any object).
pub fn list_active_by_subject_relation(
    conn: &Connection,
    namespace: &str,
    subject_id: &str,
    relation: &str,
) -> Result<Vec<KgFact>> {
    let mut stmt = conn.prepare(&format!(
        r#"SELECT {FACT_SELECT}
           FROM kg_facts f
           WHERE f.namespace = ?1
             AND f.subject_id = ?2
             AND f.status IN ('active','planned','paused')
           ORDER BY f.updated_at_ms DESC"#
    ))?;
    let rows = stmt.query_map(params![namespace, subject_id], |row| {
        row_to_fact(conn, namespace, row)
    })?;
    let wanted = super::ontology::normalize_relation(relation);
    let mut out = Vec::new();
    for row in rows {
        let fact = row?;
        // Canonical new writes must find historic known synonyms too.
        // Unknown open predicates retain their exact spelling.
        if super::ontology::normalize_relation(&fact.relation) == wanted {
            out.push(fact);
        }
    }
    Ok(out)
}

/// Active facts for relation + object entity (any subject).
pub fn list_active_by_relation_object(
    conn: &Connection,
    namespace: &str,
    relation: &str,
    object_entity_id: &str,
) -> Result<Vec<KgFact>> {
    let mut stmt = conn.prepare(&format!(
        r#"SELECT {FACT_SELECT}
           FROM kg_facts f
           WHERE f.namespace = ?1
             AND f.relation = ?2
             AND f.object_entity_id = ?3
             AND f.status IN ('active','planned','paused')
           ORDER BY f.updated_at_ms DESC"#
    ))?;
    let rows = stmt.query_map(params![namespace, relation, object_entity_id], |row| {
        row_to_fact(conn, namespace, row)
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

fn insert_fact_version(
    conn: &Connection,
    namespace: &str,
    fact_id: &str,
    revision: i64,
    operation: &str,
    fact: &KgFact,
) -> Result<()> {
    conn.execute(
        r#"INSERT OR IGNORE INTO kg_fact_versions(
             namespace, id, fact_id, revision, operation, snapshot_json, created_at_ms
           ) VALUES (?1,?2,?3,?4,?5,?6,?7)"#,
        params![
            namespace,
            Uuid::now_v7().to_string(),
            fact_id,
            revision,
            operation,
            serde_json::to_string(fact)?,
            now_ms(),
        ],
    )?;
    Ok(())
}

fn semantic_fingerprint(fact: &KgFact) -> String {
    let mut domains = fact
        .domains
        .iter()
        .map(|d| d.to_ascii_lowercase())
        .collect::<Vec<_>>();
    domains.sort();
    domains.dedup();
    // Ingest stamps valid_from with now on each assertion; explicit temporal
    // attributes, not that default timestamp, participate in identity.
    serde_json::json!([
        fact.subject_id,
        super::ontology::normalize_relation(&fact.relation),
        fact.object,
        fact.polarity.as_str(),
        fact.status.as_str(),
        fact.condition_text,
        fact.exception_text,
        fact.reason_text,
        fact.owner_member_id,
        fact.privacy.as_str(),
        domains,
        fact.attrs,
        fact.valid_to_ms,
    ])
    .to_string()
}

pub fn row_to_fact(
    conn: &Connection,
    namespace: &str,
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<KgFact> {
    let id_s: String = row.get("id")?;
    let relation_s: String = row.get("relation")?;
    let object_entity_id: Option<String> = row.get("object_entity_id")?;
    let object_value_type: Option<String> = row.get("object_value_type")?;
    let object_value_json: Option<String> = row.get("object_value_json")?;
    let object = if let Some(eid) = object_entity_id {
        FactObject::Entity { entity_id: eid }
    } else {
        FactObject::Value {
            value_type: object_value_type.unwrap_or_else(|| "text".into()),
            value: object_value_json
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(Value::Null),
        }
    };
    let polarity_s: String = row.get("polarity")?;
    let status_s: String = row.get("status")?;
    let privacy_s: String = row.get("privacy")?;
    let source_memory_id: Option<String> = row.get("source_memory_id")?;
    let attrs_json: String = row.get("attrs_json")?;
    let id = Uuid::parse_str(&id_s).unwrap_or(Uuid::nil());
    let domains = list_resource_domains(conn, namespace, "fact", &id_s).unwrap_or_default();
    Ok(KgFact {
        id,
        subject_id: row.get("subject_id")?,
        relation: super::ontology::normalize_relation(&relation_s),
        object,
        owner_member_id: row.get("owner_member_id")?,
        domains,
        polarity: FactPolarity::parse(&polarity_s).unwrap_or(FactPolarity::Affirmative),
        status: FactStatus::parse(&status_s).unwrap_or(FactStatus::Active),
        valid_from_ms: row.get("valid_from_ms")?,
        valid_to_ms: row.get("valid_to_ms")?,
        condition_text: row.get("condition_text")?,
        exception_text: row.get("exception_text")?,
        reason_text: row.get("reason_text")?,
        evidence_text: row.get("evidence_text")?,
        evidence_summary: row.get("evidence_summary")?,
        source_memory_id: source_memory_id.and_then(|s| Uuid::parse_str(&s).ok()),
        confidence: row.get("confidence")?,
        privacy: PrivacyLevel::parse(&privacy_s).unwrap_or(PrivacyLevel::FamilyShared),
        revision: row.get("revision")?,
        created_at_ms: row.get("created_at_ms")?,
        updated_at_ms: row.get("updated_at_ms")?,
        attrs: serde_json::from_str(&attrs_json).unwrap_or(Value::Object(Default::default())),
    })
}

const FACT_SELECT: &str = r#"
  f.id AS id, f.subject_id AS subject_id, f.relation AS relation,
  f.object_entity_id AS object_entity_id, f.object_value_type AS object_value_type,
  f.object_value_json AS object_value_json, f.owner_member_id AS owner_member_id,
  f.polarity AS polarity, f.status AS status, f.valid_from_ms AS valid_from_ms,
  f.valid_to_ms AS valid_to_ms, f.condition_text AS condition_text,
  f.exception_text AS exception_text, f.reason_text AS reason_text,
  f.evidence_text AS evidence_text, f.evidence_summary AS evidence_summary,
  f.source_memory_id AS source_memory_id, f.confidence AS confidence,
  f.privacy AS privacy, f.revision AS revision, f.created_at_ms AS created_at_ms,
  f.updated_at_ms AS updated_at_ms, f.attrs_json AS attrs_json
"#;

pub fn list_facts_for_person(
    conn: &Connection,
    namespace: &str,
    person_entity_id: &str,
    owner_member_id: Option<&str>,
    include_ended: bool,
    domains: &[String],
) -> Result<Vec<KgFact>> {
    let status_filter = if include_ended {
        ""
    } else {
        "AND f.status IN ('active','planned','paused')"
    };
    let sql = format!(
        r#"SELECT {FACT_SELECT}
           FROM kg_facts f
           WHERE f.namespace = ?1
             AND (
               f.subject_id = ?2
               OR f.object_entity_id = ?2
               OR (?3 IS NOT NULL AND f.owner_member_id = ?3)
             )
             {status_filter}
           ORDER BY f.updated_at_ms DESC"#
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        params![namespace, person_entity_id, owner_member_id],
        |row| row_to_fact(conn, namespace, row),
    )?;
    let mut facts = Vec::new();
    for r in rows {
        let fact = r?;
        if !domains.is_empty()
            && !fact
                .domains
                .iter()
                .any(|d| domains.iter().any(|want| want.eq_ignore_ascii_case(d)))
        {
            // Also allow relation default domain match when domains column empty.
            if fact.domains.is_empty() {
                let defaults = RelationType::parse_known(&fact.relation)
                    .map(|r| r.default_domains())
                    .unwrap_or(&[]);
                if !defaults
                    .iter()
                    .any(|d| domains.iter().any(|want| want.eq_ignore_ascii_case(d)))
                {
                    continue;
                }
            } else {
                continue;
            }
        }
        facts.push(fact);
    }
    Ok(facts)
}

/// Compatibility store search. ACL-aware callers use collect_search_facts.
pub fn search_facts(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    limit: usize,
) -> Result<Vec<KgFact>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut facts = collect_search_facts(conn, namespace, text, member_id, domains, |_| Ok(true))?;
    facts.truncate(limit);
    Ok(facts)
}

/// No cap before the caller's ACL/source filter. Hidden edges cannot be bridges.
pub fn collect_search_facts(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    mut visible: impl FnMut(&KgFact) -> Result<bool>,
) -> Result<Vec<KgFact>> {
    let mut facts = Vec::new();
    for fact in collect_facts_overview(conn, namespace, false, domains)? {
        if visible(&fact)? {
            facts.push(fact);
        }
    }
    rank_recall_facts(conn, namespace, text, member_id, facts, |_| Ok(Vec::new()))
}

/// ACL-aware search with a caller-supplied, viewer-authorized name resolver for
/// person anchors. Raw person aliases have no reliable provenance and are never
/// read by the ranking path. Non-person entity names remain graph-local labels.
pub fn collect_search_facts_with_person_names(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    mut visible: impl FnMut(&KgFact) -> Result<bool>,
    mut person_names: impl FnMut(&KgEntity) -> Result<Vec<String>>,
) -> Result<Vec<KgFact>> {
    let mut facts = Vec::new();
    for fact in collect_facts_overview(conn, namespace, false, domains)? {
        if visible(&fact)? {
            facts.push(fact);
        }
    }
    rank_recall_facts(conn, namespace, text, member_id, facts, |entity| {
        person_names(entity)
    })
}

/// Recall-path person name resolver: canonical name plus the roster member's
/// display name and registered aliases. Roster rows carry provenance (they are
/// registered household data), unlike raw kg_aliases rows which are never read
/// here. Nicknames such as "航航" therefore anchor the right person without
/// opening the alias table to unvetted extraction output.
pub fn person_names_for_entity(
    entity: &KgEntity,
    roster: &[crate::household::HouseholdMember],
) -> Vec<String> {
    let mut out = vec![entity.canonical_name.trim().to_string()];
    let member_id = entity
        .owner_member_id
        .as_deref()
        .map(str::to_string)
        .or_else(|| entity.id.strip_prefix("person:").map(str::to_string));
    if let Some(mid) = member_id {
        if let Some(member) = roster.iter().find(|m| m.member_id == mid) {
            out.push(member.display_name.trim().to_string());
            for alias in &member.aliases {
                out.push(alias.trim().to_string());
            }
        }
    }
    out.retain(|n| !n.is_empty());
    out.sort();
    out.dedup();
    out
}

fn fact_matches_domains(fact: &KgFact, domains: &[String]) -> bool {    domains.is_empty()
        || fact
            .domains
            .iter()
            .any(|d| domains.iter().any(|w| w.eq_ignore_ascii_case(d)))
        || (fact.domains.is_empty()
            && RelationType::parse_known(&fact.relation)
                .map(|r| r.default_domains())
                .unwrap_or(&[])
                .iter()
                .any(|d| domains.iter().any(|w| w.eq_ignore_ascii_case(d))))
}

fn search_words(text: &str) -> Vec<String> {
    static TOKENIZER: std::sync::OnceLock<jieba_rs::Jieba> = std::sync::OnceLock::new();
    let mut out = std::collections::BTreeSet::new();
    for word in TOKENIZER
        .get_or_init(jieba_rs::Jieba::new)
        .cut_for_search(text, false)
    {
        let word = word.trim().to_lowercase();
        if word.chars().count() < 2 || !word.chars().any(char::is_alphanumeric) {
            continue;
        }
        if matches!(
            word.as_str(),
            "什么"
                | "哪些"
                | "怎么"
                | "怎样"
                | "如何"
                | "多少"
                | "哪里"
                | "哪个"
                | "有没有"
                | "是不是"
                | "是否"
                | "可以"
                | "能够"
                | "现在"
                | "目前"
                | "一下"
                | "告诉"
                | "告诉我"
                | "请问"
                | "关于"
                | "情况"
                | "事情"
                | "最近"
                | "what"
                | "which"
                | "where"
                | "when"
                | "how"
                | "does"
                | "the"
                | "are"
                | "is"
                | "of"
                | "to"
                | "please"
                | "tell"
                | "me"
                | "about"
                | "and"
        ) {
            continue;
        }
        out.insert(word);
    }
    // 同义词扩展：口语/书面互通（妈妈↔母亲），词面通道不再因用词差异漏召回。
    let mut extra = std::collections::BTreeSet::new();
    for word in &out {
        for synonym in crate::query_rewrite::expand_word(word) {
            extra.insert((*synonym).to_string());
        }
    }
    out.extend(extra);
    out.into_iter().collect()
}

fn mentions(text: &str, name: &str) -> bool {
    let name = name.trim().to_lowercase();
    if name.is_empty() {
        return false;
    }
    if name.chars().count() < 2 {
        return text.trim() == name;
    }
    if name.is_ascii() {
        text.match_indices(&name).any(|(i, _)| {
            !text[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric())
                && !text[i + name.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric())
        })
    } else {
        text.contains(&name)
    }
}

fn relation_intents(text: &str) -> std::collections::BTreeSet<String> {
    let mut result = std::collections::BTreeSet::new();
    let groups: &[(&[&str], &[&str])] = &[
        // 家庭关系：查询里的称谓词直接映射到存储的规范关系标签。
        (&["妈妈", "母亲", "妈", "娘"], &["妈妈", "儿子", "女儿"]),
        (&["爸爸", "父亲", "爸", "爹"], &["爸爸", "儿子", "女儿"]),
        (&["儿子"], &["儿子", "妈妈", "爸爸"]),
        (&["女儿"], &["女儿", "妈妈", "爸爸"]),
        (
            &["老婆", "妻子", "夫人", "爱人", "结婚", "配偶", "夫妻"],
            &["老婆", "配偶"],
        ),
        (
            &["老公", "丈夫", "爱人", "结婚", "配偶", "夫妻"],
            &["老公", "配偶"],
        ),
        (&["父母", "爸妈", "双亲"], &["爸爸", "妈妈"]),
        (&["过敏", "allergy", "allergic"],
            &["allergic_to", "contraindicated"],
        ),
        (
            &[
                "服药",
                "用药",
                "吃药",
                "什么药",
                "哪些药",
                "剂量",
                "频次",
                "服用",
                "medication",
                "medicine",
                "dosage",
            ],
            &["takes_medication", "contraindicated"],
        ),
        (
            &[
                "疾病",
                "什么病",
                "患有",
                "诊断",
                "病史",
                "condition",
                "diagnosis",
            ],
            &["has_condition"],
        ),
        (
            &["学校", "小学", "中学", "上学", "就读", "school"],
            &["attends_school"],
        ),
        (
            &["课程", "上课", "课外班", "course", "class"],
            &["enrolled_in", "taught_by"],
        ),
        (
            &["老师", "教师", "谁教", "teacher", "teaches"],
            &["taught_by", "school_contact"],
        ),
        (
            &[
                "学费",
                "缴费",
                "付款",
                "支付",
                "费用",
                "多少钱",
                "payment",
                "cost",
            ],
            &["pays_for", "debited_from", "owes"],
        ),
        (
            &["账户", "银行卡", "account"],
            &["owns", "opened_at", "debited_from"],
        ),
        (
            &["贷款", "欠款", "负债", "debt", "loan"],
            &["owes", "pays_for"],
        ),
        (
            &["保险", "保单", "insurance"],
            &[
                "insures",
                "insured_by",
                "underwrites",
                "benefits",
                "covered_by",
            ],
        ),
        (&["预算", "budget"], &["sets_budget", "budget_for"]),
        (
            &["喜欢", "爱好", "偏好", "likes", "prefer"],
            &["likes", "risk_preference"],
        ),
        (&["原因", "为什么", "why"], &["paused_because"]),
    ];
    for (words, relations) in groups {
        if words.iter().any(|w| mentions(text, w)) {
            result.extend(
                relations
                    .iter()
                    .map(|r| super::ontology::normalize_relation(r)),
            );
        }
    }
    if let Some(r) = RelationType::parse_known(text) {
        result.insert(r.as_zh_str().to_string());
    }
    result
}

fn rank_recall_facts(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    facts: Vec<KgFact>,
    mut person_names: impl FnMut(&KgEntity) -> Result<Vec<String>>,
) -> Result<Vec<KgFact>> {
    use std::collections::{BTreeMap, BTreeSet};
    let text = text.trim().to_lowercase();
    // Resolve explicit names even when their facts are hidden or outside the
    // requested domain. Otherwise a hidden person's name could degrade into a
    // global topic search and return another person's facts.
    let entities = list_entities(conn, namespace, i64::MAX as usize)?;
    let by_id: BTreeMap<_, _> = entities.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut anchors = BTreeSet::new();
    let mut people = BTreeSet::new();
    let mut remainder = text.clone();
    let member = member_id.map(str::trim).filter(|m| !m.is_empty());
    if let Some(m) = member {
        let id = if m.starts_with("person:") {
            m.to_string()
        } else {
            person_entity_id(m)
        };
        people.insert(id.clone());
        anchors.insert(id);
    }
    // Resolve people first, then disambiguate business instances by their owner.
    // Consume a name only after every same-name candidate has been considered.
    let mut person_name_cache: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut names: BTreeMap<String, Vec<&KgEntity>> = BTreeMap::new();
    for entity in &entities {
        let entity_names = if entity.entity_type == EntityType::Person {
            let authorized = person_names(entity)?;
            person_name_cache.insert(entity.id.clone(), authorized.clone());
            authorized
        } else {
            std::iter::once(entity.canonical_name.clone())
                .chain(entity.aliases.iter().cloned())
                .collect()
        };
        for name in entity_names
            .into_iter()
            .map(|name| name.trim().to_lowercase())
        {
            if name.is_empty() {
                continue;
            }
            names.entry(name).or_default().push(entity);
        }
    }
    let mut names: Vec<_> = names.into_iter().collect();
    names.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
    for (name, candidates) in &names {
        if mentions(&text, name) {
            for entity in candidates
                .iter()
                .filter(|e| e.entity_type == EntityType::Person)
            {
                let names = person_name_cache
                    .get(&entity.id)
                    .cloned()
                    .unwrap_or_default();
                if names.iter().any(|n| mentions(&text, n))
                    && (member.is_none() || people.contains(&entity.id))
                {
                    people.insert(entity.id.clone());
                    anchors.insert(entity.id.clone());
                }
            }
        }
    }
    for (name, candidates) in names {
        if !mentions(&remainder, &name) {
            continue;
        }
        let matched: Vec<_> = candidates
            .into_iter()
            .filter(|entity| {
                if entity.entity_type == EntityType::Person {
                    return people.contains(&entity.id);
                }
                if RelationType::owner_scoped_entity(entity.entity_type)
                    || entity.privacy == PrivacyLevel::OwnerPrivate
                {
                    return entity.owner_member_id.as_deref().is_some_and(|owner| {
                        people.contains(&person_entity_id(owner))
                            || member
                                .is_some_and(|m| m.strip_prefix("person:").unwrap_or(m) == owner)
                    });
                }
                true
            })
            .collect();
        if !matched.is_empty() {
            for entity in matched {
                anchors.insert(entity.id.clone());
            }
            // Chinese canonicalization can leave a short possessive such as 的;
            // names are lowercased above and replacement is for retrieval only.
            remainder = remainder.replace(&name, " ");
        }
        // An ambiguous business name stays a search term rather than choosing
        // whichever owner's object happened to be updated most recently.
    }
    // Explicit member/name anchors are themselves sufficient evidence of intent
    // when no residual topical word survives (e.g. “张三的工资卡”).
    let has_explicit_entity_topic = anchors.iter().any(|id| !people.contains(id));
    let intents = relation_intents(&text);
    let words = search_words(&remainder);
    let topics: BTreeSet<_> = anchors.difference(&people).cloned().collect();
    let has_topic =
        !intents.is_empty() || !words.is_empty() || !topics.is_empty() || has_explicit_entity_topic;
    let ends = |fact: &KgFact, ids: &BTreeSet<String>| {
        ids.contains(&fact.subject_id)
            || matches!(&fact.object, FactObject::Entity { entity_id } if ids.contains(entity_id))
    };
    let person_direct = |fact: &KgFact| {
        ends(fact, &people)
            || member.is_some_and(|m| fact.owner_member_id.as_deref() == Some(m))
            || fact
                .owner_member_id
                .as_ref()
                .is_some_and(|m| people.contains(&person_entity_id(m)))
    };
    let mut topic_scores = Vec::new();
    for fact in &facts {
        let relation = super::ontology::normalize_relation(&fact.relation);
        let mut score = if intents.contains(&relation) { 80 } else { 0 };
        if ends(fact, &topics) {
            score += 100;
        }
        let (_, _, display) = display_fact(conn, namespace, fact)?;
        let haystack = format!("{} {}", display, relation).to_lowercase();
        score += words.iter().filter(|w| mentions(&haystack, w)).count() * 12;
        // Scalar legacy evidence may belong to a different, unauthorized
        // source of the same fact. Never use it as a ranking side channel.
        // Source text lookup remains the viewer-filtered text-memory path.
        topic_scores.push(score);
    }
    // A single outward hop, through non-person objects only. Do not travel
    // through siblings or other people into their unrelated private subgraphs.
    let mut bridges: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, fact) in facts.iter().enumerate() {
        if !people.is_empty() && person_direct(fact) {
            if let FactObject::Entity { entity_id } = &fact.object {
                if by_id
                    .get(entity_id.as_str())
                    .is_some_and(|e| e.entity_type != EntityType::Person)
                {
                    bridges.entry(entity_id.clone()).or_default().push(index);
                }
            }
        }
    }
    let mut ranked: BTreeMap<usize, usize> = BTreeMap::new();
    for (index, fact) in facts.iter().enumerate() {
        let topic = topic_scores[index];
        let direct = if !people.is_empty() {
            person_direct(fact)
        } else {
            ends(fact, &anchors)
        };
        if text.is_empty() {
            if member.is_none() || direct {
                ranked.insert(index, 1);
            }
        } else if !anchors.is_empty() {
            if direct && (!has_topic || topic > 0) {
                ranked.insert(index, 200 + topic);
            }
            if let Some(bridge_ids) = bridges.get(&fact.subject_id) {
                if has_topic && topic > 0 && !direct {
                    ranked.insert(index, 100 + topic);
                    for bridge in bridge_ids {
                        ranked.entry(*bridge).or_insert(20);
                    }
                }
            }
        } else if topic > 0 {
            ranked.insert(index, topic);
        }
    }
    let mut selected: Vec<_> = facts
        .into_iter()
        .enumerate()
        .filter_map(|(i, f)| ranked.get(&i).map(|s| (*s, f)))
        .collect();
    selected.sort_by(|(sa, a), (sb, b)| {
        sb.cmp(sa)
            .then_with(|| b.confidence.total_cmp(&a.confidence))
            .then_with(|| b.updated_at_ms.cmp(&a.updated_at_ms))
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(selected.into_iter().map(|(_, f)| f).collect())
}
pub fn collect_related_entities(
    conn: &Connection,
    namespace: &str,
    facts: &[KgFact],
) -> Result<Vec<KgEntity>> {
    let mut ids = std::collections::BTreeSet::new();
    for f in facts {
        ids.insert(f.subject_id.clone());
        if let FactObject::Entity { entity_id } = &f.object {
            ids.insert(entity_id.clone());
        }
    }
    let mut out = Vec::new();
    for id in ids {
        if let Some(e) = get_entity(conn, namespace, &id)? {
            out.push(e);
        }
    }
    Ok(out)
}

/// List facts in a namespace (newest first). `limit` caps rows; returns whether more exist.
pub fn list_facts_overview(
    conn: &Connection,
    namespace: &str,
    include_ended: bool,
    domains: &[String],
    limit: usize,
) -> Result<(Vec<KgFact>, bool)> {
    let status_filter = if include_ended {
        ""
    } else {
        "AND f.status IN ('active','planned','paused')"
    };
    // Domain filtering precedes the quota; callers with ACLs use the uncapped
    // collector below and apply their visibility predicate before truncation.
    let sql = format!(
        r#"SELECT {FACT_SELECT}
           FROM kg_facts f
           WHERE f.namespace = ?1
             {status_filter}
           ORDER BY f.updated_at_ms DESC, f.id"#
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![namespace], |row| row_to_fact(conn, namespace, row))?;
    let mut facts = Vec::new();
    for r in rows {
        let fact = r?;
        if !domains.is_empty() {
            let hit = fact
                .domains
                .iter()
                .any(|d| domains.iter().any(|w| w.eq_ignore_ascii_case(d)))
                || (fact.domains.is_empty()
                    && RelationType::parse_known(&fact.relation)
                        .map(|r| r.default_domains())
                        .unwrap_or(&[])
                        .iter()
                        .any(|d| domains.iter().any(|w| w.eq_ignore_ascii_case(d))));
            if !hit {
                continue;
            }
        }
        facts.push(fact);
    }
    let truncated = facts.len() > limit;
    if truncated {
        facts.truncate(limit);
    }
    Ok((facts, truncated))
}

/// Domain-filtered, uncapped facts for service-layer ACL/source filtering.
pub fn collect_facts_overview(
    conn: &Connection,
    namespace: &str,
    include_ended: bool,
    domains: &[String],
) -> Result<Vec<KgFact>> {
    Ok(list_facts_overview(conn, namespace, include_ended, domains, usize::MAX)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster_member(id: &str, display: &str, aliases: &[&str]) -> crate::household::HouseholdMember {
        crate::household::HouseholdMember {
            member_id: id.to_string(),
            display_name: display.to_string(),
            role: crate::household::MemberRole::Other,
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            age_band: None,
        }
    }

    fn person_entity(id: &str, canonical: &str, owner: Option<&str>) -> KgEntity {
        KgEntity {
            id: id.to_string(),
            entity_type: EntityType::Person,
            subtype: None,
            canonical_name: canonical.to_string(),
            aliases: vec![],
            status: "active".into(),
            domains: vec![],
            owner_member_id: owner.map(str::to_string),
            privacy: PrivacyLevel::FamilyShared,
            first_seen_at_ms: 0,
            last_confirmed_at_ms: 0,
            attrs: serde_json::Value::Null,
        }
    }

    #[test]
    fn person_names_include_roster_display_and_aliases() {
        let roster = vec![
            roster_member("林子航", "林子航", &["航航", "儿子"]),
            roster_member("主驾", "主驾", &["主驾驶", "司机"]),
        ];
        let entity = person_entity("person:林子航", "林子航", Some("林子航"));
        let names = person_names_for_entity(&entity, &roster);
        for expected in ["林子航", "航航", "儿子"] {
            assert!(names.iter().any(|n| n == expected), "names={names:?}");
        }
        // 与 roster 无关的成员不泄露名字。
        let stranger = person_entity("person:路人", "路人", Some("路人"));
        let names = person_names_for_entity(&stranger, &roster);
        assert_eq!(names, vec!["路人".to_string()]);
    }

    #[test]
    fn search_words_expand_family_terms() {
        let words = search_words("航航的妈妈是谁");
        for expected in ["妈妈", "母亲"] {
            assert!(words.iter().any(|w| w == expected), "words={words:?}");
        }
    }

    #[test]
    fn relation_intents_map_family_questions() {
        let intents = relation_intents("航航的妈妈是谁");
        assert!(intents.contains("妈妈"), "intents={intents:?}");
        // 反向表述同样是答案。
        assert!(intents.contains("儿子") || intents.contains("女儿"));

        let intents = relation_intents("林子航的爸爸是谁");
        assert!(intents.contains("爸爸"), "intents={intents:?}");
        assert!(intents.contains("儿子") || intents.contains("女儿"));

        assert!(relation_intents("林子航负责整理自己的书包").is_empty());
    }
}

/// Safe no-fact hubs are roster people only, never arbitrary extracted people
/// or private disease/account nodes. Attributes are not authorized by a roster.
pub fn list_safe_roster_people(conn: &Connection, namespace: &str) -> Result<Vec<KgEntity>> {
    let mut stmt = conn.prepare("SELECT id FROM kg_entities WHERE namespace = ?1 AND entity_type = 'person' AND subtype = 'household_member' AND privacy = 'family_shared' AND status = 'active' ORDER BY id")?;
    let ids = stmt
        .query_map(params![namespace], |r| r.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut people = Vec::new();
    for id in ids {
        if let Some(mut person) = get_entity(conn, namespace, &id)? {
            if person.owner_member_id.as_deref().is_some_and(|member_id| {
                crate::household::is_shared_member_id(member_id) || member_id.trim() == "__public__"
            }) {
                continue;
            }
            person.attrs = serde_json::json!({});
            people.push(person);
        }
    }
    Ok(people)
}

/// All entities in a namespace. Raw persistence API; never use as an ACL fallback.
pub fn list_entities(conn: &Connection, namespace: &str, limit: usize) -> Result<Vec<KgEntity>> {
    let mut stmt = conn.prepare(
        r#"SELECT id FROM kg_entities
           WHERE namespace = ?1 AND status != 'merged'
           ORDER BY updated_at_ms DESC
           LIMIT ?2"#,
    )?;
    let ids: Vec<String> = stmt
        .query_map(params![namespace, limit as i64], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut out = Vec::new();
    for id in ids {
        if let Some(e) = get_entity(conn, namespace, &id)? {
            out.push(e);
        }
    }
    Ok(out)
}

pub fn insert_grant(conn: &Connection, namespace: &str, input: &GrantInput) -> Result<KgGrant> {
    let grantor = input.grantor_member_id.trim();
    let grantee = input.grantee_member_id.trim();
    if grantor.is_empty() || grantee.is_empty() {
        return Err(MemoraError::InvalidState(
            "grantor_member_id and grantee_member_id are required".into(),
        ));
    }
    let scope_kind = normalize_scope_kind(&input.scope_kind)?;
    if scope_kind != "all" && input.scope_value.trim().is_empty() {
        return Err(MemoraError::InvalidState(
            "scope_value required unless scope_kind=all".into(),
        ));
    }
    let scope_value = if scope_kind == "relation" {
        normalize_relation(&input.scope_value)
    } else {
        input.scope_value.trim().to_string()
    };
    let ops = if input.operations.is_empty() {
        vec!["view".to_string()]
    } else {
        input
            .operations
            .iter()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let now = now_ms();
    let id = Uuid::now_v7().to_string();
    let scope_json = serde_json::json!({
        "kind": scope_kind,
        "value": scope_value,
    });
    conn.execute(
        r#"INSERT INTO kg_grants(
             namespace, id, grantor_member_id, grantee_member_id, scope_json, operations_json,
             valid_from_ms, valid_to_ms, revoked, evidence_text, created_at_ms
           ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,0,?9,?10)"#,
        params![
            namespace,
            id,
            grantor,
            grantee,
            serde_json::to_string(&scope_json)?,
            serde_json::to_string(&ops)?,
            input.valid_from_ms,
            input.valid_to_ms,
            input.evidence_text,
            now,
        ],
    )?;
    Ok(KgGrant {
        id,
        grantor_member_id: grantor.to_string(),
        grantee_member_id: grantee.to_string(),
        scope_kind: scope_kind.to_string(),
        scope_value,
        operations: ops,
        valid_from_ms: input.valid_from_ms,
        valid_to_ms: input.valid_to_ms,
        revoked: false,
        evidence_text: input.evidence_text.clone(),
        created_at_ms: now,
    })
}

pub fn revoke_grants(
    conn: &Connection,
    namespace: &str,
    input: &RevokeGrantInput,
) -> Result<usize> {
    if let Some(id) = input
        .grant_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let n = conn.execute(
            "UPDATE kg_grants SET revoked = 1 WHERE namespace = ?1 AND id = ?2 AND revoked = 0",
            params![namespace, id],
        )?;
        return Ok(n);
    }
    let grantor = input.grantor_member_id.as_deref().unwrap_or("");
    let grantee = input.grantee_member_id.as_deref().unwrap_or("");
    if grantor.is_empty() || grantee.is_empty() {
        return Err(MemoraError::InvalidState(
            "revoke requires grant_id or grantor+grantee".into(),
        ));
    }
    let mut sql = String::from(
        "UPDATE kg_grants SET revoked = 1 WHERE namespace = ?1 AND grantor_member_id = ?2 AND grantee_member_id = ?3 AND revoked = 0",
    );
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![
        Box::new(namespace.to_string()),
        Box::new(grantor.to_string()),
        Box::new(grantee.to_string()),
    ];
    if let Some(kind) = input
        .scope_kind
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sql.push_str(" AND scope_json LIKE ?4");
        params_vec.push(Box::new(format!("%\"kind\":\"{kind}\"%")));
    }
    let mut stmt = conn.prepare(&sql)?;
    let refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|b| b.as_ref()).collect();
    Ok(stmt.execute(refs.as_slice())?)
}

pub fn list_grants(
    conn: &Connection,
    namespace: &str,
    member_id: Option<&str>,
    as_grantor: bool,
) -> Result<Vec<KgGrant>> {
    let sql = if member_id.is_some() {
        if as_grantor {
            "SELECT id, grantor_member_id, grantee_member_id, scope_json, operations_json, valid_from_ms, valid_to_ms, revoked, evidence_text, created_at_ms FROM kg_grants WHERE namespace = ?1 AND grantor_member_id = ?2 ORDER BY created_at_ms DESC"
        } else {
            "SELECT id, grantor_member_id, grantee_member_id, scope_json, operations_json, valid_from_ms, valid_to_ms, revoked, evidence_text, created_at_ms FROM kg_grants WHERE namespace = ?1 AND grantee_member_id = ?2 ORDER BY created_at_ms DESC"
        }
    } else {
        "SELECT id, grantor_member_id, grantee_member_id, scope_json, operations_json, valid_from_ms, valid_to_ms, revoked, evidence_text, created_at_ms FROM kg_grants WHERE namespace = ?1 ORDER BY created_at_ms DESC"
    };
    let mut stmt = conn.prepare(sql)?;
    let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<KgGrant> {
        let scope_json: String = row.get(3)?;
        let ops_json: String = row.get(4)?;
        let scope: Value = serde_json::from_str(&scope_json).unwrap_or(Value::Null);
        let kind = scope
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("all")
            .to_string();
        let value = scope
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let operations: Vec<String> = serde_json::from_str(&ops_json).unwrap_or_default();
        let revoked: i64 = row.get(7)?;
        Ok(KgGrant {
            id: row.get(0)?,
            grantor_member_id: row.get(1)?,
            grantee_member_id: row.get(2)?,
            scope_kind: kind,
            scope_value: value,
            operations,
            valid_from_ms: row.get(5)?,
            valid_to_ms: row.get(6)?,
            revoked: revoked != 0,
            evidence_text: row.get(8)?,
            created_at_ms: row.get(9)?,
        })
    };
    let rows = if let Some(mid) = member_id {
        stmt.query_map(params![namespace, mid], map_row)?
    } else {
        stmt.query_map(params![namespace], map_row)?
    };
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Grants can change visibility without a data mutation. Future grants count
/// too, so cached misses cannot survive activation of a timed authorization.
pub fn has_unexpired_timed_grants(conn: &Connection, namespace: &str, now_ms: i64) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM kg_grants WHERE namespace=?1 AND revoked=0 AND (valid_from_ms IS NOT NULL OR valid_to_ms IS NOT NULL) AND (valid_to_ms IS NULL OR valid_to_ms>=?2))",
        params![namespace, now_ms],
        |row| row.get::<_, bool>(0),
    )?)
}

pub fn has_view_grant(
    conn: &Connection,
    namespace: &str,
    owner_member_id: &str,
    viewer_member_id: &str,
    fact: &KgFact,
    now: i64,
) -> Result<bool> {
    let grants = list_grants(conn, namespace, Some(viewer_member_id), false)?;
    for g in grants {
        if g.revoked || g.grantor_member_id != owner_member_id {
            continue;
        }
        if !g.operations.is_empty()
            && !g
                .operations
                .iter()
                .any(|op| op.eq_ignore_ascii_case("view") || op.eq_ignore_ascii_case("read"))
        {
            continue;
        }
        if let Some(from) = g.valid_from_ms {
            if now < from {
                continue;
            }
        }
        if let Some(to) = g.valid_to_ms {
            if now > to {
                continue;
            }
        }
        let hit = match g.scope_kind.as_str() {
            "all" => true,
            "fact" => g.scope_value == fact.id.to_string(),
            "relation" => {
                !g.scope_value.trim().is_empty()
                    && !fact.relation.trim().is_empty()
                    && normalize_relation(&g.scope_value) == normalize_relation(&fact.relation)
            }
            "domain" => {
                fact.domains
                    .iter()
                    .any(|d| d.eq_ignore_ascii_case(&g.scope_value))
                    || (fact.domains.is_empty()
                        && RelationType::parse_known(&fact.relation)
                            .map(|r| r.default_domains())
                            .unwrap_or(&[])
                            .iter()
                            .any(|d| d.eq_ignore_ascii_case(&g.scope_value)))
            }
            _ => false,
        };
        if hit {
            return Ok(true);
        }
    }
    Ok(false)
}

fn normalize_scope_kind(raw: &str) -> Result<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "all" => Ok("all"),
        "fact" => Ok("fact"),
        "relation" => Ok("relation"),
        "domain" => Ok("domain"),
        other => Err(MemoraError::InvalidState(format!(
            "unknown grant scope_kind: {other}"
        ))),
    }
}

fn display_entity_name(conn: &Connection, namespace: &str, entity_id: &str) -> Result<String> {
    let Some(entity) = get_entity(conn, namespace, entity_id)? else {
        return Ok(entity_id.to_string());
    };
    if entity.entity_type == EntityType::Person {
        // Historical alias ingestion could overwrite a shared person's canonical
        // label with a private nickname. Public labels come from the host roster.
        if let Some(member) = crate::household::list_members(conn, namespace)?
            .into_iter()
            .find(|member| person_entity_id(&member.member_id) == entity.id)
        {
            return Ok(member.display_name);
        }
    }
    Ok(entity.canonical_name)
}

pub fn display_fact(
    conn: &Connection,
    namespace: &str,
    fact: &KgFact,
) -> Result<(String, String, String)> {
    let subject_name = display_entity_name(conn, namespace, &fact.subject_id)?;
    let object_label = match &fact.object {
        FactObject::Entity { entity_id } => display_entity_name(conn, namespace, entity_id)?,
        FactObject::Value { value, .. } => match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
    };
    let subject_name = bounded_text(&subject_name, 96);
    let object_label = bounded_text(&object_label, 160);
    let polarity = if fact.polarity == FactPolarity::Negative {
        "不"
    } else {
        ""
    };
    let status = match fact.status {
        FactStatus::Paused => "（已暂停）",
        FactStatus::Ended => "（已结束）",
        FactStatus::Planned => "（计划中）",
        FactStatus::Corrected => "（已纠正）",
        FactStatus::Withdrawn => "（已撤回）",
        _ => "",
    };
    let mut details = Vec::new();
    for (label, value, cap) in [
        ("条件", fact.condition_text.as_deref(), 192),
        ("例外", fact.exception_text.as_deref(), 192),
        ("原因", fact.reason_text.as_deref(), 160),
    ] {
        if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
            details.push(format!("{label}：{}", bounded_qualifier(value, cap)));
        }
    }
    if fact.valid_from_ms.is_some() || fact.valid_to_ms.is_some() {
        let date = |value: Option<i64>| {
            value.and_then(|ms| {
                chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
                    .map(|dt| dt.with_timezone(&chrono::Local).format("%Y-%m-%d").to_string())
            })
        };
        let line = match (date(fact.valid_from_ms), date(fact.valid_to_ms)) {
            (Some(from), Some(to)) => format!("有效时间：{from} 至 {to}"),
            (Some(from), None) => format!("有效时间：{from} 至今"),
            (None, Some(to)) => format!("有效时间：至 {to}"),
            (None, None) => String::new(),
        };
        if !line.is_empty() {
            details.push(line);
        }
    }
    let details = if details.is_empty() {
        String::new()
    } else {
        format!("（{}）", details.join("；"))
    };
    let display = format!(
        "{subject_name} {polarity}{} {object_label}{status}{details}{}",
        bounded_text(&relation_label(&fact.relation), 64),
        format_fact_attrs(&fact.attrs)
    );
    Ok((subject_name, object_label, display))
}

fn bounded_text(text: &str, cap: usize) -> String {
    if text.chars().count() <= cap {
        return text.to_string();
    }
    let mut out: String = text.chars().take(cap.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn bounded_qualifier(text: &str, cap: usize) -> String {
    if text.chars().count() <= cap {
        return text.to_string();
    }
    // Do not silently turn a truncated negative/conditional assertion positive.
    // Retain a window around the first negation plus the tail and flag omission.
    let chars: Vec<_> = text.chars().collect();
    let markers = [
        "不", "无", "未", "禁", "除非", "仅", "not ", "never", "unless", "except", "only",
    ];
    let lower = text.to_lowercase();
    let negative = markers
        .iter()
        .filter_map(|m| lower.find(m))
        .min()
        .map(|byte| lower[..byte].chars().count());
    let head: String = chars.iter().take(48).collect();
    let tail: String = chars.iter().skip(chars.len().saturating_sub(40)).collect();
    let middle = negative
        .map(|index| {
            chars
                .iter()
                .skip(index.saturating_sub(8))
                .take(48)
                .collect::<String>()
        })
        .unwrap_or_default();
    bounded_text(
        &format!("{head}…{middle}…{tail} [条件有省略，勿忽略限制]"),
        cap,
    )
}

fn format_fact_attrs(attrs: &Value) -> String {
    let Some(obj) = attrs.as_object() else {
        return String::new();
    };
    // Safety qualifiers go first: truncating a long dose/schedule list must not
    // silently remove a contraindication or an exception. Preserve structured
    // values as JSON rather than discarding nested dose/unit or frequency data.
    let keys = [
        "condition",
        "condition_text",
        "exception",
        "exception_text",
        "reason",
        "reason_text",
        "context",
        "instructions",
        "contraindications",
        "restrictions",
        "route",
        "注意事项",
        "条件",
        "例外",
        "原因",
        "dose",
        "dosage",
        "frequency",
        "time",
        "schedule",
        "timing",
        "amount",
        "currency",
        "unit",
        "event_at",
        "event_at_ms",
        "start_time",
        "end_time",
        "valid_from",
        "valid_to",
        "duration",
        "grade",
        "class",
        "剂量",
        "频次",
        "频率",
        "时间",
        "金额",
        "币种",
        "单位",
        "周期",
    ];
    let mut parts = Vec::new();
    if obj.get("assertion_basis").and_then(Value::as_str) == Some("context_inference") {
        parts.push("上下文推断".to_string());
    }
    for key in keys {
        if let Some(v) = obj.get(key).filter(|v| !v.is_null()) {
            let value = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if !value.trim().is_empty() {
                parts.push(format!("{key}:{}", bounded_qualifier(&value, 160)));
            }
        }
    }
    // Open extractor attributes are not a closed ontology. Unknown contextual
    // properties are retained as well, excluding only internal alias metadata.
    for (key, value) in obj {
        if keys.contains(&key.as_str())
            || matches!(
                key.as_str(),
                "alias_kind" | "aliases" | "assertion_basis" | "inference_note" | "evidence_spans"
            )
            || key.starts_with('_')
            || value.is_null()
        {
            continue;
        }
        let text = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if !text.trim().is_empty() {
            parts.push(format!(
                "{}:{}",
                bounded_text(key, 48),
                bounded_qualifier(&text, 160)
            ));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("（{}）", parts.join("，"))
    }
}
fn relation_label(r: &str) -> String {
    if super::ontology::alias_kind_for_relation(r).is_some() {
        return normalize_relation(r);
    }
    let zh = match RelationType::parse_known(r) {
        Some(RelationType::FamilyMember) => "家庭成员",
        Some(RelationType::Kinship) => "亲属",
        Some(RelationType::HasAttribute) => "属性/称呼",
        Some(RelationType::InvolvesPerson) => "涉及",
        Some(RelationType::RelatedTo) => "相关",
        Some(RelationType::AllergicTo) => "过敏",
        Some(RelationType::HasCondition) => "患有",
        Some(RelationType::TakesMedication) => "服用",
        Some(RelationType::Visits) => "就诊于",
        Some(RelationType::ReceivesProcedure) => "接受处置",
        Some(RelationType::CaresFor) => "照护",
        Some(RelationType::EmergencyContact) => "紧急联系",
        Some(RelationType::Owns) => "拥有",
        Some(RelationType::Owes) => "欠款",
        Some(RelationType::PaysFor) => "支付",
        Some(RelationType::AttendsSchool) => "就读于",
        Some(RelationType::EnrolledIn) => "参加课程",
        Some(RelationType::TaughtBy) => "授课教师",
        Some(other) => other.as_zh_str(),
        None => return normalize_relation(r),
    };
    zh.to_string()
}

/// Attach model-approved aliases; only empty values are structurally invalid.
pub fn attach_aliases(
    conn: &Connection,
    namespace: &str,
    entity_id: &str,
    names: &[String],
    scope_member_id: Option<&str>,
) -> Result<usize> {
    let mut n = 0usize;
    for raw in names {
        let a = raw.trim();
        if a.is_empty() {
            continue;
        }
        upsert_alias(conn, namespace, entity_id, a, scope_member_id)?;
        n += 1;
    }
    Ok(n)
}

/// Merge loser into keeper: aliases move, facts retarget, loser status=merged.
pub fn merge_entities(
    conn: &Connection,
    namespace: &str,
    keeper_id: &str,
    loser_id: &str,
) -> Result<()> {
    if keeper_id == loser_id {
        return Ok(());
    }
    let Some(loser) = get_entity(conn, namespace, loser_id)? else {
        return Ok(());
    };
    let Some(keeper) = get_entity(conn, namespace, keeper_id)? else {
        return Err(MemoraError::NotFound(keeper_id.into()));
    };
    // A name collision is not evidence that two owners share a private object.
    // Ownerless historic nodes may be adopted by their unambiguous owner, but
    // two explicit, different owners must never be collapsed by this raw API.
    if keeper.entity_type != loser.entity_type
        || matches!((&keeper.owner_member_id, &loser.owner_member_id),
            (Some(a), Some(b)) if a != b)
    {
        return Err(MemoraError::InvalidState(
            "cannot merge incompatible entity types or owners".into(),
        ));
    }
    // Demote loser's canonical + aliases onto keeper.
    let mut names = loser.aliases.clone();
    names.push(loser.canonical_name.clone());
    attach_aliases(
        conn,
        namespace,
        keeper_id,
        &names,
        keeper.owner_member_id.as_deref(),
    )?;
    // Retarget facts.
    conn.execute(
        r#"UPDATE kg_facts SET subject_id = ?3, updated_at_ms = ?4
           WHERE namespace = ?1 AND subject_id = ?2"#,
        params![namespace, loser_id, keeper_id, now_ms()],
    )?;
    conn.execute(
        r#"UPDATE kg_facts SET object_entity_id = ?3, updated_at_ms = ?4
           WHERE namespace = ?1 AND object_entity_id = ?2"#,
        params![namespace, loser_id, keeper_id, now_ms()],
    )?;
    // Move aliases rows to keeper (ignore conflicts).
    let mut stmt = conn.prepare(
        "SELECT alias, scope_member_id FROM kg_aliases WHERE namespace = ?1 AND entity_id = ?2",
    )?;
    let rows: Vec<(String, String)> = stmt
        .query_map(params![namespace, loser_id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (alias, scope) in rows {
        let scope_opt = if scope.is_empty() {
            None
        } else {
            Some(scope.as_str())
        };
        upsert_alias(conn, namespace, keeper_id, &alias, scope_opt)?;
    }
    conn.execute(
        "DELETE FROM kg_aliases WHERE namespace = ?1 AND entity_id = ?2",
        params![namespace, loser_id],
    )?;
    // Mark loser merged.
    let mut attrs = loser.attrs.clone();
    if let Some(obj) = attrs.as_object_mut() {
        obj.insert("merged_into".into(), Value::String(keeper_id.to_string()));
    } else {
        attrs = serde_json::json!({ "merged_into": keeper_id });
    }
    conn.execute(
        r#"UPDATE kg_entities SET
             status = 'merged',
             attrs_json = ?3,
             updated_at_ms = ?4
           WHERE namespace = ?1 AND id = ?2"#,
        params![
            namespace,
            loser_id,
            serde_json::to_string(&attrs)?,
            now_ms()
        ],
    )?;
    let _ = keeper.canonical_name;
    Ok(())
}

/// Find entity by canonical or alias, optional type filter (None = any type).
pub fn find_entity_by_any_name(
    conn: &Connection,
    namespace: &str,
    name: &str,
    entity_type: Option<EntityType>,
    owner_member_id: Option<&str>,
) -> Result<Option<KgEntity>> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    if let Some(et) = entity_type {
        return find_entity_by_name_owned(conn, namespace, et, name, owner_member_id);
    }
    // Prefer a uniquely owner-compatible person; all other types use the same
    // owner checks, never an arbitrary canonical/alias LIMIT 1.
    if let Some(entity) = find_named_candidate(
        conn,
        namespace,
        name,
        Some(EntityType::Person),
        owner_member_id,
    )? {
        return Ok(Some(entity));
    }
    find_named_candidate(conn, namespace, name, None, owner_member_id)
}
