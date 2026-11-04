//! Knowledge-graph ingest and query orchestration.

use rusqlite::Connection;
use uuid::Uuid;

use crate::error::Result;
use crate::household::HouseholdMember;

use super::ontology::{EntityType, FactPolarity, FactStatus, RelationType};
use super::policy;
use super::store;
use super::types::{
    ExtractedEntity, ExtractedFact, ExtractedTriple, FactObject, GrantInput, GraphOverview,
    GraphOverviewQuery, IngestReport, KgEntity, KgFact, KgGrant, PersonGraph, PersonGraphQuery,
    RecalledGraphFact, RevokeGrantInput,
};

/// Align roster person nodes only. Rule-based text extract is disabled;
/// callers must run an LLM `GraphExtractor` then [`ingest_graph_batch`].
pub fn ingest_text(
    conn: &Connection,
    namespace: &str,
    _text: &str,
    roster: &[HouseholdMember],
    _speaker_member_id: Option<&str>,
    _source_memory_id: Option<Uuid>,
) -> Result<IngestReport> {
    for m in roster.iter().filter(|m| {
        !crate::household::is_shared_member_id(&m.member_id) && m.member_id != "__public__"
    }) {
        store::ensure_person_entity(conn, namespace, &m.member_id, &m.display_name, &m.aliases)?;
    }
    Ok(IngestReport {
        extractor_id: Some("empty-kg-v1".into()),
        ..Default::default()
    })
}

pub fn ingest_extracted(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    extracted: &[ExtractedFact],
    source_memory_id: Option<Uuid>,
    extractor_id: &str,
    pending_count: usize,
) -> Result<IngestReport> {
    for m in roster.iter().filter(|m| {
        !crate::household::is_shared_member_id(&m.member_id) && m.member_id != "__public__"
    }) {
        store::ensure_person_entity(conn, namespace, &m.member_id, &m.display_name, &m.aliases)?;
    }
    let mut report = apply_extracted(conn, namespace, roster, extracted, source_memory_id)?;
    report.extractor_id = Some(extractor_id.to_string());
    report.pending = report.pending.saturating_add(pending_count);
    Ok(report)
}

pub fn ingest_graph_batch(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    entities: &[ExtractedEntity],
    triples: &[ExtractedTriple],
    source_memory_id: Option<Uuid>,
) -> Result<IngestReport> {
    super::ingest::ingest_graph_batch(conn, namespace, roster, entities, triples, source_memory_id)
}

pub fn apply_extracted(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    extracted: &[ExtractedFact],
    source_memory_id: Option<Uuid>,
) -> Result<IngestReport> {
    let (entities, triples) = super::ingest::facts_to_graph(extracted);
    super::ingest::ingest_graph_batch(
        conn,
        namespace,
        roster,
        &entities,
        &triples,
        source_memory_id,
    )
}

pub fn recall_graph(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    viewer_member_id: Option<&str>,
    limit: usize,
) -> Result<Vec<RecalledGraphFact>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut out =
        collect_recall_graph(conn, namespace, text, member_id, domains, viewer_member_id)?;
    out.truncate(limit);
    Ok(out)
}

/// Collect all ranked visible candidates. Hybrid recall applies memory filters
/// and its final top_k afterwards; no fixed over-fetch factor can ensure that.
pub fn collect_recall_graph(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    viewer_member_id: Option<&str>,
) -> Result<Vec<RecalledGraphFact>> {
    collect_recall_graph_with_names(
        conn,
        namespace,
        text,
        member_id,
        domains,
        viewer_member_id,
        &[],
    )
}

/// Same as [`collect_recall_graph`], but person anchors resolve through the
/// namespace roster (display name + registered aliases). Roster data carries
/// provenance — unlike raw kg_aliases rows — so nicknames such as 小名 alias
/// facts or spoken forms ("航航") can anchor the right person and take the
/// high-confidence `person_direct` path instead of degrading into a global
/// topic search.
pub fn collect_recall_graph_with_names(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    viewer_member_id: Option<&str>,
    roster: &[HouseholdMember],
) -> Result<Vec<RecalledGraphFact>> {
    let facts = store::collect_search_facts_with_person_names(
        conn,
        namespace,
        text,
        member_id,
        domains,
        |fact| policy::fact_visible(conn, namespace, fact, viewer_member_id),
        |entity| Ok(store::person_names_for_entity(entity, roster)),
    )?;
    let mut out = Vec::new();
    for mut fact in facts {
        policy::redact_source_evidence(conn, namespace, &mut fact, viewer_member_id)?;
        let (subject_name, object_label, display_text) =
            store::display_fact(conn, namespace, &fact)?;
        let score = fact.confidence;
        out.push(RecalledGraphFact {
            fact,
            subject_name,
            object_label,
            display_text,
            score,
        });
    }
    Ok(out)
}

/// Collect all ranked visible candidates (legacy text-only ranking).
pub fn collect_recall_graph_text_only(
    conn: &Connection,
    namespace: &str,
    text: &str,
    member_id: Option<&str>,
    domains: &[String],
    viewer_member_id: Option<&str>,
) -> Result<Vec<RecalledGraphFact>> {
    let facts = store::collect_search_facts(conn, namespace, text, member_id, domains, |fact| {
        policy::fact_visible(conn, namespace, fact, viewer_member_id)
    })?;
    let mut out = Vec::new();
    for mut fact in facts {
        policy::redact_source_evidence(conn, namespace, &mut fact, viewer_member_id)?;
        let (subject_name, object_label, display_text) =
            store::display_fact(conn, namespace, &fact)?;
        let score = fact.confidence;
        out.push(RecalledGraphFact {
            fact,
            subject_name,
            object_label,
            display_text,
            score,
        });
    }
    Ok(out)
}

pub fn query_person_graph(
    conn: &Connection,
    namespace: &str,
    query: &PersonGraphQuery,
) -> Result<PersonGraph> {
    let member_id = query.member_id.trim();
    let person_id = if member_id.starts_with("person:") {
        member_id.to_string()
    } else {
        store::person_entity_id(member_id)
    };
    let person = store::get_entity(conn, namespace, &person_id)?;
    let owner = person
        .as_ref()
        .and_then(|p| p.owner_member_id.clone())
        .or_else(|| {
            if member_id.starts_with("person:") {
                None
            } else {
                Some(member_id.to_string())
            }
        });
    let facts = store::list_facts_for_person(
        conn,
        namespace,
        &person_id,
        owner.as_deref(),
        query.include_ended,
        &query.domains,
    )?;
    let viewer = query.viewer_member_id.as_deref();
    let mut visible = Vec::new();
    for mut f in facts {
        if policy::fact_visible(conn, namespace, &f, viewer)? {
            policy::redact_source_evidence(conn, namespace, &mut f, viewer)?;
            visible.push(f);
        }
    }
    let alias_facts = store::collect_facts_overview(conn, namespace, true, &[])?;
    let mut entities = store::collect_related_entities(conn, namespace, &visible)?;
    sanitize_entities(
        conn,
        namespace,
        &mut entities,
        viewer,
        &visible,
        &alias_facts,
    )?;
    let person = match person {
        Some(mut person) if policy::entity_visible(conn, namespace, &person, viewer, &visible)? => {
            sanitize_entity(conn, namespace, &mut person, viewer, &alias_facts)?;
            Some(person)
        }
        _ => None,
    };
    Ok(PersonGraph {
        member_id: member_id.to_string(),
        person,
        entities,
        facts: visible,
    })
}

pub fn query_graph_overview(
    conn: &Connection,
    namespace: &str,
    query: &GraphOverviewQuery,
) -> Result<GraphOverview> {
    let limit = query.limit.unwrap_or(400).min(2000);
    if limit == 0 {
        return Ok(GraphOverview {
            entities: Vec::new(),
            facts: Vec::new(),
            truncated: false,
        });
    }
    let facts =
        store::collect_facts_overview(conn, namespace, query.include_ended, &query.domains)?;
    let viewer = query.viewer_member_id.as_deref();
    let mut visible = Vec::new();
    for mut f in facts {
        if policy::fact_visible(conn, namespace, &f, viewer)? {
            policy::redact_source_evidence(conn, namespace, &mut f, viewer)?;
            visible.push(f);
        }
    }
    let truncated = visible.len() > limit;
    visible.truncate(limit);
    // Only entities reachable from visible facts — do not dump private disease/drug nodes.
    let entities = store::collect_related_entities(conn, namespace, &visible)?;
    // Optionally include person hubs with no facts yet (roster anchors), still non-sensitive.
    let mut entities = entities;
    let mut seen: std::collections::BTreeSet<String> =
        entities.iter().map(|e| e.id.clone()).collect();
    for e in store::list_safe_roster_people(conn, namespace)? {
        if seen.insert(e.id.clone()) {
            entities.push(e);
        }
    }
    let alias_facts = store::collect_facts_overview(conn, namespace, true, &[])?;
    sanitize_entities(
        conn,
        namespace,
        &mut entities,
        viewer,
        &visible,
        &alias_facts,
    )?;
    Ok(GraphOverview {
        entities,
        facts: visible,
        truncated,
    })
}

/// Alias assertions describe the existing hub; they never require a nickname
/// node. Raw kg_aliases/attrs have no reliable legacy provenance and are not an
/// authorization boundary.
fn alias_names(fact: &KgFact) -> Vec<String> {
    if super::ontology::alias_kind_for_relation(&fact.relation).is_none()
        && (RelationType::parse_known(&fact.relation) != Some(RelationType::HasAttribute)
            || !fact
                .attrs
                .get("alias_kind")
                .and_then(|v| v.as_str())
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "alias" | "nickname" | "address" | "name" | "别名" | "昵称" | "小名"
                    )
                }))
    {
        return Vec::new();
    }
    let mut names = Vec::new();
    if let FactObject::Value { value, .. } = &fact.object {
        if let Some(name) = value.as_str() {
            names.push(name.to_string());
        }
    }
    if let Some(aliases) = fact.attrs.get("aliases").and_then(|v| v.as_array()) {
        names.extend(
            aliases
                .iter()
                .filter_map(|v| v.as_str())
                .map(str::to_string),
        );
    }
    names
}

fn sanitize_entities(
    conn: &Connection,
    namespace: &str,
    entities: &mut Vec<KgEntity>,
    viewer: Option<&str>,
    visible_facts: &[KgFact],
    alias_facts: &[KgFact],
) -> Result<()> {
    let mut safe = Vec::new();
    for mut entity in entities.drain(..) {
        if policy::entity_visible(conn, namespace, &entity, viewer, visible_facts)? {
            sanitize_entity(conn, namespace, &mut entity, viewer, alias_facts)?;
            safe.push(entity);
        }
    }
    *entities = safe;
    Ok(())
}

fn sanitize_entity(
    conn: &Connection,
    namespace: &str,
    entity: &mut KgEntity,
    viewer: Option<&str>,
    alias_facts: &[KgFact],
) -> Result<()> {
    entity.attrs = serde_json::json!({});
    entity.aliases.clear();
    if entity.entity_type == EntityType::Person {
        if let Some(member) = crate::household::list_members(conn, namespace)?
            .into_iter()
            .find(|m| store::person_entity_id(&m.member_id) == entity.id)
        {
            // A prior ingest may have overwritten the canonical with a private
            // nickname or polluted the roster aliases. Restore its roster name
            // and remove aliases with graph provenance before adding them back
            // under the current source/ACL checks (including forgotten sources).
            entity.canonical_name = member.display_name;
            let sourced: std::collections::BTreeSet<_> = alias_facts
                .iter()
                .filter(|f| f.subject_id == entity.id)
                .flat_map(alias_names)
                .collect();
            entity
                .aliases
                .extend(member.aliases.into_iter().filter(|a| !sourced.contains(a)));
        }
    }
    if !entity.canonical_name.trim().is_empty() {
        entity.aliases.push(entity.canonical_name.clone());
    }
    for fact in alias_facts.iter().filter(|f| f.subject_id == entity.id) {
        if fact.status != FactStatus::Active
            || fact.polarity != FactPolarity::Affirmative
            || !fact
                .condition_text
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
            || !policy::fact_visible(conn, namespace, fact, viewer)?
        {
            continue;
        }
        entity.aliases.extend(alias_names(fact));
    }
    entity.aliases.retain(|a| !a.trim().is_empty());
    entity.aliases.sort();
    entity.aliases.dedup();
    Ok(())
}

pub fn grant_access(conn: &Connection, namespace: &str, input: &GrantInput) -> Result<KgGrant> {
    policy::create_grant(conn, namespace, input)
}

pub fn revoke_access(
    conn: &Connection,
    namespace: &str,
    input: &RevokeGrantInput,
) -> Result<usize> {
    policy::revoke_grant(conn, namespace, input)
}

pub fn list_grants(
    conn: &Connection,
    namespace: &str,
    member_id: Option<&str>,
    as_grantor: bool,
) -> Result<Vec<KgGrant>> {
    policy::list_grants(conn, namespace, member_id, as_grantor)
}
