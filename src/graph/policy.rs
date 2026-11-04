//! Privacy and grant checks for knowledge-graph resources.

use rusqlite::Connection;

use crate::core::now_ms;
use crate::error::Result;

use super::ontology::PrivacyLevel;
use super::store;
use super::types::{FactObject, GrantInput, KgEntity, KgFact, KgGrant, RevokeGrantInput};

fn member_in_namespace(conn: &Connection, namespace: &str, member: &str) -> Result<bool> {
    Ok(crate::household::list_members(conn, namespace)?
        .iter()
        .any(|m| m.member_id == member))
}

fn owner_or_grant(
    conn: &Connection,
    namespace: &str,
    owner: Option<&str>,
    viewer: Option<&str>,
    fact: &KgFact,
) -> Result<bool> {
    let Some(owner) = owner.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(false);
    };
    // Match text privacy: an empty viewer means unlocked / single-chat listing.
    let Some(viewer) = viewer.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(true);
    };
    if !member_in_namespace(conn, namespace, owner)? {
        return Ok(false);
    }
    Ok(owner == viewer || store::has_view_grant(conn, namespace, owner, viewer, fact, now_ms())?)
}

pub fn fact_visible(
    conn: &Connection,
    namespace: &str,
    fact: &KgFact,
    viewer_member_id: Option<&str>,
) -> Result<bool> {
    // Source authorization includes graph grants; it must not reject a grantee
    // before the fact-level grant check can run.
    if !source_visible(conn, namespace, fact, viewer_member_id)? {
        return Ok(false);
    }
    if fact.privacy == PrivacyLevel::OwnerPrivate
        && !owner_or_grant(
            conn,
            namespace,
            fact.owner_member_id.as_deref(),
            viewer_member_id,
            fact,
        )?
    {
        return Ok(false);
    }
    // A shared edge is not permission to disclose a private endpoint's name or
    // attributes (including through display_fact / related entity collection).
    let mut ids = vec![fact.subject_id.as_str()];
    if let FactObject::Entity { entity_id } = &fact.object {
        ids.push(entity_id);
    }
    for id in ids {
        if let Some(entity) = store::get_entity(conn, namespace, id)? {
            if !entity_visible(
                conn,
                namespace,
                &entity,
                viewer_member_id,
                std::slice::from_ref(fact),
            )? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub fn entity_visible(
    conn: &Connection,
    namespace: &str,
    entity: &KgEntity,
    viewer_member_id: Option<&str>,
    supporting_facts: &[KgFact],
) -> Result<bool> {
    if entity.status == "merged"
        || (entity.entity_type == super::ontology::EntityType::Person
            && entity
                .owner_member_id
                .as_deref()
                .is_some_and(|id| crate::household::is_shared_member_id(id) || id == "__public__"))
    {
        // Reserved attribution scopes are not people, including legacy roster hubs.
        // This is an identity-role check, not a ban on real names containing 公共.
        return Ok(false);
    }
    if entity.privacy == PrivacyLevel::FamilyShared {
        return Ok(true);
    }
    let Some(owner) = entity.owner_member_id.as_deref() else {
        return Ok(false);
    };
    if !member_in_namespace(conn, namespace, owner)? {
        return Ok(false);
    }
    if viewer_member_id.map(str::trim) == Some(owner) {
        return Ok(true);
    }
    for fact in supporting_facts {
        let related = fact.subject_id == entity.id
            || matches!(&fact.object, FactObject::Entity { entity_id } if entity_id == &entity.id);
        if related && owner_or_grant(conn, namespace, Some(owner), viewer_member_id, fact)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A graph-only assertion or independently authorized live memory can support
/// a fact. Missing, foreign-namespace, forgotten and superseded sources cannot.
pub fn source_visible(
    conn: &Connection,
    namespace: &str,
    fact: &KgFact,
    viewer_member_id: Option<&str>,
) -> Result<bool> {
    if store::has_graph_only_source(conn, namespace, &fact.id.to_string())? {
        return Ok(true);
    }
    Ok(!visible_source_memory_ids(conn, namespace, fact, viewer_member_id)?.is_empty())
}

/// Independently visible memory sources, also used by hybrid source filters.
pub fn visible_source_memory_ids(
    conn: &Connection,
    namespace: &str,
    fact: &KgFact,
    viewer_member_id: Option<&str>,
) -> Result<Vec<uuid::Uuid>> {
    let roster = crate::household::list_members(conn, namespace)?;
    let mut visible = Vec::new();
    for id in store::list_fact_source_memory_ids(conn, namespace, &fact.id.to_string())? {
        let Some(memory) = crate::store::get_memory(conn, id)? else {
            continue;
        };
        if memory.namespace != namespace
            || matches!(
                memory.state,
                crate::types::MemoryState::Forgotten | crate::types::MemoryState::Superseded
            )
        {
            continue;
        }
        let projected: std::collections::BTreeSet<_> = memory
            .subject_ids
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        let metadata_subjects = crate::household::subject_id_list(&memory.metadata);
        let current: std::collections::BTreeSet<_> = metadata_subjects
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        // Fail closed during metadata reprojection: do not use an old graph
        // owner or stale subject index to manufacture a grantor.
        if !projected.is_empty() && projected != current {
            continue;
        }
        let subjects = if current.is_empty() {
            &projected
        } else {
            &current
        };
        if subjects
            .iter()
            .any(|s| !roster.iter().any(|m| m.member_id == *s))
        {
            continue;
        }
        let memory_private = crate::household::is_privacy(&memory.metadata);
        // Private facts/sources must be about a subject on the memory. Shared
        // family facts may be spoken by a different member than the curated
        // subject list (speaker ≠ owner); do not drop their provenance.
        if fact.privacy == PrivacyLevel::OwnerPrivate || memory_private {
            let Some(owner) = fact.owner_member_id.as_deref() else {
                continue;
            };
            if subjects.is_empty() || !subjects.contains(owner) {
                continue;
            }
        }
        if !memory_private {
            visible.push(id);
            continue;
        }
        let Some(owner) = fact.owner_member_id.as_deref() else {
            continue;
        };
        if owner_or_grant(conn, namespace, Some(owner), viewer_member_id, fact)? {
            visible.push(id);
        }
    }
    Ok(visible)
}

/// Select provenance and evidence together. A visible source never authorizes
/// the representative row's text overwritten by a different, hidden source.
pub fn redact_source_evidence(
    conn: &Connection,
    namespace: &str,
    fact: &mut KgFact,
    viewer_member_id: Option<&str>,
) -> Result<()> {
    let fact_id = fact.id.to_string();
    let visible = visible_source_memory_ids(conn, namespace, fact, viewer_member_id)?;
    let source = fact
        .source_memory_id
        .filter(|id| visible.contains(id))
        .or_else(|| visible.first().copied());
    if let Some(source) = source {
        let (text, summary) = store::fact_source_evidence(conn, namespace, &fact_id, Some(source))?;
        fact.source_memory_id = Some(source);
        fact.evidence_text = text;
        fact.evidence_summary = summary;
    } else if store::has_graph_only_source(conn, namespace, &fact_id)? {
        let (text, summary) = store::fact_source_evidence(conn, namespace, &fact_id, None)?;
        fact.source_memory_id = None;
        fact.evidence_text = text;
        fact.evidence_summary = summary;
    } else {
        fact.source_memory_id = None;
        fact.evidence_text = None;
        fact.evidence_summary = None;
    }
    Ok(())
}

pub fn create_grant(conn: &Connection, namespace: &str, input: &GrantInput) -> Result<KgGrant> {
    store::insert_grant(conn, namespace, input)
}

pub fn revoke_grant(conn: &Connection, namespace: &str, input: &RevokeGrantInput) -> Result<usize> {
    store::revoke_grants(conn, namespace, input)
}

pub fn list_grants(
    conn: &Connection,
    namespace: &str,
    member_id: Option<&str>,
    as_grantor: bool,
) -> Result<Vec<KgGrant>> {
    store::list_grants(conn, namespace, member_id, as_grantor)
}
