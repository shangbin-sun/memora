//! Entity + triple ingest with alias merge.

use std::collections::{HashMap, HashSet};

use rusqlite::Connection;
use uuid::Uuid;

use crate::core::now_ms;
use crate::error::Result;
use crate::household::HouseholdMember;

use super::ontology::{
    alias_kind_for_relation, default_privacy_for_relation, normalize_assertion_relation,
    normalize_relation, EntityType, FactPolarity, FactStatus, PrivacyLevel, RelationType,
};
use super::store;
use super::types::{
    ExtractedEntity, ExtractedFact, ExtractedTriple, FactObject, IngestReport, KgFact,
};

pub fn ingest_graph_batch(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    entities: &[ExtractedEntity],
    triples: &[ExtractedTriple],
    source_memory_id: Option<Uuid>,
) -> Result<IngestReport> {
    let mut report = IngestReport::default();
    let now = now_ms();
    let mut roster = roster.to_vec();
    let mut id_map: HashMap<String, String> = HashMap::new();
    let declared_ids: HashSet<&str> = entities
        .iter()
        .map(|entity| entity.temp_id.as_str())
        .collect();
    let source_private = if let Some(id) = source_memory_id {
        crate::store::get_memory(conn, id)?
            .is_some_and(|m| crate::household::is_privacy(&m.metadata))
    } else {
        false
    };

    for member in &roster {
        if shared_scope_member(&member.member_id) {
            continue;
        }
        store::ensure_person_entity(
            conn,
            namespace,
            &member.member_id,
            &member.display_name,
            &member.aliases,
        )?;
    }
    for ent in entities {
        // Owner-scoped instances are resolved only after the subject owner is
        // known. Precreating an explicitly foreign owner could mutate another
        // member's account even when the eventual triple is rejected.
        if RelationType::owner_scoped_entity(ent.entity_type) {
            continue;
        }
        if triples.iter().any(|t| {
            entity_ref_matches(ent, &t.object)
                && is_alias_triple(t, &normalize_relation(&t.relation))
        }) && !triples.iter().any(|t| {
            entity_ref_matches(ent, &t.subject)
                || (entity_ref_matches(ent, &t.object)
                    && !is_alias_triple(t, &normalize_relation(&t.relation)))
        }) {
            continue;
        }
        let name = ent.name.trim();
        if name.is_empty() {
            continue;
        }
        let aliases: Vec<String> = ent
            .aliases
            .iter()
            .map(|a| a.trim().to_string())
            .filter(|a| !a.is_empty())
            .collect();

        let private_aliases = source_private
            || triples.iter().any(|t| {
                (t.subject == ent.temp_id
                    || t.subject == ent.name
                    || t.object == ent.temp_id
                    || ent.member_id.as_deref() == Some(t.subject.as_str()))
                    && t.privacy.unwrap_or_else(|| {
                        default_privacy_for_relation(&normalize_relation(&t.relation))
                    }) == PrivacyLevel::OwnerPrivate
            });
        let explicit_alias = triples.iter().any(|t| {
            is_alias_triple(t, &normalize_relation(&t.relation))
                && (t.subject == ent.temp_id
                    || t.subject == ent.name
                    || ent.member_id.as_deref() == Some(t.subject.as_str()))
        });
        let persist_aliases = source_memory_id.is_none() && !private_aliases && !explicit_alias;
        let Some(eid) = resolve_entity_candidate(
            conn,
            namespace,
            &mut roster,
            ent,
            source_memory_id.is_some() || private_aliases,
            &mut report,
        )?
        else {
            continue;
        };
        if persist_aliases {
            attach_candidate_aliases(
                conn,
                namespace,
                &eid,
                ent.entity_type,
                &aliases,
                ent.member_id.as_deref(),
            )?;
        }
        if let Some(entity) = store::get_entity(conn, namespace, &eid)? {
            let mut source_aliases = aliases.clone();
            if ent.entity_type == EntityType::Person && name != entity.canonical_name {
                source_aliases.push(name.to_string());
            }
            for alias in &source_aliases {
                // An explicit alias triple owns its lifecycle/evidence. Do not
                // synthesize an active copy of a paused/ended/negative alias.
                if triples.iter().any(|t| {
                    is_alias_triple(t, &normalize_relation(&t.relation))
                        && (t.subject == ent.temp_id
                            || t.subject == ent.name
                            || ent.member_id.as_deref() == Some(t.subject.as_str()))
                        && (entities
                            .iter()
                            .find(|e| e.temp_id == t.object)
                            .map(|e| e.name.as_str())
                            .unwrap_or(t.object.as_str())
                            == alias.as_str()
                            || t.attrs
                                .get("aliases")
                                .and_then(|v| v.as_array())
                                .is_some_and(|names| {
                                    names.iter().any(|v| v.as_str() == Some(alias.as_str()))
                                }))
                }) {
                    continue;
                }
                write_alias_fact(
                    conn,
                    namespace,
                    &eid,
                    alias,
                    entity.owner_member_id.as_deref(),
                    if private_aliases {
                        PrivacyLevel::OwnerPrivate
                    } else {
                        PrivacyLevel::FamilyShared
                    },
                    source_memory_id,
                    &mut report,
                )?;
            }
        }
        id_map.insert(ent.temp_id.clone(), eid.clone());
        // A bare name is not a unique reference for owner-scoped objects.
        if !RelationType::owner_scoped_entity(ent.entity_type) {
            if !declared_ids.contains(name) {
                id_map.insert(name.to_string(), eid.clone());
            }
            for a in &aliases {
                if !declared_ids.contains(a.as_str()) {
                    id_map.insert(a.clone(), eid.clone());
                }
            }
        }
        if ent.entity_type == EntityType::Person {
            if let Some(mid) = &ent.member_id {
                if !declared_ids.contains(mid.as_str()) {
                    id_map.insert(mid.clone(), eid.clone());
                }
            }
        }
        report.entities_upserted += 1;
    }

    for t in triples {
        let rel = normalize_assertion_relation(&t.relation, &t.attrs);
        let restricted_identity = source_memory_id.is_some()
            || source_private
            || t.privacy
                .unwrap_or_else(|| default_privacy_for_relation(&rel))
                == PrivacyLevel::OwnerPrivate;
        // An explicitly owned business subject may introduce its own scope;
        // this hint is never allowed to override a person's object owner below.
        let subject_owner = entities
            .iter()
            .find(|e| e.temp_id == t.subject && RelationType::owner_scoped_entity(e.entity_type))
            .and_then(|e| e.member_id.as_deref());
        let subj = resolve_batch_ref(
            conn,
            namespace,
            &roster,
            &id_map,
            entities,
            &t.subject,
            EntityType::Person,
            subject_owner,
            source_memory_id,
            restricted_identity,
            &mut report,
        )?;
        let Some(sid) = subj else {
            report.pending += 1;
            report
                .messages
                .push(format!("pending unresolved subject `{}`", t.subject));
            continue;
        };
        let owner_mid = store::get_entity(conn, namespace, &sid)?.and_then(|e| e.owner_member_id);
        let object = if is_alias_triple(t, &rel) {
            let alias_entity = unique_alias_value(entities, &t.object);
            let label = match alias_entity {
                Some(entity) if !entity.name.trim().is_empty() => entity.name.trim(),
                None if entities.is_empty() => t.object.trim(),
                _ => {
                    report.pending += 1;
                    report
                        .messages
                        .push("pending undefined alias value reference".into());
                    continue;
                }
            };
            let mut names = vec![label.to_string()];
            if let Some(arr) = t.attrs.get("aliases").and_then(|v| v.as_array()) {
                names.extend(arr.iter().filter_map(|v| v.as_str()).map(str::to_string));
            }
            if source_memory_id.is_none()
                && !source_private
                && t.polarity == FactPolarity::Affirmative
                && t.status == FactStatus::Active
                && t.condition_text.as_deref().unwrap_or("").trim().is_empty()
                && t.privacy
                    .unwrap_or_else(|| default_privacy_for_relation(&rel))
                    == PrivacyLevel::FamilyShared
            {
                if let Some(subject) = store::get_entity(conn, namespace, &sid)? {
                    attach_candidate_aliases(
                        conn,
                        namespace,
                        &sid,
                        subject.entity_type,
                        &names,
                        owner_mid.as_deref(),
                    )?;
                }
            }
            // Keep an auditable, source-linked assertion even when the alias
            // has been folded into the subject rather than a separate node.
            FactObject::text(label)
        } else {
            let obj_type = t.object_type.unwrap_or_else(|| guess_object_type(&rel));
            let obj = resolve_batch_ref(
                conn,
                namespace,
                &roster,
                &id_map,
                entities,
                &t.object,
                obj_type,
                owner_mid.as_deref(),
                source_memory_id,
                restricted_identity,
                &mut report,
            )?;
            let Some(oid) = obj else {
                report.pending += 1;
                report.messages.push(format!(
                    "pending unresolved endpoints for {} -[{}]-> {}",
                    t.subject, rel, t.object
                ));
                continue;
            };
            FactObject::entity(oid)
        };

        let privacy = t
            .privacy
            .unwrap_or_else(|| default_privacy_for_relation(&rel));
        let mut domains = t.domains.clone();
        if domains.is_empty() {
            if let Some(known) = RelationType::parse_known(&rel) {
                domains = known
                    .default_domains()
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();
            }
        }
        let fact = KgFact {
            id: Uuid::now_v7(),
            subject_id: sid,
            relation: rel,
            object,
            owner_member_id: owner_mid,
            domains,
            polarity: t.polarity,
            status: t.status,
            valid_from_ms: Some(now),
            valid_to_ms: None,
            condition_text: t.condition_text.clone(),
            exception_text: None,
            reason_text: None,
            evidence_text: Some(t.evidence_text.clone()),
            evidence_summary: Some(t.evidence_text.chars().take(80).collect()),
            source_memory_id,
            confidence: t.confidence,
            privacy,
            revision: 1,
            created_at_ms: now,
            updated_at_ms: now,
            attrs: t.attrs.clone(),
        };
        let superseded = reconcile_prior_facts(conn, namespace, &fact)?;
        report.facts_confirmed += superseded;
        store::write_fact(conn, namespace, &fact)?;
        report.facts_written += 1;
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn write_alias_fact(
    conn: &Connection,
    namespace: &str,
    entity_id: &str,
    alias: &str,
    owner: Option<&str>,
    privacy: PrivacyLevel,
    source_memory_id: Option<Uuid>,
    report: &mut IngestReport,
) -> Result<()> {
    let alias = alias.trim();
    if alias.is_empty() {
        return Ok(());
    }
    let now = now_ms();
    let fact = KgFact {
        id: Uuid::now_v7(),
        subject_id: entity_id.to_string(),
        relation: "别名".to_string(),
        object: FactObject::text(alias),
        owner_member_id: owner.map(str::to_string),
        domains: vec!["household".into()],
        polarity: FactPolarity::Affirmative,
        status: FactStatus::Active,
        valid_from_ms: Some(now),
        valid_to_ms: None,
        condition_text: None,
        exception_text: None,
        reason_text: None,
        evidence_text: Some(alias.to_string()),
        evidence_summary: Some(alias.to_string()),
        source_memory_id,
        confidence: 0.8,
        privacy,
        revision: 1,
        created_at_ms: now,
        updated_at_ms: now,
        attrs: serde_json::json!({"alias_kind":"alias"}),
    };
    store::write_fact(conn, namespace, &fact)?;
    report.facts_written += 1;
    Ok(())
}

fn reconcile_prior_facts(conn: &Connection, namespace: &str, new_fact: &KgFact) -> Result<usize> {
    let mut ids = Vec::new();
    let unique = RelationType::parse_known(&new_fact.relation)
        .is_some_and(|r| r.unique_object_per_subject());
    for old in store::list_active_by_subject_relation(
        conn,
        namespace,
        &new_fact.subject_id,
        &new_fact.relation,
    )? {
        if old.condition_text.as_deref().unwrap_or("")
            != new_fact.condition_text.as_deref().unwrap_or("")
        {
            continue;
        }
        let same_object = match (&old.object, &new_fact.object) {
            (FactObject::Entity { entity_id: a }, FactObject::Entity { entity_id: b }) => a == b,
            (
                FactObject::Value {
                    value_type: at,
                    value: av,
                },
                FactObject::Value {
                    value_type: bt,
                    value: bv,
                },
            ) => at == bt && av == bv,
            _ => false,
        };
        let close = if same_object {
            match new_fact.status {
                // Current assertions replace an opposite current assertion or
                // resume a paused fact. An affirmative also realizes its plan.
                FactStatus::Active => {
                    (old.status != FactStatus::Planned
                        && (old.polarity != new_fact.polarity || old.status == FactStatus::Paused))
                        || (old.status == FactStatus::Planned && old.polarity == new_fact.polarity)
                }
                // Future intentions do not negate present facts.
                FactStatus::Planned => {
                    old.status == FactStatus::Planned && old.polarity != new_fact.polarity
                }
                FactStatus::Paused => {
                    old.polarity == new_fact.polarity
                        && matches!(old.status, FactStatus::Active | FactStatus::Planned)
                }
                FactStatus::Ended => old.polarity == new_fact.polarity,
                _ => false,
            }
        } else {
            // Uniqueness applies only to current affirmative facts under the
            // same condition, never to negatives or future plans.
            unique
                && new_fact.polarity == FactPolarity::Affirmative
                && new_fact.status == FactStatus::Active
                && old.polarity == FactPolarity::Affirmative
                && matches!(old.status, FactStatus::Active | FactStatus::Paused)
        };
        if close {
            ids.push(old.id.to_string());
        }
    }
    // End the prior assertion and write the new state separately. Mutating an
    // active assertion to paused would leave its active semantic key reusable
    // by write_fact, hiding later resumes and losing the state transition.
    store::supersede_facts(conn, namespace, &ids, FactStatus::Ended, now_ms())
}

fn unique_alias_value<'a>(
    entities: &'a [ExtractedEntity],
    reference: &str,
) -> Option<&'a ExtractedEntity> {
    for priority in 0..3 {
        let matches: Vec<_> = entities
            .iter()
            .filter(|entity| match priority {
                0 => entity.temp_id == reference,
                1 => entity.name == reference || entity.member_id.as_deref() == Some(reference),
                _ => entity.aliases.iter().any(|alias| alias == reference),
            })
            .collect();
        match matches.as_slice() {
            [] => continue,
            [entity] => return Some(*entity),
            _ => return None,
        }
    }
    None
}

fn entity_ref_matches(entity: &ExtractedEntity, reference: &str) -> bool {
    entity.temp_id == reference
        || entity.name == reference
        || entity.member_id.as_deref() == Some(reference)
        || entity.aliases.iter().any(|alias| alias == reference)
}

fn is_alias_triple(triple: &ExtractedTriple, relation: &str) -> bool {
    alias_kind_for_relation(relation).is_some()
        || RelationType::parse_known(relation) == Some(RelationType::HasAttribute)
            && triple
                .attrs
                .get("alias_kind")
                .and_then(|v| v.as_str())
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "alias" | "nickname" | "address" | "name" | "别名" | "昵称" | "小名"
                    )
                })
}
pub fn facts_to_graph(facts: &[ExtractedFact]) -> (Vec<ExtractedEntity>, Vec<ExtractedTriple>) {
    let mut entities = Vec::new();
    let mut triples = Vec::new();
    let mut seen = HashSet::new();
    for f in facts {
        if f.subject_name == "家庭" {
            if seen.insert("家庭".to_string()) {
                entities.push(ExtractedEntity {
                    temp_id: "family".into(),
                    name: "家庭".into(),
                    entity_type: EntityType::Family,
                    aliases: vec![],
                    member_id: None,
                });
            }
        } else if !f.subject_name.trim().is_empty() && seen.insert(f.subject_name.clone()) {
            entities.push(ExtractedEntity {
                temp_id: format!("s:{}", f.subject_name),
                name: f.subject_name.clone(),
                entity_type: EntityType::Person,
                aliases: vec![],
                member_id: f.subject_member_id.clone(),
            });
        }
        let mut obj_aliases = Vec::new();
        if let Some(arr) = f.attrs.get("aliases").and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    obj_aliases.push(s.to_string());
                }
            }
        }
        if f.object_type == EntityType::Person {
            if !f.object_name.trim().is_empty() && seen.insert(f.object_name.clone()) {
                entities.push(ExtractedEntity {
                    temp_id: format!("o:{}", f.object_name),
                    name: f.object_name.clone(),
                    entity_type: EntityType::Person,
                    aliases: obj_aliases,
                    member_id: None,
                });
            }
        } else if seen.insert(format!("t:{}:{}", f.object_type.as_str(), f.object_name)) {
            entities.push(ExtractedEntity {
                temp_id: format!("o:{}", f.object_name),
                name: f.object_name.clone(),
                entity_type: f.object_type,
                aliases: vec![],
                member_id: None,
            });
        }
        let subj_ref = if f.subject_name == "家庭" {
            "家庭".into()
        } else {
            f.subject_member_id
                .clone()
                .unwrap_or_else(|| f.subject_name.clone())
        };
        triples.push(ExtractedTriple {
            subject: subj_ref,
            relation: f.relation.clone(),
            object: f.object_name.clone(),
            object_type: Some(f.object_type),
            polarity: f.polarity,
            status: f.status,
            domains: f.domains.clone(),
            privacy: Some(f.privacy),
            condition_text: f.condition_text.clone(),
            evidence_text: f.evidence_text.clone(),
            confidence: f.confidence,
            attrs: f.attrs.clone(),
        });
    }
    (entities, triples)
}

fn resolve_entity_candidate(
    conn: &Connection,
    namespace: &str,
    roster: &mut Vec<HouseholdMember>,
    ent: &ExtractedEntity,
    restricted_identity: bool,
    report: &mut IngestReport,
) -> Result<Option<String>> {
    if ent.entity_type == EntityType::Person
        && (ent.member_id.as_deref().is_some_and(shared_scope_member)
            || resolve_roster_id(roster, None, &ent.name)
                .as_deref()
                .is_some_and(shared_scope_member))
    {
        report
            .messages
            .push("pending shared scope is not a person entity".into());
        return Ok(None);
    }
    if ent.entity_type == EntityType::Family || ent.name == "家庭" {
        return Ok(Some(store::resolve_or_create_entity(
            conn,
            namespace,
            EntityType::Family,
            "家庭",
            None,
            &["household".into()],
            PrivacyLevel::FamilyShared,
        )?));
    }
    if ent.entity_type == EntityType::Person {
        let alias_members: HashSet<_> = ent
            .aliases
            .iter()
            .filter_map(|a| resolve_roster_id(roster, None, a))
            .collect();
        let alias_member = if alias_members.len() == 1 {
            alias_members.into_iter().next()
        } else {
            None
        };
        if let Some(mid) = ent
            .member_id
            .clone()
            .or_else(|| resolve_roster_id(roster, None, &ent.name))
            .or(alias_member)
        {
            if shared_scope_member(&mid) {
                return Ok(None);
            }
            if let Some(m) = roster.iter().find(|m| m.member_id == mid).cloned() {
                let label = m.display_name.as_str();
                return Ok(Some(store::ensure_person_entity(
                    conn, namespace, &mid, label, &m.aliases,
                )?));
            }
        }
    }
    if ent.entity_type == EntityType::Person {
        if let Some(existing) = store::find_entity_by_any_name(
            conn,
            namespace,
            &ent.name,
            Some(EntityType::Person),
            ent.member_id.as_deref(),
        )? {
            return Ok(Some(existing.id));
        }
        if restricted_identity {
            report.pending += 1;
            report.messages.push(format!(
                "pending sourced/private person without a roster identity `{}`",
                ent.name
            ));
            return Ok(None);
        }
        ensure_roster_person(conn, namespace, roster, &ent.name, &[])?;
        if let Some(mid) = resolve_roster_id(roster, None, &ent.name) {
            return Ok(Some(store::ensure_person_entity(
                conn,
                namespace,
                &mid,
                &ent.name,
                &[],
            )?));
        }
    }
    if let Some(existing) = store::find_entity_by_any_name(
        conn,
        namespace,
        &ent.name,
        Some(ent.entity_type),
        ent.member_id.as_deref(),
    )? {
        return Ok(Some(existing.id));
    }
    Ok(Some(store::resolve_or_create_entity(
        conn,
        namespace,
        ent.entity_type,
        &ent.name,
        ent.member_id.as_deref(),
        &[],
        PrivacyLevel::FamilyShared,
    )?))
}

fn attach_candidate_aliases(
    conn: &Connection,
    namespace: &str,
    entity_id: &str,
    entity_type: EntityType,
    aliases: &[String],
    owner: Option<&str>,
) -> Result<()> {
    // Find targets before attaching aliases, so the keeper cannot mask a loser.
    for alias in aliases {
        if let Some(other) =
            store::find_entity_by_any_name(conn, namespace, alias, Some(entity_type), owner)?
        {
            if other.id != entity_id
                && other.status != "merged"
                && (!RelationType::owner_scoped_entity(entity_type)
                    || other.owner_member_id.as_deref() == owner)
            {
                store::merge_entities(conn, namespace, entity_id, &other.id)?;
            }
        }
    }
    store::attach_aliases(conn, namespace, entity_id, aliases, owner)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_batch_ref(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    id_map: &HashMap<String, String>,
    entities: &[ExtractedEntity],
    refer: &str,
    hint_type: EntityType,
    owner: Option<&str>,
    source_memory_id: Option<Uuid>,
    restricted_identity: bool,
    report: &mut IngestReport,
) -> Result<Option<String>> {
    let refer = refer.trim();
    let mut candidate = None;
    for priority in 0..3 {
        let mut matches: Vec<_> = entities
            .iter()
            .filter(|entity| match priority {
                0 => entity.temp_id == refer,
                1 => {
                    entity.entity_type == hint_type
                        && (entity.name == refer || entity.member_id.as_deref() == Some(refer))
                }
                _ => {
                    entity.entity_type == hint_type
                        && entity.aliases.iter().any(|alias| alias == refer)
                }
            })
            .collect();
        // A direct name may be shared by owner-scoped instances. Narrow only
        // after exact temp-reference precedence, preserving conflict rejection.
        if priority > 0 && matches.len() > 1 {
            let scoped: Vec<_> = matches
                .iter()
                .copied()
                .filter(|entity| {
                    !RelationType::owner_scoped_entity(entity.entity_type)
                        || entity.member_id.is_none()
                        || entity.member_id.as_deref() == owner
                })
                .collect();
            if !scoped.is_empty() {
                matches = scoped;
            }
        }
        match matches.as_slice() {
            [] => continue,
            [entity] => {
                candidate = Some(*entity);
                break;
            }
            _ => {
                report
                    .messages
                    .push("pending ambiguous structured entity reference".into());
                return Ok(None);
            }
        }
    }
    let entity_type = candidate.map(|e| e.entity_type).unwrap_or(hint_type);
    if !RelationType::owner_scoped_entity(entity_type) {
        // A declared temp ID is a reference, never a fallback entity name. If
        // its candidate was rejected/skipped above, keep the edge unresolved.
        if let Some(candidate) = candidate {
            return Ok(id_map.get(&candidate.temp_id).cloned());
        }
        return resolve_ref(
            conn,
            namespace,
            roster,
            id_map,
            refer,
            entity_type,
            restricted_identity,
            entities.is_empty(),
            report,
        );
    }
    let owner = owner.map(str::trim).filter(|s| !s.is_empty());
    let candidate_owner = candidate
        .and_then(|e| e.member_id.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if candidate_owner.is_some() && candidate_owner != owner {
        report.messages.push(format!("pending owner conflict for `{refer}`: subject {owner:?}, candidate {candidate_owner:?}"));
        return Ok(None);
    }
    let Some(owner) = owner else {
        report.messages.push(format!("pending owner for `{refer}`"));
        return Ok(None);
    };
    if shared_scope_member(owner) || !roster.iter().any(|m| m.member_id == owner) {
        report
            .messages
            .push(format!("pending non-roster owner for `{refer}`"));
        return Ok(None);
    };
    let matches: Vec<_> = entities
        .iter()
        .filter(|e| {
            e.entity_type == entity_type
                && (e.temp_id == refer || e.name == refer || e.aliases.iter().any(|a| a == refer))
        })
        .filter(|e| {
            e.member_id
                .as_deref()
                .map(str::trim)
                .map_or(true, |mid| mid == owner)
        })
        .collect();
    let candidate = match matches.as_slice() {
        [candidate] => Some(*candidate),
        [] => candidate,
        _ => {
            report.messages.push(format!(
                "pending ambiguous owner-scoped reference `{refer}`"
            ));
            return Ok(None);
        }
    };
    let name = candidate.map(|e| e.name.as_str()).unwrap_or(refer).trim();
    if candidate.is_none() && !entities.is_empty() {
        // Structured batches may resolve an existing direct name for legacy
        // callers, but never create a node from an undeclared endpoint token.
        return Ok(store::find_entity_by_any_name(
            conn,
            namespace,
            name,
            Some(entity_type),
            Some(owner),
        )?
        .map(|entity| entity.id));
    }
    if name.is_empty() {
        return Ok(None);
    }
    let id = store::resolve_or_create_entity(
        conn,
        namespace,
        entity_type,
        name,
        Some(owner),
        &[],
        PrivacyLevel::OwnerPrivate,
    )?;
    if let Some(candidate) = candidate {
        for alias in &candidate.aliases {
            write_alias_fact(
                conn,
                namespace,
                &id,
                alias,
                Some(owner),
                PrivacyLevel::OwnerPrivate,
                source_memory_id,
                report,
            )?;
        }
    }
    Ok(Some(id))
}

fn resolve_ref(
    conn: &Connection,
    namespace: &str,
    roster: &[HouseholdMember],
    id_map: &HashMap<String, String>,
    refer: &str,
    hint_type: EntityType,
    restricted_identity: bool,
    allow_create_direct_name: bool,
    report: &mut IngestReport,
) -> Result<Option<String>> {
    let refer = refer.trim();
    if refer.is_empty() {
        return Ok(None);
    }
    if let Some(id) = id_map.get(refer) {
        return Ok(Some(id.clone()));
    }
    if hint_type == EntityType::Person {
        if let Some(mid) = resolve_roster_id(roster, Some(refer), refer) {
            if shared_scope_member(&mid) {
                return Ok(None);
            }
            let m = roster.iter().find(|m| m.member_id == mid).unwrap();
            let label = m.display_name.as_str();
            return Ok(Some(store::ensure_person_entity(
                conn, namespace, &mid, label, &m.aliases,
            )?));
        }
    }
    if refer == "家庭" {
        return Ok(Some(store::resolve_or_create_entity(
            conn,
            namespace,
            EntityType::Family,
            "家庭",
            None,
            &["household".into()],
            PrivacyLevel::FamilyShared,
        )?));
    }
    if let Some(e) = store::find_entity_by_any_name(conn, namespace, refer, Some(hint_type), None)?
    {
        return Ok(Some(e.id));
    }
    if let Some(e) = store::find_entity_by_any_name(conn, namespace, refer, None, None)? {
        return Ok(Some(e.id));
    }
    if !allow_create_direct_name {
        report
            .messages
            .push("pending undefined structured entity reference".into());
        return Ok(None);
    }
    if hint_type == EntityType::Person && restricted_identity {
        report.messages.push(format!(
            "pending sourced/private person reference without roster identity `{refer}`"
        ));
        return Ok(None);
    }
    Ok(Some(store::resolve_or_create_entity(
        conn,
        namespace,
        hint_type,
        refer,
        None,
        &[],
        PrivacyLevel::FamilyShared,
    )?))
}

fn guess_object_type(relation: &str) -> EntityType {
    match RelationType::parse_known(relation) {
        Some(r) => r
            .allowed_object_types()
            .first()
            .copied()
            .unwrap_or(EntityType::Concept),
        None => EntityType::Concept,
    }
}

// These values identify an attribution scope, not a semantic name blacklist.
fn shared_scope_member(member_id: &str) -> bool {
    crate::household::is_shared_member_id(member_id) || member_id.trim() == "__public__"
}

fn resolve_roster_id(
    roster: &[HouseholdMember],
    member_id: Option<&str>,
    name: &str,
) -> Option<String> {
    if let Some(mid) = member_id.map(str::trim).filter(|s| !s.is_empty()) {
        if roster.iter().any(|m| m.member_id == mid) {
            return Some(mid.to_string());
        }
    }
    let name = name.trim();
    roster
        .iter()
        .find(|m| {
            m.display_name == name || m.member_id == name || m.aliases.iter().any(|a| a == name)
        })
        .map(|m| m.member_id.clone())
}

fn ensure_roster_person(
    conn: &Connection,
    namespace: &str,
    roster: &mut Vec<HouseholdMember>,
    name: &str,
    aliases: &[String],
) -> Result<()> {
    let name = name.trim();
    if name.is_empty() || name == "家庭" || shared_scope_member(name) {
        return Ok(());
    }
    if resolve_roster_id(roster, None, name).is_some() {
        if let Some(m) = roster.iter_mut().find(|m| {
            m.display_name == name || m.member_id == name || m.aliases.iter().any(|a| a == name)
        }) {
            for a in aliases {
                let a = a.trim();
                if a.is_empty() || a == name {
                    continue;
                }
                if !m.aliases.iter().any(|x| x == a) {
                    m.aliases.push(a.to_string());
                }
            }
            crate::household::upsert_member(conn, namespace, m)?;
        }
        return Ok(());
    }
    let member_id = name.to_string();
    let mut aliases = aliases.to_vec();
    if !aliases.iter().any(|a| a == name) {
        aliases.push(name.to_string());
    }
    let member = HouseholdMember {
        member_id: member_id.clone(),
        display_name: name.to_string(),
        role: crate::household::MemberRole::Other,
        aliases: aliases.clone(),
        age_band: None,
    };
    crate::household::upsert_member(conn, namespace, &member)?;
    store::ensure_person_entity(conn, namespace, &member_id, name, &aliases)?;
    roster.push(member);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::household::{HouseholdMember, MemberRole};
    use tempfile::tempdir;

    #[test]
    fn open_relation_and_alias_merge() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("t.db");
        let pool = crate::store::open_pool(&db).unwrap();
        let mut conn = pool.get().unwrap();
        crate::store::migrate(&mut conn).unwrap();
        let ns = "default";
        let roster = vec![HouseholdMember {
            member_id: "child1".into(),
            display_name: "小航".into(),
            role: MemberRole::Child,
            aliases: vec![],
            age_band: None,
        }];
        store::ensure_person_entity(&conn, ns, "child1", "小航", &[]).unwrap();

        let entities = vec![
            ExtractedEntity {
                temp_id: "e1".into(),
                name: "小航".into(),
                entity_type: EntityType::Person,
                aliases: vec!["航航".into()],
                member_id: Some("child1".into()),
            },
            ExtractedEntity {
                temp_id: "e2".into(),
                name: "北京".into(),
                entity_type: EntityType::Location,
                aliases: vec![],
                member_id: None,
            },
        ];
        let triples = vec![ExtractedTriple {
            subject: "e1".into(),
            relation: "住在".into(),
            object: "e2".into(),
            object_type: Some(EntityType::Location),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec![],
            privacy: None,
            condition_text: None,
            evidence_text: "小航住在北京".into(),
            confidence: 0.9,
            attrs: serde_json::json!({}),
        }];
        let report = ingest_graph_batch(&conn, ns, &roster, &entities, &triples, None).unwrap();
        assert!(report.facts_written >= 1);
        let person = store::get_entity(&conn, ns, &store::person_entity_id("child1"))
            .unwrap()
            .unwrap();
        assert!(
            person.aliases.iter().any(|a| a == "航航"),
            "aliases={:?}",
            person.aliases
        );
        // Second ingest with nickname-as-name should merge, not duplicate person.
        let entities2 = vec![ExtractedEntity {
            temp_id: "n1".into(),
            name: "航航".into(),
            entity_type: EntityType::Person,
            aliases: vec!["小航".into()],
            member_id: None,
        }];
        let _ = ingest_graph_batch(&conn, ns, &roster, &entities2, &[], None).unwrap();
        let all = store::list_entities(&conn, ns, 50).unwrap();
        let people: Vec<_> = all
            .iter()
            .filter(|e| e.entity_type == EntityType::Person && e.status != "merged")
            .collect();
        assert_eq!(people.len(), 1, "people={people:?}");
    }
}
