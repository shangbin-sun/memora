//! Graph privacy regressions: source-local evidence, live aliases and grants.
//! Build the real schema through MemoryEngine, then exercise graph APIs directly
//! so extractor heuristics do not obscure the authorization being tested.

use memora::graph::{self, ExtractedEntity, ExtractedTriple};
use memora::{
    EntityType, FactObject, FactPolarity, FactStatus, ForgetReason, GrantInput,
    GraphOverviewQuery, HouseholdMember, MemberRole, MemoryEngine, MemoryService,
    PersonGraphQuery, PrivacyLevel, RememberInput,
};
use rusqlite::{params, Connection};
use uuid::Uuid;

const NS: &str = "default";

fn roster() -> Vec<HouseholdMember> {
    vec![
        HouseholdMember { member_id: "alice".into(), display_name: "小明".into(),
            role: MemberRole::Other, aliases: vec!["名册别称".into()], age_band: None },
        HouseholdMember { member_id: "bob".into(), display_name: "小红".into(),
            role: MemberRole::Other, aliases: vec![], age_band: None },
    ]
}

async fn setup() -> (tempfile::TempDir, MemoryEngine, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("privacy.db");
    let engine = MemoryEngine::builder().db_path(&path).build().await.unwrap();
    for member in roster() { engine.upsert_member(None, member).await.unwrap(); }
    let conn = Connection::open(&path).unwrap();
    for member in roster() {
        graph::store::ensure_person_entity(&conn, NS, &member.member_id, &member.display_name, &member.aliases).unwrap();
    }
    (dir, engine, conn)
}

async fn source(engine: &MemoryEngine, content: &str, private: bool) -> Uuid {
    let mut input = RememberInput::episodic(content);
    input.member_id = Some("alice".into());
    input.skip_consolidation = true;
    input.metadata = serde_json::json!({"privacy": private, "attribution":"manual",
        "subjects":[{"member_id":"alice","confidence":1.0}], "scope":"members"});
    engine.remember(input).await.unwrap()
}

fn person(aliases: &[&str]) -> ExtractedEntity {
    ExtractedEntity { temp_id: "p".into(), name: "小明".into(), entity_type: EntityType::Person,
        aliases: aliases.iter().map(|s| (*s).to_string()).collect(), member_id: Some("alice".into()) }
}

fn triple(relation: &str, object: &str, object_type: EntityType, evidence: &str) -> ExtractedTriple {
    ExtractedTriple { subject: "p".into(), relation: relation.into(), object: object.into(),
        object_type: Some(object_type), polarity: FactPolarity::Affirmative, status: FactStatus::Active,
        domains: vec!["household".into()], privacy: Some(PrivacyLevel::FamilyShared),
        condition_text: None, evidence_text: evidence.into(), confidence: 0.9,
        attrs: serde_json::json!({}) }
}

fn person_query(viewer: &str) -> PersonGraphQuery {
    PersonGraphQuery { member_id: "alice".into(), viewer_member_id: Some(viewer.into()),
        ..Default::default() }
}

fn overview_query(viewer: &str) -> GraphOverviewQuery {
    GraphOverviewQuery { viewer_member_id: Some(viewer.into()), ..Default::default() }
}

#[tokio::test]
async fn private_aliases_do_not_leak_through_person_overview_or_legacy_attributes() {
    let (_dir, engine, conn) = setup().await;
    let id = source(&engine, "私密称呼记录", true).await;
    let entities = vec![person(&["隐秘实体别名"]), ExtractedEntity {
        temp_id: "nick".into(), name: "隐秘昵称".into(), entity_type: EntityType::Concept,
        aliases: vec![], member_id: None,
    }];
    let mut alias = triple("有属性", "nick", EntityType::Concept, "隐秘昵称证明");
    alias.privacy = Some(PrivacyLevel::OwnerPrivate);
    alias.attrs = serde_json::json!({"alias_kind":"nickname"});
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[alias], Some(id)).unwrap();
    // Emulate a legacy database polluted by the old alias attachment path.
    graph::store::upsert_alias(&conn, NS, "person:alice", "无来源raw私密别名", Some("alice")).unwrap();
    conn.execute("UPDATE kg_entities SET attrs_json = ?1 WHERE namespace = ?2 AND id = 'person:alice'",
        params![r#"{"nickname":"私密raw属性"}"#, NS]).unwrap();
    conn.execute("UPDATE household_members SET aliases_json = ?1 WHERE namespace = ?2 AND member_id = 'alice'",
        params![r#"["名册别称","隐秘昵称"]"#, NS]).unwrap();

    let self_graph = graph::query_person_graph(&conn, NS, &person_query("alice")).unwrap();
    let hub = self_graph.person.unwrap();
    assert!(hub.aliases.iter().any(|a| a == "隐秘实体别名"));
    assert!(hub.aliases.iter().any(|a| a == "隐秘昵称"));
    assert!(!self_graph.entities.iter().any(|e| e.canonical_name == "隐秘昵称"),
        "alias assertions must describe the hub, not fragment it into nodes");

    let other = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    let serialized = serde_json::to_string(&other).unwrap();
    for secret in ["隐秘实体别名", "隐秘昵称", "无来源raw私密别名", "私密raw属性"] {
        assert!(!serialized.contains(secret), "person leaked {secret}: {serialized}");
    }
    assert!(other.person.unwrap().aliases.iter().any(|a| a == "名册别称"));
    let overview = graph::query_graph_overview(&conn, NS, &overview_query("bob")).unwrap();
    let serialized = serde_json::to_string(&overview).unwrap();
    for secret in ["隐秘实体别名", "隐秘昵称", "无来源raw私密别名", "私密raw属性"] {
        assert!(!serialized.contains(secret), "overview leaked {secret}: {serialized}");
    }
}

#[tokio::test]
async fn forgetting_memory_alias_removes_it_even_with_include_ended() {
    let (_dir, engine, conn) = setup().await;
    let id = source(&engine, "共享昵称来源记录", false).await;
    graph::ingest_graph_batch(&conn, NS, &roster(), &[person(&["待忘昵称"])], &[], Some(id)).unwrap();
    let before = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    assert!(before.person.unwrap().aliases.iter().any(|a| a == "待忘昵称"));
    engine.forget(id, ForgetReason::User { reason: "withdraw alias".into() }).await.unwrap();
    let after = graph::query_person_graph(&conn, NS, &PersonGraphQuery {
        include_ended: true, ..person_query("alice")
    }).unwrap();
    assert!(!serde_json::to_string(&after).unwrap().contains("待忘昵称"));
    let overview = graph::query_graph_overview(&conn, NS, &GraphOverviewQuery {
        include_ended: true, ..overview_query("bob")
    }).unwrap();
    assert!(!serde_json::to_string(&overview).unwrap().contains("待忘昵称"));
}

#[tokio::test]
async fn dual_sources_choose_visible_evidence_and_graph_grant_expires() {
    let (_dir, engine, conn) = setup().await;
    let hidden = source(&engine, "私密来源原文", true).await;
    let shared = source(&engine, "公开来源原文", false).await;
    let entities = vec![person(&[]), ExtractedEntity {
        temp_id: "place".into(), name: "北京".into(), entity_type: EntityType::Location,
        aliases: vec![], member_id: None,
    }];
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities,
        &[triple("lives_at", "place", EntityType::Location, "PRIVATE_SOURCE_EVIDENCE")], Some(hidden)).unwrap();
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities,
        &[triple("lives_at", "place", EntityType::Location, "PUBLIC_SOURCE_EVIDENCE")], Some(shared)).unwrap();
    let graph = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    let facts: Vec<_> = graph.facts.iter().filter(|f| f.relation == "住在").collect();
    assert_eq!(facts.len(), 1, "independent sources must confirm the same fact");
    let fact_id = facts[0].id;
    assert_eq!(facts[0].source_memory_id, Some(shared));
    assert_eq!(facts[0].evidence_text.as_deref(), Some("PUBLIC_SOURCE_EVIDENCE"));
    // Emulate a legacy representative pointer/text overwritten by hidden input.
    conn.execute("UPDATE kg_facts SET source_memory_id = ?1, evidence_text = 'PRIVATE_SOURCE_EVIDENCE' WHERE namespace = ?2 AND id = ?3",
        params![hidden.to_string(), NS, fact_id.to_string()]).unwrap();
    let overview = graph::query_graph_overview(&conn, NS, &overview_query("bob")).unwrap();
    let fact = overview.facts.iter().find(|f| f.id == fact_id).unwrap();
    assert_eq!(fact.source_memory_id, Some(shared));
    assert_eq!(fact.evidence_text.as_deref(), Some("PUBLIC_SOURCE_EVIDENCE"));
    assert!(!serde_json::to_string(&overview).unwrap().contains("PRIVATE_SOURCE_EVIDENCE"));

    let grant = engine.grant_access(GrantInput {
        grantor_member_id: "alice".into(), grantee_member_id: "bob".into(),
        scope_kind: "fact".into(), scope_value: fact_id.to_string(), operations: vec!["view".into()],
        ..Default::default()
    }).await.unwrap();
    let granted = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    let fact = granted.facts.iter().find(|f| f.id == fact_id).unwrap();
    assert_eq!(fact.source_memory_id, Some(hidden));
    assert_eq!(fact.evidence_text.as_deref(), Some("PRIVATE_SOURCE_EVIDENCE"));
    conn.execute("UPDATE kg_grants SET valid_to_ms = 1 WHERE namespace = ?1 AND id = ?2",
        params![NS, grant.id]).unwrap();
    let expired = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    let fact = expired.facts.iter().find(|f| f.id == fact_id).unwrap();
    assert_eq!(fact.source_memory_id, Some(shared));
    assert_eq!(fact.evidence_text.as_deref(), Some("PUBLIC_SOURCE_EVIDENCE"));
}

#[tokio::test]
async fn private_source_allows_valid_grant_but_not_expired_or_foreign_subject_owner() {
    let (_dir, engine, conn) = setup().await;
    let id = source(&engine, "私密昵称授权来源", true).await;
    graph::ingest_graph_batch(&conn, NS, &roster(), &[person(&["授权私密昵称"])], &[], Some(id)).unwrap();
    let grant = engine.grant_access(GrantInput {
        grantor_member_id: "alice".into(), grantee_member_id: "bob".into(),
        scope_kind: "all".into(), operations: vec!["view".into()], ..Default::default()
    }).await.unwrap();
    let granted = graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap();
    assert!(granted.person.unwrap().aliases.iter().any(|a| a == "授权私密昵称"));
    assert!(serde_json::to_string(&graph::query_graph_overview(&conn, NS, &overview_query("bob")).unwrap())
        .unwrap().contains("授权私密昵称"));
    conn.execute("UPDATE kg_grants SET valid_to_ms = 1 WHERE namespace = ?1 AND id = ?2",
        params![NS, grant.id]).unwrap();
    assert!(!serde_json::to_string(&graph::query_person_graph(&conn, NS, &person_query("bob")).unwrap())
        .unwrap().contains("授权私密昵称"));
    assert!(!serde_json::to_string(&graph::query_graph_overview(&conn, NS, &overview_query("bob")).unwrap())
        .unwrap().contains("授权私密昵称"));
    // The graph still says alice, but the current source now belongs to bob.
    // Neither old owner self-access nor an old grant may authorize that source.
    conn.execute("UPDATE memories SET metadata_json = ?1, subject_ids = '[\"bob\"]' WHERE id = ?2",
        params![r#"{"privacy":true,"subjects":[{"member_id":"bob","confidence":1.0}]}"#, id.to_string()]).unwrap();
    assert!(!serde_json::to_string(&graph::query_person_graph(&conn, NS, &person_query("alice")).unwrap())
        .unwrap().contains("授权私密昵称"));
    // A private source referencing a non-roster subject is not an owner grant.
    conn.execute("UPDATE memories SET metadata_json = ?1, subject_ids = '[\"foreign-owner\"]' WHERE id = ?2",
        params![r#"{"privacy":true,"subjects":[{"member_id":"foreign-owner","confidence":1.0}]}"#, id.to_string()]).unwrap();
    conn.execute("UPDATE kg_facts SET owner_member_id = 'foreign-owner' WHERE namespace = ?1 AND source_memory_id = ?2",
        params![NS, id.to_string()]).unwrap();
    let foreign = graph::query_graph_overview(&conn, NS, &overview_query("foreign-owner")).unwrap();
    assert!(!serde_json::to_string(&foreign).unwrap().contains("授权私密昵称"));
}

#[tokio::test]
async fn explicit_business_owner_conflict_stays_pending_without_touching_other_account() {
    let (_dir, _engine, conn) = setup().await;
    let entities = vec![person(&[]), ExtractedEntity {
        temp_id: "account".into(), name: "工资账户".into(), entity_type: EntityType::Account,
        aliases: vec!["不该写入的别名".into()], member_id: Some("bob".into()),
    }];
    let mut owns = triple("owns", "account", EntityType::Account, "owner conflict");
    owns.privacy = Some(PrivacyLevel::OwnerPrivate);
    let report = graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[owns], None).unwrap();
    assert!(report.pending > 0);
    assert!(report.messages.iter().any(|m| m.contains("owner conflict")));
    assert!(!graph::store::list_entities(&conn, NS, 100).unwrap().iter()
        .any(|e| e.entity_type == EntityType::Account));
    assert!(graph::query_person_graph(&conn, NS, &person_query("alice")).unwrap().facts.is_empty());
}

#[tokio::test]
async fn shared_fact_does_not_disclose_private_cross_owner_endpoint() {
    let (_dir, _engine, conn) = setup().await;
    let account = graph::store::resolve_or_create_entity(&conn, NS, EntityType::Account,
        "另一人的私密账户", Some("bob"), &[], PrivacyLevel::OwnerPrivate).unwrap();
    let now = 1;
    let fact = memora::KgFact {
        id: Uuid::now_v7(), subject_id: "person:alice".into(), relation: "相关".into(),
        object: FactObject::entity(account), owner_member_id: Some("alice".into()),
        domains: vec![], polarity: FactPolarity::Affirmative, status: FactStatus::Active,
        valid_from_ms: Some(now), valid_to_ms: None, condition_text: None,
        exception_text: None, reason_text: None, evidence_text: None, evidence_summary: None,
        source_memory_id: None, confidence: 0.9, privacy: PrivacyLevel::FamilyShared,
        revision: 1, created_at_ms: now, updated_at_ms: now, attrs: serde_json::json!({}),
    };
    graph::store::write_fact(&conn, NS, &fact).unwrap();
    let graph = graph::query_person_graph(&conn, NS, &person_query("alice")).unwrap();
    assert!(graph.facts.is_empty());
    assert!(!serde_json::to_string(&graph).unwrap().contains("另一人的私密账户"));
}

#[tokio::test]
async fn graph_only_evidence_is_not_replaced_by_a_superseded_memory_and_acl_precedes_limit() {
    let (_dir, engine, conn) = setup().await;
    let hidden = source(&engine, "即将失效来源", true).await;
    let entities = vec![person(&[]), ExtractedEntity {
        temp_id: "place".into(), name: "南京".into(), entity_type: EntityType::Location,
        aliases: vec![], member_id: None,
    }];
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities,
        &[triple("lives_at", "place", EntityType::Location, "GRAPH_ONLY_EVIDENCE")], None).unwrap();
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities,
        &[triple("lives_at", "place", EntityType::Location, "SUPERSEDED_PRIVATE_EVIDENCE")], Some(hidden)).unwrap();
    conn.execute("UPDATE memories SET state = 'superseded' WHERE id = ?1", params![hidden.to_string()]).unwrap();
    graph::ingest_graph_batch(&conn, NS, &roster(), &[person(&["隐藏占位别名"])], &[], Some(hidden)).unwrap();
    let snapshot = graph::query_graph_overview(&conn, NS, &GraphOverviewQuery {
        limit: Some(1), ..overview_query("bob")
    }).unwrap();
    assert_eq!(snapshot.facts.len(), 1, "hidden newer assertions cannot consume the quota");
    assert!(!snapshot.truncated);
    assert_eq!(snapshot.facts[0].source_memory_id, None);
    assert_eq!(snapshot.facts[0].evidence_text.as_deref(), Some("GRAPH_ONLY_EVIDENCE"));
    let serialized = serde_json::to_string(&snapshot).unwrap();
    assert!(!serialized.contains("SUPERSEDED_PRIVATE_EVIDENCE"));
    assert!(!serialized.contains("隐藏占位别名"));
}
