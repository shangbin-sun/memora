//! Structural references, specific Chinese aliases and provenance-safe family edges.
use memora::graph::{
    self, EntityType, ExtractedEntity, ExtractedTriple, FactObject, FactPolarity, FactStatus,
    PrivacyLevel,
};
use memora::{
    ForgetReason, GrantInput, HouseholdMember, MemberRole, MemoryEngine, MemoryService,
    PersonGraphQuery, RememberInput,
};
use rusqlite::Connection;
use serde_json::json;
use uuid::Uuid;

const NS: &str = "entity-relations";
fn member(id: &str, name: &str) -> HouseholdMember {
    HouseholdMember {
        member_id: id.into(),
        display_name: name.into(),
        role: MemberRole::Other,
        aliases: vec![],
        age_band: None,
    }
}
fn roster() -> Vec<HouseholdMember> {
    vec![
        member("child", "林子航"),
        member("dad", "林志远"),
        member("mom", "宋娜娜"),
        member("公共", "公共"),
        member("__public__", "公共归属"),
    ]
}
async fn setup() -> (tempfile::TempDir, MemoryEngine, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.db");
    let engine = MemoryEngine::builder()
        .db_path(&path)
        .namespace(NS)
        .graph_projection_worker(false)
        .build()
        .await
        .unwrap();
    for row in roster() {
        engine.upsert_member(Some(NS), row).await.unwrap();
    }
    let conn = Connection::open(path).unwrap();
    (dir, engine, conn)
}
fn entity(id: &str, name: &str, kind: EntityType, member: Option<&str>) -> ExtractedEntity {
    ExtractedEntity {
        temp_id: id.into(),
        name: name.into(),
        entity_type: kind,
        aliases: vec![],
        member_id: member.map(str::to_string),
    }
}
fn triple(subject: &str, relation: &str, object: &str, kind: EntityType) -> ExtractedTriple {
    ExtractedTriple {
        subject: subject.into(),
        relation: relation.into(),
        object: object.into(),
        object_type: Some(kind),
        polarity: FactPolarity::Affirmative,
        status: FactStatus::Active,
        domains: vec!["household".into()],
        privacy: Some(PrivacyLevel::FamilyShared),
        condition_text: None,
        evidence_text: "原文依据".into(),
        confidence: 1.0,
        attrs: json!({}),
    }
}
fn person_query(viewer: &str) -> PersonGraphQuery {
    PersonGraphQuery {
        member_id: "child".into(),
        viewer_member_id: Some(viewer.into()),
        ..Default::default()
    }
}
async fn source(engine: &MemoryEngine, private: bool) -> Uuid {
    let mut input = RememberInput::episodic("林子航的小名是航航，爸爸是林志远。");
    input.member_id = Some("child".into());
    input.skip_consolidation = true;
    input.metadata = json!({"skip_graph":true,"privacy":private,"attribution":"manual",
        "scope":"members","subjects":[{"member_id":"child","confidence":1.0}]});
    engine.remember(input).await.unwrap()
}

#[tokio::test]
async fn undefined_references_and_shared_member_scopes_never_become_people() {
    let (_dir, engine, conn) = setup().await;
    let entities = vec![
        entity("p", "林子航", EntityType::Person, Some("child")),
        entity("scope", "公共", EntityType::Person, Some("公共")),
        entity("legacy", "公共归属", EntityType::Person, Some("__public__")),
        entity("empty", "", EntityType::Concept, None),
        entity("transport", "公共交通", EntityType::Concept, None),
    ];
    let edges = vec![
        triple("p", "喜欢", "e2", EntityType::Concept),
        triple("p", "喜欢", "empty", EntityType::Concept),
        triple("scope", "喜欢", "transport", EntityType::Concept),
        triple("legacy", "喜欢", "transport", EntityType::Concept),
        triple("p", "喜欢", "transport", EntityType::Concept),
    ];
    let result = graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &edges, None).unwrap();
    assert_eq!(result.facts_written, 1);
    assert!(result.pending >= 4);
    let nodes = graph::store::list_entities(&conn, NS, 100).unwrap();
    assert!(!nodes.iter().any(|e| matches!(
        e.canonical_name.as_str(),
        "e2" | "empty" | "公共" | "公共归属"
    )));
    assert!(nodes.iter().any(|e| e.canonical_name == "公共交通"));
    // Existing bad scope hubs remain persisted but are not safe overview anchors.
    graph::store::ensure_person_entity(&conn, NS, "公共", "公共", &[]).unwrap();
    assert!(!graph::store::list_safe_roster_people(&conn, NS)
        .unwrap()
        .iter()
        .any(|e| e.owner_member_id.as_deref() == Some("公共")));
    engine.shutdown();
}

#[tokio::test]
async fn direct_name_compatibility_does_not_reject_real_e2_model_names() {
    let (_dir, engine, conn) = setup().await;
    let edges = vec![triple("林子航", "设备型号", "E2", EntityType::Concept)];
    let result = graph::ingest_graph_batch(&conn, NS, &roster(), &[], &edges, None).unwrap();
    assert_eq!(result.facts_written, 1);
    assert!(graph::store::list_entities(&conn, NS, 100)
        .unwrap()
        .iter()
        .any(|e| e.canonical_name == "E2"));
    engine.shutdown();
}

#[tokio::test]
async fn concrete_nickname_folds_to_person_with_source_privacy_and_forget() {
    let (_dir, engine, conn) = setup().await;
    let id = source(&engine, true).await;
    let entities = vec![
        entity("p", "林子航", EntityType::Person, Some("child")),
        entity("nick", "航航", EntityType::Person, None),
    ];
    let mut alias = triple("p", "小名", "nick", EntityType::Person);
    alias.privacy = Some(PrivacyLevel::OwnerPrivate);
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[alias], Some(id)).unwrap();
    assert!(!graph::store::list_entities(&conn, NS, 100)
        .unwrap()
        .iter()
        .any(|e| e.canonical_name == "航航"));
    let own = graph::query_person_graph(&conn, NS, &person_query("child")).unwrap();
    assert!(own.person.unwrap().aliases.iter().any(|a| a == "航航"));
    assert!(own.facts.iter().any(|f| f.relation == "小名"
        && matches!(&f.object,FactObject::Value{value,..} if value==&json!("航航"))));
    assert!(!serde_json::to_string(
        &graph::query_person_graph(&conn, NS, &person_query("dad")).unwrap()
    )
    .unwrap()
    .contains("航航"));
    engine
        .forget(
            id,
            ForgetReason::User {
                reason: "withdraw evidence".into(),
            },
        )
        .await
        .unwrap();
    let after = graph::query_person_graph(
        &conn,
        NS,
        &PersonGraphQuery {
            include_ended: true,
            ..person_query("child")
        },
    )
    .unwrap();
    assert!(!serde_json::to_string(&after).unwrap().contains("航航"));
    engine.shutdown();
}

#[tokio::test]
async fn legacy_nickname_is_specific_and_negative_or_ended_is_not_promoted() {
    let (_dir, engine, conn) = setup().await;
    let entities = vec![
        entity("p", "林子航", EntityType::Person, Some("child")),
        entity("a", "旧小名", EntityType::Concept, None),
        entity("b", "误传小名", EntityType::Concept, None),
    ];
    let mut ended = triple("p", "has_attribute", "a", EntityType::Concept);
    ended.attrs = json!({"alias_kind":"nickname"});
    ended.status = FactStatus::Ended;
    let mut negative = triple("p", "小名", "b", EntityType::Concept);
    negative.polarity = FactPolarity::Negative;
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[ended, negative], None).unwrap();
    let own = graph::query_person_graph(
        &conn,
        NS,
        &PersonGraphQuery {
            include_ended: true,
            ..person_query("child")
        },
    )
    .unwrap();
    let aliases = own.person.unwrap().aliases;
    assert!(!aliases.iter().any(|a| a == "旧小名" || a == "误传小名"));
    assert!(own.facts.iter().all(|f| f.relation == "小名"));
    assert!(!graph::store::list_entities(&conn, NS, 100)
        .unwrap()
        .iter()
        .any(|e| e.canonical_name == "旧小名" || e.canonical_name == "误传小名"));
    engine.shutdown();
}

#[tokio::test]
async fn chinese_relation_grant_matches_legacy_english_without_generic_scope_expansion() {
    let (_dir, engine, conn) = setup().await;
    let entities = vec![
        entity("p", "林子航", EntityType::Person, Some("child")),
        entity("n", "航航", EntityType::Concept, None),
    ];
    let mut alias = triple("p", "nickname", "n", EntityType::Concept);
    alias.privacy = Some(PrivacyLevel::OwnerPrivate);
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[alias], None).unwrap();
    let own = graph::query_person_graph(&conn, NS, &person_query("child")).unwrap();
    let fact = own.facts.iter().find(|f| f.relation == "小名").unwrap();
    let mut grant = GrantInput {
        namespace: Some(NS.into()),
        grantor_member_id: "child".into(),
        grantee_member_id: "dad".into(),
        scope_kind: "relation".into(),
        scope_value: "has_attribute".into(),
        operations: vec!["view".into()],
        valid_from_ms: None,
        valid_to_ms: None,
        evidence_text: None,
    };
    graph::store::insert_grant(&conn, NS, &grant).unwrap();
    assert!(!graph::store::has_view_grant(&conn, NS, "child", "dad", fact, 1).unwrap());
    grant.scope_value = "nickname".into();
    let row = graph::store::insert_grant(&conn, NS, &grant).unwrap();
    assert_eq!(row.scope_value, "小名");
    assert!(graph::store::has_view_grant(&conn, NS, "child", "dad", fact, 1).unwrap());
    // Legacy persisted English scopes are normalized only at exact relation level.
    conn.execute(
        "UPDATE kg_grants SET scope_json=?1 WHERE id=?2",
        rusqlite::params![
            r#"{"kind":"relation","value":"nickname"}"#,
            row.id.to_string()
        ],
    )
    .unwrap();
    assert!(graph::store::has_view_grant(&conn, NS, "child", "dad", fact, 1).unwrap());
    grant.scope_value = "  ".into();
    assert!(graph::store::insert_grant(&conn, NS, &grant).is_err());
    let mut displayed = fact.clone();
    displayed.relation = "称呼".into();
    displayed.attrs = json!({"assertion_basis":"context_inference","inference_note":"proof note", "evidence_spans":["proof span"]});
    let (_, _, text) = graph::store::display_fact(&conn, NS, &displayed).unwrap();
    assert!(text.contains("称呼") && !text.contains("属性/称呼"));
    assert!(text.contains("上下文推断") && !text.contains("proof"));
    displayed.relation = "likes".into();
    assert!(graph::store::display_fact(&conn, NS, &displayed)
        .unwrap()
        .2
        .contains("喜欢"));
    engine.shutdown();
}

#[tokio::test]
async fn family_relations_keep_direction_and_do_not_grant_private_reverse_source_access() {
    let (_dir, engine, conn) = setup().await;
    let entities = vec![
        entity("p", "林子航", EntityType::Person, Some("child")),
        entity("d", "林志远", EntityType::Person, Some("dad")),
        entity("m", "宋娜娜", EntityType::Person, Some("mom")),
    ];
    let public_edges = vec![
        triple("p", "爸爸", "d", EntityType::Person),
        triple("p", "妈妈", "m", EntityType::Person),
        triple("d", "儿子", "p", EntityType::Person),
        triple("d", "老婆", "m", EntityType::Person),
    ];
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &public_edges, None).unwrap();
    let child = graph::query_person_graph(&conn, NS, &person_query("child")).unwrap();
    assert!(child
        .facts
        .iter()
        .any(|f| f.subject_id == "person:child" && f.relation == "爸爸"));
    let dad = graph::query_person_graph(
        &conn,
        NS,
        &PersonGraphQuery {
            member_id: "dad".into(),
            viewer_member_id: Some("dad".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(dad
        .facts
        .iter()
        .any(|f| f.subject_id == "person:dad" && f.relation == "儿子"));
    assert!(dad
        .facts
        .iter()
        .any(|f| f.subject_id == "person:dad" && f.relation == "老婆"));
    let id = source(&engine, true).await;
    let mut private_reverse = triple("d", "儿子", "p", EntityType::Person);
    private_reverse.privacy = Some(PrivacyLevel::OwnerPrivate);
    private_reverse.evidence_text = "PRIVATE_REVERSE_SOURCE".into();
    graph::ingest_graph_batch(
        &conn,
        NS,
        &roster(),
        &entities,
        &[private_reverse],
        Some(id),
    )
    .unwrap();
    let dad_after = graph::query_person_graph(
        &conn,
        NS,
        &PersonGraphQuery {
            member_id: "dad".into(),
            viewer_member_id: Some("dad".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!serde_json::to_string(&dad_after)
        .unwrap()
        .contains("PRIVATE_REVERSE_SOURCE"));
    engine.shutdown();
}

#[tokio::test]
async fn declared_temp_ids_win_over_person_aliases_and_direct_name_alias_values() {
    let (_dir, engine, conn) = setup().await;
    let mut child = entity("p", "林子航", EntityType::Person, Some("child"));
    child.aliases = vec!["航航".into(), "d".into(), "n".into()];
    let entities = vec![
        entity("d", "林志远", EntityType::Person, Some("dad")),
        child,
        entity("n", "航航", EntityType::Concept, None),
    ];
    let edges = vec![
        triple("p", "爸爸", "d", EntityType::Person),
        triple("p", "小名", "n", EntityType::Concept),
    ];
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &edges, None).unwrap();
    let own = graph::query_person_graph(&conn, NS, &person_query("child")).unwrap();
    assert!(own.facts.iter().any(|f| f.relation == "爸爸"
        && matches!(&f.object,
        FactObject::Entity{entity_id} if entity_id=="person:dad")));
    assert!(own.facts.iter().any(|f| f.relation == "小名"
        && matches!(&f.object,
        FactObject::Value{value,..} if value==&json!("航航"))));
    let direct = triple("p", "别名", "航航", EntityType::Concept);
    graph::ingest_graph_batch(&conn, NS, &roster(), &entities, &[direct], None).unwrap();
    let own = graph::query_person_graph(&conn, NS, &person_query("child")).unwrap();
    assert!(own.facts.iter().any(|f| f.relation == "别名"
        && matches!(&f.object,
        FactObject::Value{value,..} if value==&json!("航航"))));
    engine.shutdown();
}
