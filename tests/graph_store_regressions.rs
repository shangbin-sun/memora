//! Offline persistence/formatting regressions through a real migrated engine DB.
use memora::graph::{self, EntityType, FactObject, FactPolarity, FactStatus, KgFact, PrivacyLevel};
use memora::{HouseholdMember, MemberRole, MemoryEngine, MemoryService};
use rusqlite::{params, Connection};
use serde_json::json;
use uuid::Uuid;

fn fact(subject: &str) -> KgFact {
    KgFact {
        id: Uuid::now_v7(),
        subject_id: subject.into(),
        relation: "takes_medication".into(),
        object: FactObject::text("示例药物"),
        owner_member_id: Some("owner".into()),
        domains: vec!["medical".into()],
        polarity: FactPolarity::Affirmative,
        status: FactStatus::Active,
        valid_from_ms: None,
        valid_to_ms: None,
        condition_text: None,
        exception_text: None,
        reason_text: None,
        evidence_text: Some("公开证据".into()),
        evidence_summary: Some("公开摘要".into()),
        source_memory_id: None,
        confidence: 1.0,
        privacy: PrivacyLevel::FamilyShared,
        revision: 1,
        created_at_ms: 1,
        updated_at_ms: 1,
        attrs: json!({}),
    }
}

async fn engine() -> (tempfile::TempDir, MemoryEngine, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph-store.db");
    let engine = MemoryEngine::builder()
        .db_path(&path)
        .namespace("test")
        .build()
        .await
        .unwrap();
    let conn = Connection::open(path).unwrap();
    (dir, engine, conn)
}

#[tokio::test]
async fn jobs_persist_exact_input_backoff_and_revision_guard() {
    let (_dir, engine, conn) = engine().await;
    let id = Uuid::now_v7();
    graph::store::enqueue_graph_projection_job(
        &conn,
        id,
        "other",
        2,
        "完整附件正文",
        Some("owner"),
    )
    .unwrap();
    let jobs = graph::store::due_graph_projection_jobs(&conn, i64::MAX, 1).unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].memory_id, id);
    assert_eq!(jobs[0].namespace, "other");
    assert_eq!(jobs[0].revision, 2);
    assert_eq!(jobs[0].source_text, "完整附件正文");
    assert_eq!(jobs[0].speaker_member_id.as_deref(), Some("owner"));
    graph::store::fail_graph_projection_job(&conn, id, 2, "暂时不可用").unwrap();
    assert_eq!(
        graph::store::due_graph_projection_jobs(&conn, 0, 1)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        graph::store::due_graph_projection_jobs(&conn, i64::MAX, 1).unwrap()[0].attempts,
        1
    );
    graph::store::enqueue_graph_projection_job(&conn, id, "other", 1, "旧正文", None).unwrap();
    assert_eq!(
        graph::store::delete_graph_projection_job(&conn, id, 1).unwrap(),
        0
    );
    assert_eq!(
        graph::store::due_graph_projection_jobs(&conn, i64::MAX, 1).unwrap()[0].source_text,
        "完整附件正文"
    );
    graph::store::enqueue_graph_projection_job(&conn, id, "other", 3, "新正文", None).unwrap();
    let job = graph::store::due_graph_projection_jobs(&conn, 0, 1)
        .unwrap()
        .remove(0);
    assert_eq!(job.revision, 3);
    assert_eq!(job.attempts, 0);
    assert_eq!(
        graph::store::delete_graph_projection_job(&conn, id, 2).unwrap(),
        0
    );
    assert_eq!(
        graph::store::delete_graph_projection_job(&conn, id, 3).unwrap(),
        1
    );
    assert!(graph::store::due_graph_projection_jobs(&conn, i64::MAX, 1)
        .unwrap()
        .is_empty());
    engine.shutdown();
}

#[tokio::test]
async fn timed_grants_include_future_activation_and_expire_without_mutation() {
    let (_dir, engine, conn) = engine().await;
    conn.execute("INSERT INTO kg_grants(namespace,id,grantor_member_id,grantee_member_id,scope_json,operations_json,valid_from_ms,valid_to_ms,revoked,created_at_ms) VALUES ('test','g','owner','viewer','{\"kind\":\"all\",\"value\":\"\"}','[\"view\"]',100,200,0,1)",[]).unwrap();
    assert!(graph::store::has_unexpired_timed_grants(&conn, "test", 0).unwrap());
    assert!(graph::store::has_unexpired_timed_grants(&conn, "test", 150).unwrap());
    assert!(!graph::store::has_unexpired_timed_grants(&conn, "test", 201).unwrap());
    assert!(!graph::store::has_unexpired_timed_grants(&conn, "other", 150).unwrap());
    conn.execute("UPDATE kg_grants SET revoked=1", []).unwrap();
    assert!(!graph::store::has_unexpired_timed_grants(&conn, "test", 150).unwrap());
    engine.shutdown();
}

#[tokio::test]
async fn shared_person_display_uses_roster_not_historical_private_canonical_name() {
    let (_dir, engine, conn) = engine().await;
    engine
        .upsert_member(
            Some("test"),
            HouseholdMember {
                member_id: "owner".into(),
                display_name: "张三".into(),
                role: MemberRole::Self_,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
    let person = graph::store::person_entity_id("owner");
    graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    conn.execute(
        "UPDATE kg_entities SET canonical_name='秘密昵称' WHERE namespace='test' AND id=?1",
        [&person],
    )
    .unwrap();
    let mut row = fact(&person);
    row.object = FactObject::entity(&person);
    let (subject, object, display) = graph::store::display_fact(&conn, "test", &row).unwrap();
    assert_eq!(subject, "张三");
    assert_eq!(object, "张三");
    assert!(!display.contains("秘密昵称"));
    engine.shutdown();
}

#[tokio::test]
async fn structured_dose_frequency_and_open_qualifiers_are_not_dropped() {
    let (_dir, engine, conn) = engine().await;
    let mut row = fact("person:owner");
    row.attrs = json!({"dose":{"value":5,"unit":"mg"},"frequency":{"per_day":1},"route":"口服","context":"肾功能低下时不可用","open_safety_rule":"仅饭后服用"});
    let (_, _, display) = graph::store::display_fact(&conn, "test", &row).unwrap();
    for required in [
        "5",
        "mg",
        "per_day",
        "口服",
        "肾功能低下时不可用",
        "仅饭后服用",
    ] {
        assert!(display.contains(required), "lost {required}: {display}");
    }
    engine.shutdown();
}

#[tokio::test]
async fn historical_known_predicates_reconcile_with_canonical_new_writes() {
    let (_dir, engine, conn) = engine().await;
    let person = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let row = fact(&person);
    graph::store::write_fact(&conn, "test", &row).unwrap();
    conn.execute(
        "UPDATE kg_facts SET relation='服药' WHERE namespace='test' AND id=?1",
        [row.id.to_string()],
    )
    .unwrap();
    let matches =
        graph::store::list_active_by_subject_relation(&conn, "test", &person, "TAKES-MEDICATION")
            .unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].relation, "服用");
    assert!(
        graph::store::list_active_by_subject_relation(&conn, "test", &person, "Unknown-Open")
            .unwrap()
            .is_empty()
    );
    engine.shutdown();
}

#[tokio::test]
async fn per_source_evidence_never_falls_back_to_another_source() {
    let (_dir, engine, conn) = engine().await;
    let person = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let row = fact(&person);
    graph::store::write_fact(&conn, "test", &row).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    graph::store::link_fact_source_with_evidence(
        &conn,
        "test",
        &row.id.to_string(),
        Some(a),
        Some("公开证据"),
        Some("公开摘要"),
    )
    .unwrap();
    graph::store::link_fact_source_with_evidence(
        &conn,
        "test",
        &row.id.to_string(),
        Some(b),
        Some("私密家庭细节"),
        Some("私密摘要"),
    )
    .unwrap();
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &row.id.to_string(), Some(a))
            .unwrap()
            .0
            .as_deref(),
        Some("公开证据")
    );
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &row.id.to_string(), Some(b))
            .unwrap()
            .0
            .as_deref(),
        Some("私密家庭细节")
    );
    assert_eq!(
        graph::store::fact_source_evidence(
            &conn,
            "test",
            &row.id.to_string(),
            Some(Uuid::now_v7())
        )
        .unwrap(),
        (None, None)
    );
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &row.id.to_string(), None)
            .unwrap()
            .0
            .as_deref(),
        Some("公开证据")
    );
    let mut confirm = row.clone();
    confirm.id = Uuid::now_v7();
    confirm.source_memory_id = Some(b);
    confirm.evidence_text = Some("私密家庭细节".into());
    confirm.evidence_summary = Some("私密摘要".into());
    graph::store::write_fact(&conn, "test", &confirm).unwrap();
    let persisted = graph::store::get_fact(&conn, "test", &row.id.to_string())
        .unwrap()
        .unwrap();
    assert_eq!(
        persisted.source_memory_id, None,
        "graph-only representative must remain graph-only"
    );
    assert_eq!(persisted.evidence_text.as_deref(), Some("公开证据"));
    assert_eq!(
        graph::store::withdraw_facts_by_source_memory(&conn, "test", b).unwrap(),
        0
    );
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &row.id.to_string(), Some(b)).unwrap(),
        (None, None)
    );
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &row.id.to_string(), None)
            .unwrap()
            .0
            .as_deref(),
        Some("公开证据")
    );
    let count: i64 = conn
        .query_row(
            "SELECT count(*) FROM kg_fact_sources WHERE namespace=?1 AND fact_id=?2",
            params!["test", row.id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2);
    engine.shutdown();
}

#[tokio::test]
async fn merged_loan_lookup_resolves_keeper_and_deduplicates_same_owner_confirmation() {
    let (_dir, engine, conn) = engine().await;
    let owner = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let old = graph::store::resolve_or_create_entity(
        &conn,
        "test",
        EntityType::Liability,
        "房贷",
        Some("owner"),
        &[],
        PrivacyLevel::OwnerPrivate,
    )
    .unwrap();
    let keeper = graph::store::resolve_or_create_entity(
        &conn,
        "test",
        EntityType::Liability,
        "住房贷款",
        Some("owner"),
        &[],
        PrivacyLevel::OwnerPrivate,
    )
    .unwrap();
    let mut loan = fact(&owner);
    loan.relation = "owes".into();
    loan.object = FactObject::entity(&old);
    loan.domains = vec!["finance".into()];
    graph::store::write_fact(&conn, "test", &loan).unwrap();
    graph::store::merge_entities(&conn, "test", &keeper, &old).unwrap();
    let found = graph::store::find_entity_by_name_owned(
        &conn,
        "test",
        EntityType::Liability,
        "房贷",
        Some("owner"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(found.id, keeper);
    assert_ne!(found.status, "merged");
    assert!(graph::store::find_entity_by_name_owned(
        &conn,
        "test",
        EntityType::Liability,
        "房贷",
        Some("other")
    )
    .unwrap()
    .is_none());
    assert!(graph::store::find_entity_by_name_owned(
        &conn,
        "test",
        EntityType::Liability,
        "房贷",
        None
    )
    .unwrap()
    .is_none());
    let persisted = graph::store::get_fact(&conn, "test", &loan.id.to_string())
        .unwrap()
        .unwrap();
    assert!(matches!(&persisted.object,FactObject::Entity { entity_id } if entity_id == &keeper));
    let mut confirm = loan.clone();
    confirm.id = Uuid::now_v7();
    // Even a caller carrying the old ID must resolve to the canonical keeper.
    graph::store::write_fact(&conn, "test", &confirm).unwrap();
    let active =
        graph::store::list_active_by_subject_relation(&conn, "test", &owner, "owes").unwrap();
    assert_eq!(active.len(), 1);
    assert!(matches!(&active[0].object,FactObject::Entity { entity_id } if entity_id == &keeper));
    engine.shutdown();
}

#[tokio::test]
async fn same_name_accounts_use_requested_owner_not_most_recent_entity() {
    let (_dir, engine, conn) = engine().await;
    let zhang = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let li = graph::store::ensure_person_entity(&conn, "test", "other", "李四", &[]).unwrap();
    let a = graph::store::resolve_or_create_entity(
        &conn,
        "test",
        EntityType::Account,
        "工资卡",
        Some("owner"),
        &[],
        PrivacyLevel::OwnerPrivate,
    )
    .unwrap();
    let b = graph::store::resolve_or_create_entity(
        &conn,
        "test",
        EntityType::Account,
        "工资卡",
        Some("other"),
        &[],
        PrivacyLevel::OwnerPrivate,
    )
    .unwrap();
    conn.execute(
        "UPDATE kg_entities SET updated_at_ms=9999999999999 WHERE namespace='test' AND id=?1",
        [&b],
    )
    .unwrap();
    for (person, card, owner) in [(&zhang, &a, "owner"), (&li, &b, "other")] {
        let mut owns = fact(person);
        owns.relation = "owns".into();
        owns.object = FactObject::entity(card);
        owns.owner_member_id = Some(owner.into());
        owns.domains = vec!["finance".into()];
        graph::store::write_fact(&conn, "test", &owns).unwrap();
    }
    assert!(
        graph::store::merge_entities(&conn, "test", &a, &b).is_err(),
        "same label is not permission to merge different owners"
    );
    for (text, member) in [
        ("张三的工资卡", None),
        ("工资卡", Some("owner")),
        ("工资卡", Some("person:owner")),
    ] {
        let hits = if text == "张三的工资卡" {
            graph::store::collect_search_facts_with_person_names(
                &conn,
                "test",
                text,
                member,
                &[],
                |_| Ok(true),
                |entity| {
                    Ok(if entity.id == zhang {
                        vec!["张三".into()]
                    } else if entity.id == li {
                        vec!["李四".into()]
                    } else {
                        Vec::new()
                    })
                },
            )
            .unwrap()
        } else {
            graph::store::search_facts(&conn, "test", text, member, &[], 10).unwrap()
        };
        assert_eq!(hits.len(), 1, "wrong match for {text}/{member:?}");
        assert_eq!(hits[0].subject_id, zhang);
        assert!(matches!(&hits[0].object,FactObject::Entity { entity_id } if entity_id == &a));
    }
    engine.shutdown();
}

#[tokio::test]
async fn unauthorized_raw_person_alias_cannot_be_a_search_anchor() {
    let (_dir, engine, conn) = engine().await;
    let owner = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    graph::store::upsert_alias(&conn, "test", &owner, "秘密昵称", Some("owner")).unwrap();
    let row = fact(&owner);
    graph::store::write_fact(&conn, "test", &row).unwrap();

    let secret = graph::store::collect_search_facts_with_person_names(
        &conn,
        "test",
        "秘密昵称",
        None,
        &[],
        |_| Ok(true),
        |_| Ok(Vec::new()),
    )
    .unwrap();
    assert!(
        secret.is_empty(),
        "unprovenanced person alias must not anchor recall"
    );

    let visible = graph::store::collect_search_facts_with_person_names(
        &conn,
        "test",
        "张三",
        None,
        &[],
        |_| Ok(true),
        |entity| {
            Ok(if entity.id == owner {
                vec!["张三".into()]
            } else {
                Vec::new()
            })
        },
    )
    .unwrap();
    assert_eq!(
        visible.len(),
        1,
        "caller-authorized roster name remains searchable"
    );
    engine.shutdown();
}

#[tokio::test]
async fn removing_one_support_preserves_other_then_last_withdraw_cannot_resurrect() {
    let (_dir, engine, conn) = engine().await;
    let person = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let a = Uuid::now_v7();
    let b = Uuid::now_v7();
    let mut first = fact(&person);
    first.source_memory_id = Some(a);
    graph::store::write_fact(&conn, "test", &first).unwrap();
    let mut second = first.clone();
    second.id = Uuid::now_v7();
    second.source_memory_id = Some(b);
    second.evidence_text = Some("第二来源的证据".into());
    second.evidence_summary = Some("第二摘要".into());
    graph::store::write_fact(&conn, "test", &second).unwrap();
    let id = first.id.to_string();
    let before = graph::store::get_fact(&conn, "test", &id).unwrap().unwrap();
    assert_eq!(before.source_memory_id, Some(a));
    assert_eq!(before.evidence_text.as_deref(), Some("公开证据"));
    assert_eq!(
        graph::store::list_fact_source_memory_ids(&conn, "test", &id)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        graph::store::withdraw_facts_by_source_memory(&conn, "test", a).unwrap(),
        0
    );
    let after = graph::store::get_fact(&conn, "test", &id).unwrap().unwrap();
    assert_eq!(after.status, FactStatus::Active);
    assert_eq!(after.source_memory_id, Some(b));
    assert_eq!(after.evidence_text.as_deref(), Some("第二来源的证据"));
    assert_eq!(
        graph::store::fact_source_evidence(&conn, "test", &id, Some(a)).unwrap(),
        (None, None)
    );
    assert_eq!(
        graph::store::withdraw_facts_by_source_memory(&conn, "test", b).unwrap(),
        1
    );
    let last = graph::store::get_fact(&conn, "test", &id).unwrap().unwrap();
    assert_eq!(last.status, FactStatus::Withdrawn);
    assert_eq!(last.source_memory_id, None);
    assert_eq!(last.evidence_text, None);
    assert!(
        graph::store::list_fact_source_memory_ids(&conn, "test", &id)
            .unwrap()
            .is_empty()
    );
    assert!(!graph::store::has_graph_only_source(&conn, "test", &id).unwrap());
    assert_eq!(
        graph::store::withdraw_facts_by_source_memory(&conn, "test", a).unwrap(),
        0
    );
    assert!(!graph::store::has_graph_only_source(&conn, "test", &id).unwrap());
    engine.shutdown();
}

#[tokio::test]
async fn scalar_source_evidence_is_not_an_unauthorized_query_side_channel() {
    let (_dir, engine, conn) = engine().await;
    let person = graph::store::ensure_person_entity(&conn, "test", "owner", "张三", &[]).unwrap();
    let mut row = fact(&person);
    row.relation = "likes".into();
    row.object = FactObject::text("蓝色");
    row.evidence_text = Some("秘密代号NebulaX".into());
    graph::store::write_fact(&conn, "test", &row).unwrap();
    let hits =
        graph::store::search_facts(&conn, "test", "张三 NebulaX", Some("owner"), &[], 10).unwrap();
    assert!(
        hits.is_empty(),
        "scalar evidence must not influence graph matching"
    );
    engine.shutdown();
}
