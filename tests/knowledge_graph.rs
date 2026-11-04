//! Knowledge-graph integration: shared remember/recall + person subgraph.
//!
//! Offline tests inject [`HeuristicGraphExtractor`] as a stand-in for the LLM.
//! Production remember→graph uses LLM-only (`QianfanGraphExtractor`); rules are not merged.

use std::sync::Arc;

#[allow(deprecated)]
use memora::HeuristicGraphExtractor;
use memora::{
    FactObject, FactStatus, HouseholdMember, MemberRole, MemoryEngine, MemoryService,
    PersonGraphQuery, PrivacyLevel, RecallQuery, RelationType, RememberInput,
};

#[allow(deprecated)]
async fn test_engine(db_name: &str) -> (tempfile::TempDir, MemoryEngine) {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join(db_name))
        .graph_extractor(Arc::new(HeuristicGraphExtractor))
        .graph_projection_worker(false)
        .build()
        .await
        .unwrap();
    (dir, engine)
}

#[tokio::test]
async fn remember_writes_graph_and_dual_recall_returns_facts() {
    let (_dir, engine) = test_engine("kg.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec!["孩子".into()],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut input = RememberInput::episodic("小航花生过敏，在实验小学三年级二班。");
    input.member_id = Some("child1".into());
    let _id = engine.remember(input).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let recalled = engine
        .recall(RecallQuery {
            text: "过敏".into(),
            member_id: Some("child1".into()),
            fts_only: true,
            top_k: Some(8),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        !recalled.hits.is_empty() || !recalled.graph_hits.is_empty(),
        "expected memory and/or graph hits"
    );
    assert!(
        recalled
            .graph_hits
            .iter()
            .any(|g| g.fact.relation == RelationType::AllergicTo.as_zh_str()),
        "graph path should recall allergy fact: {:?}",
        recalled.graph_hits
    );
    assert!(
        recalled
            .context
            .items
            .iter()
            .any(|i| i.text.contains("过敏")),
        "context should include graph fact text"
    );

    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "child1".into(),
            include_ended: false,
            viewer_member_id: Some("child1".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(graph.person.is_some());
    assert!(graph
        .facts
        .iter()
        .any(|f| f.relation == RelationType::AllergicTo.as_zh_str()));
    assert!(graph
        .facts
        .iter()
        .any(|f| f.relation == RelationType::AttendsSchool.as_zh_str()));
    assert!(graph.entities.len() >= 2);
}

#[tokio::test]
async fn person_graph_hides_private_facts_from_other_viewers() {
    let (_dir, engine) = test_engine("kg_priv.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "grandma".into(),
                display_name: "姥姥".into(),
                role: MemberRole::Parent,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut input = RememberInput::episodic("姥姥有高血压，每天吃氨氯地平，早上一次。");
    input.member_id = Some("grandma".into());
    engine.remember(input).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let as_self = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "grandma".into(),
            viewer_member_id: Some("grandma".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!as_self.facts.is_empty());

    let as_other = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "grandma".into(),
            viewer_member_id: Some("someone_else".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(
        as_other.facts.is_empty()
            || as_other
                .facts
                .iter()
                .all(|f| f.privacy != PrivacyLevel::OwnerPrivate),
        "other viewer must not see owner_private medical facts"
    );
}

#[tokio::test]
async fn school_facts_coexist_without_explicit_correction() {
    let (_dir, engine) = test_engine("kg_school_coexist.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut a = RememberInput::episodic("小航在实验小学上学。");
    a.member_id = Some("child1".into());
    engine.remember(a).await.unwrap();

    let mut b = RememberInput::episodic("小航也在育才小学上课。");
    b.member_id = Some("child1".into());
    engine.remember(b).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "child1".into(),
            viewer_member_id: Some("child1".into()),
            include_ended: true,
            ..Default::default()
        })
        .await
        .unwrap();

    let school_facts: Vec<_> = graph
        .facts
        .iter()
        .filter(|f| f.relation == RelationType::AttendsSchool.as_zh_str())
        .collect();
    assert!(
        school_facts.len() >= 1,
        "expected school fact(s): {:?}",
        graph.facts
    );
}

#[tokio::test]
async fn school_fact_reassigns_with_explicit_correction() {
    let (_dir, engine) = test_engine("kg_school_fix.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut a = RememberInput::episodic("小航在实验小学上学。");
    a.member_id = Some("child1".into());
    engine.remember(a).await.unwrap();

    let mut b = RememberInput::episodic("说错了，小航在育才小学上学。");
    b.member_id = Some("child1".into());
    engine.remember(b).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "child1".into(),
            viewer_member_id: Some("child1".into()),
            include_ended: true,
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        graph
            .entities
            .iter()
            .any(|e| e.canonical_name.contains("育才")),
        "育才小学 should be present: {:?}",
        graph.entities
    );
}

#[tokio::test]
async fn school_transfer_replaces_prior_school_for_same_child() {
    let (_dir, engine) = test_engine("kg_school_xfer.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut a = RememberInput::episodic("小航在实验小学上学。");
    a.member_id = Some("child1".into());
    engine.remember(a).await.unwrap();

    let mut b = RememberInput::episodic("小航在育才小学上学。");
    b.member_id = Some("child1".into());
    engine.remember(b).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "child1".into(),
            viewer_member_id: Some("child1".into()),
            include_ended: false,
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        graph
            .entities
            .iter()
            .any(|e| e.canonical_name.contains("育才")),
        "育才小学 should be present: {:?}",
        graph.entities
    );
    assert!(
        !graph.entities.iter().any(|e| {
            e.canonical_name.contains("实验") && graph.facts.iter().any(|f| {
                f.relation == RelationType::AttendsSchool.as_zh_str()
                    && f.status == FactStatus::Active
                    && matches!(&f.object, FactObject::Entity { entity_id } if entity_id == &e.id)
            })
        }),
        "实验小学 should no longer be an active school edge"
    );
}

#[tokio::test]
async fn nickname_merges_into_person_aliases() {
    let (_dir, engine) = test_engine("kg_alias.db").await;

    engine
        .upsert_member(
            None,
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();

    let mut input = RememberInput::episodic("我儿子叫小航，小名航航。");
    input.member_id = Some("child1".into());
    engine.remember(input).await.unwrap();
    engine.retry_graph_projections(32).await.unwrap();

    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "child1".into(),
            viewer_member_id: Some("child1".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    let person = graph.person.expect("person entity");
    assert!(
        person
            .aliases
            .iter()
            .any(|a| a.contains("航航") || a == "航航")
            || person.canonical_name.contains("小航"),
        "aliases should include nickname: {:?}",
        person.aliases
    );
    assert!(
        !graph
            .entities
            .iter()
            .any(|e| e.canonical_name == "主驾" || e.canonical_name == "我"),
        "must not create 我/主驾 person nodes: {:?}",
        graph.entities
    );
}
