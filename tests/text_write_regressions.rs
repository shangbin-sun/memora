//! Text-curation regressions. No remote model is needed for these checks.
use memora::{
    HouseholdMember, MemberRole, MemoryEngine, MemoryService, RecallQuery, RememberInput,
};
use serde_json::json;

// Exercise private curator gates in the same integration target as engine writes.
#[allow(dead_code)]
#[path = "../server/src/reconcile.rs"]
mod reconcile;

async fn engine() -> (tempfile::TempDir, MemoryEngine) {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("memory.db"))
        .build()
        .await
        .unwrap();
    for (name, role) in [
        ("主驾", MemberRole::Self_),
        ("老公", MemberRole::Spouse),
        ("老婆", MemberRole::Spouse),
    ] {
        engine
            .upsert_member(
                None,
                HouseholdMember {
                    member_id: name.into(),
                    display_name: name.into(),
                    role,
                    aliases: vec![],
                    age_band: None,
                },
            )
            .await
            .unwrap();
    }
    (dir, engine)
}

fn fact(owner: &str, text: &str, tag: &str) -> RememberInput {
    let mut input = RememberInput::episodic(text);
    input.skip_consolidation = true;
    // speaker/member_id remains only a hint; resolved metadata must win.
    input.member_id = Some("主驾".into());
    input.metadata = json!({"attribution":"manual", "subjects":[{"member_id":owner,"confidence":1.0}],
        "scope":"members", "tag":tag});
    input
}

#[tokio::test]
async fn confirmed_owner_survives_subjectless_content_and_speaker_hint() {
    let (_dir, engine) = engine().await;
    let id = engine
        .remember(fact("老公", "喜欢吃肉", "饮食"))
        .await
        .unwrap();
    let row = engine.get(id).await.unwrap().unwrap();
    assert_eq!(row.subject_ids, vec!["老公"]);
    assert_eq!(row.metadata["attribution"], "manual");
    assert_eq!(row.tag, "饮食");
    assert_eq!(row.metadata["domains"][0]["id"], "diet");
}

#[tokio::test]
async fn two_owners_with_identical_text_coexist_and_recall_independently() {
    let (_dir, engine) = engine().await;
    let a = engine
        .remember(fact("老公", "喜欢吃肉", "饮食"))
        .await
        .unwrap();
    let b = engine
        .remember(fact("老婆", "喜欢吃肉", "饮食"))
        .await
        .unwrap();
    assert_ne!(a, b);
    for (owner, id, other) in [("老公", a, b), ("老婆", b, a)] {
        let found = engine
            .recall(RecallQuery {
                text: "喜欢吃肉".into(),
                member_id: Some(owner.into()),
                domains: vec!["diet".into()],
                skip_graph: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(found.hits.iter().any(|h| h.memory.id == id));
        assert!(found.hits.iter().all(|h| h.memory.id != other));
    }
}

#[tokio::test]
async fn english_domain_queries_match_validated_chinese_tags() {
    let (_dir, engine) = engine().await;
    // The tag is intentional: there are no medical keywords to rescue it.
    let id = engine
        .remember(fact("老公", "每周三下午两点固定安排", "医疗健康"))
        .await
        .unwrap();
    let row = engine.get(id).await.unwrap().unwrap();
    assert_eq!(row.tag, "医疗健康");
    assert!(row.metadata["domains"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["id"] == "medical"));
    let found = engine
        .recall(RecallQuery {
            text: "每周三下午两点固定安排".into(),
            domains: vec!["medical".into()],
            skip_graph: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(found.hits.iter().any(|h| h.memory.id == id));
}
