//! Recall presentation regressions through the public local-only API.
//! SQLite fixtures fix creation/update times independently of wall-clock time.
//! No remote embedder, LLM, dependency installation, or toolchain download.

use memora::{
    render_memory_snippet, Config, ContextRole, HouseholdMember, MemberRole, MemoryEngine,
    MemoryId, MemoryService, RecallQuery, RecalledMemory, RememberInput,
};
use rusqlite::{params, Connection};
use serde_json::json;
use std::time::Duration;

const DB_NAME: &str = "recall-display.db";
const CREATION_RFC3339: &str = "2026-10-26T03:20:59Z";
const UPDATE_RFC3339: &str = "2027-02-03T04:05:06Z";
const EXPECTED: &str = "[主驾 · 记录于2026-10-26 11:20] 我有点头疼";

fn timestamp(text: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(text)
        .unwrap()
        .timestamp_millis()
}

async fn build(dir: &tempfile::TempDir, budget: usize) -> MemoryEngine {
    MemoryEngine::builder()
        .db_path(dir.path().join(DB_NAME))
        .config(Config {
            token_budget: budget,
            query_cache_ttl: Duration::from_secs(300),
            ..Default::default()
        })
        .graph_projection_worker(false)
        .build()
        .await
        .unwrap()
}

async fn member(engine: &MemoryEngine, namespace: &str, id: &str, name: &str) {
    engine
        .upsert_member(
            Some(namespace),
            HouseholdMember {
                member_id: id.into(),
                display_name: name.into(),
                role: MemberRole::Self_,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
}

fn input(namespace: &str, text: &str) -> RememberInput {
    let mut input = RememberInput::episodic(text);
    input.namespace = Some(namespace.into());
    input.member_id = Some("driver".into());
    input.importance = 0.9;
    input.confidence = 1.0;
    input.skip_consolidation = true;
    input.metadata = json!({
        "attribution": "manual", "scope": "members", "tag": "其他", "skip_graph": true,
        "subjects": [{"member_id": "driver", "confidence": 1.0}]
    });
    input
}

fn fixed_times(dir: &tempfile::TempDir, id: MemoryId) {
    let connection = Connection::open(dir.path().join(DB_NAME)).unwrap();
    connection.busy_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        connection
            .execute(
                "UPDATE memories SET created_at_ms = ?1, updated_at_ms = ?2 WHERE id = ?3",
                params![
                    timestamp(CREATION_RFC3339),
                    timestamp(UPDATE_RFC3339),
                    id.to_string()
                ],
            )
            .unwrap(),
        1
    );
}

fn query(namespace: &str, text: &str) -> RecallQuery {
    RecallQuery {
        namespace: Some(namespace.into()),
        text: text.into(),
        fts_only: true,
        skip_graph: true,
        ..Default::default()
    }
}

fn estimated_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(2).max(1)
}

#[tokio::test]
async fn fts_hit_and_context_share_member_created_time_without_rewriting_raw_memory() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "default", "driver", "主驾").await;
    let id = engine
        .remember(input("default", "我有点头疼"))
        .await
        .unwrap();
    fixed_times(&dir, id);
    let before = engine.get(id).await.unwrap().unwrap();
    let result = engine.recall(query("default", "头疼")).await.unwrap();
    let hit = result.hits.iter().find(|hit| hit.memory.id == id).unwrap();
    assert_eq!(hit.display_text, EXPECTED);
    assert_eq!(hit.memory.content, before.content);
    assert_eq!(hit.memory.summary, before.summary);
    assert_eq!(hit.memory.metadata, before.metadata);
    assert_eq!(hit.memory.created_at_ms, timestamp(CREATION_RFC3339));
    assert_eq!(hit.memory.updated_at_ms, timestamp(UPDATE_RFC3339));
    let context = result
        .context
        .items
        .iter()
        .find(|item| item.memory_id == id)
        .unwrap();
    assert_eq!(context.text, hit.display_text);
    assert_eq!(context.role, ContextRole::Atomic);
    assert_eq!(context.tokens, estimated_tokens(EXPECTED));
    let after = engine.get(id).await.unwrap().unwrap();
    assert_eq!(after.content, before.content);
    assert_eq!(after.summary, before.summary);
    assert_eq!(after.created_at_ms, before.created_at_ms);
    assert_eq!(after.updated_at_ms, before.updated_at_ms);
    engine.shutdown();
}

#[tokio::test]
async fn summary_is_preferred_and_event_marker_is_display_only_not_creation_time() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "default", "driver", "主驾").await;
    let mut fact = input("default", "原始细节：我有点头疼");
    fact.summary = Some("我有点头疼 ⌈event:2030-01-02⌋".into());
    let id = engine.remember(fact).await.unwrap();
    fixed_times(&dir, id);
    let before = engine.get(id).await.unwrap().unwrap();
    let result = engine.recall(query("default", "原始细节")).await.unwrap();
    let hit = result.hits.iter().find(|hit| hit.memory.id == id).unwrap();
    assert_eq!(hit.display_text, EXPECTED);
    assert_eq!(hit.memory.summary, before.summary);
    assert!(hit
        .memory
        .summary
        .as_deref()
        .unwrap()
        .contains("⌈event:2030-01-02⌋"));
    let item = result
        .context
        .items
        .iter()
        .find(|item| item.memory_id == id)
        .unwrap();
    assert_eq!(item.text, EXPECTED);
    assert_eq!(
        engine.get(id).await.unwrap().unwrap().summary,
        before.summary
    );
    engine.shutdown();
}

#[tokio::test]
async fn member_rename_invalidates_cached_hit_and_context_not_durable_content_or_time() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "default", "driver", "主驾").await;
    let id = engine
        .remember(input("default", "我有点头疼"))
        .await
        .unwrap();
    fixed_times(&dir, id);
    let q = query("default", "头疼");
    let first = engine.recall(q.clone()).await.unwrap();
    assert_eq!(
        first
            .hits
            .iter()
            .find(|hit| hit.memory.id == id)
            .unwrap()
            .display_text,
        EXPECTED
    );
    // Same query remains in the long-lived local cache until the roster mutation.
    assert_eq!(
        engine.recall(q.clone()).await.unwrap().hits[0].display_text,
        first.hits[0].display_text
    );
    member(&engine, "default", "driver", "驾驶员小李").await;
    let current = engine.recall(q).await.unwrap();
    let expected = "[驾驶员小李 · 记录于2026-10-26 11:20] 我有点头疼";
    let hit = current.hits.iter().find(|hit| hit.memory.id == id).unwrap();
    assert_eq!(hit.display_text, expected);
    let item = current
        .context
        .items
        .iter()
        .find(|item| item.memory_id == id)
        .unwrap();
    assert_eq!(item.text, expected);
    assert_eq!(hit.memory.content, "我有点头疼");
    assert_eq!(hit.memory.subject_ids, ["driver"]);
    assert_eq!(hit.memory.created_at_ms, timestamp(CREATION_RFC3339));
    engine
        .remove_member(Some("default"), "driver")
        .await
        .unwrap();
    let deleted = engine.recall(query("default", "头疼")).await.unwrap();
    assert_eq!(
        deleted
            .hits
            .iter()
            .find(|hit| hit.memory.id == id)
            .unwrap()
            .display_text,
        "[driver · 记录于2026-10-26 11:20] 我有点头疼"
    );
    engine.shutdown();
}

#[tokio::test]
async fn pinned_stm_context_uses_durable_creation_not_synthetic_recall_now() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "default", "driver", "主驾").await;
    let mut fact = input("default", "我有点头疼");
    fact.pin_to_stm = true;
    fact.session_id = Some("trip-session".into());
    let id = engine.remember(fact).await.unwrap();
    fixed_times(&dir, id);
    let result = engine
        .recall(RecallQuery {
            session_id: Some("trip-session".into()),
            ..query("default", "头疼")
        })
        .await
        .unwrap();
    let stm = result
        .stm_hits
        .iter()
        .find(|memory| memory.id == id)
        .unwrap();
    assert_eq!(stm.created_at_ms, timestamp(CREATION_RFC3339));
    assert_eq!(stm.updated_at_ms, timestamp(UPDATE_RFC3339));
    assert_eq!(stm.content, "我有点头疼");
    let item = result
        .context
        .items
        .iter()
        .find(|item| item.memory_id == id && item.role == ContextRole::StmFact)
        .unwrap();
    assert_eq!(item.text, EXPECTED);
    assert_eq!(item.tokens, estimated_tokens(EXPECTED));
    engine.shutdown();
}

#[tokio::test]
async fn context_budget_counts_member_and_timestamp_prefix_not_only_body() {
    let body_budget = estimated_tokens("我有点头疼");
    assert!(body_budget < estimated_tokens(EXPECTED));
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, body_budget).await;
    member(&engine, "default", "driver", "主驾").await;
    let id = engine
        .remember(input("default", "我有点头疼"))
        .await
        .unwrap();
    fixed_times(&dir, id);
    let result = engine.recall(query("default", "头疼")).await.unwrap();
    assert_eq!(
        result
            .hits
            .iter()
            .find(|hit| hit.memory.id == id)
            .unwrap()
            .display_text,
        EXPECTED
    );
    assert!(
        result.context.items.is_empty(),
        "body fits but prefixed snippet does not"
    );
    assert_eq!(result.context.estimated_tokens, 0);
    assert_eq!(result.context.budget, body_budget);
    engine.shutdown();

    let exact_dir = tempfile::tempdir().unwrap();
    let exact_budget = estimated_tokens(EXPECTED);
    let engine = build(&exact_dir, exact_budget).await;
    member(&engine, "default", "driver", "主驾").await;
    let id = engine
        .remember(input("default", "我有点头疼"))
        .await
        .unwrap();
    fixed_times(&exact_dir, id);
    let exact = engine.recall(query("default", "头疼")).await.unwrap();
    assert_eq!(exact.context.items.len(), 1);
    assert_eq!(exact.context.items[0].text, EXPECTED);
    assert_eq!(exact.context.items[0].tokens, exact_budget);
    assert_eq!(exact.context.estimated_tokens, exact_budget);
    assert_eq!(exact.context.budget, exact_budget);
    engine.shutdown();
}

#[tokio::test]
async fn namespace_local_roster_does_not_leak_names_for_same_member_id() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "A", "driver", "甲家庭主驾").await;
    member(&engine, "B", "driver", "乙家庭主驾").await;
    let a = engine.remember(input("A", "我有点头疼")).await.unwrap();
    let b = engine.remember(input("B", "我有点头疼")).await.unwrap();
    fixed_times(&dir, a);
    fixed_times(&dir, b);
    for (namespace, id, other, name) in [("A", a, b, "甲家庭主驾"), ("B", b, a, "乙家庭主驾")]
    {
        let result = engine.recall(query(namespace, "头疼")).await.unwrap();
        let expected = format!("[{name} · 记录于2026-10-26 11:20] 我有点头疼");
        assert_eq!(
            result
                .hits
                .iter()
                .find(|hit| hit.memory.id == id)
                .unwrap()
                .display_text,
            expected
        );
        assert!(result.hits.iter().all(|hit| hit.memory.id != other));
        assert!(result
            .context
            .items
            .iter()
            .all(|item| item.memory_id != other));
        assert!(result
            .context
            .items
            .iter()
            .filter(|item| item.memory_id == id)
            .all(|item| item.text == expected));
    }
    engine.shutdown();
}

#[tokio::test]
async fn exported_renderer_subject_precedence_metadata_fallback_and_legacy_hit_default() {
    let dir = tempfile::tempdir().unwrap();
    let engine = build(&dir, 2000).await;
    member(&engine, "default", "driver", "主驾").await;
    member(&engine, "default", "child", "小航").await;
    let id = engine
        .remember(input("default", "我有点头疼"))
        .await
        .unwrap();
    fixed_times(&dir, id);
    let mut record = engine.get(id).await.unwrap().unwrap();
    let roster = engine.list_members(Some("default")).await.unwrap();
    record.metadata = json!({"subjects": [{"member_id": "child", "confidence": 1.0}]});
    // Indexed subjects take precedence over unrelated metadata subjects.
    assert_eq!(render_memory_snippet(&record, &roster), EXPECTED);
    record.subject_ids.clear();
    assert_eq!(
        render_memory_snippet(&record, &roster),
        "[小航 · 记录于2026-10-26 11:20] 我有点头疼"
    );
    record.metadata = json!({});
    assert_eq!(
        render_memory_snippet(&record, &roster),
        "[未归属 · 记录于2026-10-26 11:20] 我有点头疼"
    );
    record.subject_ids = vec![
        "driver".into(),
        "child".into(),
        "missing-member".into(),
        "driver".into(),
    ];
    assert_eq!(
        render_memory_snippet(&record, &roster),
        "[主驾、小航、missing-member · 记录于2026-10-26 11:20] 我有点头疼"
    );
    let result = engine.recall(query("default", "头疼")).await.unwrap();
    let mut legacy =
        serde_json::to_value(result.hits.iter().find(|hit| hit.memory.id == id).unwrap()).unwrap();
    legacy.as_object_mut().unwrap().remove("display_text");
    let legacy_hit: RecalledMemory = serde_json::from_value(legacy).unwrap();
    assert!(legacy_hit.display_text.is_empty());
    assert_eq!(legacy_hit.memory.content, "我有点头疼");
    engine.shutdown();
}
