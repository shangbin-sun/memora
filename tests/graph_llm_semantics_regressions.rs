//! LLM semantic decisions are not reclassified by keyword filters.
use memora::graph::{kg_prompt_with_context, parse_and_validate_llm_json};
use memora::{HouseholdMember, MemberRole};
use serde_json::{json, Value};

fn roster() -> Vec<HouseholdMember> {
    vec![HouseholdMember {
        member_id: "child1".into(),
        display_name: "小航".into(),
        role: MemberRole::Child,
        aliases: vec![],
        age_band: None,
    }]
}

#[test]
fn birthday_graph_keeps_llm_date_and_original_evidence_despite_gift_question() {
    let source = "小航下个月6号就要过生日了，请问送什么礼物好呢";
    let raw = json!({"entities":[
        {"temp_id":"p", "name":"小航", "type":"person", "member_id":"child1"},
        {"temp_id":"b", "name":"生日", "type":"event"}],
        "triples":[{"subject":"p", "relation":"有生日", "object":"b",
            "object_type":"event", "evidence_text":"小航下个月6号就要过生日了",
            "attrs":{"date":"2026-11-06", "original_time":"下个月6号", "calendar":"unspecified"}}],
        "pending":[], "complete":true})
    .to_string();
    let output = parse_and_validate_llm_json(&raw, source, &roster()).unwrap();
    assert_eq!(output.triples.len(), 1);
    assert_eq!(output.triples[0].attrs["date"], "2026-11-06");
    assert_eq!(output.triples[0].evidence_text, "小航下个月6号就要过生日了");
    assert!(output.pending.is_empty());
}

#[test]
fn legacy_llm_normalized_date_need_not_be_literal_evidence_substring() {
    let source = "小航下个月6号就要过生日了，请问送什么礼物好呢";
    let raw = json!({"facts":[{"subject_member_id":"child1", "subject_name":"小航",
        "relation":"birthday_at", "object_type":"concept", "object_name":"2026-11-06",
        "evidence_text":"小航下个月6号就要过生日了", "attrs":{"calendar":"unspecified"}}]})
    .to_string();
    let output = parse_and_validate_llm_json(&raw, source, &roster()).unwrap();
    assert_eq!(output.facts.len(), 1);
    assert_eq!(output.triples.len(), 1);
    assert_eq!(output.facts[0].object_name, "2026-11-06");
}

#[test]
fn model_supplied_semantic_decision_not_rejected_by_question_or_label_keywords() {
    let source = "‘晚’是小航的正式昵称，他不喜欢别人读错名字，怎么办？";
    let raw = json!({"entities":[
        {"temp_id":"child1", "name":"小航", "type":"person", "member_id":"child1", "aliases":["晚"]},
        {"temp_id":"alias", "name":"晚", "type":"concept"}],
        "triples":[{"subject":"child1", "relation":"有属性", "object":"alias",
            "evidence_text":"‘晚’是小航的正式昵称", "attrs":{"alias_kind":"nickname"}}],
        "complete":true, "pending":[]}).to_string();
    let output = parse_and_validate_llm_json(&raw, source, &roster()).unwrap();
    assert_eq!(output.entities.len(), 2);
    assert_eq!(output.entities[0].aliases, ["晚"]);
    assert_eq!(output.triples.len(), 1);
}

#[test]
fn structural_evidence_still_fail_closed_and_unknown_member_id_keeps_person_by_name() {
    let raw = json!({"entities":[{"temp_id":"p", "name":"小航", "type":"person", "member_id":"forged"}],
        "triples":[{"subject":"p", "relation":"有生日", "object":"生日", "evidence_text":"不存在的证据"}],
        "pending":[], "complete":true}).to_string();
    let output = parse_and_validate_llm_json(&raw, "小航下个月6号就要过生日了", &roster()).unwrap();
    // Forged member_id is dropped; content name uniquely merges onto roster.
    assert_eq!(output.entities.len(), 1);
    assert_eq!(output.entities[0].name, "小航");
    assert_eq!(output.entities[0].member_id.as_deref(), Some("child1"));
    assert!(output.triples.is_empty());
    assert_eq!(output.pending.len(), 1);
}

#[test]
fn active_prompt_uses_versioned_rules_and_host_time_json_without_inventing_missing_anchor() {
    let roster_json = serde_json::to_string(&roster()).unwrap();
    let prompt = kg_prompt_with_context(
        "小航下个月6号就要过生日了，请问送什么礼物好呢",
        &roster_json,
        None,
        Some(1_792_899_000_000),
    );
    assert!(prompt.contains(memora::prompt_latest().kg_extract_system));
    assert!(prompt.contains("有生日"));
    assert!(prompt.contains("2026-11-06"));
    let payload: Value = serde_json::from_str(prompt.rsplit_once("：\n").unwrap().1).unwrap();
    assert_eq!(payload["message_at_ms"], 1_792_899_000_000i64);
    assert!(payload["message_at_china"]
        .as_str()
        .unwrap()
        .ends_with("+08:00"));
    let no_time = kg_prompt_with_context("生日在下个月6号", &roster_json, None, None);
    let payload: Value = serde_json::from_str(no_time.rsplit_once("：\n").unwrap().1).unwrap();
    assert!(payload["message_at_ms"].is_null());
    assert!(payload["message_at_china"].is_null());
}
