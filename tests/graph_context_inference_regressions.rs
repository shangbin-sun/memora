//! Production parser/ontology regressions; model decisions are fixtures, not rules.
//! No network/model call, generated inverse edge, or keyword entity deny-list.
use memora::graph::ontology::{
    alias_kind_for_relation, normalize_assertion_relation, normalize_relation,
};
use memora::graph::{
    kg_prompt_with_context, parse_and_validate_llm_json, EntityType, GraphExtractOutput,
};
use memora::{HouseholdMember, MemberRole};
use serde_json::{json, Value};

fn member(id: &str, name: &str, role: MemberRole) -> HouseholdMember {
    HouseholdMember {
        member_id: id.into(),
        display_name: name.into(),
        role,
        aliases: vec![],
        age_band: None,
    }
}
fn roster() -> Vec<HouseholdMember> {
    vec![
        member("child", "林子航", MemberRole::Child),
        member("father", "林志远", MemberRole::Other),
        member("mother", "宋娜娜", MemberRole::Other),
    ]
}
fn people() -> Value {
    json!([
        {"temp_id":"p1","name":"林子航","type":"person","member_id":"child"},
        {"temp_id":"p2","name":"林志远","type":"person","member_id":"father"},
        {"temp_id":"p3","name":"宋娜娜","type":"person","member_id":"mother"}
    ])
}
fn parse(source: &str, entities: Value, triples: Value) -> GraphExtractOutput {
    parse_and_validate_llm_json(
        &json!({"entities":entities,"triples":triples,"pending":[],"complete":true}).to_string(),
        source,
        &roster(),
    )
    .unwrap()
}
fn edge(subject: &str, relation: &str, object: &str, evidence: &str, attrs: Value) -> Value {
    json!({"subject":subject,"relation":relation,"object":object,"evidence_text":evidence,"attrs":attrs})
}
fn assert_pending(output: &GraphExtractOutput, fragment: &str) {
    assert!(
        output.triples.is_empty(),
        "unexpected accepted edge: {:?}",
        output.triples
    );
    assert!(
        output
            .pending
            .iter()
            .any(|item| item.reason.contains(fragment)),
        "missing pending reason {fragment}: {:?}",
        output.pending
    );
}
#[test]
fn duplicate_temp_id_with_same_identity_is_collapsed_before_ingest() {
    let source = "林子航的小名航航，又叫小航，他的爸爸是林志远。";
    let mut entities = people();
    entities[0]["aliases"] = json!(["航航"]);
    let mut duplicate = entities[0].clone();
    duplicate["aliases"] = json!(["小航", "航航"]);
    entities.as_array_mut().unwrap().push(duplicate);
    let output = parse(
        source,
        entities,
        json!([edge("p1", "爸爸", "p2", source, json!({}))]),
    );
    assert!(output.pending.is_empty());
    assert_eq!(output.entities.len(), 3);
    assert_eq!(output.entities[0].aliases, ["航航", "小航"]);
    assert_eq!(output.triples.len(), 1);
    assert_eq!(output.triples[0].object, "p2");
}

#[test]
fn conflicting_duplicate_temp_id_cannot_rebind_a_family_endpoint() {
    let source = "林子航的爸爸是林志远，妈妈是宋娜娜。";
    let mut entities = people();
    let mut conflicting = entities[2].clone();
    conflicting["temp_id"] = json!("p2");
    entities.as_array_mut().unwrap().push(conflicting);
    let error = parse_and_validate_llm_json(
        &json!({"entities":entities,
        "triples":[edge("p1","爸爸","p2",source,json!({}))],"pending":[],"complete":true})
        .to_string(),
        source,
        &roster(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("ambiguous duplicate entity temp_id"));
}

const CONTEXT: &str = "林子航说：爸爸叫林志远。林志远说：宋娜娜是我老婆，她也是林子航的妈妈。";
const FIRST: &str = "林子航说：爸爸叫林志远";
const SECOND: &str = "林志远说：宋娜娜是我老婆，她也是林子航的妈妈";
fn inference_attrs() -> Value {
    json!({"assertion_basis":"context_inference","inference_note":"两句共同明确三人的身份，第二句由林志远说明配偶及孩子母亲。","evidence_spans":[FIRST,SECOND],"source_scope":"current_input"})
}

#[test]
fn chinese_specific_predicates_and_all_model_supplied_family_directions_are_preserved() {
    let source = "林子航的小名是航航，爸爸是林志远，妈妈是宋娜娜。林子航是林志远和宋娜娜的儿子，林志远的老婆是宋娜娜。";
    let mut entities = people();
    entities
        .as_array_mut()
        .unwrap()
        .push(json!({"temp_id":"alias","name":"航航","type":"concept"}));
    let expected = [
        ("p1", "小名", "alias"),
        ("p1", "爸爸", "p2"),
        ("p1", "妈妈", "p3"),
        ("p2", "儿子", "p1"),
        ("p3", "儿子", "p1"),
        ("p2", "老婆", "p3"),
        ("p3", "老公", "p2"),
    ];
    let triples: Vec<_> = expected
        .iter()
        .map(|(s, r, o)| edge(s, r, o, source, json!({"assertion_basis":"direct"})))
        .collect();
    let output = parse(source, entities, json!(triples));
    assert!(output.pending.is_empty());
    assert_eq!(output.triples.len(), expected.len());
    for (triple, (s, r, o)) in output.triples.iter().zip(expected) {
        assert_eq!(
            (&*triple.subject, &*triple.relation, &*triple.object),
            (s, r, o)
        );
    }
}

#[test]
fn a_single_model_supplied_parent_edge_does_not_create_inverse_or_spouse_edges() {
    let source = "林子航的爸爸是林志远，妈妈是宋娜娜。";
    let output = parse(
        source,
        people(),
        json!([edge("p1", "爸爸", "p2", source, json!({}))]),
    );
    assert!(output.pending.is_empty());
    assert_eq!(output.triples.len(), 1);
    assert_eq!(
        (
            &*output.triples[0].subject,
            &*output.triples[0].relation,
            &*output.triples[0].object
        ),
        ("p1", "爸爸", "p2")
    );
}

#[test]
fn cross_sentence_context_inference_retains_all_current_source_provenance() {
    let attrs = inference_attrs();
    let output = parse(
        CONTEXT,
        people(),
        json!([edge("p1", "妈妈", "p3", SECOND, attrs.clone())]),
    );
    assert!(output.pending.is_empty());
    assert_eq!(output.triples.len(), 1);
    assert_eq!(output.triples[0].relation, "妈妈");
    assert_eq!(output.triples[0].evidence_text, SECOND);
    assert_eq!(output.triples[0].attrs, attrs);
}

#[test]
fn inference_spans_must_each_be_continuous_current_source_and_include_primary_evidence() {
    for (spans, expected) in [
        (json!([SECOND, "林志远去年已离婚"]), "not found in source"),
        (
            json!([SECOND, "林子航说宋娜娜是我老婆"]),
            "not found in source",
        ),
        (json!([FIRST]), "evidence_text must be one"),
        (json!([SECOND, ""]), "invalid evidence span"),
        (json!([SECOND, 42]), "invalid evidence span"),
    ] {
        let mut attrs = inference_attrs();
        attrs["evidence_spans"] = spans;
        let output = parse(
            CONTEXT,
            people(),
            json!([edge("p1", "妈妈", "p3", SECOND, attrs)]),
        );
        assert_pending(&output, expected);
    }
}

#[test]
fn inference_missing_note_unknown_basis_and_missing_spans_are_pending() {
    let mut missing_note = inference_attrs();
    missing_note
        .as_object_mut()
        .unwrap()
        .remove("inference_note");
    let mut blank_note = inference_attrs();
    blank_note["inference_note"] = json!("  ");
    let mut missing_spans = inference_attrs();
    missing_spans
        .as_object_mut()
        .unwrap()
        .remove("evidence_spans");
    let mut empty_spans = inference_attrs();
    empty_spans["evidence_spans"] = json!([]);
    let mut unknown = inference_attrs();
    unknown["assertion_basis"] = json!("external_knowledge");
    let mut wrong_type = inference_attrs();
    wrong_type["assertion_basis"] = json!(true);
    for (attrs, expected) in [
        (missing_note, "inference_note"),
        (blank_note, "inference_note"),
        (missing_spans, "evidence_spans"),
        (empty_spans, "evidence_spans"),
        (unknown, "invalid assertion_basis"),
        (wrong_type, "invalid assertion_basis"),
    ] {
        let output = parse(
            CONTEXT,
            people(),
            json!([edge("p1", "妈妈", "p3", SECOND, attrs)]),
        );
        assert_pending(&output, expected);
    }
}

#[test]
fn direct_and_legacy_omitted_basis_remain_compatible_but_supplied_spans_are_validated() {
    for attrs in [json!({}), json!({"assertion_basis":"direct"})] {
        let output = parse(
            CONTEXT,
            people(),
            json!([edge("p1", "爸爸", "p2", FIRST, attrs)]),
        );
        assert!(output.pending.is_empty());
        assert_eq!(output.triples.len(), 1);
    }
    let output = parse(
        CONTEXT,
        people(),
        json!([edge(
            "p1",
            "爸爸",
            "p2",
            FIRST,
            json!({"assertion_basis":"direct","evidence_spans":["并不存在"]})
        )]),
    );
    assert_pending(&output, "not found in source");
}

#[test]
fn undefined_temp_id_and_name_instead_of_declared_id_cannot_create_nodes_or_edges() {
    let source = "林子航喜欢公共交通。";
    let entities = json!([{"temp_id":"p","name":"林子航","type":"person","member_id":"child"},{"temp_id":"transport","name":"公共交通","type":"concept"}]);
    for object in ["e2", "公共交通"] {
        let output = parse(
            source,
            entities.clone(),
            json!([edge("p", "喜欢", object, source, json!({}))]),
        );
        assert_pending(&output, "declared temp_id");
        assert!(!output.entities.iter().any(|entity| entity.name == "e2"));
    }
}

#[test]
fn copied_temp_id_name_without_source_support_is_pending_and_cannot_leave_an_edge() {
    let source = "林子航喜欢公共交通。";
    let output = parse(
        source,
        json!([{"temp_id":"p","name":"林子航","type":"person","member_id":"child"},{"temp_id":"e2","name":"e2","type":"concept"}]),
        json!([edge("p", "喜欢", "e2", source, json!({}))]),
    );
    assert_pending(&output, "identifier used as an unsupported name");
    assert_eq!(output.entities.len(), 1);
    assert!(output
        .pending
        .iter()
        .any(|item| item.reason.contains("declared temp_id")));
}

#[test]
fn real_e2_designation_and_public_transport_are_not_rejected_by_keyword_or_id_pattern() {
    let source = "林子航使用E2型号设备，也喜欢公共交通。";
    let output = parse(
        source,
        json!([{"temp_id":"p","name":"林子航","type":"person","member_id":"child"},{"temp_id":"E2","name":"E2","type":"concept"},{"temp_id":"transport","name":"公共交通","type":"concept"}]),
        json!([
            edge("p", "使用型号", "E2", source, json!({})),
            edge("p", "喜欢", "transport", source, json!({}))
        ]),
    );
    assert!(output.pending.is_empty());
    assert_eq!(output.entities.len(), 3);
    assert_eq!(output.triples.len(), 2);
    assert!(output.entities.iter().any(|entity| entity.name == "E2"));
    assert!(output
        .entities
        .iter()
        .any(|entity| entity.name == "公共交通"));
}

#[test]
fn shared_attribution_ids_cannot_be_people_but_explicit_real_family_entities_are_allowed() {
    for id in ["公共", "__public__"] {
        let mut members = roster();
        members.push(member(id, "公共", MemberRole::Other));
        let raw = json!({"entities":[{"temp_id":"scope","name":"公共","type":"person","member_id":id},{"temp_id":"family","name":"林家","type":"family","member_id":id}],"triples":[],"pending":[],"complete":true});
        let output =
            parse_and_validate_llm_json(&raw.to_string(), "林家是这一家的称呼。", &members)
                .unwrap();
        assert_eq!(output.entities.len(), 1);
        assert_eq!(output.entities[0].name, "林家");
        assert_eq!(output.entities[0].entity_type, EntityType::Family);
        assert!(output.entities[0].member_id.is_none());
        assert!(output
            .pending
            .iter()
            .any(|item| item.reason.contains("shared attribution")));
    }
}

#[test]
fn shared_roster_entries_never_become_prompt_people_or_speaker_anchors() {
    let mut members = roster();
    members.push(member("公共", "公共", MemberRole::Other));
    members.push(member("__public__", "旧公共归属", MemberRole::Other));
    for speaker in ["公共", "__public__"] {
        let prompt = kg_prompt_with_context(
            "林家喜欢公共交通",
            &serde_json::to_string(&members).unwrap(),
            Some(speaker),
            None,
        );
        let data: Value = serde_json::from_str(prompt.rsplit_once("：\n").unwrap().1).unwrap();
        assert_eq!(data["roster"].as_array().unwrap().len(), 3);
        assert!(data["speaker_member_id"].is_null());
        assert_eq!(data["source_text"], "林家喜欢公共交通");
    }
}

#[test]
fn explicit_alias_predicates_and_legacy_subtype_keep_specific_chinese_relation() {
    let source = "林子航的小名是航航。";
    for (relation, attrs, expected) in [
        ("小名", json!({}), "小名"),
        ("nickname", json!({}), "小名"),
        ("有属性", json!({"alias_kind":"nickname"}), "小名"),
        ("属性", json!({"alias_kind":"昵称"}), "小名"),
        ("has_attribute", json!({"alias_kind":"alias"}), "别名"),
        ("has_attribute", json!({"alias_kind":"address"}), "称呼"),
        ("has_attribute", json!({"alias_kind":"name"}), "姓名"),
    ] {
        assert_eq!(normalize_assertion_relation(relation, &attrs), expected);
        let raw = json!({"facts":[{"subject_member_id":"child","subject_name":"林子航","relation":relation,"object_type":"concept","object_name":"航航","evidence_text":source,"attrs":attrs}]});
        let output = parse_and_validate_llm_json(&raw.to_string(), source, &roster()).unwrap();
        assert!(output.pending.is_empty());
        assert_eq!(output.facts.len(), 1);
        assert_eq!(output.triples.len(), 1);
        assert_eq!(output.facts[0].relation, expected);
        assert_eq!(output.triples[0].relation, expected);
    }
    assert_eq!(alias_kind_for_relation("小名"), Some("nickname"));
    assert_eq!(alias_kind_for_relation("别名"), Some("alias"));
    assert_eq!(alias_kind_for_relation("工作单位"), None);
    // An English address may mean a street address, not a person's appellation.
    assert_eq!(alias_kind_for_relation("address"), None);
    assert_eq!(normalize_relation("address"), "address");
}

#[test]
fn known_english_predicates_translate_without_reversing_or_inventing_unknown_relations() {
    let source = "林子航的爸爸是林志远，林子航喜欢和林志远玩。";
    for (relation, expected) in [
        ("father", "爸爸"),
        ("likes", "喜欢"),
        ("unlisted_open_predicate", "unlisted_open_predicate"),
        ("周末陪练", "周末陪练"),
    ] {
        let output = parse(
            source,
            people(),
            json!([edge("p1", relation, "p2", source, json!({}))]),
        );
        assert!(output.pending.is_empty());
        assert_eq!(output.triples.len(), 1);
        let triple = &output.triples[0];
        assert_eq!(
            (&*triple.subject, &*triple.relation, &*triple.object),
            ("p1", expected, "p2")
        );
        assert_eq!(normalize_relation(relation), expected);
    }
}

#[test]
fn empty_relation_is_pending_in_new_and_legacy_protocols_not_generic_related_to() {
    let source = "林子航的小名是航航。";
    let output = parse(
        source,
        people(),
        json!([edge("p1", "  ", "p2", source, json!({}))]),
    );
    assert_pending(&output, "empty relation");
    let raw = json!({"facts":[{"subject_member_id":"child","subject_name":"林子航","relation":"","object_type":"concept","object_name":"航航","evidence_text":source}]});
    let legacy = parse_and_validate_llm_json(&raw.to_string(), source, &roster()).unwrap();
    assert!(legacy.facts.is_empty());
    assert_pending(&legacy, "empty relation");
}

#[test]
fn legacy_context_inference_uses_real_conversion_and_preserves_provenance_or_goes_pending() {
    let attrs = inference_attrs();
    let valid = json!({"subject_member_id":"child","subject_name":"林子航","relation":"妈妈","object_type":"person","object_name":"宋娜娜","evidence_text":SECOND,"attrs":attrs});
    let output = parse_and_validate_llm_json(
        &json!({"facts":[valid.clone()]}).to_string(),
        CONTEXT,
        &roster(),
    )
    .unwrap();
    assert!(output.pending.is_empty());
    assert_eq!(output.facts.len(), 1);
    assert_eq!(output.triples.len(), 1);
    assert_eq!(output.facts[0].attrs, inference_attrs());
    assert_eq!(output.triples[0].attrs, inference_attrs());
    assert_eq!(output.triples[0].object, "宋娜娜");
    assert!(output
        .entities
        .iter()
        .any(|entity| entity.name == "宋娜娜" && entity.entity_type == EntityType::Person));
    let mut invalid = valid;
    invalid["attrs"]["evidence_spans"] = json!([SECOND, "外部历史记录"]);
    let rejected =
        parse_and_validate_llm_json(&json!({"facts":[invalid]}).to_string(), CONTEXT, &roster())
            .unwrap();
    assert!(rejected.facts.is_empty());
    assert_pending(&rejected, "not found in source");
}
