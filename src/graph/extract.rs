//! Deterministic heuristic extractor for family KG facts (MVP).
//!
//! One ingest call yields **n** person-centric relation pairs (`n` may be 0, 1, or
//! many). Heuristics accumulate every matched pattern; they never collapse a rich
//! utterance into a single summary edge.
//!
//! Hosts may later inject LLM extractors; this path guarantees controlled types
//! and does not invent amounts / doses / institution full names.

use regex::Regex;
use serde_json::json;

use crate::household::HouseholdMember;

use super::ontology::{
    default_privacy, EntityType, FactPolarity, FactStatus, PrivacyLevel, RelationType,
};
use super::types::{ExtractedEntity, ExtractedFact, ExtractedTriple};

/// Pronouns / host placeholders that must not become graph person nodes.
pub fn is_meaningless_person_label(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return true;
    }
    matches!(
        s,
        "我" | "本人"
            | "自己"
            | "咱"
            | "咱们"
            | "他"
            | "她"
            | "它"
            | "他们"
            | "她们"
            | "它们"
            | "对方"
            | "此人"
            | "该人"
            | "主驾"
            | "主驾驶"
            | "司机"
            | "公共"
            | "大家"
            | "全家"
            | "某人"
            | "有人"
            | "这人"
            | "那人"
            | "这个"
            | "那个"
            | "这"
            | "那"
    ) || s.eq_ignore_ascii_case("me")
        || s.eq_ignore_ascii_case("self")
        || s.eq_ignore_ascii_case("driver")
        || s.eq_ignore_ascii_case("i")
}

/// Names / kinship / titles worth anchoring a person-centric fact
/// (e.g. 爸爸、张三、小张、张老师). 「家庭」is a family node, not a person.
pub fn is_meaningful_person_label(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty() && s != "家庭" && !is_meaningless_person_label(s)
}

/// Single-char particles / negation fragments that must never become entities
/// (e.g. 不、没、晚 from bad spans).
pub fn is_junk_entity_label(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || is_meaningless_person_label(s) {
        return true;
    }
    let mut chars = s.chars();
    let Some(c) = chars.next() else {
        return true;
    };
    if chars.next().is_some() {
        return false;
    }
    matches!(
        c,
        '不' | '没'
            | '无'
            | '晚'
            | '早'
            | '有'
            | '在'
            | '的'
            | '了'
            | '和'
            | '与'
            | '或'
            | '很'
            | '都'
            | '也'
            | '又'
            | '再'
            | '才'
            | '就'
            | '还'
            | '被'
            | '把'
            | '让'
            | '给'
            | '从'
            | '到'
            | '向'
            | '对'
            | '为'
            | '以'
            | '及'
            | '等'
            | '中'
            | '上'
            | '下'
            | '时'
            | '年'
            | '月'
            | '日'
            | '是'
            | '会'
            | '能'
            | '要'
            | '可'
            | '得'
    )
}

fn fact_person_anchors_ok(f: &ExtractedFact) -> bool {
    if f.subject_name != "家庭" {
        if is_meaningless_person_label(&f.subject_name) {
            return false;
        }
        if f.subject_member_id
            .as_deref()
            .is_some_and(is_meaningless_person_label)
        {
            return false;
        }
    }
    if f.object_type == EntityType::Person && is_meaningless_person_label(&f.object_name) {
        return false;
    }
    true
}

pub fn extract_facts(
    text: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> Vec<ExtractedFact> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();

    // Family diet rule: 家里不吃香菜
    if let Some(cap) = re(r"家里不吃([^\s，。；、]+)").captures(text) {
        let obj = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !obj.is_empty() {
            out.push(ExtractedFact {
                subject_member_id: None,
                subject_name: "家庭".into(),
                relation: RelationType::FollowsRule.as_zh_str().into(),
                object_type: EntityType::Rule,
                object_name: format!("不吃{obj}"),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["diet".into(), "household".into()],
                privacy: PrivacyLevel::FamilyShared,
                condition_text: None,
                evidence_text: text.to_string(),
                confidence: 0.9,
                attrs: json!({}),
            });
        }
    }

    // Allergy: polarity + substance from the local clause only (not whole utterance).
    for (member_id, name, substance, polarity, evidence) in
        extract_allergies(text, roster, speaker_member_id)
    {
        out.push(ExtractedFact {
            subject_member_id: member_id,
            subject_name: name,
            relation: RelationType::AllergicTo.as_zh_str().into(),
            object_type: EntityType::Substance,
            object_name: substance,
            polarity,
            status: FactStatus::Active,
            domains: vec!["medical".into()],
            privacy: default_privacy(RelationType::AllergicTo),
            condition_text: None,
            evidence_text: evidence,
            confidence: 0.92,
            attrs: json!({}),
        });
    }

    // Condition: 姥姥有高血压
    for cap in re(r"([\u4e00-\u9fffA-Za-z0-9]{1,12})有([^\s，。；、]{2,12})").captures_iter(text)
    {
        let who = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let cond = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        if !looks_like_condition(cond) {
            continue;
        }
        let (member_id, name) = resolve_person(who, roster, speaker_member_id);
        out.push(ExtractedFact {
            subject_member_id: member_id,
            subject_name: name,
            relation: RelationType::HasCondition.as_zh_str().into(),
            object_type: EntityType::Condition,
            object_name: cond.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["medical".into()],
            privacy: default_privacy(RelationType::HasCondition),
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.85,
            attrs: json!({}),
        });
    }

    // Medication: 每天吃氨氯地平 / 姥姥每天吃氨氯地平，早上一次
    if let Some(cap) = re(r"(?:([\u4e00-\u9fffA-Za-z0-9]{1,12}))?(?:每天|每日)?(?:早上|晚上)?(?:吃|服用)([^\s，。；、0-9]{2,20})").captures(text)
    {
        let who = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let med = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        if looks_like_medication(med) {
            let (member_id, name) = if who.is_empty() {
                resolve_person("我", roster, speaker_member_id)
            } else {
                resolve_person(who, roster, speaker_member_id)
            };
            let mut attrs = json!({});
            if text.contains("早上") {
                attrs["time_of_day"] = json!("morning");
            }
            if text.contains("每天") || text.contains("每日") {
                attrs["frequency"] = json!("daily");
            }
            out.push(ExtractedFact {
                subject_member_id: member_id,
                subject_name: name,
                relation: RelationType::TakesMedication.as_zh_str().into(),
                object_type: EntityType::Medication,
                object_name: med.to_string(),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["medical".into()],
                privacy: default_privacy(RelationType::TakesMedication),
                condition_text: None,
                evidence_text: text.to_string(),
                confidence: 0.88,
                attrs,
            });
        }
    }

    // School: 小航在实验小学… / 爸爸就读实验小学 / 小航在实验小学上学
    let school_cap = re(
        r"(?:([\u4e00-\u9fffA-Za-z0-9]{1,12}))?(?:就读于?|在读于?|上学于?|在)([^\s，。；]{2,20}?(?:小学|中学|学校|幼儿园))",
    )
    .captures(text)
    .or_else(|| {
        re(r"([\u4e00-\u9fffA-Za-z0-9]{1,12})(?:就读|在读|上学)([^\s，。；]{2,20}?(?:小学|中学|学校|幼儿园))")
            .captures(text)
    });
    if let Some(cap) = school_cap {
        let who = cap
            .get(1)
            .map(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| first_person_mention(text, roster))
            .unwrap_or("我");
        let school = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        if !school.is_empty() {
            let (member_id, name) = resolve_person(who, roster, speaker_member_id);
            let mut attrs = json!({});
            if let Some(g) = re(r"([一二三四五六七八九]年级|\d年级)").captures(text) {
                attrs["grade"] = json!(g.get(1).map(|m| m.as_str()).unwrap_or(""));
            }
            if let Some(c) = re(r"([一二三四五六七八九十\d]+班)").captures(text) {
                attrs["class"] = json!(c.get(1).map(|m| m.as_str()).unwrap_or(""));
            }
            out.push(ExtractedFact {
                subject_member_id: member_id,
                subject_name: name,
                relation: RelationType::AttendsSchool.as_zh_str().into(),
                object_type: EntityType::School,
                object_name: school.to_string(),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["education".into()],
                privacy: PrivacyLevel::FamilyShared,
                condition_text: None,
                evidence_text: text.to_string(),
                confidence: 0.9,
                attrs,
            });
        }
    }

    // Course: 上钢琴课 / 暂停钢琴课 / 不再上钢琴课
    if let Some(cap) = re(r"(?:([\u4e00-\u9fffA-Za-z0-9]{1,12}))?(?:因为[^\s，。；]{0,12})?(暂停|恢复|不再上|不上|上)([^\s，。；]{1,12}课)").captures(text)
    {
        let who = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let action = cap.get(2).map(|m| m.as_str()).unwrap_or("上");
        let course = cap.get(3).map(|m| m.as_str()).unwrap_or("");
        let (member_id, name) = if who.is_empty() {
            // Try leading name before 因为
            if let Some(c2) = re(r"^([\u4e00-\u9fffA-Za-z0-9]{1,12})").captures(text) {
                resolve_person(c2.get(1).map(|m| m.as_str()).unwrap_or("我"), roster, speaker_member_id)
            } else {
                resolve_person("我", roster, speaker_member_id)
            }
        } else {
            resolve_person(who, roster, speaker_member_id)
        };
        let (status, polarity) = match action {
            "暂停" => (FactStatus::Paused, FactPolarity::Affirmative),
            "不再上" | "不上" => (FactStatus::Ended, FactPolarity::Negative),
            "恢复" | "上" => (FactStatus::Active, FactPolarity::Affirmative),
            _ => (FactStatus::Active, FactPolarity::Affirmative),
        };
        let condition = if text.contains("因为") {
            re(r"因为([^\s，。；]+)").captures(text).map(|c| {
                c.get(1)
                    .map(|m| m.as_str().trim_end_matches("暂停").to_string())
                    .unwrap_or_default()
            })
        } else {
            None
        };
        out.push(ExtractedFact {
            subject_member_id: member_id,
            subject_name: name,
            relation: RelationType::EnrolledIn.as_zh_str().into(),
            object_type: EntityType::Course,
            object_name: course.to_string(),
            polarity,
            status,
            domains: vec!["education".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: condition.filter(|s| !s.is_empty()),
            evidence_text: text.to_string(),
            confidence: 0.88,
            attrs: json!({}),
        });
    }

    // Mortgage / loan: 房贷每月8号从我招行卡扣9200
    if text.contains("房贷") || text.contains("车贷") {
        let liability = if text.contains("车贷") {
            "车贷"
        } else {
            "房贷"
        };
        let (member_id, name) = resolve_person("我", roster, speaker_member_id);
        let ended = text.contains("还清") || text.contains("结清");
        out.push(ExtractedFact {
            subject_member_id: member_id.clone(),
            subject_name: name.clone(),
            relation: RelationType::Owes.as_zh_str().into(),
            object_type: EntityType::Liability,
            object_name: liability.into(),
            polarity: FactPolarity::Affirmative,
            status: if ended {
                FactStatus::Ended
            } else {
                FactStatus::Active
            },
            domains: vec!["finance".into()],
            privacy: default_privacy(RelationType::Owes),
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.9,
            attrs: json!({}),
        });
        if let Some(cap) = re(r"(?:从)?(?:我的?)?([^\s，。；]{1,12}卡)").captures(text) {
            let account = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            out.push(ExtractedFact {
                subject_member_id: member_id,
                subject_name: name,
                relation: RelationType::Owns.as_zh_str().into(),
                object_type: EntityType::Account,
                object_name: account.to_string(),
                polarity: FactPolarity::Affirmative,
                status: FactStatus::Active,
                domains: vec!["finance".into()],
                privacy: default_privacy(RelationType::Owns),
                condition_text: None,
                evidence_text: text.to_string(),
                confidence: 0.85,
                attrs: json!({}),
            });
        }
    }

    // Likes with optional condition: 喜欢辣 / 胃不舒服时不吃辣
    if let Some(cap) =
        re(r"(?:([\u4e00-\u9fffA-Za-z0-9]{1,12}))?喜欢([^\s，。；]{1,12})").captures(text)
    {
        let who = cap.get(1).map(|m| m.as_str()).unwrap_or("我");
        let obj = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        // Prefer named person; otherwise household-shared rule node.
        let (member_id, name) = resolve_person_or_family(who, roster, speaker_member_id);
        out.push(ExtractedFact {
            subject_member_id: member_id,
            subject_name: name,
            relation: RelationType::Likes.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: obj.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["diet".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: Some("通常情况".into()),
            evidence_text: text.to_string(),
            confidence: 0.8,
            attrs: json!({}),
        });
    }
    if let Some(cap) =
        re(r"((?:[\u4e00-\u9fffA-Za-z0-9]{1,12})?时)不吃([^\s，。；]{1,12})").captures(text)
    {
        let cond = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let obj = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let who = first_person_mention(text, roster).unwrap_or("我");
        let (member_id, name) = resolve_person_or_family(who, roster, speaker_member_id);
        out.push(ExtractedFact {
            subject_member_id: member_id,
            subject_name: name,
            relation: RelationType::FollowsRule.as_zh_str().into(),
            object_type: EntityType::Rule,
            object_name: format!("不吃{obj}"),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["diet".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: Some(cond.to_string()),
            evidence_text: text.to_string(),
            confidence: 0.85,
            attrs: json!({}),
        });
    }

    out.extend(extract_family_and_aliases(text, roster, speaker_member_id));

    // Person-centric: drop pronoun / placeholder subjects (我/主驾/他…).
    // Unscoped household diet rules may keep subject「家庭」.
    out.retain(fact_person_anchors_ok);

    // Deduplicate by relation+object+subject+condition+polarity+status
    let mut seen = std::collections::HashSet::new();
    out.retain(|f| {
        let key = format!(
            "{}|{}|{}|{}|{}|{:?}",
            f.subject_name,
            f.relation,
            f.object_name,
            f.condition_text.as_deref().unwrap_or(""),
            f.polarity.as_str(),
            f.status
        );
        seen.insert(key)
    });
    out
}

/// Heuristic extract as entities + open-relation triples.
pub fn extract_graph(
    text: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> (Vec<ExtractedEntity>, Vec<ExtractedTriple>) {
    let facts = extract_facts(text, roster, speaker_member_id);
    facts_to_entities_triples(&facts)
}

fn facts_to_entities_triples(
    facts: &[ExtractedFact],
) -> (Vec<ExtractedEntity>, Vec<ExtractedTriple>) {
    let mut entities = Vec::new();
    let mut triples = Vec::new();
    let mut seen = std::collections::HashSet::new();
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
        } else if is_meaningful_person_label(&f.subject_name) && seen.insert(f.subject_name.clone())
        {
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
        // Only explicit, unconditional current aliases describe the person hub.
        // Ordinary attributes and negative/planned nicknames remain facts.
        if RelationType::parse_known(&f.relation) == Some(RelationType::HasAttribute)
            && f.object_type == EntityType::Concept
            && f.polarity == FactPolarity::Affirmative
            && f.status == FactStatus::Active
            && f.condition_text.as_deref().unwrap_or("").trim().is_empty()
            && f.attrs
                .get("alias_kind")
                .and_then(|v| v.as_str())
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "alias" | "nickname" | "address" | "name" | "别名" | "昵称" | "小名"
                    )
                })
        {
            if let Some(ent) = entities.iter_mut().find(|e| {
                e.name == f.subject_name
                    || (f.subject_member_id.is_some()
                        && e.member_id.as_deref() == f.subject_member_id.as_deref())
            }) {
                let nick = f.object_name.trim();
                if !nick.is_empty() && !ent.aliases.iter().any(|a| a == nick) {
                    ent.aliases.push(nick.to_string());
                }
                for a in obj_aliases {
                    if !ent.aliases.iter().any(|x| x == &a) {
                        ent.aliases.push(a);
                    }
                }
            }
            // Still emit has_attribute triple so ingest can attach_aliases path.
        } else if f.object_type == EntityType::Person {
            if is_meaningful_person_label(&f.object_name) && seen.insert(f.object_name.clone()) {
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

fn kinship_role(label: &str) -> Option<&'static str> {
    match label.trim() {
        "儿子" | "小子" => Some("son"),
        "女儿" | "闺女" => Some("daughter"),
        "孩子" | "娃" => Some("child"),
        "老公" | "丈夫" | "先生" => Some("spouse"),
        "老婆" | "妻子" | "太太" => Some("spouse"),
        "爸" | "爸爸" | "父亲" | "爹" => Some("father"),
        "妈" | "妈妈" | "母亲" | "娘" => Some("mother"),
        "爷爷" | "祖父" => Some("grandfather"),
        "奶奶" | "祖母" => Some("grandmother"),
        "姥爷" | "外公" => Some("grandfather"),
        "姥姥" | "外婆" => Some("grandmother"),
        _ => None,
    }
}

/// Longest kinship titles first so 「妈妈」does not match inside longer tokens.
const KINSHIP_TITLE_TOKENS: &[&str] = &[
    "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "丈夫", "妻子",
    "老公", "老婆", "儿子", "女儿", "孩子", "闺女", "小子", "爸", "妈",
];

fn kinship_title_in_text(text: &str) -> Option<(&str, &'static str)> {
    for token in KINSHIP_TITLE_TOKENS {
        if text.contains(token) {
            if let Some(kin) = kinship_role(token) {
                return Some((*token, kin));
            }
        }
    }
    None
}

/// Markdown pipe tables: `| 林志远 | 爸爸，男 … | … |` — identity in column 2.
fn extract_markdown_roster_kinship(
    text: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> Vec<ExtractedFact> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with('|') || line.contains("---") {
            continue;
        }
        let cells: Vec<&str> = line
            .split('|')
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .collect();
        if cells.len() < 2 {
            continue;
        }
        if cells[0].contains("姓名") || cells.get(1).is_some_and(|c| c.contains("身份")) {
            continue;
        }
        let name = cells[0].trim();
        if !is_meaningful_person_label(name) || kinship_role(name).is_some() {
            continue;
        }
        let identity = cells[1];
        let Some((role_zh, kin)) = kinship_title_in_text(identity) else {
            continue;
        };
        let (obj_mid, obj_name) = resolve_person(name, roster, speaker_member_id);
        let obj_name = if is_meaningful_person_label(&obj_name) {
            obj_name
        } else {
            name.to_string()
        };
        let evidence = line.to_string();
        out.push(ExtractedFact {
            subject_member_id: None,
            subject_name: "家庭".into(),
            relation: RelationType::FamilyMember.as_zh_str().into(),
            object_type: EntityType::Person,
            object_name: obj_name.clone(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: evidence.clone(),
            confidence: 0.88,
            attrs: json!({
                "kinship": kin,
                "aliases": [role_zh, name],
                "alias_kind": "address",
            }),
        });
        out.push(ExtractedFact {
            subject_member_id: obj_mid,
            subject_name: obj_name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: role_zh.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: evidence,
            confidence: 0.86,
            attrs: json!({
                "alias_kind": "address",
                "aliases": [role_zh, name],
                "kinship": kin,
            }),
        });
    }
    out
}

/// Family membership edges + nickname/alias attributes.
fn extract_family_and_aliases(
    text: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> Vec<ExtractedFact> {
    let mut out = Vec::new();
    out.extend(extract_markdown_roster_kinship(
        text,
        roster,
        speaker_member_id,
    ));
    // Anchor on meaningful person, else「家庭」— never 我/主驾.
    let (anchor_id, anchor_name) = resolve_person_or_family("我", roster, speaker_member_id);

    // 「我儿子叫小航」「儿子叫小航」
    for cap in re(
        r"(?:我的?)?(儿子|女儿|孩子|娃|老公|老婆|丈夫|妻子|爸|妈|爸爸|妈妈|父亲|母亲|爷爷|奶奶|姥姥|姥爷|外公|外婆)叫([^\s，。；、的]{1,12})",
    )
    .captures_iter(text)
    {
        let role_zh = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let name = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
        let Some(kin) = kinship_role(role_zh) else {
            continue;
        };
        if !is_meaningful_person_label(name) {
            continue;
        }
        let (obj_mid, obj_name) = resolve_person(name, roster, speaker_member_id);
        let obj_name = if is_meaningful_person_label(&obj_name) {
            obj_name
        } else {
            name.to_string()
        };
        out.push(ExtractedFact {
            subject_member_id: anchor_id.clone(),
            subject_name: anchor_name.clone(),
            relation: RelationType::FamilyMember.as_zh_str().into(),
            object_type: EntityType::Person,
            object_name: obj_name.clone(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.9,
            attrs: json!({
                "kinship": kin,
                "aliases": [role_zh, name],
                "alias_kind": "address",
            }),
        });
        // Nickname / given name on the related person.
        out.push(ExtractedFact {
            subject_member_id: obj_mid,
            subject_name: obj_name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: name.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.88,
            attrs: json!({
                "alias_kind": "alias",
                "aliases": [name, role_zh],
                "kinship": kin,
            }),
        });
    }

    // 「小航是我儿子」「航航是我女儿」
    for cap in re(
        r"([^\s，。；、的]{1,12})是我的?(儿子|女儿|孩子|娃|老公|老婆|丈夫|妻子|爸|妈|爸爸|妈妈|父亲|母亲)",
    )
    .captures_iter(text)
    {
        let name = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let role_zh = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let Some(kin) = kinship_role(role_zh) else {
            continue;
        };
        if !is_meaningful_person_label(name) {
            continue;
        }
        let (obj_mid, obj_name) = resolve_person(name, roster, speaker_member_id);
        let obj_name = if is_meaningful_person_label(&obj_name) {
            obj_name
        } else {
            name.to_string()
        };
        out.push(ExtractedFact {
            subject_member_id: anchor_id.clone(),
            subject_name: anchor_name.clone(),
            relation: RelationType::FamilyMember.as_zh_str().into(),
            object_type: EntityType::Person,
            object_name: obj_name.clone(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.9,
            attrs: json!({
                "kinship": kin,
                "aliases": [role_zh, name],
                "alias_kind": "address",
            }),
        });
        out.push(ExtractedFact {
            subject_member_id: obj_mid,
            subject_name: obj_name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: name.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.88,
            attrs: json!({
                "alias_kind": "alias",
                "aliases": [name, role_zh],
                "kinship": kin,
            }),
        });
    }

    // 「我们家小航」「我家小航」→ family_member + alias cue for roster
    for cap in re(r"(?:我们家|我家)([^\s，。；、在的]{1,12})").captures_iter(text) {
        let name = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if !is_meaningful_person_label(name) || name.chars().count() > 8 {
            continue;
        }
        if kinship_role(name).is_some() {
            continue;
        }
        let (obj_mid, obj_name) = resolve_person(name, roster, speaker_member_id);
        let obj_name = if is_meaningful_person_label(&obj_name) {
            obj_name
        } else {
            name.to_string()
        };
        out.push(ExtractedFact {
            subject_member_id: None,
            subject_name: "家庭".into(),
            relation: RelationType::FamilyMember.as_zh_str().into(),
            object_type: EntityType::Person,
            object_name: obj_name.clone(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.82,
            attrs: json!({
                "kinship": "family",
                "aliases": [name],
                "alias_kind": "address",
            }),
        });
        out.push(ExtractedFact {
            subject_member_id: obj_mid,
            subject_name: obj_name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: name.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.8,
            attrs: json!({
                "alias_kind": "alias",
                "aliases": [name],
            }),
        });
    }

    // 「小航小名叫航航」「航航爱称/绰号/外号叫…」
    for cap in re(
        r"([^\s，。；、的]{1,12})(?:的)?(?:小名|爱称|绰号|外号|昵称)(?:是|叫)?([^\s，。；、的]{1,12})",
    )
    .captures_iter(text)
    {
        let who = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let nick = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
        if who.is_empty() || nick.is_empty() {
            continue;
        }
        let (mid, name) = resolve_person(who, roster, speaker_member_id);
        out.push(ExtractedFact {
            subject_member_id: mid,
            subject_name: name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: nick.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.92,
            attrs: json!({
                "alias_kind": "nickname",
                "aliases": [nick, who],
            }),
        });
    }

    // 「大家都叫他航航」— only when a meaningful person is already named in the utterance.
    for cap in re(r"(?:大家)?(?:都)?叫(?:他|她)([^\s，。；、的]{1,12})").captures_iter(text)
    {
        let nick = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        if nick.is_empty() || !is_meaningful_person_label(nick) {
            continue;
        }
        let Some(who) =
            first_person_mention(text, roster).filter(|n| is_meaningful_person_label(n))
        else {
            continue;
        };
        let (mid, name) = resolve_person(who, roster, speaker_member_id);
        if !is_meaningful_person_label(&name) {
            continue;
        }
        out.push(ExtractedFact {
            subject_member_id: mid,
            subject_name: name,
            relation: RelationType::HasAttribute.as_zh_str().into(),
            object_type: EntityType::Concept,
            object_name: nick.to_string(),
            polarity: FactPolarity::Affirmative,
            status: FactStatus::Active,
            domains: vec!["household".into()],
            privacy: PrivacyLevel::FamilyShared,
            condition_text: None,
            evidence_text: text.to_string(),
            confidence: 0.8,
            attrs: json!({
                "alias_kind": "nickname",
                "aliases": [nick],
            }),
        });
    }

    out
}

fn re(pat: &str) -> Regex {
    Regex::new(pat).expect("kg extract regex")
}

fn extract_allergies(
    text: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> Vec<(Option<String>, String, String, FactPolarity, String)> {
    let mut out = Vec::new();
    let mut names: Vec<(Option<String>, String)> = roster
        .iter()
        .flat_map(|m| {
            std::iter::once((Some(m.member_id.clone()), m.display_name.clone())).chain(
                m.aliases
                    .iter()
                    .filter(|a| !a.is_empty())
                    .map(|a| (Some(m.member_id.clone()), a.clone())),
            )
        })
        .collect();
    // Also try kinship titles present in text (爸爸/奶奶…) even before roster register.
    for title in [
        "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "儿子",
        "女儿", "老公", "老婆", "丈夫", "妻子", "爸", "妈",
    ] {
        if text.contains(title)
            && !names.iter().any(|(_, n)| n == title)
            && is_meaningful_person_label(title)
        {
            names.push((None, title.to_string()));
        }
    }
    names.retain(|(_, name)| is_meaningful_person_label(name));
    names.sort_by(|a, b| b.1.chars().count().cmp(&a.1.chars().count()));
    for (mid, name) in &names {
        // Negative first so 「牛奶不过敏」不落入过敏分支。
        let neg_pats = [
            format!("{name}对([^\\s，。；、]{{1,12}}?)不过敏"),
            format!("{name}([^\\s，。；、对]{{1,12}}?)不过敏"),
        ];
        let mut hit = false;
        for pat in &neg_pats {
            if let Some(cap) = Regex::new(pat).ok().and_then(|r| r.captures(text)) {
                let substance = cap
                    .get(1)
                    .map(|m| m.as_str().trim())
                    .filter(|s| !s.is_empty() && *s != "不")
                    .unwrap_or("");
                if substance.is_empty() {
                    continue;
                }
                let (member_id, display) = resolve_person(name, roster, speaker_member_id);
                let evidence = cap.get(0).map(|m| m.as_str()).unwrap_or(text).to_string();
                out.push((
                    member_id.or_else(|| mid.clone()),
                    display,
                    substance.to_string(),
                    FactPolarity::Negative,
                    evidence,
                ));
                hit = true;
                break;
            }
        }
        if hit {
            continue;
        }
        let pos_pats = [
            format!("{name}对([^\\s，。；、不]{{1,12}})过敏"),
            format!("{name}([^\\s，。；、对不]{{1,12}})过敏"),
        ];
        for pat in &pos_pats {
            if let Some(cap) = Regex::new(pat).ok().and_then(|r| r.captures(text)) {
                let substance = cap
                    .get(1)
                    .map(|m| m.as_str().trim())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("");
                if substance.is_empty() {
                    continue;
                }
                let (member_id, display) = resolve_person(name, roster, speaker_member_id);
                let evidence = cap.get(0).map(|m| m.as_str()).unwrap_or(text).to_string();
                out.push((
                    member_id.or_else(|| mid.clone()),
                    display,
                    substance.to_string(),
                    FactPolarity::Affirmative,
                    evidence,
                ));
                break;
            }
        }
    }
    // A valid supplied speaker may resolve first person, but only that explicit
    // identity (never Self_ or the first roster row) is an authorized fallback.
    let (speaker_id, speaker_name) = resolve_person("我", roster, speaker_member_id);
    if speaker_id.is_some() && is_meaningful_person_label(&speaker_name) {
        for (pattern, polarity) in [
            (
                r"(?:我|本人)对([^\s，。；、]{1,12}?)不过敏",
                FactPolarity::Negative,
            ),
            (
                r"(?:我|本人)对([^\s，。；、不]{1,12})过敏",
                FactPolarity::Affirmative,
            ),
        ] {
            for cap in re(pattern).captures_iter(text) {
                let substance = cap.get(1).map(|m| m.as_str()).unwrap_or("");
                if !substance.is_empty() {
                    out.push((
                        speaker_id.clone(),
                        speaker_name.clone(),
                        substance.into(),
                        polarity,
                        cap.get(0).map(|m| m.as_str()).unwrap_or(text).into(),
                    ));
                }
            }
        }
    }
    // Unknown speaker/pronouns never create a 我/主驾 node.
    out.retain(|(_, name, _, _, _)| is_meaningful_person_label(name));
    out
}

fn first_person_mention<'a>(text: &'a str, roster: &'a [HouseholdMember]) -> Option<&'a str> {
    for m in roster {
        if text.contains(&m.display_name) {
            return Some(m.display_name.as_str());
        }
        for a in &m.aliases {
            if !a.is_empty() && text.contains(a) {
                return Some(m.display_name.as_str());
            }
        }
    }
    None
}

fn resolve_person(
    mention: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> (Option<String>, String) {
    let mention = mention.trim().trim_start_matches(['是', '为', '由']).trim();
    if mention.is_empty() || is_meaningless_person_label(mention) {
        // Third-person pronouns do not identify the speaker. Leave them
        // unresolved instead of turning "他/她" into an assertion about me.
        if !mention.is_empty()
            && !matches!(mention, "我" | "本人" | "自己" | "主驾" | "主驾驶" | "司机")
            && !mention.eq_ignore_ascii_case("i")
            && !mention.eq_ignore_ascii_case("me")
            && !mention.eq_ignore_ascii_case("self")
            && !mention.eq_ignore_ascii_case("driver")
        {
            return (None, mention.to_string());
        }
        // Map first person to a valid supplied speaker only, never infer Self_
        // or select the first roster row when the host omitted its identity.
        let candidate = speaker_member_id
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .and_then(|id| {
                let mut matches = roster.iter().filter(|m| m.member_id == id);
                let first = matches.next();
                if matches.next().is_some() {
                    None
                } else {
                    first
                }
            });
        if let Some(m) = candidate {
            if is_meaningful_person_label(&m.display_name) {
                return (Some(m.member_id.clone()), m.display_name.clone());
            }
        }
        return (None, mention.to_string());
    }
    let mut matches = roster
        .iter()
        .filter(|m| m.display_name == mention || m.aliases.iter().any(|a| a == mention));
    if let Some(m) = matches.next() {
        if matches.next().is_some() {
            return (None, mention.to_string());
        }
        // Prefer a meaningful surface mention (爸爸), but never pick the first
        // roster row when a shared alias describes several people.
        let label = if is_meaningful_person_label(mention) {
            mention.to_string()
        } else if is_meaningful_person_label(&m.display_name) {
            m.display_name.clone()
        } else {
            mention.to_string()
        };
        return (Some(m.member_id.clone()), label);
    }
    (None, mention.to_string())
}

/// Personal fact subject, or「家庭」for shared rules when no meaningful person.
fn resolve_person_or_family(
    mention: &str,
    roster: &[HouseholdMember],
    speaker_member_id: Option<&str>,
) -> (Option<String>, String) {
    let (id, name) = resolve_person(mention, roster, speaker_member_id);
    if is_meaningful_person_label(&name) {
        (id, name)
    } else {
        (None, "家庭".into())
    }
}

fn looks_like_condition(s: &str) -> bool {
    const HINTS: &[&str] = &[
        "高血压",
        "糖尿病",
        "哮喘",
        "过敏",
        "不适",
        "疼痛",
        "炎症",
        "感冒",
        "发烧",
        "膝盖",
    ];
    HINTS.iter().any(|h| s.contains(h)) || s.ends_with('炎') || s.ends_with('症')
}

fn looks_like_medication(s: &str) -> bool {
    let s = s.trim();
    if s.chars().count() < 2
        || s.contains("饭")
        || s.contains("东西")
        || s.contains("西红柿")
        || s.contains("番茄")
        || s.contains("水果")
        || s.contains("蔬菜")
    {
        return false;
    }
    // Do not treat ordinary food as medication just because name length ≥ 3.
    const FOOD: &[&str] = &[
        "西红柿",
        "番茄",
        "苹果",
        "香蕉",
        "鸡蛋",
        "牛奶",
        "面包",
        "米饭",
        "面条",
        "饺子",
        "西瓜",
        "橙子",
        "葡萄",
        "草莓",
        "黄瓜",
        "土豆",
        "白菜",
        "青菜",
    ];
    if FOOD.iter().any(|f| s.contains(f)) {
        return false;
    }
    // Only explicit medication cues — never length alone.
    s.ends_with('药')
        || s.ends_with('片')
        || s.contains("胶囊")
        || s.contains("口服液")
        || s.contains("冲剂")
        || s.contains("氨氯地平")
        || s.contains("阿司匹林")
        || s.contains("布洛芬")
        || s.contains("抗生素")
        || s.contains("青霉素")
        || s.contains("维生素")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::household::{HouseholdMember, MemberRole};

    fn roster() -> Vec<HouseholdMember> {
        vec![
            HouseholdMember {
                member_id: "child1".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec!["孩子".into()],
                age_band: None,
            },
            HouseholdMember {
                member_id: "self".into(),
                display_name: "主驾".into(),
                role: MemberRole::Self_,
                aliases: vec!["我".into()],
                age_band: None,
            },
        ]
    }

    #[test]
    fn extracts_allergy_and_school() {
        let facts = extract_facts(
            "小航花生过敏，在实验小学三年级二班。",
            &roster(),
            Some("self"),
        );
        assert!(facts
            .iter()
            .any(|f| f.relation == RelationType::AllergicTo.as_zh_str()));
        assert!(facts
            .iter()
            .any(|f| f.relation == RelationType::AttendsSchool.as_zh_str()));
    }

    #[test]
    fn extracts_family_member_and_alias() {
        let facts = extract_facts("我儿子叫小航，小名航航。", &roster(), Some("self"));
        assert!(facts.iter().any(|f| {
            f.relation == RelationType::FamilyMember.as_zh_str() && f.object_name.contains("小航")
        }));
        assert!(facts.iter().any(|f| {
            f.relation == RelationType::HasAttribute.as_zh_str()
                && (f.object_name == "航航" || f.object_name == "小航")
        }));
    }

    #[test]
    fn extracts_our_family_mention() {
        let facts = extract_facts("我们家小航在实验中学上学", &roster(), Some("self"));
        assert!(facts
            .iter()
            .any(|f| f.relation == RelationType::FamilyMember.as_zh_str()));
        assert!(facts
            .iter()
            .any(|f| f.relation == RelationType::AttendsSchool.as_zh_str()));
    }

    #[test]
    fn extracts_conditional_diet_rule() {
        let facts = extract_facts("胃不舒服时不吃辣", &roster(), Some("self"));
        assert!(facts.iter().any(|f| {
            f.relation == RelationType::FollowsRule.as_zh_str()
                && f.condition_text.as_deref() == Some("胃不舒服时")
                && f.subject_name == "家庭"
        }));
    }

    #[test]
    fn skips_pronoun_allergy_without_meaningful_person() {
        let facts = extract_facts("我对花生过敏", &roster(), Some("self"));
        assert!(!facts
            .iter()
            .any(|f| f.relation == RelationType::AllergicTo.as_zh_str()));
    }

    #[test]
    fn keeps_kinship_and_given_name() {
        let facts = extract_facts("爸爸对牛奶过敏", &roster(), Some("self"));
        // 爸爸 not in roster → pending at ingest; heuristic still emits meaningful label.
        assert!(facts.iter().any(|f| {
            f.relation == RelationType::AllergicTo.as_zh_str() && f.subject_name == "爸爸"
        }));
    }
}
