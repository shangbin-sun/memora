//! Household roster and memory facets (member × domain).
//!
//! Cognitive [`crate::types::MemoryKind`] stays episodic/semantic/pattern/summary.
//! Who a memory is about, and which life domain it belongs to, live in `metadata`.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::core::now_ms;
use crate::error::{MemoraError, Result};
use crate::types::MemoryRecord;

pub const CONFIDENCE_LOCK: f64 = 0.6;

/// Shared / household-level subject when no specific person can be inferred.
pub const PUBLIC_MEMBER_ID: &str = "公共";
pub(crate) const SHARED_MEMBER_IDS: &[&str] = &[PUBLIC_MEMBER_ID, "大家", "全家", "家庭", "共享", "public"];

pub const DOMAINS: &[&str] = &[
    "medical",
    "education",
    "travel",
    "lifestyle",
    "diet",
    "finance",
    "social",
    "household",
    "unknown",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberRole {
    #[serde(rename = "self")]
    Self_,
    Spouse,
    Child,
    Parent,
    Other,
}

impl MemberRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Self_ => "self",
            Self::Spouse => "spouse",
            Self::Child => "child",
            Self::Parent => "parent",
            Self::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim() {
            "self" => Ok(Self::Self_),
            "spouse" => Ok(Self::Spouse),
            "child" => Ok(Self::Child),
            "parent" => Ok(Self::Parent),
            "other" => Ok(Self::Other),
            other => Err(MemoraError::InvalidState(format!(
                "unknown member role: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HouseholdMember {
    pub member_id: String,
    pub display_name: String,
    pub role: MemberRole,
    pub aliases: Vec<String>,
    #[serde(default)]
    pub age_band: Option<String>,
}

#[derive(Debug, Clone)]
struct Hit {
    member_id: String,
    confidence: f64,
    speaker: bool,
}

pub fn validate_member_id(id: &str) -> Result<String> {
    let id = id.trim();
    if id.is_empty() || id.contains('|') || id.contains(',') || id.contains('"') {
        return Err(MemoraError::InvalidState(
            "member_id must be a non-empty token without comma or pipe".into(),
        ));
    }
    Ok(id.to_string())
}

/// Normalize an explicit owner field, never words inside the fact's content.
pub fn canonical_owner_id(id: &str) -> Result<String> {
    let id = validate_member_id(id)?;
    Ok(if is_shared_member_id(&id) { PUBLIC_MEMBER_ID.into() } else { id })
}

/// Model placeholders are not identities and must not create household members.
pub fn is_unresolved_owner_id(id: &str) -> bool {
    matches!(id.trim().to_ascii_lowercase().as_str(),
        "我" | "本人" | "用户" | "当前用户" | "说话人" | "未知" | "未知用户" |
        "匿名" | "匿名用户" | "未识别说话人" | "他" | "她" | "它" |
        "user" | "self" | "me" | "unknown" | "anonymous" | "speaker")
}

/// One ownership representation for extracted facts and explicit JNI writes.
/// An explicitly empty subject list means public; an absent owner is not passed here.
pub fn with_ownership(metadata: &Value, owners: &[String]) -> Result<Value> {
    let mut ids = owners.iter().map(|id| canonical_owner_id(id)).collect::<Result<Vec<_>>>()?;
    ids.sort();
    ids.dedup();
    if ids.is_empty() { ids.push(PUBLIC_MEMBER_ID.into()); }
    if ids.len() > 1 && ids.iter().any(|id| id == PUBLIC_MEMBER_ID) {
        return Err(MemoraError::InvalidState("public and personal owners cannot be combined".into()));
    }
    let mut meta = metadata.as_object().cloned().unwrap_or_default();
    meta.insert("subjects".into(), json!(ids.iter().map(|id| json!({"member_id":id,"confidence":1.0})).collect::<Vec<_>>()));
    meta.insert("attribution".into(), json!("manual"));
    meta.insert("scope".into(), json!(if ids == [PUBLIC_MEMBER_ID] { "public" } else { "members" }));
    Ok(Value::Object(meta))
}

/// Canonical English domain, accepting the Chinese tags used by legacy rows.
pub fn canonical_domain(raw: &str) -> Option<&'static str> {
    let t = raw.trim().to_ascii_lowercase();
    DOMAINS
        .iter()
        .copied()
        .find(|d| *d == t)
        .or_else(|| match t.as_str() {
            "医疗健康" | "医疗" | "健康" => Some("medical"),
            "工作学习" | "工作" | "学习" | "教育" | "职业" => Some("education"),
            "出行" => Some("travel"),
            "生活习惯" => Some("lifestyle"),
            "饮食" => Some("diet"),
            "金融" | "收入" | "消费" | "支出" => Some("finance"),
            "娱乐" | "社交" => Some("social"),
            "家务" | "家庭" => Some("household"),
            "其他" => Some("unknown"),
            _ => None,
        })
}

/// Convert a built-in tag (or legacy English domain) to its storage domain.
pub fn domain_for_tag(raw: &str) -> Option<&'static str> {
    canonical_domain(raw)
}

/// Convert a canonical/legacy domain to its built-in Chinese display tag.
pub fn tag_for_domain(raw: &str) -> Option<&'static str> {
    match canonical_domain(raw)? {
        "medical" => Some("医疗健康"),
        "education" => Some("工作学习"),
        "travel" => Some("出行"),
        "lifestyle" | "household" => Some("生活习惯"),
        "diet" => Some("饮食"),
        "finance" => Some("金融"),
        "social" => Some("娱乐"),
        "unknown" => Some("其他"),
        _ => None,
    }
}

pub fn list_members(conn: &Connection, namespace: &str) -> Result<Vec<HouseholdMember>> {
    let mut stmt = conn.prepare(
        r#"SELECT member_id, display_name, role, aliases_json, age_band
           FROM household_members WHERE namespace = ?1
           ORDER BY created_at_ms"#,
    )?;
    let rows = stmt.query_map(params![namespace], |row| {
        let role_s: String = row.get(2)?;
        let aliases_json: String = row.get(3)?;
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
        let role = MemberRole::parse(&role_s).unwrap_or(MemberRole::Other);
        Ok(HouseholdMember {
            member_id: row.get(0)?,
            display_name: row.get(1)?,
            role,
            aliases,
            age_band: row.get(4)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn upsert_member(conn: &Connection, namespace: &str, member: &HouseholdMember) -> Result<()> {
    let id = validate_member_id(&member.member_id)?;
    let name = member.display_name.trim();
    if name.is_empty() {
        return Err(MemoraError::InvalidState("display_name is required".into()));
    }
    let aliases: Vec<String> = member
        .aliases
        .iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    let now = now_ms();
    conn.execute(
        r#"INSERT INTO household_members(
             namespace, member_id, display_name, role, aliases_json, age_band, created_at_ms, updated_at_ms
           ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
           ON CONFLICT(namespace, member_id) DO UPDATE SET
             display_name = excluded.display_name,
             role = excluded.role,
             aliases_json = excluded.aliases_json,
             age_band = excluded.age_band,
             updated_at_ms = excluded.updated_at_ms"#,
        params![
            namespace,
            id,
            name,
            member.role.as_str(),
            serde_json::to_string(&aliases)?,
            member.age_band.as_deref().map(str::trim).filter(|s| !s.is_empty()),
            now,
        ],
    )?;
    Ok(())
}

pub fn remove_member(conn: &Connection, namespace: &str, member_id: &str) -> Result<bool> {
    let id = validate_member_id(member_id)?;
    let n = conn.execute(
        "DELETE FROM household_members WHERE namespace = ?1 AND member_id = ?2",
        params![namespace, id],
    )?;
    Ok(n > 0)
}

/// Fill `subjects` / `domains` / `attribution` / `sensitivity` without dropping other metadata keys.
///
/// Ownership rule: when the host did **not** pass explicit `subjects` (`attribution=manual`),
/// infer the member from content + household roster (names / aliases / kinship), **not** from
/// the speaking actor. `member_id` is only a soft fallback when nothing in the text matches.
pub fn annotate(
    roster: &[HouseholdMember],
    content: &str,
    entities: &[String],
    actor: Option<&str>,
    metadata: &Value,
    explicit_member: Option<&str>,
    explicit_domains: &[String],
) -> Value {
    let mut meta = metadata.as_object().cloned().unwrap_or_default();
    normalize_domains(&mut meta);
    let requested: Vec<Value> = explicit_domains
        .iter()
        .filter_map(|d| canonical_domain(d))
        .filter(|d| *d != "unknown")
        .map(|id| json!({"id": id, "confidence": 1.0}))
        .collect();
    if !requested.is_empty() {
        meta.insert("domains".into(), Value::Array(requested));
    }
    // Host-provided subjects (JNI/API `subjects` field) win and are not re-inferred.
    let host_subjects = meta.get("attribution").and_then(|v| v.as_str()) == Some("manual")
        && meta.get("subjects").and_then(|v| v.as_array()).is_some();
    if host_subjects || meta.get("attribution").and_then(|v| v.as_str()) == Some("classifier") {
        apply_sensitivity(&mut meta);
        return Value::Object(meta);
    }

    let explicit_member = explicit_member.map(str::trim).filter(|s| !s.is_empty());
    // Content + household relation inference (no speaker-as-default).
    let hits = rule_subjects(roster, content, entities, actor);
    if !hits.is_empty() {
        meta.insert(
            "subjects".into(),
            Value::Array(
                hits.iter()
                    .map(|h| json!({ "member_id": h.member_id, "confidence": h.confidence }))
                    .collect(),
            ),
        );
        meta.insert("attribution".into(), json!("rule"));
        meta.insert("scope".into(), json!("members"));
    } else if first_person_wo(content) {
        // 「我」→ speaker / explicit member; else public.
        let id = explicit_member
            .or(actor.map(str::trim).filter(|s| !s.is_empty()))
            .unwrap_or(PUBLIC_MEMBER_ID);
        if let Ok(id) = validate_member_id(id) {
            meta.insert(
                "subjects".into(),
                json!([{ "member_id": id, "confidence": 0.85 }]),
            );
            meta.insert("attribution".into(), json!("rule"));
            meta.insert(
                "scope".into(),
                json!(if id == PUBLIC_MEMBER_ID {
                    "public"
                } else {
                    "members"
                }),
            );
        }
    } else {
        // No specific person → shared public bucket.
        meta.insert(
            "subjects".into(),
            json!([{ "member_id": PUBLIC_MEMBER_ID, "confidence": 0.9 }]),
        );
        meta.insert("attribution".into(), json!("rule"));
        meta.insert("scope".into(), json!("public"));
    }

    if meta.get("domains").is_none() {
        let domains = keyword_domains(content, entities);
        meta.insert("domains".into(), domains);
    }
    apply_sensitivity(&mut meta);
    Value::Object(meta)
}

pub fn needs_household_pass(metadata: &Value) -> bool {
    match metadata.get("attribution").and_then(|v| v.as_str()) {
        Some("manual") => false,
        Some("classifier") => domain_uncertain(metadata),
        Some("rule") | Some("speaker") => {
            subjects(metadata).iter().any(|(_, c)| *c < CONFIDENCE_LOCK)
                || domain_uncertain(metadata)
        }
        _ => true,
    }
}

/// Apply classifier subjects/domains onto memories that are still uncertain.
pub fn merge_classifier_facets(
    metadata: &Value,
    roster: &[HouseholdMember],
    subjects_json: Option<&Value>,
    domains_json: Option<&Value>,
) -> Value {
    if !needs_household_pass(metadata) {
        return metadata.clone();
    }
    let mut meta = metadata.as_object().cloned().unwrap_or_default();
    let known: std::collections::HashSet<&str> =
        roster.iter().map(|m| m.member_id.as_str()).collect();
    let subject_locked = subjects(metadata)
        .iter()
        .any(|(_, c)| *c >= CONFIDENCE_LOCK);
    if !subject_locked {
        if let Some(arr) = subjects_json.and_then(|v| v.as_array()) {
            let cleaned: Vec<Value> = arr
                .iter()
                .filter_map(|item| {
                    let id = item.get("member_id").and_then(|v| v.as_str())?;
                    if !known.contains(id) {
                        return None;
                    }
                    let confidence = item
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.7)
                        .clamp(0.0, 1.0);
                    Some(json!({ "member_id": id, "confidence": confidence }))
                })
                .collect();
            meta.insert("subjects".into(), Value::Array(cleaned));
        }
    }
    if domain_uncertain(metadata) {
        if let Some(arr) = domains_json.and_then(|v| v.as_array()) {
            let cleaned: Vec<Value> = arr
                .iter()
                .filter_map(|item| {
                    let id =
                        canonical_domain(item.get("id").and_then(|v| v.as_str()).unwrap_or(""))?;
                    if id == "unknown" {
                        return None;
                    }
                    let confidence = item
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.7)
                        .clamp(0.0, 1.0);
                    Some(json!({ "id": id, "confidence": confidence }))
                })
                .collect();
            if !cleaned.is_empty() {
                meta.insert("domains".into(), Value::Array(cleaned));
            }
        }
    }
    meta.insert("attribution".into(), json!("classifier"));
    apply_sensitivity(&mut meta);
    Value::Object(meta)
}

pub fn cluster_key(metadata: &Value) -> String {
    let mut ids: Vec<&str> = subjects(metadata).into_iter().map(|(id, _)| id).collect();
    ids.sort_unstable();
    ids.dedup();
    format!("{}|{}", ids.join(","), primary_domain(metadata))
}

pub fn matches_filter(metadata: &Value, member_id: Option<&str>, domains: &[String]) -> bool {
    matches_member_and_tag(metadata, &[], "", member_id, None, domains)
}

/// Filter by searchable member / Chinese tag columns, with metadata fallback.
pub fn matches_member_and_tag(
    metadata: &Value,
    subject_ids: &[String],
    tag: &str,
    member_id: Option<&str>,
    want_tag: Option<&str>,
    domains: &[String],
) -> bool {
    if let Some(mid) = member_id.map(str::trim).filter(|s| !s.is_empty()) {
        let same = |id: &str| id == mid || (is_shared_member_id(mid) && is_shared_member_id(id));
        let in_column = subject_ids.iter().any(|id| same(id));
        let in_meta = subjects(metadata).iter().any(|(id, _)| same(id));
        // Old JNI explicit-public writes stored an empty subjects array.
        let legacy_public = is_shared_member_id(mid) && subject_ids.is_empty()
            && subjects(metadata).is_empty() && metadata.get("scope").and_then(Value::as_str) == Some("public");
        if !in_column && !in_meta && !legacy_public {
            return false;
        }
    }
    if let Some(want) = want_tag.map(str::trim).filter(|s| !s.is_empty()) {
        let have = if !tag.trim().is_empty() {
            tag.trim()
        } else {
            memory_tag(metadata)
        };
        if have != want {
            return false;
        }
    }
    if !domains.is_empty() {
        let have = domain_ids(metadata);
        if !domains.iter().any(|d| {
            canonical_domain(d).is_some_and(|c| {
                have.contains(&c)
                    || domain_for_tag(tag) == Some(c)
                    || domain_for_tag(memory_tag(metadata)) == Some(c)
            }) || have.iter().any(|h| *h == d.as_str())
                || memory_tag(metadata) == d
        }) {
            return false;
        }
    }
    true
}

pub fn is_sensitive(metadata: &Value) -> bool {
    metadata.get("sensitivity").and_then(|v| v.as_str()) == Some("high")
        || domain_ids(metadata).iter().any(|d| *d == "medical")
        || memory_tag(metadata) == "医疗健康"
}

pub fn subject_id_list(metadata: &Value) -> Vec<String> {
    subjects(metadata)
        .into_iter()
        .map(|(id, _)| id.to_string())
        .collect()
}

pub fn memory_tag(metadata: &Value) -> &str {
    if let Some(tag) = metadata.get("tag").and_then(|v| v.as_str()).map(str::trim) {
        if !tag.is_empty() {
            return tag;
        }
    }
    tag_for_domain(primary_domain(metadata)).unwrap_or("其他")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTag {
    pub tag_id: String,
    pub display_name: String,
    pub aliases: Vec<String>,
}

pub const DEFAULT_TAGS: &[(&str, &[&str])] = &[
    ("娱乐", &["娱乐", "休闲", "游戏", "影视", "兴趣"]),
    ("饮食", &["饮食", "吃饭", "口味", "忌口", "菜谱"]),
    (
        "工作学习",
        &[
            "工作学习",
            "工作",
            "学习",
            "学校",
            "作业",
            "补习",
            "上课",
            "上班",
        ],
    ),
    (
        "医疗健康",
        &[
            "医疗健康",
            "医疗",
            "健康",
            "医院",
            "感冒",
            "看病",
            "体检",
            "用药",
            "医生",
            "药物",
            "过敏",
            "血压",
            "诊断",
            "复诊",
            "复查",
        ],
    ),
    (
        "出行",
        &["出行", "旅行", "旅游", "机票", "高铁", "开车", "行程"],
    ),
    (
        "生活习惯",
        &["生活习惯", "习惯", "作息", "睡眠", "运动", "家务"],
    ),
    ("金融", &["金融", "账单", "工资", "花销", "缴费", "报销"]),
    ("其他", &["其他", "杂项", "日常"]),
];

pub fn list_tags(conn: &Connection, namespace: &str) -> Result<Vec<MemoryTag>> {
    let mut stmt = conn.prepare(
        r#"SELECT tag_id, display_name, aliases_json
           FROM memory_tags WHERE namespace = ?1
           ORDER BY created_at_ms"#,
    )?;
    let rows = stmt.query_map(params![namespace], |row| {
        let aliases_json: String = row.get(2)?;
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
        Ok(MemoryTag {
            tag_id: row.get(0)?,
            display_name: row.get(1)?,
            aliases,
        })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

pub fn upsert_tag(conn: &Connection, namespace: &str, tag: &MemoryTag) -> Result<()> {
    let id = validate_member_id(&tag.tag_id)?;
    let name = tag.display_name.trim();
    if name.is_empty() {
        return Err(MemoraError::InvalidState(
            "tag display_name is required".into(),
        ));
    }
    let mut aliases: Vec<String> = tag
        .aliases
        .iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    if !aliases.iter().any(|a| a == name) {
        aliases.push(name.to_string());
    }
    let now = now_ms();
    conn.execute(
        r#"INSERT INTO memory_tags(
             namespace, tag_id, display_name, aliases_json, created_at_ms, updated_at_ms
           ) VALUES (?1, ?2, ?3, ?4, ?5, ?5)
           ON CONFLICT(namespace, tag_id) DO UPDATE SET
             display_name = excluded.display_name,
             aliases_json = excluded.aliases_json,
             updated_at_ms = excluded.updated_at_ms"#,
        params![namespace, id, name, serde_json::to_string(&aliases)?, now],
    )?;
    Ok(())
}

pub fn ensure_default_tags(conn: &Connection, namespace: &str) -> Result<()> {
    for (name, aliases) in DEFAULT_TAGS {
        upsert_tag(
            conn,
            namespace,
            &MemoryTag {
                tag_id: (*name).into(),
                display_name: (*name).into(),
                aliases: aliases.iter().map(|s| (*s).into()).collect(),
            },
        )?;
    }
    Ok(())
}

pub fn resolve_tag_id(catalog: &[MemoryTag], raw: &str) -> String {
    let token = raw.trim();
    if token.is_empty() {
        return "其他".into();
    }
    for tag in catalog {
        if tag.tag_id == token
            || tag.display_name == token
            || tag.aliases.iter().any(|a| a == token)
        {
            return tag.tag_id.clone();
        }
    }
    tag_for_domain(token).unwrap_or(token).to_string()
}

/// Validate explicit metadata against the tag catalog (including built-ins).
/// Unknown model-produced strings must not become new tags implicitly.
pub fn validated_metadata_tag(catalog: &[MemoryTag], metadata: &Value) -> Option<String> {
    if let Some(raw) = metadata.get("tag").and_then(Value::as_str).map(str::trim).filter(|v| *v != "错题集") {
        if let Some(tag) = catalog.iter().find(|tag| {
            tag.tag_id != "错题集" && (tag.tag_id == raw || tag.display_name == raw || tag.aliases.iter().any(|a| a == raw))
        }) {
            return Some(tag.tag_id.clone());
        }
        if let Some(tag) = tag_for_domain(raw) {
            return Some(tag.to_string());
        }
    }
    let domain = primary_domain(metadata);
    (domain != "unknown").then(|| tag_for_domain(domain).unwrap_or("其他").to_string())
}

/// Use validated facets before heuristic text inference. Call before `annotate`
/// when selecting a tag for a new write, so inferred domains are not mistaken for
/// host-provided metadata. For revise, pass the merged metadata (or the explicit
/// patch when the host intentionally requests reclassification).
pub fn select_memory_tag(
    catalog: &[MemoryTag],
    metadata: &Value,
    content: &str,
    entities: &[String],
) -> String {
    validated_metadata_tag(catalog, metadata)
        .unwrap_or_else(|| resolve_tag_id(catalog, &infer_tag_from_text(content, entities)))
}

pub fn infer_tag_from_text(content: &str, entities: &[String]) -> String {
    // Use the same domain evidence as annotation, prioritizing medical safety
    // over incidental diet/travel words (e.g. 食物过敏 or 去医院).
    let inferred = keyword_domains(content, entities);
    let domain_meta = json!({"domains": inferred});
    if let Some(tag) = validated_metadata_tag(&[], &domain_meta) {
        return tag;
    }
    let blob = format!("{content} {}", entities.join(" "));
    for (name, aliases) in DEFAULT_TAGS {
        if *name == "其他" {
            continue;
        }
        if aliases.iter().any(|a| blob.contains(a)) {
            return (*name).into();
        }
    }
    "其他".into()
}

pub fn roster_prompt(roster: &[HouseholdMember]) -> String {
    // Shared placeholder rows (e.g. storage bucket) must not appear as people
    // the model can pick as fact owners.
    let brief: Vec<Value> = roster
        .iter()
        .filter(|m| !is_shared_member_id(&m.member_id))
        .map(|m| {
            json!({
                "member_id": m.member_id,
                "display_name": m.display_name,
                "role": m.role.as_str(),
                "aliases": m.aliases,
            })
        })
        .collect();
    serde_json::to_string(&brief).unwrap_or_else(|_| "[]".into())
}

fn rule_subjects(
    roster: &[HouseholdMember],
    content: &str,
    entities: &[String],
    actor: Option<&str>,
) -> Vec<Hit> {
    let actor = actor.map(str::trim).filter(|s| !s.is_empty());
    let mut named_hits = Vec::new();
    let mut kinship_hits = Vec::new();
    let mut pronoun_hits = Vec::new();
    for member in roster {
        let named = named_mention(member, content, entities);
        let via_kinship = !named && kinship_mention(member, roster, content);
        let via_pronoun = !named && !via_kinship && pronoun_mention(member, content, entities);
        let is_speaker = actor.is_some_and(|a| {
            eq_name(a, &member.member_id)
                || eq_name(a, &member.display_name)
                || member.aliases.iter().any(|alias| eq_name(a, alias))
        });
        if named {
            named_hits.push(Hit {
                member_id: member.member_id.clone(),
                confidence: 0.95,
                speaker: is_speaker,
            });
        } else if via_kinship {
            kinship_hits.push(Hit {
                member_id: member.member_id.clone(),
                confidence: 0.9,
                speaker: is_speaker,
            });
        } else if via_pronoun {
            pronoun_hits.push(Hit {
                member_id: member.member_id.clone(),
                confidence: 0.9,
                speaker: is_speaker,
            });
        }
        // Intentionally no "actor/speaker default" — subjects 未传时只看语义归属。
    }
    // 「我们家小航…」→ 小航；「儿子在实验中学」→ role=child 成员。
    if !named_hits.is_empty() {
        return named_hits;
    }
    if !kinship_hits.is_empty() {
        return kinship_hits;
    }
    if !pronoun_hits.is_empty() {
        return pronoun_hits;
    }
    Vec::new()
}

/// Map kinship words in text to roster members.
///
/// Priority: (1) a present title equals this member's name/alias;
/// (2) role-wide fallback only when the title does not name anyone else and
/// this is the unique roster member of that role (e.g. 「儿子」→ sole Child).
///
/// Short titles like 「爸」must not fire inside longer titles like 「爸爸」.
fn kinship_mention(member: &HouseholdMember, roster: &[HouseholdMember], content: &str) -> bool {
    let present: Vec<&str> = kinship_labels_for_role(member.role)
        .iter()
        .copied()
        .filter(|label| contains_kinship_title(content, label))
        .collect();
    if present.is_empty() {
        return false;
    }
    if present.iter().any(|label| member_has_label(member, label)) {
        return true;
    }
    // Title already names another member → do not attach via shared role.
    if present.iter().any(|label| {
        roster
            .iter()
            .any(|other| other.member_id != member.member_id && member_has_label(other, label))
    }) {
        return false;
    }
    let same_role = roster.iter().filter(|m| m.role == member.role).count();
    same_role == 1
}

fn member_has_label(member: &HouseholdMember, label: &str) -> bool {
    eq_name(&member.display_name, label)
        || eq_name(&member.member_id, label)
        || member.aliases.iter().any(|a| eq_name(a, label))
}

/// True when `title` appears as its own kinship mention, not merely as a
/// substring of a longer kinship title that is also present
/// (「爸」inside 「爸爸」, 「妈」inside 「妈妈」).
fn contains_kinship_title(content: &str, title: &str) -> bool {
    if !contains_ci(content, title) {
        return false;
    }
    // Longer titles that embed this short form; if the longer form is present,
    // require an occurrence of `title` outside that longer span.
    const LONGER: &[&str] = &[
        "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "老公",
        "老婆", "丈夫", "妻子", "儿子", "女儿", "孩子", "闺女",
    ];
    for longer in LONGER {
        if *longer == title || !longer.contains(title) || !contains_ci(content, longer) {
            continue;
        }
        if !content.replace(longer, "").contains(title) {
            return false;
        }
    }
    true
}

fn kinship_labels_for_role(role: MemberRole) -> &'static [&'static str] {
    match role {
        MemberRole::Child => &["儿子", "女儿", "孩子", "娃", "闺女", "小子"],
        MemberRole::Spouse => &["老公", "老婆", "丈夫", "妻子", "太太", "先生"],
        MemberRole::Parent => &[
            "爸", "妈", "爸爸", "妈妈", "父亲", "母亲", "爹", "娘",
            // Grandparents often registered as parent/elder in household roster.
            "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷",
        ],
        MemberRole::Self_ => &[],
        // Unspecified elders / relatives addressed by kinship title.
        MemberRole::Other => &[
            "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "叔叔", "阿姨", "舅舅", "姑姑",
        ],
    }
}

fn named_mention(member: &HouseholdMember, content: &str, entities: &[String]) -> bool {
    std::iter::once(member.display_name.as_str())
        .chain(member.aliases.iter().map(String::as_str))
        .filter(|n| !is_self_pronoun(n))
        .any(|n| mentioned(n, content, entities))
}

fn pronoun_mention(member: &HouseholdMember, content: &str, entities: &[String]) -> bool {
    std::iter::once(member.display_name.as_str())
        .chain(member.aliases.iter().map(String::as_str))
        .filter(|n| is_self_pronoun(n))
        .any(|n| mentioned(n, content, entities))
}

fn is_self_pronoun(name: &str) -> bool {
    matches!(name.trim(), "我" | "本人" | "自己")
}

/// Infer memory owner from who the content is about (not who is speaking).
/// Uses display names, aliases, and kinship role words; first-person 「我」maps to
/// `speaker_id` / role=self only when no other member is implicated.
pub fn infer_content_owner(
    roster: &[HouseholdMember],
    content: &str,
    speaker_id: Option<&str>,
) -> Option<String> {
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let hits = rule_subjects(roster, content, &[], speaker_id);
    // A paragraph naming several people does not identify a single fact owner.
    // Let the caller use fact-local evidence / explicit owner instead of roster order.
    if hits.len() > 1 {
        return None;
    }
    if let Some(hit) = hits.first() {
        // Resolve a first-person-only alias to the actual speaker when supplied.
        if first_person_wo(content)
            && !roster
                .iter()
                .any(|m| named_mention(m, content, &[]) || kinship_mention(m, roster, content))
        {
            if let Some(speaker) = speaker_id.map(str::trim).filter(|s| !s.is_empty()) {
                return Some(speaker.to_string());
            }
        }
        return Some(hit.member_id.clone());
    }
    first_person_wo(content)
        .then(|| {
            speaker_id
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .flatten()
}

/// Shared / household-level subject bucket — not a specific person.
///
/// Extractors often emit these tokens when the predicate was subject-stripped
/// (`content="喜欢喝茶"`); they must not preempt grounding against the source utterance.
pub fn is_shared_member_id(id: &str) -> bool {
    SHARED_MEMBER_IDS.iter().any(|alias| id.trim().eq_ignore_ascii_case(alias))
}

fn compact_text(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// When fact text has the subject stripped, locate the clause in `source` that
/// contains `fact_content` and infer a unique owner from that local window.
///
/// Example: fact=`喜欢喝茶`, source=`爸爸喜欢喝茶` → `爸爸`.
pub fn ground_owner_from_source(
    roster: &[HouseholdMember],
    fact_content: &str,
    source: &str,
    speaker_id: Option<&str>,
) -> Option<String> {
    let fact_content = fact_content.trim();
    let source = source.trim();
    if fact_content.is_empty() || source.is_empty() {
        return None;
    }
    let needle = compact_text(fact_content);
    if needle.chars().count() < 2 {
        return None;
    }
    let mut owners = Vec::new();
    for clause in source.split(|c| {
        matches!(
            c,
            '。' | '！' | '!' | '?' | '？' | '\n' | '；' | ';' | '，' | ','
        )
    }) {
        let clause = clause.trim();
        if clause.is_empty() || !compact_text(clause).contains(&needle) {
            continue;
        }
        if let Some(id) = infer_content_owner(roster, clause, speaker_id) {
            owners.push(id);
        }
    }
    owners.sort();
    owners.dedup();
    (owners.len() == 1).then(|| owners.remove(0))
}

/// Rebuild a self-contained fact string when the extractor stripped the subject
/// into `owner_id` (e.g. content=`喜欢喝茶`, owner=`爸爸` → `爸爸喜欢喝茶`).
pub fn complete_owned_content(
    roster: &[HouseholdMember],
    content: &str,
    owner_id: &str,
    evidence: &str,
    source: &str,
) -> String {
    let content = content.trim();
    if content.is_empty() || is_shared_member_id(owner_id) {
        return content.to_string();
    }
    if infer_content_owner(roster, content, None).as_deref() == Some(owner_id) {
        return content.to_string();
    }
    let needle = compact_text(content);
    let evidence = evidence.trim();
    if !evidence.is_empty()
        && compact_text(evidence).contains(&needle)
        && infer_content_owner(roster, evidence, None).as_deref() == Some(owner_id)
    {
        return evidence.to_string();
    }
    for clause in source.split(|c| {
        matches!(
            c,
            '。' | '！' | '!' | '?' | '？' | '\n' | '；' | ';' | '，' | ','
        )
    }) {
        let clause = clause.trim();
        if clause.is_empty() || !compact_text(clause).contains(&needle) {
            continue;
        }
        if infer_content_owner(roster, clause, None).as_deref() == Some(owner_id) {
            return clause.to_string();
        }
    }
    let label = roster
        .iter()
        .find(|m| m.member_id == owner_id)
        .map(|m| m.display_name.as_str())
        .unwrap_or(owner_id)
        .trim();
    if label.is_empty() || compact_text(content).contains(&compact_text(label)) {
        content.to_string()
    } else {
        format!("{label}{content}")
    }
}

fn mentioned(name: &str, content: &str, entities: &[String]) -> bool {
    let name = name.trim();
    let chars = name.chars().count();
    // Allow common single-char Chinese kinship / self pronouns.
    let allow_short = matches!(name, "我" | "爸" | "妈" | "爹" | "娘");
    if chars < 2 && !allow_short {
        return false;
    }
    if name == "我" {
        return first_person_wo(content)
            || entities
                .iter()
                .any(|e| first_person_wo(e) || eq_name(e, "我"));
    }
    contains_ci(content, name)
        || entities
            .iter()
            .any(|e| contains_ci(e, name) || contains_ci(name, e))
}

/// True when 「我」appears as a self pronoun, not inside 「我们 / 我家 / 我们家」.
fn first_person_wo(text: &str) -> bool {
    let mut rest = text;
    while let Some(idx) = rest.find('我') {
        let after = &rest[idx + '我'.len_utf8()..];
        if after.starts_with('们') || after.starts_with('家') {
            rest = after;
            continue;
        }
        return true;
    }
    false
}

fn eq_name(a: &str, b: &str) -> bool {
    !a.trim().is_empty() && a.trim().eq_ignore_ascii_case(b.trim())
}

fn contains_ci(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    if hay.contains(needle) {
        return true;
    }
    hay.to_lowercase().contains(&needle.to_lowercase())
}

fn keyword_domains(content: &str, entities: &[String]) -> Value {
    let blob = format!("{content} {}", entities.join(" "));
    const TABLE: &[(&str, &[&str])] = &[
        (
            "medical",
            &[
                "医院",
                "就医",
                "过敏",
                "发烧",
                "体检",
                "血压",
                "糖尿病",
                "感冒",
                "症状",
                "医生",
                "用药",
                "处方",
                "复诊",
                "复查",
                "诊断",
                "哮喘",
                "药物",
                "治疗",
            ],
        ),
        (
            "education",
            &[
                "学校",
                "作业",
                "考试",
                "老师",
                "课程",
                "成绩",
                "幼儿园",
                "辅导",
                "上课",
                "补习",
                "补习班",
            ],
        ),
        (
            "travel",
            &[
                "出行", "机票", "高铁", "火车", "酒店", "旅游", "航班", "自驾", "行程",
            ],
        ),
        (
            "lifestyle",
            &["作息", "睡眠", "运动", "起床", "熬夜", "习惯"],
        ),
        (
            "diet",
            &[
                "饮食",
                "口味",
                "忌口",
                "早餐",
                "午餐",
                "晚餐",
                "菜谱",
                "不吃",
                "爱吃",
                "喜欢吃",
                "喜欢喝",
                "爱喝",
                "不喝",
                "吃辣",
                "素食",
            ],
        ),
        ("finance", &["账单", "工资", "花销", "报销", "缴费"]),
        ("social", &["朋友", "聚会", "亲戚", "做客"]),
        ("household", &["家务", "打扫", "物业", "维修"]),
    ];
    let mut found = Vec::new();
    for (id, words) in TABLE {
        if words.iter().any(|w| blob.contains(w)) {
            found.push(json!({ "id": id, "confidence": 0.75 }));
        }
    }
    if found.is_empty() {
        json!([{ "id": "unknown", "confidence": 0.0 }])
    } else {
        Value::Array(found)
    }
}

fn subjects(metadata: &Value) -> Vec<(&str, f64)> {
    metadata
        .get("subjects")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let id = item.get("member_id").and_then(|v| v.as_str())?;
                    let c = item
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    Some((id, c))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn domain_ids(metadata: &Value) -> Vec<&str> {
    metadata
        .get("domains")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    item.get("id")
                        .and_then(Value::as_str)
                        .or_else(|| item.as_str())
                })
                .filter_map(canonical_domain)
                .collect()
        })
        .unwrap_or_default()
}

fn primary_domain(metadata: &Value) -> &str {
    let mut best: Option<(&str, f64)> = None;
    if let Some(arr) = metadata.get("domains").and_then(|v| v.as_array()) {
        for item in arr {
            let Some(id) = item
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| item.as_str())
                .and_then(canonical_domain)
            else {
                continue;
            };
            if id == "unknown" {
                continue;
            }
            let c = item
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let replace = match best {
                None => true,
                Some((_, bc)) => c > bc,
            };
            if replace {
                best = Some((id, c));
            }
        }
    }
    best.map(|(id, _)| id).unwrap_or("unknown")
}

fn domain_uncertain(metadata: &Value) -> bool {
    let ids = domain_ids(metadata);
    if ids.is_empty() || ids.iter().all(|d| *d == "unknown") {
        return true;
    }
    let mut max_known = 0.0;
    if let Some(arr) = metadata.get("domains").and_then(|v| v.as_array()) {
        for item in arr {
            if item.get("id").and_then(|v| v.as_str()) == Some("unknown") {
                continue;
            }
            let c = item
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if c > max_known {
                max_known = c;
            }
        }
    }
    max_known < CONFIDENCE_LOCK
}

fn apply_sensitivity(meta: &mut serde_json::Map<String, Value>) {
    let medical = meta
        .get("domains")
        .and_then(|v| v.as_array())
        .is_some_and(|arr| {
            arr.iter().any(|d| {
                matches!(
                    d.get("id").and_then(|v| v.as_str()),
                    Some("medical" | "医疗健康")
                )
            })
        })
        || meta.get("tag").and_then(|v| v.as_str()) == Some("医疗健康");
    if medical {
        // Sensitivity is informational only. Product rule: memories default to
        // public; 「隐藏」is host-only (manual revise). Do not auto-set privacy.
        meta.insert("sensitivity".into(), json!("high"));
    }
}

/// True when metadata marks the memory as private / 隐藏.
pub fn is_privacy(metadata: &Value) -> bool {
    if metadata
        .get("privacy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }
    metadata
        .get("attributes")
        .and_then(|v| v.as_array())
        .is_some_and(|arr| {
            arr.iter().any(|a| {
                a.as_str()
                    .map(|s| s.eq_ignore_ascii_case("privacy"))
                    .unwrap_or(false)
            })
        })
}

/// Privacy visibility: non-private → shared; private → only when viewer is a subject.
/// No viewer (empty) keeps private rows visible (single-chat / unlocked listing).
pub fn visible_to_viewer(metadata: &Value, subject_ids: &[String], viewer: Option<&str>) -> bool {
    if !is_privacy(metadata) {
        return true;
    }
    let Some(viewer) = viewer.map(str::trim).filter(|s| !s.is_empty()) else {
        return true;
    };
    subject_ids.iter().any(|id| id == viewer)
}

fn normalize_domains(meta: &mut serde_json::Map<String, Value>) {
    let Some(domains) = meta.get("domains").and_then(Value::as_array) else {
        return;
    };
    let mut seen = std::collections::HashSet::new();
    let canonical: Vec<Value> = domains
        .iter()
        .filter_map(|item| {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| item.as_str())
                .and_then(canonical_domain)?;
            if !seen.insert(id) {
                return None;
            }
            let confidence = item
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            Some(json!({"id": id, "confidence": confidence}))
        })
        .collect();
    meta.insert("domains".into(), Value::Array(canonical));
}

/// Store a Chinese display tag with canonical English domains, never Chinese
/// domain IDs. Preserve matching explicit domains (notably `household`, whose
/// display tag is shared with `lifestyle`) and canonical facets for custom tags.
pub fn apply_tag(metadata: &Value, tag: &str) -> Value {
    let mut meta = metadata.as_object().cloned().unwrap_or_default();
    normalize_domains(&mut meta);
    let tag = tag.trim();
    let tag = if tag.is_empty() { "其他" } else { tag };
    let tag = tag_for_domain(tag).unwrap_or(tag);
    let known = domain_ids(&Value::Object(meta.clone()))
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if let Some(domain) = domain_for_tag(tag) {
        if !known.iter().any(|id| tag_for_domain(id) == Some(tag)) {
            meta.insert(
                "domains".into(),
                json!([{ "id": domain, "confidence": 1.0 }]),
            );
        }
    } else if known.is_empty() {
        meta.insert(
            "domains".into(),
            json!([{ "id": "unknown", "confidence": 0.0 }]),
        );
    }
    meta.insert("tag".into(), json!(tag));
    apply_sensitivity(&mut meta);
    Value::Object(meta)
}

impl MemoryRecord {
    pub fn household_sensitive(&self) -> bool {
        is_sensitive(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_ownership_is_shared_by_public_and_personal_writes() {
        let original = json!({"privacy":true,"source":{"message_id":"m1"},"custom":7});
        for alias in ["公共", "家庭", "全家", "大家", "共享", " public ", "PUBLIC"] {
            let meta = with_ownership(&original, &[alias.into()]).unwrap();
            assert_eq!(subject_id_list(&meta), [PUBLIC_MEMBER_ID]);
            assert_eq!(meta["scope"], "public");
            assert_eq!(meta["source"], original["source"]);
            assert_eq!(meta["custom"], 7);
            assert_eq!(meta["privacy"], true);
        }
        let meta = with_ownership(&original, &["zhang-san".into()]).unwrap();
        assert_eq!(subject_id_list(&meta), ["zhang-san"]);
        assert_eq!(meta["scope"], "members");
        assert_eq!(meta["privacy"], true);
        assert_eq!(subject_id_list(&with_ownership(&json!({}), &[]).unwrap()), [PUBLIC_MEMBER_ID]);
        assert!(with_ownership(&original, &[" ".into()]).is_err());
        assert!(with_ownership(&original, &["bad|id".into()]).is_err());
        assert!(with_ownership(&original, &["家庭".into(),"zhang-san".into()]).is_err());
        assert_eq!(canonical_owner_id("家庭医生张三").unwrap(), "家庭医生张三");
    }

    #[test]
    fn shared_filter_reads_legacy_aliases_without_matching_personal_rows() {
        let legacy = json!({"subjects":[],"scope":"public","attribution":"manual"});
        let alias = json!({"subjects":[{"member_id":"家庭","confidence":1.0}],"scope":"members"});
        let personal = with_ownership(&json!({}), &["zhang-san".into()]).unwrap();
        for owner in ["公共","家庭","PUBLIC"] {
            assert!(matches_member_and_tag(&legacy, &[], "", Some(owner), None, &[]));
            assert!(matches_member_and_tag(&alias, &["家庭".into()], "", Some(owner), None, &[]));
            assert!(!matches_member_and_tag(&personal, &["zhang-san".into()], "", Some(owner), None, &[]));
        }
        assert!(!matches_member_and_tag(&legacy, &[], "", Some("zhang-san"), None, &[]));
        assert!(!matches_member_and_tag(&json!({}), &[], "", Some("公共"), None, &[]));
    }

    fn roster() -> Vec<HouseholdMember> {
        vec![
            HouseholdMember {
                member_id: "主驾".into(),
                display_name: "主驾".into(),
                role: MemberRole::Self_,
                aliases: vec!["我".into()],
                age_band: None,
            },
            HouseholdMember {
                member_id: "小航".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec!["儿子".into()],
                age_band: None,
            },
        ]
    }

    #[test]
    fn content_owner_prefers_mentioned_child() {
        let id = infer_content_owner(&roster(), "我们家小航在实验中学上学", Some("主驾"));
        assert_eq!(id.as_deref(), Some("小航"));
    }

    #[test]
    fn content_owner_no_mention_stays_empty() {
        // 未点名任何人时不因 actor/speaker 强行归属
        let id = infer_content_owner(&roster(), "明天去医院体检", Some("主驾"));
        assert_eq!(id, None);
    }

    #[test]
    fn content_owner_first_person_maps_to_speaker() {
        let id = infer_content_owner(&roster(), "我喜欢吃辣", Some("主驾"));
        assert_eq!(id.as_deref(), Some("主驾"));
    }

    #[test]
    fn content_owner_kinship_maps_to_child() {
        let id = infer_content_owner(&roster(), "儿子在实验中学上学", Some("主驾"));
        assert_eq!(id.as_deref(), Some("小航"));
    }

    #[test]
    fn short_kinship_title_does_not_match_inside_longer_title() {
        let roster = vec![
            HouseholdMember {
                member_id: "爸爸".into(),
                display_name: "爸爸".into(),
                role: MemberRole::Parent,
                aliases: vec![],
                age_band: None,
            },
            HouseholdMember {
                member_id: "妈妈".into(),
                display_name: "妈妈".into(),
                role: MemberRole::Parent,
                aliases: vec![],
                age_band: None,
            },
        ];
        assert_eq!(
            infer_content_owner(&roster, "爸爸喜欢喝茶", None).as_deref(),
            Some("爸爸")
        );
        assert_eq!(
            ground_owner_from_source(&roster, "喜欢喝茶", "爸爸喜欢喝茶", None).as_deref(),
            Some("爸爸")
        );
        assert_eq!(
            complete_owned_content(&roster, "喜欢喝茶", "爸爸", "喜欢喝茶", "爸爸喜欢喝茶"),
            "爸爸喜欢喝茶"
        );
    }

    #[test]
    fn medical_annotate_sets_sensitivity_not_privacy() {
        let meta = annotate(
            &roster(),
            "小航对花生过敏",
            &[],
            Some("主驾"),
            &json!({}),
            None,
            &[],
        );
        let tagged = apply_tag(&meta, "医疗健康");
        // Default public: medical no longer auto-hides.
        assert!(!is_privacy(&tagged));
        assert!(is_sensitive(&tagged));
        assert!(visible_to_viewer(
            &tagged,
            &subject_id_list(&tagged),
            Some("主驾"),
        ));
        assert!(visible_to_viewer(
            &tagged,
            &subject_id_list(&tagged),
            Some("小航"),
        ));
        assert!(visible_to_viewer(&tagged, &subject_id_list(&tagged), None));
    }

    #[test]
    fn explicit_host_privacy_survives_medical_tag() {
        let meta = annotate(
            &roster(),
            "小航对花生过敏",
            &[],
            Some("主驾"),
            &json!({ "privacy": true, "attributes": ["privacy"] }),
            None,
            &[],
        );
        let tagged = apply_tag(&meta, "医疗健康");
        assert!(is_privacy(&tagged));
        assert!(is_sensitive(&tagged));
    }

    #[test]
    fn annotate_subjects_follow_content_mention() {
        let meta = annotate(
            &roster(),
            "我们家小航在实验中学上学",
            &[],
            Some("主驾"),
            &json!({}),
            None,
            &[],
        );
        let ids = subject_id_list(&meta);
        assert_eq!(ids, vec!["小航".to_string()]);
        assert_eq!(
            meta.get("attribution").and_then(|v| v.as_str()),
            Some("rule")
        );
    }

    #[test]
    fn annotate_without_subjects_ignores_member_id_when_content_names_child() {
        let meta = annotate(
            &roster(),
            "我们家小航在实验中学上学",
            &[],
            Some("主驾"),
            &json!({}),
            Some("主驾"), // 旧设计会锁死主驾；现应以正文为准
            &[],
        );
        assert_eq!(subject_id_list(&meta), vec!["小航".to_string()]);
    }

    #[test]
    fn annotate_host_subjects_manual_respected() {
        let meta = annotate(
            &roster(),
            "我们家小航在实验中学上学",
            &[],
            Some("主驾"),
            &json!({
                "subjects": [{"member_id": "主驾", "confidence": 1.0}],
                "attribution": "manual",
            }),
            None,
            &[],
        );
        assert_eq!(subject_id_list(&meta), vec!["主驾".to_string()]);
    }

    #[test]
    fn annotate_unscoped_content_uses_public_member() {
        let meta = annotate(
            &roster(),
            "明天去医院体检",
            &[],
            Some("主驾"),
            &json!({}),
            Some("主驾"),
            &[],
        );
        assert_eq!(subject_id_list(&meta), vec![PUBLIC_MEMBER_ID.to_string()]);
        assert_eq!(meta.get("scope").and_then(|v| v.as_str()), Some("public"));
    }
}

#[cfg(test)]
mod text_facet_regressions {
    use super::*;

    #[test]
    fn english_domains_match_legacy_chinese_facets() {
        let legacy = json!({"tag":"医疗健康", "domains":[{"id":"医疗健康", "confidence":1.0}]});
        assert!(matches_filter(&legacy, None, &["medical".into()]));
        let canonical = apply_tag(&legacy, "医疗健康");
        assert_eq!(canonical["domains"][0]["id"], "medical");
        assert!(matches_filter(&canonical, None, &["医疗健康".into()]));
        assert_eq!(tag_for_domain("medical"), Some("医疗健康"));
        assert_eq!(domain_for_tag("饮食"), Some("diet"));
        assert!(is_sensitive(&legacy));
    }

    #[test]
    fn validated_metadata_precedes_text_and_invalid_tags_do_not() {
        assert_eq!(
            select_memory_tag(&[], &json!({"tag":"医疗健康"}), "每周三固定安排", &[]),
            "医疗健康"
        );
        assert_eq!(
            select_memory_tag(
                &[],
                &json!({"domains":[{"id":"diet", "confidence":1.0}]}),
                "固定安排",
                &[]
            ),
            "饮食"
        );
        assert_eq!(
            select_memory_tag(&[], &json!({"tag":"模型杜撰标签"}), "喜欢吃肉", &[]),
            "饮食"
        );
        assert_eq!(infer_tag_from_text("吃花生会过敏", &[]), "医疗健康");
        assert_eq!(infer_tag_from_text("喜欢吃肉", &[]), "饮食");
        let custom = vec![MemoryTag {
            tag_id: "自定义".into(),
            display_name: "我的分类".into(),
            aliases: vec!["别名".into()],
        }];
        assert_eq!(
            select_memory_tag(&custom, &json!({"tag":"别名"}), "喜欢吃肉", &[]),
            "自定义"
        );
    }

    #[test]
    fn canonical_household_domain_survives_shared_display_tag() {
        let meta = apply_tag(
            &json!({"domains":[{"id":"household", "confidence":0.9}]}),
            "生活习惯",
        );
        assert_eq!(meta["domains"][0]["id"], "household");
        assert!(matches_filter(&meta, None, &["household".into()]));
    }
}
