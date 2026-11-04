//! Versioned curator prompts (extract / OCR / ops / knowledge-graph).
//!
//! Source of truth: `prompts/v1/*.txt`. Hosts should load via [`bundle`] rather
//! than hardcoding strings. Bump [`CURRENT_VERSION`] when the text changes.

pub const CURRENT_VERSION: &str = include_str!("../prompts/v1/VERSION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptBundle {
    pub version: &'static str,
    pub extract_system: &'static str,
    pub extract_rules: &'static str,
    pub extract_plan_system: &'static str,
    pub ocr_system: &'static str,
    pub ops_system: &'static str,
    pub ops_rules: &'static str,
    pub kg_extract_system: &'static str,
    pub kg_extract_rules: &'static str,
}

const V1: PromptBundle = PromptBundle {
    version: CURRENT_VERSION,
    extract_system: include_str!("../prompts/v1/extract_system.txt"),
    extract_rules: include_str!("../prompts/v1/extract_rules.txt"),
    extract_plan_system: include_str!("../prompts/v1/extract_plan_system.txt"),
    ocr_system: include_str!("../prompts/v1/ocr_system.txt"),
    ops_system: include_str!("../prompts/v1/ops_system.txt"),
    ops_rules: include_str!("../prompts/v1/ops_rules.txt"),
    kg_extract_system: include_str!("../prompts/v1/kg_extract_system.txt"),
    kg_extract_rules: include_str!("../prompts/v1/kg_extract_rules.txt"),
};

/// Return the prompt bundle for `version` (trim). Empty / `"latest"` / current → v1.
pub fn bundle(version: Option<&str>) -> Option<&'static PromptBundle> {
    let v = version
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("latest");
    let current = CURRENT_VERSION.trim();
    if matches!(v, "latest" | "v1") || v == current {
        Some(&V1)
    } else {
        None
    }
}

pub fn latest() -> &'static PromptBundle {
    &V1
}

/// Structured extraction repeats fields and exact evidence for each atomic fact.
/// A half-input budget truncates dense profiles even within one short chunk.
/// Keep ordinary calls bounded; long-text recovery may explicitly request more.
pub fn output_budget(chars: usize) -> usize {
    2400usize
        .saturating_add(chars.saturating_mul(3))
        .clamp(4096, 8192)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_extraction_budget_scales_then_saturates_without_overflow() {
        assert_eq!(output_budget(0), 4096);
        assert_eq!(output_budget(1200), 6000);
        assert_eq!(output_budget(2500), 8192);
        assert_eq!(output_budget(usize::MAX), 8192);
    }

    #[test]
    fn latest_has_content() {
        let b = latest();
        assert!(b.extract_system.contains("事实保全"));
        assert!(b.extract_rules.contains("原子拆分"));
        assert!(b.ocr_system.contains("OCR"));
        assert!(b.ops_rules.contains("noop"));
        assert!(b.kg_extract_system.contains("知识图谱"));
        assert!(b.kg_extract_system.contains("实体 Entity"));
        assert!(b.kg_extract_system.contains("关系 Relation"));
        assert!(b.kg_extract_system.contains("话语目的 ≠ 断言集合"));
        assert!(b.kg_extract_rules.contains("可指称"));
        assert!(b.kg_extract_rules.contains("有向"));
        assert!(b.kg_extract_rules.contains("一次调用 = n 条"));
        assert!(b.kg_extract_rules.contains("嵌入断言判定"));
        assert!(b.kg_extract_rules.contains("evidence_text"));
        assert!(b.kg_extract_rules.contains("complete 必填"));
        assert!(!b.version.trim().is_empty());
    }

    #[test]
    fn compact_prompts_keep_protocol_and_safety_contracts() {
        let b = latest();
        let memory = format!("{}\n{}", b.extract_system, b.extract_rules);
        let graph = format!("{}\n{}", b.kg_extract_system, b.kg_extract_rules);
        assert!(memory.chars().count() <= 2_000, "memory prompt regrew");
        assert!(graph.chars().count() <= 2_000, "graph prompt regrew");
        for marker in [
            "NDJSON",
            "{\"done\":true}",
            "\"o\"",
            "\"t\"",
            "\"c\"",
            "\"e\"",
            "\"k\":\"r\"",
            "\"k\":\"f\"",
            "YYYY-MM-DD",
            "连续原文",
            "speaker_id",
            "member_id",
            "不猜",
            "附件",
            "凭证",
            "2026-10-25",
            "代词",
            "不兜底具体个人",
            "第一人称",
            "不得映射名册成员",
            "自报身份",
        ] {
            assert!(memory.contains(marker), "missing memory contract: {marker}");
        }
        for marker in [
            "entities",
            "triples",
            "pending",
            "complete",
            "evidence_text",
            "temp_id",
            "subject",
            "object",
            "relation",
            "member_id",
            "polarity",
            "status",
            "condition_text",
            "attrs",
            "domains",
            "owner_private",
            "message_at_china",
            "有生日",
            "2026-11-06",
            "不猜测",
            "连续原文",
        ] {
            assert!(graph.contains(marker), "missing graph contract: {marker}");
        }
    }

    #[test]
    fn graph_uses_specific_chinese_predicates_and_contextual_family_directions() {
        let rules = latest().kg_extract_rules;
        for marker in [
            "具体中文有向谓词",
            "不限白名单",
            "属性/有属性/关系/关联",
            "林子航→小名→航航",
            "出生日期/工作单位/喜欢",
            "宾语是主语的什么人",
            "林子航→爸爸→林志远",
            "林志远→儿子→林子航",
            "林志远→老婆→宋娜娜",
            "宋娜娜→老公→林志远",
            "妈妈及反向须母子依据",
            "性别不明用子女",
            "不因共同孩子认定夫妻",
            "可靠上下文无需逐字明述",
            "domains=[\"household\"]",
            "不靠名册角色或其他来源",
            "推理不放宽来源隐私",
        ] {
            assert!(
                rules.contains(marker),
                "missing family/predicate contract: {marker}"
            );
        }
        assert!(!rules.contains("须输出「家庭成员」或「亲属」边"));
        assert!(!rules.contains("并用「有属性」边"));
    }

    #[test]
    fn graph_ids_aliases_and_inferred_evidence_have_explicit_contracts() {
        let rules = latest().kg_extract_rules;
        for marker in [
            "本批 entities 已定义的唯一 temp_id",
            "非成员 ID/名称",
            "不补未定义 e2实体",
            "「公共」成员 ID 只归属",
            "公共交通/E2设备则保留",
            "不按词面排除",
            "不按关键词筛选",
            "昵称值可为 concept，非另一人物",
            "小名/别名/称呼/姓名",
            "attrs.alias_kind=nickname|alias|address|name",
            "仅肯定/当前有效/无条件时更新别名",
            "推理边必须在 attrs 内加",
            "assertion_basis=\"context_inference\"",
            "inference_note=\"简短可核验依据，非思维链\"",
            "evidence_spans=[\"连续原文\",...]",
            "spans 非空",
            "每段连续出自 source_text",
            "evidence_text 取其中一段",
            "直接断言不必加这些字段",
        ] {
            assert!(
                rules.contains(marker),
                "missing graph evidence/identity contract: {marker}"
            );
        }
        assert!(!rules.contains("两端引用本批 temp_id、成员 ID 或实体名"));
    }

    #[test]
    fn unknown_version_none() {
        assert!(bundle(Some("1999.01.01")).is_none());
        assert!(bundle(Some("latest")).is_some());
    }
}
