//! 明确身份的短句偏好兜底。整句不能安全覆盖时交回端侧抽取。
use crate::reconcile::{EvidenceSource, ExtractionProgress, ReconcileReport, ReconcileRequest};
use crate::{HouseholdMember, ListQuery, MemoryEngine, MemoryService, RememberInput, Result};
use serde_json::json;
fn fragments(text: &str) -> Option<Vec<String>> {
    if text.chars().count() > 80
        || !text.starts_with('我')
        || [
            "？",
            "?",
            "什么",
            "吗",
            "呢",
            "可能",
            "也许",
            "如果",
            "假如",
            "以前",
            "不再",
            "忘掉",
            "别记",
            "不要记",
            "他说",
            "她说",
            "我老婆",
            "我老公",
            "我女儿",
            "我儿子",
        ]
        .iter()
        .any(|w| text.contains(w))
    {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    for part in text
        .trim_end_matches(['。', '！', '!'])
        .split(['，', ',', '。', '；', ';'])
    {
        let p = part
            .trim()
            .trim_start_matches('我')
            .trim_start_matches('也');
        if p.is_empty() {
            continue;
        }
        let core = p.trim_start_matches("周末").trim_start_matches("最近");
        if [
            "喜欢",
            "爱",
            "不喝",
            "不吃",
            "不喜欢",
            "不爱",
            "讨厌",
            "迷上了",
        ]
        .iter()
        .any(|v| core.starts_with(v))
            && core.chars().count() > 2
        {
            out.push(p.to_string());
        } else if !out.is_empty()
            && (p == "喝了睡不着"
                || p.starts_with("每天下午")
                || p.starts_with("每天早上")
                || p.starts_with("买了一套"))
        {
            out.last_mut().unwrap().push_str(&format!("，{}", p));
        } else {
            return None;
        }
    }
    (!out.is_empty()).then_some(out)
}
pub async fn remember(
    engine: &MemoryEngine,
    req: &ReconcileRequest,
    members: &[HouseholdMember],
    sources: &[EvidenceSource],
) -> Result<Option<ReconcileReport>> {
    if sources.len() != 1
        || sources[0].kind != "user_text"
        || sources[0].text.trim() != req.query.trim()
    {
        return Ok(None);
    }
    let Some(owner) = req.speaker_id.as_deref().and_then(|id| {
        members
            .iter()
            .find(|m| m.member_id == id && id != "主驾" && id != "公共")
    }) else {
        return Ok(None);
    };
    let Some(parts) = fragments(req.query.trim()) else {
        return Ok(None);
    };
    let mut written = Vec::new();
    let mut facts = Vec::new();
    let mut provenance = Vec::new();
    for part in parts {
        let content = format!("{}{}", owner.display_name, part);
        let existing = engine
            .list(ListQuery {
                namespace: req.namespace.clone(),
                member_id: Some(owner.member_id.clone()),
                keyword: Some(content.clone()),
                limit: Some(100),
                ..Default::default()
            })
            .await?;
        let id = if let Some(old) = existing.items.iter().find(|r| r.content == content) {
            old.id
        } else {
            let mut input = RememberInput::episodic(&content);
            input.member_id = Some(owner.member_id.clone());
            input.namespace = req.namespace.clone();
            input.session_id = req.session_id.clone();
            input.skip_consolidation = true;
            input.metadata = json!({"skip_graph":true,"rule":"short_preference_v1","source":{"query":req.query,"message_id":req.message_id,"session_id":req.session_id,"speaker_member_id":owner.member_id},"subjects":[{"member_id":owner.member_id,"confidence":1.0}]});
            let id = engine.remember(input).await?;
            written.push(id.to_string());
            id
        };
        facts.push(json!({"owner":owner.member_id,"content":content,"evidence":req.query}));
        provenance.push(json!({"memory_id":id,"source_id":sources[0].id,"evidence":req.query,"rule":"short_preference_v1"}));
    }
    Ok(Some(ReconcileReport {
        staged: false,
        skipped: false,
        skip_reason: None,
        speaker_id: owner.member_id.clone(),
        extract_raw: None,
        registers: 0,
        memories_extracted: facts.len(),
        memories_kept: facts.len(),
        retractions: 0,
        ops_raw: None,
        written_ids: written,
        revised_ids: vec![],
        forgotten: 0,
        fallback_used: true,
        warnings: vec![],
        facts,
        provenance,
        chunk_ids: vec![],
        extraction: ExtractionProgress::default(),
    }))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_preferences_preserves_negation_and_rejects_ambiguous() {
        assert_eq!(fragments("我喜欢吃酸菜鱼").unwrap(), vec!["喜欢吃酸菜鱼"]);
        assert_eq!(
            fragments("我周末喜欢打羽毛球，也爱看《繁花》这种年代剧")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            fragments("我不喝咖啡，喝了睡不着").unwrap(),
            vec!["不喝咖啡，喝了睡不着"]
        );
        for s in [
            "我喜欢吃什么菜",
            "我老婆喜欢吃鱼",
            "我可能喜欢跑步",
            "不要记我喜欢吃鱼",
            "我喜欢吃鱼，但现在不吃了",
        ] {
            assert!(fragments(s).is_none(), "{s}");
        }
    }
}
#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::{MemberRole, RecallQuery};
    #[tokio::test]
    async fn personal_preferences_store_split_dedup_and_recall_without_chat() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MemoryEngine::builder()
            .db_path(dir.path().join("p.db"))
            .build()
            .await
            .unwrap();
        let member = HouseholdMember {
            member_id: "speaker1".into(),
            display_name: "测试成员".into(),
            role: MemberRole::Other,
            aliases: vec![],
            age_band: None,
        };
        engine.upsert_member(None, member.clone()).await.unwrap();
        for text in [
            "我喜欢吃酸菜鱼",
            "我周末喜欢打羽毛球，也爱看《繁花》这种年代剧",
            "我不喝咖啡，喝了睡不着",
            "我最近迷上了拼乐高，买了一套故宫的",
            "我喜欢喝无糖拿铁，每天下午一杯",
        ] {
            let req: ReconcileRequest =
                serde_json::from_value(json!({"query":text,"speaker_id":"speaker1"})).unwrap();
            let source: EvidenceSource = serde_json::from_value(
                json!({"id":"s","kind":"user_text","name":"用户原话","text":text}),
            )
            .unwrap();
            let r = remember(&engine, &req, &[member.clone()], &[source.clone()])
                .await
                .unwrap()
                .unwrap();
            assert!(r.fallback_used);
            assert!(!r.written_ids.is_empty());
            let r = remember(&engine, &req, &[member.clone()], &[source])
                .await
                .unwrap()
                .unwrap();
            assert!(r.written_ids.is_empty());
        }
        for (query, expected) in [
            ("我喜欢吃什么菜", "酸菜鱼"),
            ("我周末喜欢什么运动", "羽毛球"),
            ("我爱看什么电视剧", "繁花"),
            ("我喝咖啡吗", "不喝咖啡"),
            ("我最近在玩什么", "乐高"),
            ("我下午喜欢喝什么", "无糖拿铁"),
        ] {
            let r = engine
                .recall(RecallQuery {
                    text: query.into(),
                    member_id: Some("speaker1".into()),
                    fts_only: true,
                    skip_graph: true,
                    top_k: Some(8),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert!(
                r.hits.iter().any(|h| h.memory.content.contains(expected)),
                "{query} missing {expected}"
            );
        }
    }
}
