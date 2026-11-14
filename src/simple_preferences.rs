//! 明确身份的八类短句事实兜底。整句不能安全覆盖时交回抽取与逐事实规则验证。
use crate::reconcile::{EvidenceSource, ExtractionProgress, ReconcileReport, ReconcileRequest};
use crate::{HouseholdMember, ListQuery, MemoryEngine, MemoryService, RememberInput, Result};
use serde_json::json;
#[cfg(test)]
fn fragments(text: &str) -> Option<Vec<String>> {
    let member=HouseholdMember {member_id:"test".into(),display_name:"测试".into(),role:crate::MemberRole::Other,aliases:vec![],age_band:None};
    let result=crate::memory_rules::analyze(text,&[member],Some("test"));
    result.complete.then(||result.facts.into_iter().map(|f|f.content.trim_start_matches("测试").to_owned()).collect())
}
pub async fn remember(
    engine: &MemoryEngine,
    req: &ReconcileRequest,
    members: &[HouseholdMember],
    sources: &[EvidenceSource],
) -> Result<Option<ReconcileReport>> {
    if sources.len() != 1
        || sources[0].kind != "user_text"
        || sources[0].truncated
        || sources[0].text.trim() != req.query.trim()
    {
        return Ok(None);
    }
    let analysis=crate::memory_rules::analyze(req.query.trim(),members,req.speaker_id.as_deref());
    if !analysis.complete { return Ok(None); }
    let mut written = Vec::new();
    let mut facts = Vec::new();
    let mut provenance = Vec::new();
    for fact in analysis.facts {
        let owner=members.iter().find(|m|m.member_id==fact.owner).expect("rule owner is roster member");
        let content=fact.content;

        let existing = engine
            .list(ListQuery {
                namespace: req.namespace.clone(),
                member_id: Some(owner.member_id.clone()),
                keyword: Some(content.clone()),
                limit: Some(100),
                ..Default::default()
            })
            .await?;
        let canonical_old=crate::turn_control::duplicate(engine,req.namespace.clone(),&content,&[owner.member_id.clone()],&fact.tag).await?;
        let id = if let Some(old) = canonical_old.as_ref().or_else(||existing.items.iter().find(|r| r.content == content)) {
            old.id
        } else {
            let mut input = RememberInput::episodic(&content);
            input.member_id = Some(owner.member_id.clone());
            input.namespace = req.namespace.clone();
            input.session_id = req.session_id.clone();
            input.skip_consolidation = true;
            input.metadata = crate::household::apply_tag(&json!({"skip_graph":true,"rule":"grounded_fact_rules_v2","source":{"query":fact.evidence,"evidence":fact.evidence,"source_id":sources[0].id,"kind":"user_text","message_id":req.message_id,"session_id":req.session_id,"speaker_member_id":req.speaker_id.as_deref().filter(|id|members.iter().any(|m|m.member_id==*id))},"subjects":[{"member_id":owner.member_id,"confidence":1.0}]}),fact.tag);
            // 规则已用原文与声纹ID确认归属，同名名册不能再次扩充成多个拥有者。
            input.metadata = crate::household::with_ownership(&input.metadata, &[owner.member_id.clone()])?;
            let id = engine.remember(input).await?;
            written.push(id.to_string());
            id
        };
        facts.push(json!({"owner":owner.member_id,"content":content,"evidence":fact.evidence,"tag":fact.tag}));
        provenance.push(json!({"memory_id":id,"source_id":sources[0].id,"evidence":fact.evidence,"tag":fact.tag,"rule":"grounded_fact_rules_v2"}));
    }
    Ok(Some(ReconcileReport {
        staged: false,
        skipped: false,
        skip_reason: None,
        speaker_id: req.speaker_id.clone().unwrap_or_default(),
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

#[cfg(test)]
mod enrolled_owner_tests {
    use super::*;
    #[tokio::test]
    async fn enrolled_id_wins_over_duplicate_display_name_and_compound_preferences_are_recallable() {
        let dir=tempfile::tempdir().unwrap();
        let engine=MemoryEngine::builder().db_path(dir.path().join("known.db")).build().await.unwrap();
        let mut members=Vec::new();
        for id in ["enrolled-mom","other-mom"] {
            let member=HouseholdMember{member_id:id.into(),display_name:"宋娜娜".into(),role:crate::MemberRole::Other,aliases:vec![],age_band:None};
            engine.upsert_member(None,member.clone()).await.unwrap();members.push(member);
        }
        let text="我喜欢吃酸菜鱼，平时爱喝桂花乌龙茶。";
        let req:ReconcileRequest=serde_json::from_value(json!({"query":text,"speaker_id":"enrolled-mom"})).unwrap();
        let source:EvidenceSource=serde_json::from_value(json!({"id":"live","kind":"user_text","text":text})).unwrap();
        let result=remember(&engine,&req,&members,&[source]).await.unwrap().unwrap();
        assert_eq!(result.written_ids.len(),2);
        let rows=engine.list(ListQuery::default()).await.unwrap();assert_eq!(rows.total,2);
        for f in &rows.items {assert_eq!(f.subject_ids,vec!["enrolled-mom"]);assert_eq!(f.tag,"饮食");}
        let query=|id:&str|crate::RecallQuery{text:"桂花乌龙茶".into(),member_id:Some(id.into()),fts_only:true,skip_graph:true,..Default::default()};
        let hits=engine.recall(query("enrolled-mom")).await.unwrap();
        assert!(hits.hits.iter().any(|h|h.memory.content=="宋娜娜平时爱喝桂花乌龙茶"));
        assert!(engine.recall(query("other-mom")).await.unwrap().hits.is_empty());
    }
}
