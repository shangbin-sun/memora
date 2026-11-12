//! 有来源标识的演示家庭预置。仅显式宿主请求启用，幂等、无模型依赖。
use crate::graph::{self, FactObject, KgEntity, KgFact};
use crate::{core::now_ms, household, store, types::*, Result};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use uuid::Uuid;

pub fn seed(conn: &Connection, ns: &str, jieba: &jieba_rs::Jieba, body: &Value) -> Result<Value> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS family_demo_versions(namespace TEXT PRIMARY KEY,version INTEGER NOT NULL)")?;
    let version = body["data"]["version"].as_i64().unwrap_or(1);
    let existing = conn
        .query_row(
            "SELECT version FROM family_demo_versions WHERE namespace=?1",
            [ns],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0);
    if existing == version {
        return Ok(json!({"seeded":true,"already_seeded":true}));
    }
    // 只替换预设来源，不触碰用户对话、上传及其他普通记忆。
    let mut stmt = conn.prepare("SELECT id FROM memories WHERE namespace=?1 AND (json_extract(metadata_json,'$.family_demo')=1 OR json_extract(metadata_json,'$.source_id')='preset:wrong_questions')")?;
    let old = stmt
        .query_map([ns], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);
    for id in old {
        let id = Uuid::parse_str(&id)
            .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
        graph::store::withdraw_facts_by_source_memory(conn, ns, id)?;
        store::delete_memory(conn, id)?;
    }
    let data = &body["data"];
    for m in data["members"].as_array().into_iter().flatten() {
        let aliases: Vec<String> = m["aliases"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        household::upsert_member(
            conn,
            ns,
            &household::HouseholdMember {
                member_id: m["id"].as_str().unwrap().into(),
                display_name: m["name"].as_str().unwrap().into(),
                role: household::MemberRole::parse(m["role"].as_str().unwrap())?,
                age_band: if m["role"] == "child" {
                    Some("child".into())
                } else {
                    Some("adult".into())
                },
                aliases: aliases.clone(),
            },
        )?;
        graph::store::ensure_person_entity(
            conn,
            ns,
            m["id"].as_str().unwrap(),
            m["name"].as_str().unwrap(),
            &aliases,
        )?;
    }
    let now = now_ms();
    let mut count = 0;
    for (i, f) in data["facts"].as_array().into_iter().flatten().enumerate() {
        let owner = f["owner"].as_str().unwrap();
        let tag = f["tag"].as_str().unwrap();
        let domain = f["domain"].as_str().unwrap();
        let content = f["content"].as_str().unwrap();
        let id = Uuid::now_v7();
        let meta = household::apply_tag(
            &json!({"family_demo":true,"seed_version":version,"seed_index":i,"source_kind":"family_demo_preset","source_label":f["source"],"keywords":f["keywords"],"subjects":[{"member_id":owner,"confidence":1.0}]}),
            tag,
        );
        let r = MemoryRecord {
            id,
            revision: 1,
            kind: MemoryKind::Episodic,
            state: MemoryState::Active,
            content: content.into(),
            summary: None,
            importance: 0.8,
            confidence: 1.0,
            salience: 0.8,
            created_at_ms: now,
            updated_at_ms: now,
            last_accessed_at_ms: None,
            next_review_at_ms: None,
            review_count: 0,
            source_hash: None,
            embedding_model: None,
            embedding_version: 1,
            embedding_pending: true,
            consolidation_pending: false,
            parent_id: None,
            namespace: ns.into(),
            session_id: Some("family-demo-preset-v1".into()),
            entities: vec![owner.into()],
            metadata: meta,
            tag: tag.into(),
            subject_ids: vec![owner.into()],
        };
        store::insert_memory(conn, &r, jieba)?;
        store::insert_version(conn, id, "create", None, &r, "family_demo", None)?;
        let target = if let Some(member) = f["target_member"].as_str() {
            graph::store::person_entity_id(member)
        } else {
            let target = format!("family_demo:{}:{}", ns, i);
            graph::store::upsert_entity(
                conn,
                ns,
                &KgEntity {
                    id: target.clone(),
                    entity_type: graph::EntityType::parse(f["entity_type"].as_str().unwrap())?,
                    subtype: None,
                    canonical_name: f["entity_name"].as_str().unwrap().into(),
                    aliases: vec![],
                    status: "active".into(),
                    domains: vec![domain.into()],
                    owner_member_id: Some(owner.into()),
                    privacy: graph::PrivacyLevel::FamilyShared,
                    first_seen_at_ms: now,
                    last_confirmed_at_ms: now,
                    attrs: json!({"source_label":f["source"]}),
                },
            )?;
            target
        };
        let fact = KgFact {
            id: Uuid::now_v7(),
            subject_id: graph::store::person_entity_id(owner),
            relation: f["relation"].as_str().unwrap().into(),
            object: FactObject::entity(target),
            owner_member_id: Some(owner.into()),
            domains: vec![domain.into()],
            polarity: Default::default(),
            status: Default::default(),
            valid_from_ms: None,
            valid_to_ms: None,
            condition_text: None,
            exception_text: None,
            reason_text: None,
            evidence_text: Some(content.into()),
            evidence_summary: Some(content.into()),
            source_memory_id: Some(id),
            confidence: 1.0,
            privacy: graph::PrivacyLevel::FamilyShared,
            revision: 1,
            created_at_ms: now,
            updated_at_ms: now,
            attrs: json!({"source_label":f["source"]}),
        };
        graph::store::write_fact(conn, ns, &fact)?;
        graph::store::link_fact_source(conn, ns, &fact.id.to_string(), Some(id))?;
        count += 1;
    }
    conn.execute("INSERT INTO family_demo_versions(namespace,version) VALUES(?1,?2) ON CONFLICT(namespace) DO UPDATE SET version=excluded.version",params![ns,version])?;
    Ok(json!({"seeded":true,"count":count}))
}

pub fn recall(conn: &Connection, ns: &str, body: &Value) -> Result<Value> {
    let query = body["query"].as_str().unwrap_or("");
    let members = household::list_members(conn, ns)?;
    // 长别名优先：航航妈妈不能同时被短别名“航航”解析成孩子。
    let named: Vec<_> = members
        .iter()
        .map(|m| {
            let length = std::iter::once(m.display_name.as_str())
                .chain(m.aliases.iter().map(String::as_str))
                .filter(|name| !name.is_empty() && query.contains(name))
                .map(|name| name.chars().count())
                .max()
                .unwrap_or(0);
            (m, length)
        })
        .collect();
    let longest = named.iter().map(|(_, n)| *n).max().unwrap_or(0);
    let mut owners: Vec<String> = named
        .iter()
        .filter(|(_, n)| *n > 0 && *n == longest)
        .map(|(m, _)| m.member_id.clone())
        .collect();
    // 预设家庭的亲属指称在主题匹配前解析；最后一个亲属词表示最终主体，
    // 因此“我儿子的妈妈”与“我妈妈的儿子”不会反向归属。
    let nearby_at = ["附近", "周边", "旁边", "周围"]
        .iter()
        .filter_map(|word| query.find(word))
        .min()
        .unwrap_or(query.len());
    let subject = &query[..nearby_at];
    let child_at = ["儿子", "孩子", "小孩", "娃", "小朋友", "宝宝"]
        .iter()
        .filter_map(|word| subject.rfind(word))
        .max();
    let mother_at = ["妈妈", "母亲", "母子"]
        .iter()
        .filter_map(|word| subject.rfind(word))
        .max();
    let relative_name = match (child_at, mother_at) {
        (Some(c), Some(m)) if c > m => Some("梁子航"),
        (_, Some(_)) => Some("宋娜娜"),
        (Some(_), None) => Some("梁子航"),
        _ => None,
    };
    if let Some(name) = relative_name {
        owners = members
            .iter()
            .filter(|m| m.display_name == name)
            .map(|m| m.member_id.clone())
            .collect();
    }
    if owners.is_empty() {
        if let Some(id) = body["viewer_member_id"].as_str() {
            owners = members
                .iter()
                .filter(|m| {
                    m.member_id == id || m.display_name == id || m.aliases.iter().any(|a| a == id)
                })
                .map(|m| m.member_id.clone())
                .collect();
        }
    }
    let workplace = ["上班", "单位", "工作", "医院", "下班", "公司"]
        .iter()
        .any(|word| subject.contains(word));
    let school = [
        "学校",
        "校门",
        "上学",
        "放学",
        "接娃",
        "接孩子",
        "接儿子",
        "小学",
    ]
    .iter()
    .any(|word| subject.contains(word));
    let mut search = subject.to_string();
    if workplace {
        search.push_str(" 工作单位");
    }
    if school {
        search.push_str(" 学校");
    }
    if owners.is_empty() {
        return Ok(json!({"hits":[]}));
    }
    let mut stmt=conn.prepare("SELECT id,metadata_json FROM memories WHERE namespace=?1 AND state='active' AND json_extract(metadata_json,'$.family_demo')=1")?;
    let rows = stmt.query_map([ns], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut hits = vec![];
    for row in rows {
        let (id, meta) = row?;
        let meta: Value = serde_json::from_str(&meta)?;
        let owner = meta["subjects"][0]["member_id"].as_str().unwrap_or("");
        if !owners.iter().any(|o| o == owner) {
            continue;
        }
        if let Some(member) = body["member_id"].as_str() {
            if owner != member {
                continue;
            }
        }
        if let Some(tag) = body["tag"].as_str() {
            if meta["tag"].as_str() != Some(tag) {
                continue;
            }
        }
        if household::is_privacy(&meta) {
            continue;
        }
        let mut score = meta["keywords"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|key| search.contains(key))
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(0);
        if score == 0 {
            continue;
        }
        // 查附近商户时，地点事实优先于“妈妈/儿子是谁”等关系事实。
        let keys = meta["keywords"].as_array();
        if keys.is_some_and(|keys| {
            keys.iter()
                .any(|k| (school && k == "学校") || (workplace && k == "单位"))
        }) {
            score += 10;
        }
        if let Some(memory) = store::get_memory(conn, Uuid::parse_str(&id).unwrap())? {
            hits.push(json!({"memory":memory,"scores":{"final_score":2.0+score as f64/10.0}}));
        }
    }
    hits.sort_by(|a, b| {
        b["scores"]["final_score"]
            .as_f64()
            .partial_cmp(&a["scores"]["final_score"].as_f64())
            .unwrap()
    });
    hits.truncate(body["top_k"].as_u64().unwrap_or(8) as usize);
    Ok(json!({"hits":hits}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ListQuery, MemoryEngine, MemoryService};
    #[tokio::test]
    async fn preset_atomic_idempotent_graph_and_recall_filters() {
        let d = tempfile::tempdir().unwrap();
        let e = MemoryEngine::builder()
            .db_path(d.path().join("m.db"))
            .build()
            .await
            .unwrap();
        let data = json!({"version":1,"members":[{"id":"child","name":"梁子航","role":"child","aliases":["航航"]}],"facts":[{"owner":"child","content":"梁子航对花生过敏。","tag":"医疗健康","domain":"medical","source":"演示补充","keywords":["过敏","不能吃"],"entity_type":"substance","entity_name":"花生","relation":"过敏"}]});
        assert_eq!(
            e.wrong_practice(json!({"action":"family_seed","data":data}))
                .await
                .unwrap()["count"],
            1
        );
        assert_eq!(
            e.wrong_practice(json!({"action":"family_seed","data":data}))
                .await
                .unwrap()["already_seeded"],
            true
        );
        let p = e.list(ListQuery::default()).await.unwrap();
        assert_eq!(p.total, 1);
        let r = e
            .wrong_practice(json!({"action":"family_recall","query":"航航不能吃什么"}))
            .await
            .unwrap();
        assert_eq!(r["hits"].as_array().unwrap().len(), 1);
        let r = e
            .wrong_practice(
                json!({"action":"family_recall","query":"航航不能吃什么","member_id":"other"}),
            )
            .await
            .unwrap();
        assert_eq!(r["hits"].as_array().unwrap().len(), 0);
        let r = e
            .wrong_practice(json!({"action":"family_recall","query":"航航不能吃什么","tag":"金融"}))
            .await
            .unwrap();
        assert_eq!(r["hits"].as_array().unwrap().len(), 0);
        let graph = e
            .query_person_graph(graph::PersonGraphQuery {
                member_id: "child".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(graph.facts.len(), 1);
        assert!(graph.entities.iter().any(|n| n.canonical_name == "花生"));
    }
    #[tokio::test]
    async fn changed_preset_replaces_only_seed_records_and_imports_questions() {
        let d = tempfile::tempdir().unwrap();
        let e = MemoryEngine::builder()
            .db_path(d.path().join("m.db"))
            .build()
            .await
            .unwrap();
        e.remember(crate::RememberInput::episodic("用户自己添加的记忆"))
            .await
            .unwrap();
        let mut data = json!({"version":200,"members":[{"id":"child","name":"梁子航","role":"child","aliases":["航航"]}],"facts":[{"owner":"child","content":"梁子航对花生过敏。","tag":"医疗健康","domain":"medical","source":"演示补充","keywords":["过敏"],"entity_type":"substance","entity_name":"花生","relation":"过敏"}]});
        let wrong =
            "错题集\n姓名：梁子航\n2. 判断：所有偶数都是合数。（√）\n正确答案：×\n解析：2是质数。";
        let result = e
            .wrong_practice(json!({"action":"family_seed","data":data,"wrong_questions":wrong}))
            .await
            .unwrap();
        assert_eq!(result["wrong_questions"], 1);
        assert_eq!(e.list(ListQuery::default()).await.unwrap().total, 3);
        // 内容hash可能比上次小，仍然应该更新。
        data["version"] = json!(100);
        data["facts"][0]["content"] = json!("梁子航对芝麻过敏。");
        data["facts"][0]["entity_name"] = json!("芝麻");
        e.wrong_practice(json!({"action":"family_seed","data":data,"wrong_questions":wrong}))
            .await
            .unwrap();
        let page = e.list(ListQuery::default()).await.unwrap();
        assert_eq!(page.total, 3);
        assert!(page.items.iter().any(|m| m.content == "用户自己添加的记忆"));
        assert!(page.items.iter().any(|m| m.content == "梁子航对芝麻过敏。"));
        assert!(!page.items.iter().any(|m| m.content == "梁子航对花生过敏。"));
        let graph = e
            .query_person_graph(graph::PersonGraphQuery {
                member_id: "child".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(graph.facts.len(), 1);
    }
    #[tokio::test]
    async fn indirect_places_resolve_aliases_and_relatives_before_keywords() {
        let d = tempfile::tempdir().unwrap();
        let e = MemoryEngine::builder()
            .db_path(d.path().join("m.db"))
            .build()
            .await
            .unwrap();
        let data = json!({"members":[
            {"id":"child","name":"梁子航","role":"child","aliases":["航航"]},
            {"id":"mom","name":"宋娜娜","role":"parent","aliases":["航航妈妈"]}],
            "facts":[
            {"owner":"mom","content":"宋娜娜的工作单位是北京同仁医院亦庄院区。","tag":"工作学习","domain":"education","source":"用户指定","keywords":["工作","单位"],"entity_type":"organization","entity_name":"北京同仁医院亦庄院区","relation":"工作单位"},
            {"owner":"child","content":"梁子航在北京亦庄实验小学读一年级。","tag":"工作学习","domain":"education","source":"用户指定","keywords":["学校"],"entity_type":"school","entity_name":"北京亦庄实验小学","relation":"就读"}]});
        e.wrong_practice(json!({"action":"family_seed","data":data}))
            .await
            .unwrap();
        for (viewer, query, expected) in [
            ("mom", "我工作单位附近有哪些咖啡店", "同仁医院"),
            ("mom", "我儿子学校附近有哪些文具店", "实验小学"),
            ("mom", "我是宋娜娜，我儿子学校旁边有什么公园", "实验小学"),
            ("child", "航航妈妈上班的地方附近有药店吗", "同仁医院"),
            ("mom", "我单位附近适合孩子吃饭的地方", "同仁医院"),
            ("child", "我妈妈的单位旁边有什么学校", "同仁医院"),
            ("mom", "我儿子的妈妈工作单位附近有什么公园", "同仁医院"),
            ("mom", "我妈妈的儿子学校附近有什么文具店", "实验小学"),
        ] {
            let r = e
                .wrong_practice(
                    json!({"action":"family_recall","query":query,"viewer_member_id":viewer}),
                )
                .await
                .unwrap();
            assert!(
                r["hits"][0]["memory"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .contains(expected),
                "{query}: {r}"
            );
        }
        let filtered=e.wrong_practice(json!({"action":"family_recall","query":"我儿子学校附近的书店","viewer_member_id":"mom","member_id":"mom"})).await.unwrap();
        assert!(filtered["hits"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn manual_revision_reembeds_the_latest_fact_and_retires_old_vector() {
        use std::sync::{Arc, Mutex};
        struct RecordingEmbedder(Mutex<Vec<String>>);
        #[async_trait::async_trait]
        impl crate::Embedder for RecordingEmbedder {
            fn model_id(&self) -> &str {
                "revision-unit"
            }
            fn dimension(&self) -> usize {
                2
            }
            async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
                self.0.lock().unwrap().extend(texts.iter().cloned());
                Ok(texts
                    .iter()
                    .map(|s| {
                        if s.contains("新单位") {
                            vec![0.0, 1.0]
                        } else {
                            vec![1.0, 0.0]
                        }
                    })
                    .collect())
            }
        }
        let d = tempfile::tempdir().unwrap();
        let db = d.path().join("m.db");
        let embed = Arc::new(RecordingEmbedder(Mutex::new(vec![])));
        let e = MemoryEngine::builder()
            .db_path(&db)
            .embedder(embed.clone())
            .build()
            .await
            .unwrap();
        let id = e
            .remember(crate::RememberInput::episodic("宋娜娜在旧单位上班"))
            .await
            .unwrap();
        for _ in 0..100 {
            if !e.get(id).await.unwrap().unwrap().embedding_pending {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(!e.get(id).await.unwrap().unwrap().embedding_pending);
        e.revise(
            id,
            crate::MemoryUpdate {
                content: Some("宋娜娜在新单位上班".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for _ in 0..100 {
            if !e.get(id).await.unwrap().unwrap().embedding_pending {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let rec = e.get(id).await.unwrap().unwrap();
        assert!(!rec.embedding_pending);
        assert_eq!(rec.embedding_version, 2);
        assert_eq!(
            *embed.0.lock().unwrap(),
            vec!["宋娜娜在旧单位上班", "宋娜娜在新单位上班"]
        );
        let conn = Connection::open(&db).unwrap();
        let (version, vector): (i64, Vec<u8>) = conn
            .query_row(
                "SELECT version,vector FROM embeddings WHERE memory_id=?1",
                [id.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(version, 2);
        assert_eq!(
            vector,
            [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat()
        );
    }
}
