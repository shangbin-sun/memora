//! 可配置的错题规则通道。正文紧凑展示，原文和题号保留在来源元数据。
use crate::{core::now_ms, household, store, types::*, Result};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use uuid::Uuid;

fn has(s: &str, words: &[&str]) -> bool {
    words.iter().any(|w| s.contains(w))
}
fn clean(s: &str) -> String {
    s.replace('\\', "").replace("**", "").trim().to_string()
}
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
pub(crate) fn refresh_edited_question(rec: &mut MemoryRecord) {
    let text = clean(&rec.content);
    let child = observed_child_answer(&text, "");
    rec.content = compact_record(&text, &child);
    rec.metadata["child_answer"] = json!(child);
    rec.metadata["question_edited"] = json!(true);
    if let Some(meta) = rec.metadata.as_object_mut() {
        meta.remove("last_presented_upload_ms");
    }
}
#[derive(Debug)]
struct ChoiceSpan {
    label: char,
    start: usize,
    body_start: usize,
    end: usize,
}
fn choice_spans(text: &str) -> Vec<ChoiceSpan> {
    let re = Regex::new(r"(?m)(?:^|[\s　])([A-D])(?:[.．、][ \t]*|[ \t]+)").unwrap();
    let matches: Vec<_> = re.captures_iter(text).collect();
    matches
        .iter()
        .enumerate()
        .map(|(i, c)| ChoiceSpan {
            label: c[1].chars().next().unwrap(),
            start: c.get(0).unwrap().start(),
            body_start: c.get(0).unwrap().end(),
            end: matches
                .get(i + 1)
                .map(|next| next.get(0).unwrap().start())
                .unwrap_or(text.len()),
        })
        .collect()
}
fn grading_mark() -> Regex {
    Regex::new(r"(?:[×✗✘❌]|[ \t]*[Xx])[ \t　。.!！\r\n]*$").unwrap()
}
fn before_answer_fields(text: &str) -> &str {
    Regex::new(r"(?:订正|訂正|改正|更正|正确答案|标准答案|参考答案|解析|孩子答案|孩子选择)[：:]")
        .unwrap()
        .find(text)
        .map(|m| &text[..m.start()])
        .unwrap_or(text)
}
fn question(text: &str) -> String {
    let normalized = text
        .replace(['Ⓐ', 'ⓐ'], "A.")
        .replace(['Ⓑ', 'ⓑ'], "B.")
        .replace(['Ⓒ', 'ⓒ'], "C.")
        .replace(['Ⓓ', 'ⓓ'], "D.");
    let normalized = Regex::new(r"[（(]([A-D])[）)]")
        .unwrap()
        .replace_all(&normalized, "$1.");
    let text = normalized.as_ref();
    let text = Regex::new(r"孩子答案[：:]")
        .unwrap()
        .find(text)
        .map(|m| &text[..m.start()])
        .unwrap_or(text);
    let t = Regex::new(r"孩子选择[：:]\s*[A-D]\s*")
        .unwrap()
        .replace_all(text, "");
    let t = Regex::new(r"(?:订正|訂正|改正|更正|正确答案|解析)[：:]")
        .unwrap()
        .split(&t)
        .next()
        .unwrap_or("")
        .to_string();
    let t = t
        .split("答案")
        .next()
        .unwrap_or(&t)
        .trim()
        .trim_end_matches("正确");
    let choices = choice_spans(t);
    if choices.len() >= 2 {
        let mark = grading_mark();
        let options: Vec<_> = choices
            .iter()
            .map(|option| {
                let body = &t[option.body_start..option.end];
                let value = if mark
                    .find(body)
                    .is_some_and(|m| !body[..m.start()].trim().is_empty())
                {
                    mark.replace(body, "")
                } else {
                    std::borrow::Cow::Borrowed(body)
                };
                format!("{}. {}", option.label, value.trim())
            })
            .collect();
        return format!("{}\n{}", t[..choices[0].start].trim(), options.join("　"));
    }
    Regex::new(r"[（(][√×][）)]|[×✗❌]")
        .unwrap()
        .replace_all(t, |caps: &regex::Captures| {
            let m = caps.get(0).unwrap();
            let left = t[..m.start()].chars().rev().find(|c| !c.is_whitespace());
            let right = t[m.end()..].chars().find(|c| !c.is_whitespace());
            if m.as_str() == "×"
                && left.is_some_and(|c| c.is_ascii_digit() || c == ')' || c == '）')
                && right.is_some_and(|c| c.is_ascii_digit() || c == '(' || c == '（')
            {
                "×"
            } else {
                "（ ）"
            }
        })
        .to_string()
}
// 同题重传允许题干的常见等价说法与批改痕迹不同，保留最新原文及上一版来源。
// 图片观察正文已经由调用方分题；保留全部作答行，不以答案语法或占位词拒绝落盘。
fn observed_child_answer(text: &str, heading: &str) -> String {
    if let Some(found) = Regex::new(r"孩子答案[：:]").unwrap().find(text) {
        return before_reference_fields(&text[found.end()..])
            .trim()
            .to_string();
    }
    child_answer(text, heading)
}
fn child_answer(text: &str, heading: &str) -> String {
    if let Some(c) = Regex::new(r"孩子(?:选择|答案)[：:][ \t]*([^\n　]*)")
        .unwrap()
        .captures(text)
    {
        let value = c[1].split("正确答案").next().unwrap_or("").trim();
        // 空白/未答不能覆盖同图中清晰的圈选和批改痕迹。
        if !has(value, &["没答上", "未作答", "未答", "空白", "无法辨认"]) && !value.is_empty()
        {
            return value.to_string();
        }
    }
    let stem = before_answer_fields(text);
    let circles: Vec<_> = Regex::new(r"[（(]\s*([A-D])\s*[）)]")
        .unwrap()
        .captures_iter(stem)
        .map(|c| c[1].to_string())
        .collect();
    let unicode_circles: Vec<_> = stem
        .chars()
        .filter_map(|c| match c {
            'Ⓐ' | 'ⓐ' => Some("A"),
            'Ⓑ' | 'ⓑ' => Some("B"),
            'Ⓒ' | 'ⓒ' => Some("C"),
            'Ⓓ' | 'ⓓ' => Some("D"),
            _ => None,
        })
        .collect();
    if unicode_circles.len() == 1 {
        return unicode_circles[0].into();
    }
    if circles.len() == 1 {
        return circles[0].clone();
    }
    if let Some(c) = Regex::new(r"[（(]([√×])[）)]").unwrap().captures(stem) {
        return c[1].to_string();
    }
    // 限定在每个选项自己的片段内，避免同一行C后的叉被贪婪归到A；X/x仅作末尾批改标记。
    let mark = grading_mark();
    let marked: Vec<_> = choice_spans(stem)
        .into_iter()
        .filter(|option| {
            let body = &stem[option.body_start..option.end];
            mark.find(body)
                .is_some_and(|m| !body[..m.start()].trim().is_empty())
        })
        .map(|option| option.label)
        .collect();
    if marked.len() == 1 {
        return marked[0].to_string();
    }
    let _ = heading;
    "没答上".into()
}
fn before_reference_fields(text: &str) -> &str {
    Regex::new(r"(?:订正|訂正|改正|更正|正确答案|标准答案|参考答案|解析)[：:]")
        .unwrap()
        .find(text)
        .map(|m| &text[..m.start()])
        .unwrap_or(text)
}
fn compact_record(text: &str, child: &str) -> String {
    let text = Regex::new(r"^\s*\d+[.．、。]\s*")
        .unwrap()
        .replace(text, "");
    let stem = question(&text);
    format!("{}\n**孩子答案：** {}", stem.trim(), child.trim())
}
fn question_key(text: &str) -> String {
    let stem = question(text).replace("每个小朋友分到几个苹果", "每人分到几个");
    let stem = Regex::new(r"^\s*\d+[.．、。]\s*(?:选择题[：:]\s*)?")
        .unwrap()
        .replace(&stem, "");
    norm(&stem)
}
fn record(
    conn: &Connection,
    ns: &str,
    jieba: &jieba_rs::Jieba,
    owner: &str,
    text: &str,
    meta: Value,
    session: Option<String>,
) -> Result<String> {
    let id = Uuid::now_v7();
    let now = now_ms();
    let meta = household::apply_tag(&meta, "错题集");
    let r = MemoryRecord {
        id,
        revision: 1,
        kind: MemoryKind::Episodic,
        state: MemoryState::Active,
        content: text.into(),
        summary: None,
        importance: 0.7,
        confidence: 1.0,
        salience: 0.7,
        created_at_ms: now,
        updated_at_ms: now,
        last_accessed_at_ms: None,
        next_review_at_ms: None,
        review_count: 0,
        source_hash: None,
        embedding_model: None,
        embedding_version: 1,
        embedding_pending: false,
        consolidation_pending: false,
        parent_id: None,
        namespace: ns.into(),
        session_id: session,
        entities: vec![owner.into()],
        metadata: meta,
        tag: "错题集".into(),
        subject_ids: vec![owner.into()],
    };
    store::insert_memory(conn, &r, jieba)?;
    store::insert_version(conn, id, "create", None, &r, "wrong_practice", None)?;
    Ok(id.to_string())
}
#[derive(Clone)]
struct Q {
    id: String,
    number: i64,
    raw: String,
    meta: Value,
}
fn load(conn: &Connection, ns: &str) -> Result<Vec<Q>> {
    let mut stmt = conn.prepare("SELECT id,content,metadata_json FROM memories WHERE namespace=?1 AND tag='错题集' AND state='active' ORDER BY created_at_ms,id")?;
    let rows = stmt.query_map([ns], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, raw, m) = row?;
        let meta: Value = serde_json::from_str(&m)?;
        if meta["practice_kind"] == "question"
            && meta["display_only"] != true
            && meta["record_type"] != "practice_results"
        {
            out.push(Q {
                id,
                number: meta["number"].as_i64().unwrap_or(0),
                raw,
                meta,
            });
        }
    }
    Ok(out)
}
fn save_state(tx: &Connection, ns: &str, sid: &str, state: &Value) -> Result<()> {
    tx.execute("INSERT INTO wrong_practice_sessions(namespace,session_id,state) VALUES(?1,?2,?3) ON CONFLICT(namespace,session_id) DO UPDATE SET state=excluded.state", params![ns,sid,state.to_string()])?;
    Ok(())
}
/// 仅宿主配置开启时调用。调用者明确区分上传与对话。
pub fn execute(
    conn: &mut Connection,
    ns: &str,
    jieba: &jieba_rs::Jieba,
    body: Value,
) -> Result<Value> {
    let start = std::time::Instant::now();
    conn.execute_batch("CREATE TABLE IF NOT EXISTS wrong_practice_sessions(namespace TEXT NOT NULL,session_id TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(namespace,session_id)); CREATE TABLE IF NOT EXISTS wrong_practice_turns(namespace TEXT NOT NULL,session_id TEXT NOT NULL,message_id TEXT NOT NULL,reply TEXT NOT NULL,PRIMARY KEY(namespace,session_id,message_id));")?;
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let mut result = if body["action"] == "family_seed" {
        let mut seeded = crate::family_demo::seed(&tx, ns, jieba, &body)?;
        if seeded["already_seeded"] != true {
            if let Some(text) = body["wrong_questions"].as_str() {
                let imported = import(
                    &tx,
                    ns,
                    jieba,
                    &json!({"text":text,"source_id":"preset:wrong_questions","name":"错题集.md","kind":"document"}),
                )?;
                if imported["handled"] != true || imported["error"].is_string() {
                    return Err(crate::error::MemoraError::InvalidState(
                        "错题预设导入失败".into(),
                    ));
                }
                seeded["wrong_questions"] = imported["count"].clone();
            }
        }
        seeded
    } else if body["action"] == "latest_uploaded_questions" {
        latest_uploaded_questions(&tx, ns, &body)?
    } else if body["action"] == "delete_question" {
        delete_question(&tx, ns, &body)?
    } else if body["action"] == "latest_health_memory" {
        latest_health_memory(&tx, ns, &body)?
    } else if body["action"] == "record_results" {
        record_results(&tx, ns, jieba, &body)?
    } else if body["action"] == "family_recall" {
        crate::family_demo::recall(&tx, ns, &body)?
    } else if body["action"] == "import" {
        import(&tx, ns, jieba, &body)?
    } else {
        turn(&tx, ns, jieba, &body)?
    };
    tx.commit()?;
    result["elapsed_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
    Ok(result)
}
fn required_person_name(body: &Value) -> Result<&str> {
    let name = body["name"]
        .as_str()
        .map(str::trim)
        .filter(|n| !n.is_empty() && n.chars().count() <= 40 && !n.chars().any(char::is_control));
    name.ok_or_else(|| {
        crate::error::MemoraError::InvalidState("name必须为1至40字的有效人物姓名字符串".into())
    })
}

fn person_owner(conn: &Connection, ns: &str, name: &str) -> Result<Option<String>> {
    let members = household::list_members(conn, ns)?;
    // 正式姓名优先；没有正式姓名匹配时才接受唯一别名，不做跨人的模糊匹配。
    let exact: Vec<_> = members.iter().filter(|m| m.display_name == name).collect();
    let owners = if exact.is_empty() {
        members
            .iter()
            .filter(|m| m.aliases.iter().any(|a| a == name))
            .collect()
    } else {
        exact
    };
    if owners.len() > 1 {
        return Err(crate::error::MemoraError::InvalidState(format!(
            "{name}匹配多个成员，无法确定归属"
        )));
    }
    Ok(owners.first().map(|m| m.member_id.clone()))
}

fn latest_uploaded_questions(conn: &Connection, ns: &str, body: &Value) -> Result<Value> {
    let name = required_person_name(body)?;
    if body
        .as_object()
        .is_some_and(|o| o.keys().any(|k| !["action", "name"].contains(&k.as_str())))
    {
        return Err(crate::error::MemoraError::InvalidState(
            "错题读取只接受name参数".into(),
        ));
    }
    let Some(owner) = person_owner(conn, ns, name)? else {
        return Ok(json!({"ok":false,"error":format!("未找到{name}最近上传的错题")}));
    };
    let all = load(conn, ns)?;
    let source_of = |q: &Q| {
        q.meta["latest_source_id"]
            .as_str()
            .unwrap_or("")
            .to_string()
    };
    let upload_of = |q: &Q| {
        q.meta["latest_upload_id"]
            .as_str()
            .unwrap_or("")
            .to_string()
    };
    let mut questions: Vec<_> = all
        .iter()
        .filter(|q| {
            q.meta["subjects"][0]["member_id"] == owner
                && !household::is_privacy(&q.meta)
                && !source_of(q).is_empty()
                && !source_of(q).starts_with("preset:")
        })
        .collect();
    questions.sort_by_key(|q| (q.meta["uploaded_at_ms"].as_i64().unwrap_or(0), q.id.clone()));
    let Some(latest) = questions.last() else {
        return Ok(json!({"ok":false,"error":format!("未找到{name}最近上传的错题")}));
    };
    let upload = upload_of(latest);
    questions.retain(|q| upload_of(q) == upload);
    questions.sort_by_key(|q| {
        (
            q.meta["source_order"].as_i64().unwrap_or(0),
            q.number,
            q.id.clone(),
        )
    });
    // 原文是记忆事实正文：保持每条content逐字不变，仅以换行连接多条事实。
    let contents: Vec<_> = questions.iter().map(|q| q.raw.as_str()).collect();
    let questions: Vec<_> = questions
        .iter()
        .map(|q| json!({"question_id":q.id,"content":q.raw}))
        .collect();
    Ok(json!({"ok":true,"content":contents.join("\n"),"questions":questions,"upload_id":upload}))
}

fn delete_question(conn: &Connection, ns: &str, body: &Value) -> Result<Value> {
    let name = required_person_name(body)?;
    let question = body["question"]
        .as_str()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| {
            crate::error::MemoraError::InvalidState("question必须为非空题目字符串".into())
        })?;
    if body
        .get("question_id")
        .is_some_and(|id| !id.is_null() && !id.is_string())
    {
        return Err(crate::error::MemoraError::InvalidState(
            "question_id必须为字符串".into(),
        ));
    }
    let Some(owner) = person_owner(conn, ns, name)? else {
        return Ok(json!({"ok":false,"error":"未找到对应人物的错题"}));
    };
    let bank: Vec<_> = load(conn, ns)?
        .into_iter()
        .filter(|q| {
            q.meta["subjects"]
                .as_array()
                .is_some_and(|a| a.iter().any(|s| s["member_id"] == owner))
                && !household::is_privacy(&q.meta)
        })
        .collect();
    let mut matches: Vec<_> = body["question_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .map(|id| bank.iter().filter(|q| q.id == id).collect())
        .unwrap_or_default();
    let by_id = !matches.is_empty();
    if matches.is_empty() {
        matches = bank.iter().filter(|q| q.raw.contains(question)).collect();
    }
    if matches.is_empty() {
        return Ok(json!({"ok":false,"error":"未找到正文包含该子串的错题，未删除"}));
    }
    if matches.len() != 1 {
        return Ok(
            json!({"ok":false,"error":"题目子串匹配到多道错题，请提供question_id，未删除","matched_count":matches.len()}),
        );
    }
    let q = matches[0];
    let id = Uuid::parse_str(&q.id)
        .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
    crate::graph::store::withdraw_facts_by_source_memory(conn, ns, id)?;
    store::delete_memory(conn, id)?;
    conn.execute("UPDATE wrong_practice_sessions SET state='{}' WHERE namespace=?1 AND json_extract(state,'$.current')=?2",params![ns,q.id])?;
    conn.execute("DELETE FROM wrong_practice_turns WHERE namespace=?1 AND CASE WHEN json_valid(reply) THEN json_extract(reply,'$.question_id') ELSE NULL END=?2",params![ns,q.id])?;
    Ok(
        json!({"ok":true,"deleted":true,"question_id":q.id,"matched_by":if by_id {"question_id"}else{"substring"}}),
    )
}

fn latest_health_memory(conn: &Connection, ns: &str, body: &Value) -> Result<Value> {
    let name = required_person_name(body)?;
    let Some(owner) = person_owner(conn, ns, name)? else {
        return Ok(json!({"ok":false,"error":format!("未找到{name}的医疗健康记忆")}));
    };
    // 与UI事实列表使用相同的最新排序，排除摘要、规律及隐藏项。
    let mut stmt = conn.prepare("SELECT id FROM memories WHERE namespace=?1 AND tag='医疗健康' AND (state='active' OR (state='archived' AND json_extract(metadata_json,'$.recall_archived')=1)) AND kind NOT IN ('summary','pattern') ORDER BY updated_at_ms DESC,id DESC")?;
    let ids = stmt
        .query_map([ns], |r| r.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let members = household::list_members(conn, ns)?;
    let display_name = members
        .iter()
        .find(|m| m.member_id == owner)
        .map(|m| m.display_name.as_str())
        .unwrap_or(name);
    for id in ids {
        let id = Uuid::parse_str(&id)
            .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
        let Some(rec) = store::get_memory(conn, id)? else {
            continue;
        };
        if !rec.subject_ids.contains(&owner)
            || household::is_privacy(&rec.metadata)
            || rec.metadata["display_only"] == true
            || rec.metadata["record_type"] == "practice_results"
        {
            continue;
        }
        if rec.content.trim().is_empty()
            || (rec.metadata["dream"] == true
                && rec.metadata["support_count"].as_i64().unwrap_or(0) > 0)
        {
            continue;
        }
        let has_person = members.iter().any(|m| {
            rec.content.contains(&m.display_name)
                || m.aliases
                    .iter()
                    .any(|a| a.chars().count() >= 2 && rec.content.contains(a))
        });
        let content = if has_person {
            rec.content
        } else {
            format!("{display_name}{}", rec.content)
        };
        return Ok(json!({"ok":true,"content":content,"id":id}));
    }
    Ok(json!({"ok":false,"error":format!("未找到{name}的医疗健康记忆")}))
}

/// 宿主工具写入已判定结果，不重新抽取、判题或拆分正文。
fn record_results(
    tx: &Connection,
    ns: &str,
    jieba: &jieba_rs::Jieba,
    body: &Value,
) -> Result<Value> {
    let invalid = |message: &str| crate::error::MemoraError::InvalidState(message.into());
    let name = required_person_name(body)?;
    let rows = body["list"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("list必须为非空数组"))?;
    let mut lines = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let fields = row.as_object().ok_or_else(|| invalid("题目项必须为对象"))?;
        if fields.len() != 2 {
            return Err(invalid("每项只接受题目、是否答对两个字段"));
        }
        let question = row["题目"]
            .as_str()
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| invalid("题目必须为非空字符串"))?;
        let correct = row["是否答对"]
            .as_bool()
            .ok_or_else(|| invalid("是否答对必须为布尔值"))?;
        let prefix = if rows.len() == 1 {
            String::new()
        } else {
            format!("{}.", i + 1)
        };
        lines.push(format!(
            "{prefix}{question} **{}**",
            if correct { "答对" } else { "答错" }
        ));
    }
    let existing = person_owner(tx, ns, name)?;
    let owner = existing
        .clone()
        .unwrap_or_else(|| format!("person_{}", Uuid::now_v7()));
    if existing.is_none() {
        household::upsert_member(
            tx,
            ns,
            &household::HouseholdMember {
                member_id: owner.clone(),
                display_name: name.into(),
                role: household::MemberRole::Other,
                aliases: vec![],
                age_band: None,
            },
        )?;
    }
    household::upsert_tag(
        tx,
        ns,
        &household::MemoryTag {
            tag_id: "错题集".into(),
            display_name: "错题集".into(),
            aliases: vec![],
        },
    )?;
    let content = lines.join("\n");
    let id = record(
        tx,
        ns,
        jieba,
        &owner,
        &content,
        json!({
            "record_type":"practice_results", "display_only":true, "source":"person_incorrect_question_record", "owner_name":name,
            "subjects":[{"member_id":owner,"confidence":1.0}], "attribution":"manual", "results":rows
        }),
        None,
    )?;
    Ok(json!({"ok":true,"id":id,"content":content,"count":rows.len()}))
}

fn import(tx: &Connection, ns: &str, jieba: &jieba_rs::Jieba, body: &Value) -> Result<Value> {
    let raw = body["text"].as_str().unwrap_or("");
    let text = clean(raw);
    if !has(&text, &["错题", "古诗词积累练习", "请补全《竹石》"]) {
        return Ok(json!({"handled":false}));
    }
    let owner = Regex::new(r"(?:姓名|学生|归属)[：:]\s*([\p{Han}]{2,8})")
        .unwrap()
        .captures(&text)
        .map(|c| c[1].to_string());
    let Some(name) = owner else {
        return Ok(json!({"handled":true,"error":"错题资料缺少明确姓名，未归属和入库"}));
    };
    let roster = household::list_members(tx, ns)?;
    let owner = roster
        .iter()
        .find(|m| m.display_name == name || m.aliases.contains(&name))
        .map(|m| m.member_id.clone())
        .unwrap_or_else(|| format!("person_{}", Uuid::now_v7()));
    if !roster.iter().any(|m| m.member_id == owner) {
        household::upsert_member(
            tx,
            ns,
            &household::HouseholdMember {
                member_id: owner.clone(),
                display_name: name.clone(),
                role: household::MemberRole::Other,
                aliases: vec![],
                age_band: None,
            },
        )?;
    }
    household::upsert_tag(
        tx,
        ns,
        &household::MemoryTag {
            tag_id: "错题集".into(),
            display_name: "错题集".into(),
            aliases: vec![],
        },
    )?;
    let source = body["source_id"].as_str().unwrap_or("");
    if source.is_empty() {
        return Ok(json!({"handled":true,"error":"缺少上传标识"}));
    }
    let re = Regex::new(r"(?m)^\s*(\d{1,3})\s*[.．、。]\s*").unwrap();
    let matches: Vec<_> = re.captures_iter(&text).collect();
    if matches.is_empty() {
        return Ok(json!({"handled":true,"error":"未找到题号，未入库"}));
    }
    let existing = load(tx, ns)?;
    let mut count = 0;
    let mut ids = Vec::new();
    for (i, c) in matches.iter().enumerate() {
        let number: i64 = c[1].parse().unwrap_or(0);
        let from = c.get(0).unwrap().start();
        let to = matches
            .get(i + 1)
            .map(|c| c.get(0).unwrap().start())
            .unwrap_or(text.len());
        let raw_q = text[from..to].split("\n##").next().unwrap_or("").trim();
        let image = body["kind"] == "image_ocr";
        let mut source_question = raw_q.to_string();
        let child = if image && text.starts_with("# 错题原文（仅转录）") {
            let value = Regex::new(r"^\s*\d+[.．、。]\s*")
                .unwrap()
                .replace(raw_q, "");
            let value = value
                .trim()
                .strip_suffix("答错")
                .unwrap_or(value.trim())
                .trim();
            if let Some((stem, child)) = value.split_once('=') {
                source_question = stem.trim().to_string();
                child.trim().to_string()
            } else {
                observed_child_answer(raw_q, "")
            }
        } else {
            observed_child_answer(raw_q, "")
        };
        let q = compact_record(&source_question, &child);
        let original = q.replace("**", "");
        let meta = json!({"practice_kind":"question","channel":"wrong_practice","subjects":[{"member_id":owner,"confidence":1.0}],"number":number,"child_answer":child,"source_id":source,"latest_source_id":source,"latest_upload_id":body["upload_id"].as_str().unwrap_or(source),"source_order":body["source_order"].as_i64().unwrap_or(0),"uploaded_at_ms":body["uploaded_at_ms"].as_i64().unwrap_or_else(now_ms),"source_name":body["name"],"original_text":original,"image":image});
        if let Some(old) = existing.iter().find(|old| {
            old.number == number
                && (old.meta["source_id"] == source || question_key(&old.raw) == question_key(&q))
                && old.meta["subjects"][0]["member_id"] == owner
        }) {
            let id = Uuid::parse_str(&old.id)
                .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
            if let Some(mut rec) = store::get_memory(tx, id)? {
                let private = household::is_privacy(&rec.metadata);
                rec.metadata = household::apply_tag(&meta, "错题集");
                if private {
                    rec.metadata["privacy"] = json!(true);
                }
                rec.content = q;
                rec.updated_at_ms = now_ms();
                store::update_memory(tx, &mut rec, jieba)?;
            }
            ids.push(old.id.clone());
        } else {
            ids.push(record(tx, ns, jieba, &owner, &q, meta, None)?);
        }
        count += 1;
    }
    Ok(json!({"handled":true,"count":count,"ids":ids,"owner":name}))
}
// 这里只检索题目；作答判定由上层学习流程处理，不保存或消费参考答案。
fn turn(tx: &Connection, ns: &str, _jieba: &jieba_rs::Jieba, body: &Value) -> Result<Value> {
    let query = body["query"].as_str().unwrap_or("").trim();
    let sid = body["session_id"].as_str().unwrap_or("default");
    let mid = body["message_id"].as_str().unwrap_or("");
    if !mid.is_empty() {
        if let Some(saved)=tx.query_row("SELECT reply FROM wrong_practice_turns WHERE namespace=?1 AND session_id=?2 AND message_id=?3",params![ns,sid,mid],|r|r.get::<_,String>(0)).optional()? {
            return Ok(serde_json::from_str(&saved)?);
        }
    }
    let saved = tx
        .query_row(
            "SELECT state FROM wrong_practice_sessions WHERE namespace=?1 AND session_id=?2",
            params![ns, sid],
            |r| r.get::<_, String>(0),
        )
        .optional()?;
    let mut state: Value = saved
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(json!({}));
    let follow = state["active"] == true;
    if follow && has(query, &["结束", "不练了", "先到这里"]) {
        state["active"] = json!(false);
        save_state(tx, ns, sid, &state)?;
        return Ok(json!({"handled":true,"reply":"本轮练习结束。"}));
    }
    let explicit = has(
        query,
        &[
            "错题",
            "出一道",
            "出一题",
            "复习上传",
            "最新上传的题",
            "刚刚上传的题",
        ],
    );
    let next = follow
        && has(
            query,
            &["下一题", "换一题", "换一道", "再来", "再练", "继续"],
        );
    if !(explicit || next) {
        return Ok(json!({"handled":false}));
    }
    let roster = household::list_members(tx, ns)?;
    let named = roster.iter().find(|m| query.contains(&m.display_name));
    let owner = named
        .map(|m| m.member_id.as_str())
        .or_else(|| state["owner"].as_str())
        .unwrap_or("")
        .to_string();
    let mut bank: Vec<_> = load(tx, ns)?
        .into_iter()
        .filter(|q| {
            !household::is_privacy(&q.meta)
                && (owner.is_empty()
                    || q.meta["subjects"]
                        .as_array()
                        .is_some_and(|v| v.iter().any(|s| s["member_id"] == owner)))
        })
        .collect();
    if has(query, &["多少", "几道", "几题", "总数", "数量"]) {
        return Ok(json!({"handled":true,"reply":format!("错题集共{}道。",bank.len())}));
    }
    let number =
        Regex::new(r"(?:第\s*|题号\s*[是为：:]?\s*|[Qq])(\d+)\s*(?:[道个]?题|号)?").unwrap();
    let selected_number = number
        .captures(query)
        .and_then(|c| c[1].parse::<i64>().ok());
    if let Some(n) = selected_number {
        bank.retain(|q| q.number == n);
    } else if has(query, &["最新", "刚上传", "上次上传"]) {
        bank.retain(|q| {
            !q.meta["source_id"]
                .as_str()
                .unwrap_or("")
                .starts_with("preset:")
        });
        if let Some(at) = bank
            .iter()
            .filter_map(|q| q.meta["uploaded_at_ms"].as_i64())
            .max()
        {
            bank.retain(|q| q.meta["uploaded_at_ms"] == at);
        }
    }
    if next && !has(query, &["再练"]) {
        bank.retain(|q| q.id != state["current"].as_str().unwrap_or(""));
    }
    let chosen = bank
        .iter()
        .max_by_key(|q| (q.meta["uploaded_at_ms"].as_i64().unwrap_or(0), q.id.clone()));
    let result = if let Some(q) = chosen {
        state = json!({"active":true,"current":q.id,"owner":q.meta["subjects"][0]["member_id"]});
        save_state(tx, ns, sid, &state)?;
        json!({"handled":true,"reply":format!("{}\n题目ID：{}",question(&q.raw),q.id),"question_id":q.id})
    } else {
        json!({"handled":true,"reply":"没有找到符合条件的错题。"})
    };
    if !mid.is_empty() {
        tx.execute(
            "INSERT INTO wrong_practice_turns VALUES(?1,?2,?3,?4)",
            params![ns, sid, mid, result.to_string()],
        )?;
    }
    Ok(result)
}
#[cfg(test)]
mod record_results_tests {
    use super::*;
    use crate::MemoryEngine;

    #[tokio::test]
    async fn record_results_preserves_format_owner_and_flags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        for (rows, expected) in [
            (json!([{"题目":"1+1=?","是否答对":true}]), "1+1=? **答对**"),
            (
                json!([{"题目":"甲题","是否答对":true},{"题目":"乙题\n第二行","是否答对":false}]),
                "1.甲题 **答对**\n2.乙题\n第二行 **答错**",
            ),
        ] {
            let result = e
                .wrong_practice(json!({"action":"record_results","name":"梁子航","list":rows}))
                .await
                .unwrap();
            assert_eq!(result["content"], expected);
            let id = Uuid::parse_str(result["id"].as_str().unwrap()).unwrap();
            let conn = Connection::open(&path).unwrap();
            let fact = store::get_memory(&conn, id).unwrap().unwrap();
            assert_eq!(fact.content, expected);
            assert_eq!(fact.tag, "错题集");
            assert!(!fact.embedding_pending && !fact.consolidation_pending);
            assert_eq!(fact.metadata["record_type"], "practice_results");
            let members = household::list_members(&conn, "default").unwrap();
            let owner = members.iter().find(|m| m.display_name == "梁子航").unwrap();
            assert_eq!(fact.subject_ids, vec![owner.member_id.clone()]);
        }
        let facts = e.list(ListQuery::default()).await.unwrap();
        assert_eq!(facts.total, 2);
    }

    #[tokio::test]
    async fn record_results_rejects_invalid_batch_without_partial_writes() {
        let dir = tempfile::tempdir().unwrap();
        let e = MemoryEngine::builder()
            .db_path(dir.path().join("invalid.db"))
            .build()
            .await
            .unwrap();
        for rows in [
            json!([]),
            json!([{"题目":"甲","是否答对":"true"}]),
            json!([{"题目":"甲","是否答对":true},{"题目":" ","是否答对":false}]),
            json!([{"题目":42,"是否答对":true}]),
            json!([{"题目":"甲","是否答对":true,"其他":1}]),
        ] {
            assert!(e
                .wrong_practice(json!({"action":"record_results","name":"梁子航","list":rows}))
                .await
                .is_err());
        }
        assert_eq!(e.list(ListQuery::default()).await.unwrap().total, 0);
    }
}

#[cfg(test)]
mod latest_memory_tools_tests {
    use super::*;
    use crate::{MemoryEngine, RecallQuery};

    async fn engine() -> (tempfile::TempDir, MemoryEngine, std::path::PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("m.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        (d, e, path)
    }
    async fn upload(e: &MemoryEngine, source: &str, group: &str, order: i64, at: i64, text: &str) {
        let r = e.wrong_practice(json!({"action":"import","source_id":source,"upload_id":group,"source_order":order,"uploaded_at_ms":at,"text":text,"kind":"document"})).await.unwrap();
        assert!(r["count"].as_i64().unwrap_or(0) > 0, "{r}");
    }
    #[tokio::test]
    async fn display_results_stay_in_ui_but_never_enter_any_recall_or_latest_upload() {
        let (_d, e, path) = engine().await;
        let r = e.wrong_practice(json!({"action":"record_results","name":"梁子航","list":[{"题目":"独特展示词紫色河马","是否答对":true}]})).await.unwrap();
        let conn = Connection::open(path).unwrap();
        let rec = store::get_memory(&conn, Uuid::parse_str(r["id"].as_str().unwrap()).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(rec.metadata["display_only"], true);
        assert!(!store::memory_is_recallable(&rec));
        let mut legacy = rec.clone();
        legacy
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("display_only");
        assert!(!store::memory_is_recallable(&legacy));
        let page = e
            .list(ListQuery {
                facts_only: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(page.total, 1);
        assert!(load(&conn, "default").unwrap().is_empty());
        assert!(e
            .recall(RecallQuery {
                text: "紫色河马".into(),
                fts_only: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .hits
            .is_empty());
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
                .await
                .unwrap()["ok"],
            false
        );
    }
    #[tokio::test]
    async fn latest_upload_returns_fact_content_and_all_pages_excludes_other_owners_and_presets() {
        let (_d, e, path) = engine().await;
        upload(
            &e,
            "old",
            "old",
            0,
            1,
            "错题\n姓名：梁子航\n1. 旧题 **原样**\n正确答案：A",
        )
        .await;
        let page1 = "# 错题\r\n姓名：梁子航\r\n1. 甲题 **原样**\r\n正确答案：B\r\n";
        let page2 = "错题\n姓名：梁子航\n2. 乙题\n正确答案：C\n";
        upload(&e, "new:1", "new", 0, 20, page1).await;
        upload(&e, "new:2", "new", 1, 20, page2).await;
        upload(
            &e,
            "someone",
            "someone",
            0,
            999,
            "错题\n姓名：宋娜娜\n1. 别人的题\n正确答案：D",
        )
        .await;
        upload(
            &e,
            "preset:wrong_questions",
            "preset",
            0,
            1000,
            "错题\n姓名：梁子航\n3. 预设题\n正确答案：A",
        )
        .await;
        let r = e
            .wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
            .await
            .unwrap();
        let conn = Connection::open(&path).unwrap();
        let mut facts = load(&conn, "default")
            .unwrap()
            .into_iter()
            .filter(|q| q.meta["latest_upload_id"] == "new")
            .collect::<Vec<_>>();
        facts.sort_by_key(|q| (q.meta["source_order"].as_i64().unwrap(), q.number));
        let expected = facts
            .iter()
            .map(|q| q.raw.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(r["content"], expected);
        assert_ne!(expected, format!("{page1}\n{page2}"));
        e.wrong_practice(json!({"action":"record_results","name":"梁子航","list":[{"题目":"更晚的展示题","是否答对":false}]})).await.unwrap();
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
                .await
                .unwrap()["content"],
            r["content"]
        );
    }
    #[tokio::test]
    async fn hidden_or_deleted_sources_are_not_exposed() {
        let (_d, e, path) = engine().await;
        let raw = "错题\n姓名：梁子航\n1. 古诗 **原样**\n正确答案：A";
        upload(&e, "legacy", "legacy", 0, 1, raw).await;
        let conn = Connection::open(path).unwrap();
        let r = e
            .wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
            .await
            .unwrap();
        assert_eq!(r["content"], load(&conn, "default").unwrap()[0].raw);
        conn.execute(
            "UPDATE memories SET metadata_json=json_set(metadata_json,'$.privacy',json('true'))",
            [],
        )
        .unwrap();
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
                .await
                .unwrap()["ok"],
            false
        );
        conn.execute("DELETE FROM memories", []).unwrap();
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
                .await
                .unwrap()["ok"],
            false
        );
    }
    fn health(content: &str, latest: bool) -> Value {
        json!({"owner":"child","content":content,"tag":"医疗健康","domain":"medical","source":"用户指定","keywords":["发烧"],"entity_type":"concept","entity_name":"健康状态","relation":"近期健康","ui_latest":latest})
    }
    #[tokio::test]
    async fn newest_health_matches_ui_order_and_only_adds_missing_person() {
        let (_d, e, path) = engine().await;
        let mut data = json!({"version":1,"members":[{"id":"child","name":"梁子航","role":"child","aliases":["航航"]}],"facts":[health("最近正在发烧。",true),health("梁子航对花生过敏。",false)]});
        e.wrong_practice(json!({"action":"family_seed","data":data}))
            .await
            .unwrap();
        let r = e
            .wrong_practice(json!({"action":"latest_health_memory","name":"梁子航"}))
            .await
            .unwrap();
        assert_eq!(r["content"], "梁子航最近正在发烧。");
        let page = e
            .list(ListQuery {
                member_id: Some("child".into()),
                tag: Some("医疗健康".into()),
                facts_only: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(page.items[0].content, "最近正在发烧。");
        for text in ["梁子航最近正在发烧。", "航航最近正在发烧。"] {
            data["version"] = json!(data["version"].as_i64().unwrap() + 1);
            data["facts"][0]["content"] = json!(text);
            e.wrong_practice(json!({"action":"family_seed","data":data}))
                .await
                .unwrap();
            assert_eq!(
                e.wrong_practice(json!({"action":"latest_health_memory","name":"梁子航"}))
                    .await
                    .unwrap()["content"],
                text
            );
        }
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "UPDATE memories SET metadata_json=json_set(metadata_json,'$.privacy',json('true'))",
            [],
        )
        .unwrap();
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_health_memory","name":"梁子航"}))
                .await
                .unwrap()["ok"],
            false
        );
    }
}

#[cfg(test)]
mod named_memory_tools_tests {
    use super::*;
    use crate::MemoryEngine;

    #[tokio::test]
    async fn all_three_actions_require_real_string_name_and_reads_do_not_create_people() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("named.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        for action in ["latest_uploaded_questions", "latest_health_memory"] {
            assert_eq!(
                e.wrong_practice(json!({"action":action,"name":"梁子航"}))
                    .await
                    .unwrap()["ok"],
                false
            );
        }
        let conn = Connection::open(&path).unwrap();
        assert!(household::list_members(&conn, "default")
            .unwrap()
            .is_empty());
        let rows = json!([{"题目":"甲题","是否答对":true}]);
        e.wrong_practice(json!({"action":"record_results","name":"梁子航","list":rows}))
            .await
            .unwrap();
        for action in [
            "record_results",
            "latest_uploaded_questions",
            "latest_health_memory",
        ] {
            for name in [
                Value::Null,
                json!(1),
                json!(true),
                json!(" "),
                json!("梁\n子航"),
            ] {
                assert!(e
                    .wrong_practice(json!({"action":action,"name":name,"list":rows}))
                    .await
                    .is_err());
            }
            assert!(e
                .wrong_practice(json!({"action":action,"list":rows}))
                .await
                .is_err());
        }
        assert_eq!(e.list(ListQuery::default()).await.unwrap().total, 1);
    }

    #[tokio::test]
    async fn liang_alias_reuses_owner_and_health_uses_canonical_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alias.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        let data = json!({"version":1,"members":[{"id":"child","name":"梁子航","role":"child","aliases":["航航"]}],"facts":[{"owner":"child","content":"最近正在发烧。","tag":"医疗健康","domain":"medical","source":"用户指定","keywords":["发烧"],"entity_type":"concept","entity_name":"发烧","relation":"近期健康状态","ui_latest":true}]});
        e.wrong_practice(json!({"action":"family_seed","data":data}))
            .await
            .unwrap();
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_health_memory","name":"航航"}))
                .await
                .unwrap()["content"],
            "梁子航最近正在发烧。"
        );
        let result=e.wrong_practice(json!({"action":"record_results","name":"航航","list":[{"题目":"甲题","是否答对":false}]})).await.unwrap();
        let conn = Connection::open(&path).unwrap();
        let rec = store::get_memory(
            &conn,
            Uuid::parse_str(result["id"].as_str().unwrap()).unwrap(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(rec.subject_ids, vec!["child"]);
        assert_eq!(household::list_members(&conn, "default").unwrap().len(), 1);
        // 正式姓名冲突必须拒绝；不能任选一人，也不能悄悄创建第三个人。
        household::upsert_member(
            &conn,
            "default",
            &household::HouseholdMember {
                member_id: "duplicate".into(),
                display_name: "梁子航".into(),
                role: household::MemberRole::Other,
                aliases: vec![],
                age_band: None,
            },
        )
        .unwrap();
        for action in [
            "record_results",
            "latest_uploaded_questions",
            "latest_health_memory",
        ] {
            assert!(e.wrong_practice(json!({"action":action,"name":"梁子航","list":[{"题目":"甲题","是否答对":true}]})).await.is_err());
        }
    }
}

#[cfg(test)]
mod question_storage_tests {
    use super::*;
    use crate::{MemoryEngine, MemoryUpdate};
    async fn setup() -> (tempfile::TempDir, MemoryEngine, std::path::PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("questions.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        (d, e, path)
    }
    fn assert_shape(row: &MemoryRecord) {
        assert_eq!(row.tag, "错题集");
        assert_eq!(row.metadata["practice_kind"], "question");
        assert!(!row.embedding_pending);
        for key in [
            "answer",
            "correct_answer",
            "explanation",
            "answer_correction",
            "question_types",
            "question_label_source",
            "question_label_reason",
        ] {
            assert!(row.metadata.get(key).is_none(), "{key}: {}", row.metadata);
        }
        for field in [
            &row.content,
            row.metadata["original_text"].as_str().unwrap_or(""),
        ] {
            assert!(!field.contains("正确答案"), "{field}");
            assert!(!field.contains("解析："), "{field}");
        }
        assert!(!Regex::new(r"^\d+[.．、。]").unwrap().is_match(&row.content));
        let bold: Vec<_> = Regex::new(r"\*\*(.*?)\*\*")
            .unwrap()
            .captures_iter(&row.content)
            .map(|c| c[1].to_string())
            .collect();
        assert_eq!(bold, vec!["孩子答案："]);
    }
    #[tokio::test]
    async fn presets_store_ten_questions_without_labels_or_reference_answers() {
        let (_d, e, path) = setup().await;
        let text = r###"# 错题集

姓名：梁子航

**1.** 计算：9.36÷1.2。

**孩子答案：** 0.78

**2.** 鸡和兔在同一个笼子里，共有24只，共有64只脚。鸡和兔各有多少只？

**孩子答案：**
假设全是鸡，脚的总数是24×2＝48（只）。
实际比假设多64－48＝16（只）脚。
每只兔有4只脚，所以兔有16÷4＝4（只）。
鸡有24－4＝20（只）。
答：鸡有20只，兔有4只。

**3.** 计算：14.4÷0.24。

**孩子答案：** 6

**4.** 一个平行四边形的底是12厘米，这条底对应的高是7厘米，另一条边长8厘米。它的面积是（　）平方厘米。

**孩子答案：** 96

**5.** 公园成人票每张10元，儿童票每张6元。一共卖出32张票，收入256元。成人票和儿童票各卖出多少张？

**孩子答案：**
假设卖出的全是儿童票，收入是32×6＝192（元）。
实际比假设多收入256－192＝64（元）。
一张成人票和一张儿童票合计10＋6＝16（元）。
成人票卖出64÷16＝4（张）。
儿童票卖出32－4＝28（张）。
答：成人票卖出4张，儿童票卖出28张。

**6.** 计算：6.84÷1.8。

**孩子答案：** 3.08

**7.** 方程4x＋6＝38的解是（　）。

A. x＝6　B. x＝8　C. x＝10　D. x＝11

**孩子答案：** C

**8.** 老师买了两种笔，共30支。签字笔每支3元，铅笔每支2元，共花了78元。两种笔各买了多少支？

**孩子答案：**
假设全买签字笔，需要30×3＝90（元）。
比实际多花90－78＝12（元）。
每支签字笔3元，所以铅笔买了12÷3＝4（支）。
签字笔买了30－4＝26（支）。
答：签字笔买了26支，铅笔买了4支。

**9.** 计算：5.46÷0.35。

**孩子答案：** 1.56

**10.** 袋子里有5个红球、3个蓝球和2个黄球，除颜色外完全相同。从中任意摸出一个球，下面说法正确的是（　）。

A. 摸到红球的可能性最大　B. 摸到蓝球的可能性最大
C. 摸到黄球的可能性最大　D. 摸到三种颜色球的可能性一样大

**孩子答案：** B
"###;
        let result = e
            .wrong_practice(
                json!({"action":"import","text":text,"source_id":"preset:wrong_questions"}),
            )
            .await
            .unwrap();
        assert_eq!(result["count"], 10);
        let conn = Connection::open(&path).unwrap();
        for q in load(&conn, "default").unwrap() {
            let row = store::get_memory(&conn, Uuid::parse_str(&q.id).unwrap())
                .unwrap()
                .unwrap();
            assert_shape(&row);
            if [2, 5, 8].contains(&q.number) {
                let child = row.metadata["child_answer"].as_str().unwrap();
                assert!(child.contains("假设") && child.contains("答："));
                assert!(child.lines().count() >= 5);
            }
        }
    }
    #[tokio::test]
    async fn image_import_keeps_observed_values_and_discards_reference_fields() {
        let (_d, e, path) = setup().await;
        for (i,text,expected) in [
            (1,"# 错题原文（仅转录）\n归属：梁子航\n1.2×2=40 答错","2×2\n**孩子答案：** 40"),
            (2,"# 图片错题（圈选）\n归属：梁子航\n1.12个苹果分给4人？\nA.2个 B.3个 C.4个X D.6个\n孩子选择：C\n订正：12÷4=3（个）\n正确答案：B\n解析：平均分。","12个苹果分给4人？\nA. 2个　B. 3个　C. 4个　D. 6个\n**孩子答案：** C"),
            (3,"# 图片错题（题目与孩子答案）\n归属：梁子航\n1.两种车共18辆46轮，各几辆？\n孩子答案：答：（46-18×2）÷（3-2）=8（辆）\n18-8=10（辆）\n自行车10辆，三轮车8辆。\n正确答案：自行车8辆，三轮车10辆。\n解析：差额法。","两种车共18辆46轮，各几辆？\n**孩子答案：** 答：（46-18×2）÷（3-2）=8（辆）\n18-8=10（辆）\n自行车10辆，三轮车8辆。")
        ] {
            let result=e.wrong_practice(json!({"action":"import","text":text,"source_id":format!("image:{i}"),"kind":"image_ocr"})).await.unwrap();
            assert_eq!(result["count"],1);
            let conn=Connection::open(&path).unwrap();
            let row=store::get_memory(&conn,Uuid::parse_str(result["ids"][0].as_str().unwrap()).unwrap()).unwrap().unwrap();
            assert_eq!(row.content,expected);
            assert_shape(&row);
        }
    }
    #[tokio::test]
    async fn repeated_upload_updates_one_id_and_latest_tool_has_only_id_and_content() {
        let (_d, e, path) = setup().await;
        let text = "# 图片错题（题目与孩子答案）\n归属：梁子航\n1.8.64÷1.2\n孩子答案：0.72";
        let mut ids = Vec::new();
        for i in 1..=2 {
            let r=e.wrong_practice(json!({"action":"import","text":text,"kind":"image_ocr","source_id":format!("page{i}"),"upload_id":format!("batch{i}"),"uploaded_at_ms":i})).await.unwrap();
            ids.push(r["ids"][0].clone());
        }
        assert_eq!(ids[0], ids[1]);
        assert_eq!(
            load(&Connection::open(path).unwrap(), "default")
                .unwrap()
                .len(),
            1
        );
        let latest = e
            .wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
            .await
            .unwrap();
        assert_eq!(latest["upload_id"], "batch2");
        let q = latest["questions"][0].as_object().unwrap();
        assert_eq!(q.len(), 2);
        assert!(q.contains_key("question_id") && q.contains_key("content"));
        assert!(e
            .wrong_practice(
                json!({"action":"latest_uploaded_questions","name":"梁子航","tags":["除法题"]})
            )
            .await
            .is_err());
        let deleted=e.wrong_practice(json!({"action":"delete_question","name":"梁子航","question":"8.64÷1.2","question_id":ids[0]})).await.unwrap();
        assert_eq!(deleted["deleted"], true);
        assert_eq!(
            e.wrong_practice(json!({"action":"latest_uploaded_questions","name":"梁子航"}))
                .await
                .unwrap()["ok"],
            false
        );
    }
    #[tokio::test]
    async fn malformed_or_empty_answers_are_preserved_without_solving() {
        let (_d, e, _path) = setup().await;
        for (i, child) in ["", "未作答", "[无法辨认]", "abc不规范算式"]
            .iter()
            .enumerate()
        {
            let r=e.wrong_practice(json!({"action":"import","text":format!("错题\n姓名：梁子航\n1.测试题{i}\n孩子答案：{child}"),"source_id":format!("empty{i}")})).await.unwrap();
            assert_eq!(r["count"], 1);
        }
        assert_eq!(e.list(ListQuery::default()).await.unwrap().total, 4);
    }
    #[tokio::test]
    async fn edit_keeps_full_answer_without_classifying_or_creating_graph_jobs() {
        let (_d, e, path) = setup().await;
        let r=e.wrong_practice(json!({"action":"import","text":"错题\n姓名：梁子航\n1.原题\n孩子答案：A","source_id":"edit"})).await.unwrap();
        let id = Uuid::parse_str(r["ids"][0].as_str().unwrap()).unwrap();
        e.revise(
            id,
            MemoryUpdate {
                content: Some("新题\n**孩子答案：** 第一步\n第二步\n答：错答。".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let conn = Connection::open(path).unwrap();
        let row = store::get_memory(&conn, id).unwrap().unwrap();
        assert_shape(&row);
        assert_eq!(row.metadata["child_answer"], "第一步\n第二步\n答：错答。");
        assert_eq!(
            conn.query_row("SELECT count(*) FROM graph_projection_jobs", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    #[tokio::test]
    async fn retrieval_counts_all_questions_and_never_judges_from_a_stored_answer() {
        let (_d, e, _path) = setup().await;
        e.wrong_practice(json!({"action":"import","text":"错题\n姓名：梁子航\n1.9.36÷1.2\n孩子答案：0.78","source_id":"turn"})).await.unwrap();
        let count = e
            .wrong_practice(json!({"action":"turn","query":"梁子航有几道错题","session_id":"s"}))
            .await
            .unwrap();
        assert_eq!(count["reply"], "错题集共1道。");
        let q=e.wrong_practice(json!({"action":"turn","query":"给梁子航出一道错题","session_id":"s","message_id":"one"})).await.unwrap();
        assert!(q["question_id"].is_string());
        assert!(!q["reply"].as_str().unwrap().contains("0.78"));
        assert_eq!(e.wrong_practice(json!({"action":"turn","query":"给梁子航出一道错题","session_id":"s","message_id":"one"})).await.unwrap()["question_id"],q["question_id"]);
        assert_eq!(
            e.wrong_practice(json!({"action":"turn","query":"7.8","session_id":"s"}))
                .await
                .unwrap()["handled"],
            false
        );
    }
}
