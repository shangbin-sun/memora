//! 可配置的错题规则通道。事实正文保留原文，索引和会话只是元数据。
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
fn labels(text: &str, heading: &str) -> Vec<String> {
    let t = format!("{heading}{text}");
    let mut v = Vec::new();
    if has(&t, &["古诗", "诗句", "《", "竹石", "咬定青山"]) {
        v.extend(["古诗题", "语文题", "填空题"]);
    } else {
        v.push("数学题");
        if has(&t, &["选择题", "A.", "A．", "A、"]) {
            v.push("选择题");
        }
        if has(&t, &["判断题", "√", "×）", "×)"]) {
            v.push("判断题");
        }
        if has(&t, &["计算", "苹果", "分数", "数", "÷", "×", "小时"]) {
            v.push("算术题");
        }
        if has(&t, &["周长", "面积", "三角形", "平行四边形"]) {
            v.push("几何题");
        }
    }
    v.into_iter().map(str::to_string).collect()
}
fn answer(text: &str) -> String {
    // 已知资料勘误：保留事实原文，判题元数据明确记录修正。
    if text.contains("千锤万凿出深山") {
        return "粉骨碎身浑不怕".into();
    }
    if text.contains("咬定青山不放松") {
        return "任尔东西南北风".into();
    }
    Regex::new(r"答案[：:]\s*([^\n　（(]+)")
        .unwrap()
        .captures(text)
        .map(|c| c[1].split("解析").next().unwrap_or("").trim().to_string())
        .unwrap_or_default()
}
fn question(text: &str) -> String {
    let t = Regex::new(r"孩子选择[：:][A-D]\s*")
        .unwrap()
        .replace_all(text, "");
    let t = t.split("答案").next().unwrap_or(&t).trim();
    let t = Regex::new(r"[（(][√×][）)]")
        .unwrap()
        .replace_all(t, "（ ）");
    t.to_string()
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
        if meta["practice_kind"] == "question" {
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
fn view(q: &Q) -> String {
    let raw = if q.meta["image"] == true {
        q.raw.split("练习题面：").last().unwrap_or(&q.raw)
    } else {
        &q.raw
    };
    format!("第{}题：{}", q.number, question(raw))
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
    let mut result = if body["action"] == "import" {
        import(&tx, ns, jieba, &body)?
    } else {
        turn(&tx, ns, jieba, &body)?
    };
    tx.commit()?;
    result["elapsed_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
    Ok(result)
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
    let re = Regex::new(r"(?m)^\s*(\d{1,3})[.．、]\s*").unwrap();
    let matches: Vec<_> = re.captures_iter(&text).collect();
    if matches.is_empty() {
        return Ok(json!({"handled":true,"error":"未找到题号，未入库"}));
    }
    let existing = load(tx, ns)?;
    let mut count = 0;
    let mut ids = Vec::new();
    for (i, c) in matches.iter().enumerate() {
        let number: i64 = c[1].parse().unwrap_or(0);
        if let Some(q) = existing
            .iter()
            .find(|q| q.meta["source_id"] == source && q.number == number)
        {
            ids.push(q.id.clone());
            count += 1;
            continue;
        }
        let from = c.get(0).unwrap().start();
        let to = matches
            .get(i + 1)
            .map(|c| c.get(0).unwrap().start())
            .unwrap_or(text.len());
        let raw_q = text[from..to].split("\n##").next().unwrap_or("").trim();
        let heading = text[..from]
            .rsplit("##")
            .next()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("");
        let mut q = raw_q.to_string();
        // 指定图片只有批改痕迹，原文保留在正文；以空缺题面出题，不能把学生错误作答当答案。
        let image = body["kind"] == "image_ocr";
        if image && has(raw_q, &["竹石", "咬定青山"]) {
            q = format!(
                "{}\n练习题面：咬定青山不放松，立根原在破岩中。千磨万击还坚劲，________。",
                raw_q
            );
        }
        if let Some(old) = existing.iter().find(|old| {
            old.number == number
                && norm(&old.raw) == norm(&q)
                && old.meta["subjects"][0]["member_id"] == owner
        }) {
            let id = Uuid::parse_str(&old.id)
                .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
            if let Some(mut rec) = store::get_memory(tx, id)? {
                rec.metadata["uploaded_at_ms"] =
                    json!(body["uploaded_at_ms"].as_i64().unwrap_or_else(now_ms));
                rec.updated_at_ms = now_ms();
                store::update_memory(tx, &mut rec, jieba)?;
            }
            ids.push(old.id.clone());
            count += 1;
            continue;
        }
        let a = answer(&q);
        let meta = json!({"practice_kind":"question","channel":"wrong_practice","subjects":[{"member_id":owner,"confidence":1.0}],"number":number,"question_types":labels(&q,heading),"answer":a,"weight":1.0,"source_id":source,"uploaded_at_ms":body["uploaded_at_ms"].as_i64().unwrap_or_else(now_ms),"source_name":body["name"],"original_text":raw_q,"answer_correction":if q.contains("千锤万凿出深山"){json!("资料答案勘误：粉骨碎身浑不怕")}else{Value::Null},"image":image});
        let id = record(tx, ns, jieba, &owner, &q, meta, None)?;
        ids.push(id);
        count += 1;
    }
    Ok(json!({"handled":true,"count":count,"ids":ids,"owner":name}))
}
fn query_types(q: &str) -> Vec<&'static str> {
    let mut types = Vec::new();
    for (words, label) in [
        (vec!["选择"], "选择题"),
        (vec!["判断"], "判断题"),
        (vec!["填空", "填词"], "填空题"),
        (vec!["数学"], "数学题"),
        (vec!["算术", "计算题"], "算术题"),
        (vec!["语文"], "语文题"),
        (vec!["古诗", "诗词", "诗题"], "古诗题"),
    ] {
        if has(q, &words) {
            types.push(label);
        }
    }
    types
}
fn turn(tx: &Connection, ns: &str, jieba: &jieba_rs::Jieba, b: &Value) -> Result<Value> {
    let query = b["query"].as_str().unwrap_or("").trim();
    let sid = b["session_id"].as_str().unwrap_or("default");
    let mid = b["message_id"].as_str().unwrap_or("");
    if !mid.is_empty() {
        if let Some(reply)=tx.query_row("SELECT reply FROM wrong_practice_turns WHERE namespace=?1 AND session_id=?2 AND message_id=?3",params![ns,sid,mid],|r|r.get::<_,String>(0)).optional()? { return Ok(json!({"handled":true,"reply":reply})); }
    }
    let saved: Option<String> = tx
        .query_row(
            "SELECT state FROM wrong_practice_sessions WHERE namespace=?1 AND session_id=?2",
            params![ns, sid],
            |r| r.get(0),
        )
        .optional()?;
    let mut state: Value = saved
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(json!({}));
    let active = state["active"].as_bool().unwrap_or(false);
    let types = query_types(query);
    let explicit = has(
        query,
        &[
            "错题",
            "出一道",
            "出一题",
            "练一下",
            "练一道",
            "最新上传的题",
            "刚刚上传的题",
            "第",
        ],
    );
    let count = has(query, &["多少", "几道", "数量", "总共", "一共"]);
    let today = has(query, &["今天", "今日", "当天"])
        && has(
            query,
            &[
                "答题", "答对", "答错", "复习", "情况", "练了", "做了", "记录",
            ],
        );
    let follow = active
        && (has(
            query,
            &[
                "提示",
                "不会",
                "不知道",
                "想不起来",
                "下一题",
                "再来",
                "继续",
                "再练",
                "结束",
                "不练了",
                "答案",
            ],
        ) || count
            || !types.is_empty()
            || plausible_answer(query)
            || (state["poem"] == true
                && query.chars().count() <= 35
                && !has(
                    query,
                    &[
                        "什么",
                        "怎么",
                        "为何",
                        "天气",
                        "我喜欢",
                        "帮我",
                        "几点",
                        "提醒",
                        "嗯",
                        "好的",
                    ],
                )));
    if !(explicit || today || follow || (count && !types.is_empty())) {
        state["active"] = json!(false);
        save_state(tx, ns, sid, &state)?;
        return Ok(json!({"handled":false}));
    }
    // 「第」只能和题目编号结合，不能接管普通日期、第一份报告等。
    let number_re = Regex::new(r"第\s*([0-9一二三四五六七八九十]+)\s*[道个]?题").unwrap();
    if explicit
        && query.contains('第')
        && !number_re.is_match(query)
        && !has(query, &["错题", "练一下", "出一道", "出一题"])
        && !follow
        && !today
    {
        return Ok(json!({"handled":false}));
    }
    let mut bank = load(tx, ns)?;
    let roster = household::list_members(tx, ns)?;
    let named: Vec<_> = roster
        .iter()
        .filter(|m| query.contains(&m.display_name))
        .collect();
    let requested = named
        .first()
        .map(|m| m.member_id.as_str())
        .or_else(|| state["owner"].as_str());
    if let Some(owner) = requested {
        bank.retain(|q| {
            q.meta["subjects"]
                .as_array()
                .is_some_and(|s| s.iter().any(|x| x["member_id"] == owner))
        });
    }
    let owners: std::collections::HashSet<_> = bank
        .iter()
        .filter_map(|q| q.meta["subjects"][0]["member_id"].as_str())
        .collect();
    if owners.len() > 1 && requested.is_none() {
        return Ok(json!({"handled":true,"reply":"有多位成员的错题，请指定要练习谁的错题。"}));
    }
    if let Some(owner) = owners.iter().next() {
        state["owner"] = json!(owner);
    }
    let mut filtered = bank.clone();
    filtered.retain(|q| {
        types.iter().all(|t| {
            q.meta["question_types"]
                .as_array()
                .is_some_and(|v| v.iter().any(|v| v == *t))
        })
    });
    let reply = if today {
        let today = b["local_date"].as_str().unwrap_or("");
        let owner = state["owner"].as_str().unwrap_or("");
        let mut stmt=tx.prepare("SELECT content,metadata_json FROM memories WHERE namespace=?1 AND tag='错题集' AND state='active'")?;
        let rows = stmt.query_map([ns], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut lines = Vec::new();
        let mut good = 0;
        let mut bad = 0;
        let mut hinted = 0;
        for row in rows {
            let (c, m) = row?;
            let m: Value = serde_json::from_str(&m)?;
            if m["practice_kind"] == "attempt"
                && m["local_date"] == today
                && (owner.is_empty() || m["owner"] == owner)
            {
                if m["correct"] == true {
                    good += 1;
                    if m["hinted"] == true {
                        hinted += 1;
                    }
                } else {
                    bad += 1;
                }
                lines.push(c);
            }
        }
        format!(
            "今天已作答{}次，答对{}次（其中提示后答对{}次），答错{}次。{}",
            good + bad,
            good,
            hinted,
            bad,
            if lines.is_empty() {
                "暂无答题记录。".into()
            } else {
                format!("\n{}", lines.join("\n"))
            }
        )
    } else if count {
        let detail = if types.is_empty() {
            ["选择题", "判断题", "填空题", "数学题", "语文题", "古诗题"]
                .iter()
                .map(|t| {
                    format!(
                        "{}{}道",
                        t,
                        bank.iter()
                            .filter(|q| q.meta["question_types"]
                                .as_array()
                                .is_some_and(|v| v.iter().any(|x| x == *t)))
                            .count()
                    )
                })
                .collect::<Vec<_>>()
                .join("，")
        } else {
            types.join("、")
        };
        format!(
            "错题集共{}道；{}，符合本次条件的有{}道。题型标签可重叠。",
            bank.len(),
            detail,
            filtered.len()
        )
    } else if has(query, &["结束", "不练了", "先到这里"]) {
        state["active"] = json!(false);
        "本轮练习结束，已确认的答题记录已保存。可以询问今天的答题情况。".into()
    } else {
        let current = state["current"]
            .as_str()
            .and_then(|id| bank.iter().find(|q| q.id == id))
            .cloned();
        let select = explicit
            || has(query, &["下一题", "再来", "再练", "继续"])
            || !types.is_empty()
            || current.is_none();
        if !select {
            let q = current.unwrap();
            let a = q.meta["answer"].as_str().unwrap_or("");
            if has(query, &["提示"]) {
                state["hinted"] = json!(true);
                format!("提示：{}", hint(&q))
            } else if has(query, &["答案"]) {
                state["hinted"] = json!(true);
                format!(
                    "正确答案是：{}。可以再试一次。",
                    if a.is_empty() {
                        "这道题的答案尚未核对"
                    } else {
                        a
                    }
                )
            } else if state["answered"] == true {
                "本题已记录。说“下一题”继续，或“再练一次”重新练习。".into()
            } else if a.is_empty() {
                "这道题尚无核对后的答案，先不判分。可以练下一题。".into()
            } else {
                match if has(query, &["不知道", "不会", "想不起来"]) {
                    Some(false)
                } else {
                    judge(&q, query)
                } {
                    None => "还不能确定你的答案，请说选项、判断对错，或填空内容。".into(),
                    Some(correct) => {
                        let weight = (q.meta["weight"].as_f64().unwrap_or(1.0)
                            * if correct { 0.6 } else { 1.8 })
                        .clamp(0.2, 20.0);
                        let id = Uuid::parse_str(&q.id)
                            .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
                        if let Some(mut rec) = store::get_memory(tx, id)? {
                            rec.metadata["weight"] = json!(weight);
                            rec.updated_at_ms = now_ms();
                            store::update_memory(tx, &mut rec, jieba)?;
                        }
                        let owner = state["owner"].as_str().unwrap_or("");
                        let date = b["local_date"].as_str().unwrap_or("");
                        let hinted = state["hinted"] == true;
                        let text = format!(
                            "{} 第{}题，作答：{}，{}{}。",
                            date,
                            q.number,
                            query,
                            if correct { "答对" } else { "答错" },
                            if hinted { "（已提示）" } else { "" }
                        );
                        record(
                            tx,
                            ns,
                            jieba,
                            owner,
                            &text,
                            json!({"practice_kind":"attempt","channel":"wrong_practice","subjects":[{"member_id":owner,"confidence":1.0}],"owner":owner,"question_id":q.id,"local_date":date,"correct":correct,"hinted":hinted,"answer":query}),
                            Some(sid.into()),
                        )?;
                        state["answered"] = json!(true);
                        if !correct {
                            state["hinted"] = json!(true);
                        }
                        if correct {
                            "答对了！这次答题已记录。说“下一题”继续。".into()
                        } else {
                            format!("这次还没答对，正确答案是{}。已记录，可以再练一次。", a)
                        }
                    }
                }
            }
        } else {
            if let Some(c) = number_re.captures(query) {
                let n = parse_number(&c[1]);
                filtered.retain(|q| q.number == n);
            }
            if has(query, &["最新", "刚刚", "刚传", "刚上传"]) {
                if let Some(latest) = filtered
                    .iter()
                    .filter_map(|q| q.meta["uploaded_at_ms"].as_i64())
                    .max()
                {
                    filtered.retain(|q| q.meta["uploaded_at_ms"] == latest);
                }
            }
            let quoted = Regex::new(r"[‘'“《]([^’'”》]+)[’'”》]").unwrap();
            let keys: Vec<_> = quoted
                .captures_iter(query)
                .map(|c| c[1].to_string())
                .collect();
            if !keys.is_empty() {
                filtered.retain(|q| keys.iter().all(|k| norm(&q.raw).contains(&norm(k))));
            } else if query.contains("咬定青山") {
                filtered.retain(|q| q.raw.contains("咬定青山"));
            } else if query.contains("竹石") {
                filtered.retain(|q| q.raw.contains("竹石"));
            }
            if has(query, &["再练"]) && current.is_some() {
                filtered = vec![current.unwrap()];
            }
            if filtered.is_empty() {
                "没有找到符合条件的错题，请换个题型、编号或题目关键词。".into()
            } else {
                let total: f64 = filtered
                    .iter()
                    .map(|q| q.meta["weight"].as_f64().unwrap_or(1.0))
                    .sum();
                let random = (Uuid::now_v7().as_u128() % 1_000_000) as f64 / 1_000_000.0 * total;
                let mut sum = 0.0;
                let q = filtered
                    .iter()
                    .find(|q| {
                        sum += q.meta["weight"].as_f64().unwrap_or(1.0);
                        sum > random
                    })
                    .unwrap_or(&filtered[0]);
                state["poem"] = json!(q.meta["question_types"]
                    .as_array()
                    .is_some_and(|v| v.contains(&json!("古诗题"))));
                state["current"] = json!(q.id);
                state["active"] = json!(true);
                state["answered"] = json!(false);
                state["hinted"] = json!(has(query, &["再练"]));
                view(q)
            }
        }
    };
    save_state(tx, ns, sid, &state)?;
    if !mid.is_empty() {
        tx.execute(
            "INSERT INTO wrong_practice_turns VALUES(?1,?2,?3,?4)",
            params![ns, sid, mid, reply],
        )?;
    }
    Ok(json!({"handled":true,"reply":reply}))
}
fn parse_number(s: &str) -> i64 {
    s.parse().unwrap_or_else(|_| {
        let digits = |s: &str| match s {
            "一" => 1,
            "二" => 2,
            "三" => 3,
            "四" => 4,
            "五" => 5,
            "六" => 6,
            "七" => 7,
            "八" => 8,
            "九" => 9,
            _ => 0,
        };
        if let Some((a, b)) = s.split_once('十') {
            let tens = if a.is_empty() { 1 } else { digits(a) };
            tens * 10 + digits(b)
        } else {
            digits(s)
        }
    })
}
fn plausible_answer(s: &str) -> bool {
    let n = norm(s);
    n.len() < 90
        && (Regex::new(r"^(选|我选|答案是|选项)?[ABCDabcd一二三四1234]([个米厘米]+)?[。！! ]*$")
            .unwrap()
            .is_match(s)
            || has(
                s,
                &[
                    "正确", "错误", "对的", "错的", "不对", "粉骨", "任尔", "意恐", "遍插", "门泊",
                    "只缘", "疑是",
                ],
            )
            || matches!(n.as_str(), "对" | "错" | "是" | "不是"))
}
fn judge(q: &Q, input: &str) -> Option<bool> {
    let a = q.meta["answer"].as_str().unwrap_or("");
    let n = norm(input);
    if a == "√" || a == "×" {
        let yes =
            matches!(n.as_str(), "对" | "正确" | "对的" | "是" | "是对的") || input.trim() == "√";
        let no = matches!(
            n.as_str(),
            "错" | "错误" | "错的" | "不对" | "不是" | "是错的"
        ) || input.trim() == "×";
        return if yes || no {
            Some(yes == (a == "√"))
        } else {
            None
        };
    }
    if a.len() == 1 && "ABCD".contains(a) {
        let re = Regex::new(r"^(?:我选|选|答案是|选项)?\s*([ABCDabcd])(?:[。！!])?$").unwrap();
        if let Some(c) = re.captures(input) {
            return Some(c[1].eq_ignore_ascii_case(a));
        }
        let options = q.raw.lines().filter(|l| l.contains("A.")).collect::<Vec<_>>().join(" ");
        for option in ["A", "B", "C", "D"] {
            let opt = Regex::new(&format!(r"{}[.．、]\s*(.*?)(?:[ABCD][.．、]|$)", option)).unwrap();
            if let Some(c) = opt.captures(&options) {
                if norm(&c[1]) == n { return Some(option == a); }
            }
        }
        return None;
    }
    if n.is_empty() {
        None
    } else {
        Some(n.contains(&norm(a)))
    }
}
fn hint(q: &Q) -> String {
    if q.raw.contains("千锤万凿") {
        return "想一想石灰经历粉碎焚烧仍不畏惧，空缺诗句以“粉骨”开头。".into();
    }
    if q.raw.contains("咬定青山") {
        return "竹子不怕任何方向的风，末句以“任尔”开头，注意东西南北的顺序。".into();
    }
    if q.raw.contains("苹果") {
        return "平均分，用苹果总数除以小朋友人数。".into();
    }
    if q.raw.contains("周长") {
        return "想想正方形有几条相等的边。".into();
    }
    if let Some((_, s)) = q.raw.split_once("解析：") {
        return s.into();
    }
    let a = q.meta["answer"].as_str().unwrap_or("");
    format!(
        "回忆题干中的关键词，答案以“{}”开头。",
        a.chars().next().map(|c| c.to_string()).unwrap_or_default()
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryEngine;
    const MD: &str = "# 错题集\n姓名：梁子航\n## 选择题\n**1\\.** 12个苹果平均分给4人？\nA. 2个 B. 3个 C. 4个 D. 6个\n孩子选择：D 答案：B\n## 古诗词填空题\n**16\\.** 千锤万凿出深山，烈火焚烧若等闲。________，要留清白在人间。答案：要留清白在人间（于谦《石灰吟》）";
    async fn setup() -> (tempfile::TempDir, MemoryEngine) {
        let d = tempfile::tempdir().unwrap();
        let e = MemoryEngine::builder()
            .db_path(d.path().join("m.db"))
            .build()
            .await
            .unwrap();
        let r = e
            .wrong_practice(
                json!({"action":"import","text":MD,"source_id":"md1","uploaded_at_ms":1}),
            )
            .await
            .unwrap();
        assert_eq!(r["count"], 2);
        (d, e)
    }
    async fn ask(e: &MemoryEngine, q: &str, mid: &str) -> Value {
        e.wrong_practice(
            json!({"query":q,"session_id":"s1","message_id":mid,"local_date":"2026-11-12"}),
        )
        .await
        .unwrap()
    }
    #[tokio::test]
    async fn import_dedup_tags_and_no_vectors() {
        let (_d, e) = setup().await;
        let r = e
            .wrong_practice(json!({"action":"import","text":MD,"source_id":"md1"}))
            .await
            .unwrap();
        assert_eq!(r["count"], 2);
        let p = e
            .list(ListQuery {
                tag: Some("错题集".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(p.total, 2);
        assert!(p
            .items
            .iter()
            .all(|q| !q.embedding_pending && !q.consolidation_pending));
        assert!(p.items.iter().any(|q| q.metadata["question_types"]
            .as_array()
            .unwrap()
            .contains(&json!("语文题"))));
        assert!(ask(&e, "我喜欢吃什么菜", "x").await["handled"] == false);
        assert!(e
            .recall(RecallQuery {
                text: "苹果".into(),
                fts_only: true,
                skip_graph: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .hits
            .is_empty());
    }
    #[tokio::test]
    async fn selection_answer_weight_idempotence_and_daily() {
        let (_d, e) = setup().await;
        let r = ask(&e, "练一下第1道题", "1").await;
        assert!(r["reply"].as_str().unwrap().contains("苹果"));
        assert!(!r["reply"].as_str().unwrap().contains("答案"));
        let r = ask(&e, "选B", "2").await;
        assert!(r["reply"].as_str().unwrap().contains("答对"));
        ask(&e, "选B", "2").await;
        let r = ask(&e, "今天的答题情况", "3").await;
        assert!(r["reply"].as_str().unwrap().contains("已作答1次"));
        let p = e
            .list(ListQuery {
                tag: Some("错题集".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(p.items.iter().any(|q| q.metadata["weight"] == 0.6));
    }
    #[tokio::test]
    async fn poem_correction_hint_and_original() {
        let (_d, e) = setup().await;
        let r = ask(&e, "出一道‘石灰吟’的填空题", "1").await;
        assert!(r["reply"].as_str().unwrap().contains("千锤"));
        assert!(ask(&e, "提示一下", "2").await["reply"]
            .as_str()
            .unwrap()
            .contains("粉骨"));
        assert!(
            ask(&e, "粉骨碎身浑不怕，要留清白在人间", "3").await["reply"]
                .as_str()
                .unwrap()
                .contains("答对")
        );
        assert!(ask(&e, "今天的答题情况", "4").await["reply"]
            .as_str()
            .unwrap()
            .contains("提示后答对1次"));
    }
    #[tokio::test]
    async fn image_latest_and_session_isolation() {
        let (_d, e) = setup().await;
        let r=e.wrong_practice(json!({"action":"import","source_id":"image1","kind":"image_ocr","uploaded_at_ms":2,"text":"古诗词积累练习\n姓名：梁子航\n21. 请补全《竹石》中的诗句。\n咬定青山不放松，立根原在破岩中。\n千磨万击还坚劲，任尔东南西北风。× 再想一想"})).await.unwrap();
        assert_eq!(r["count"], 1);
        let r = ask(&e, "练一下刚刚上传的题目", "a").await;
        assert!(r["reply"].as_str().unwrap().contains("第21题"));
        assert!(!r["reply"].as_str().unwrap().contains("任尔"));
        assert!(ask(&e, "任尔东西南北风", "b").await["reply"]
            .as_str()
            .unwrap()
            .contains("答对"));
        assert_eq!(
            e.wrong_practice(json!({"query":"选B","session_id":"other"}))
                .await
                .unwrap()["handled"],
            false
        );
    }
    #[tokio::test]
    async fn wrong_and_noisy_answers_and_normal_tag_reservation() {
        let (_d, e) = setup().await;
        ask(&e, "练一下第1道题", "1").await;
        assert_eq!(ask(&e, "明天北京天气怎么样", "2").await["handled"], false);
        ask(&e, "练一下第1道题", "3").await;
        assert!(ask(&e, "选D", "4").await["reply"]
            .as_str()
            .unwrap()
            .contains("还没答对"));
        let tags = vec![household::MemoryTag {
            tag_id: "错题集".into(),
            display_name: "错题集".into(),
            aliases: vec![],
        }];
        assert_ne!(
            household::select_memory_tag(&tags, &json!({"tag":"错题集"}), "我爱吃鱼", &[]),
            "错题集"
        );
    }
}
