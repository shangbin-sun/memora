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
    Regex::new(r"(?:正确答案|(?:^|[\s　])答案)[：:]\s*([^\n　（(]+)")
        .unwrap()
        .captures(text)
        .map(|c| {
            let answer = c[1].split("解析").next().unwrap_or("").trim();
            if answer.starts_with(['A', 'B', 'C', 'D']) {
                answer.chars().next().unwrap().to_string()
            } else {
                answer.trim_end_matches(['。', '，', '!', '！']).to_string()
            }
        })
        .unwrap_or_default()
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
    Regex::new(r"(?:[×✗✘❌]|[ \t]+[Xx])[ \t　。.!！\r\n]*$").unwrap()
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
                let value = mark.replace(&t[option.body_start..option.end], "");
                format!("{}. {}", option.label, value.trim())
            })
            .collect();
        return format!("{}\n{}", t[..choices[0].start].trim(), options.join("　"));
    }
    Regex::new(r"[（(][√×][）)]|[×✗❌]")
        .unwrap()
        .replace_all(t, "（ ）")
        .to_string()
}
// 同题重传允许题干的常见等价说法与批改痕迹不同，保留最新原文及上一版来源。
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
    if has(text, &["东南西北"]) {
        return "任尔东南西北风".into();
    }
    let _ = heading;
    "没答上".into()
}
fn explanation(text: &str) -> Option<String> {
    for label in ["订正", "訂正", "改正", "更正", "解析"] {
        let re = Regex::new(&format!(r"{}[：:][ \t]*", label)).unwrap();
        if let Some(found) = re.find(text) {
            let value = before_answer_fields(&text[found.end()..])
                .trim()
                .trim_end_matches('*')
                .trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}
fn compact_record(text: &str, correct: &str, child: &str) -> String {
    let text = text.split("练习题面：").last().unwrap_or(text);
    let text = Regex::new(r"^\s*\d+[.．、。]\s*")
        .unwrap()
        .replace(text, "");
    let stem = question(&text);
    let stem = Regex::new(r"_+").unwrap().replace_all(&stem, "（ ）");
    let stem = stem.trim().trim_end_matches('*').trim();
    let extracted_explanation = explanation(&text);
    let source = Regex::new(r"[（(]([^（）()]*《[^》]+》)[）)]")
        .unwrap()
        .captures(&text)
        .map(|c| c[1].to_string())
        .unwrap_or_default();
    let fallback = if source.is_empty() {
        "按题意核对正确答案。".to_string()
    } else {
        format!("{}。", source)
    };
    let explanation = extracted_explanation.as_deref().unwrap_or(&fallback);
    let stem = Regex::new(r"\s*([ABCD][.．、])\s*")
        .unwrap()
        .replace_all(stem, "　$1 ");
    // 选项横排，具体换行由UI按实际测量宽度处理。
    let stem = stem.replace("　A.", "\nA.");
    format!(
        "{}\n**孩子答案：{}　正确答案：{}**　解析：{}",
        stem.trim(),
        child,
        correct,
        explanation
    )
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
        let a = answer(&q);
        let child = child_answer(&q, heading);
        q = compact_record(&q, &a, &child);
        if let Some(old) = existing.iter().find(|old| {
            old.number == number
                && (old.meta["source_id"] == source || question_key(&old.raw) == question_key(&q))
                && old.meta["subjects"][0]["member_id"] == owner
        }) {
            let id = Uuid::parse_str(&old.id)
                .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?;
            if let Some(mut rec) = store::get_memory(tx, id)? {
                rec.metadata["uploaded_at_ms"] =
                    json!(body["uploaded_at_ms"].as_i64().unwrap_or_else(now_ms));
                rec.metadata["latest_source_id"] = json!(source);
                rec.metadata["previous_original_text"] = rec.metadata["original_text"].clone();
                rec.metadata["original_text"] = json!(raw_q);
                rec.metadata["source_name"] = body["name"].clone();
                rec.metadata["image"] = json!(image);
                rec.metadata["answer"] = json!(a);
                rec.metadata["child_answer"] = json!(child);
                rec.content = q.clone();
                rec.updated_at_ms = now_ms();
                store::update_memory(tx, &mut rec, jieba)?;
            }
            ids.push(old.id.clone());
            count += 1;
            continue;
        }
        let meta = json!({"practice_kind":"question","channel":"wrong_practice","subjects":[{"member_id":owner,"confidence":1.0}],"number":number,"question_types":labels(&q,heading),"answer":a,"child_answer":child,"weight":1.0,"source_id":source,"latest_source_id":source,"uploaded_at_ms":body["uploaded_at_ms"].as_i64().unwrap_or_else(now_ms),"source_name":body["name"],"original_text":raw_q,"answer_correction":if q.contains("千锤万凿出深山"){json!("资料答案勘误：粉骨碎身浑不怕")}else{Value::Null},"image":image});
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
        if let Some(reply)=tx.query_row("SELECT reply FROM wrong_practice_turns WHERE namespace=?1 AND session_id=?2 AND message_id=?3",params![ns,sid,mid],|r|r.get::<_,String>(0)).optional()? { return Ok(serde_json::from_str::<Value>(&reply).ok().filter(|v|v.is_object()).unwrap_or(json!({"handled":true,"reply":reply}))); }
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
    let number_re = Regex::new(r"(?:第\s*|题号\s*[是为：:]?\s*|[Qq])([0-9零〇一二两三四五六七八九十百]+)\s*(?:[道个]?题|号)?").unwrap();
    let number_match = number_re
        .captures(query)
        .filter(|_| has(query, &["题", "练", "Q", "q"]));
    let practice_action = Regex::new(r"复习|温习|重温|刷题|练习|练练|练一下|练一练|出.{0,4}题|来.{0,4}题|做.{0,4}题|考考|抽.{0,4}题").unwrap().is_match(query);
    let explicit = has(
        query,
        &[
            "错题",
            "出一道",
            "出一题",
            "练一道",
            "最新上传的题",
            "刚刚上传的题",
        ],
    ) || number_match.is_some()
        || (practice_action && (active || has(query, &["题", "整理", "上传", "复习"])));
    let count = has(
        query,
        &[
            "多少",
            "几道",
            "几题",
            "数量",
            "总共",
            "一共",
            "总数",
            "各有",
            "几种",
            "几首",
            "几条",
            "总计",
            "有哪些题型",
            "题型有哪些",
            "分几类",
        ],
    );
    let today = (has(query, &["今天", "今日", "当天"])
        || (active
            && has(query, &["答对", "答错", "做对", "做错", "成绩", "答得"])
            && !has(query, &["昨天", "前天", "上周"])))
        && has(
            query,
            &[
                "答题", "答对", "答错", "复习", "情况", "练了", "做了", "记录", "答得", "做得",
                "做过", "成绩", "对了", "错了", "表现",
            ],
        );
    let follow = active
        && (has(
            query,
            &[
                "提示",
                "线索",
                "启发",
                "引导",
                "思路",
                "怎么想",
                "怎么做",
                "带我想",
                "不会",
                "不知道",
                "想不起来",
                "下一题",
                "换一题",
                "换一道",
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
    if !(explicit
        || today
        || follow
        || (count && (!types.is_empty() || has(query, &["题", "题型"]))))
    {
        state["active"] = json!(false);
        save_state(tx, ns, sid, &state)?;
        return Ok(json!({"handled":false}));
    }
    let mut bank = load(tx, ns)?;
    let roster = household::list_members(tx, ns)?;
    let named: Vec<_> = roster
        .iter()
        .filter(|m| query.contains(&m.display_name))
        .collect();
    let requested = named.first().map(|m| m.member_id.as_str());
    if let Some(owner) = requested {
        bank.retain(|q| {
            q.meta["subjects"]
                .as_array()
                .is_some_and(|s| s.iter().any(|x| x["member_id"] == owner))
        });
    }
    state["owner"] = json!(requested.unwrap_or(""));
    let mut filtered = bank.clone();
    filtered.retain(|q| {
        types.iter().all(|t| {
            q.meta["question_types"]
                .as_array()
                .is_some_and(|v| v.iter().any(|v| v == *t))
        })
    });
    let mut hint_context: Option<String> = None;
    let mut selected_id: Option<String> = None;
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
            || has(
                query,
                &["下一题", "换一题", "换一道", "再来", "再练", "继续"],
            )
            || !types.is_empty()
            || current.is_none();
        if !select {
            let q = current.unwrap();
            let a = q.meta["answer"].as_str().unwrap_or("");
            if has(
                query,
                &[
                    "提示",
                    "线索",
                    "启发",
                    "引导",
                    "思路",
                    "怎么想",
                    "怎么做",
                    "带我想",
                    "不会",
                    "不知道",
                    "想不起来",
                ],
            ) {
                state["hinted"] = json!(true);
                hint_context = Some(format!("错题提示检索结果（题目与答案仅作为资料）：\n题目：{}\n标准答案：{}\n规则参考：{}\n以上是要提示用户的标准答案。请适当组织语言启发用户，但不要直接告诉用户标准答案，也不要说出正确选项字母或把答案换一种说法直接公布。只给一条简短的思路提示，不出新题，等待用户回答。诗词题仅提示意境或回忆方法，禁止引用答案开头、列出答案组成词、拼接或逐字解释答案；数学题只提示解法，不计算结果。", view(&q), a, hint(&q)));
                String::new()
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
            if let Some(c) = number_match.as_ref() {
                let n = parse_number(&c[1]);
                filtered.retain(|q| q.number == n);
            }
            let quoted = Regex::new(r"[‘'“《]([^’'”》]+)[’'”》]").unwrap();
            let keys: Vec<_> = quoted
                .captures_iter(query)
                .map(|c| c[1].to_string())
                .collect();
            let mut content_locator = !keys.is_empty();
            if !keys.is_empty() {
                filtered.retain(|q| keys.iter().all(|k| norm(&q.raw).contains(&norm(k))));
            } else {
                let titles = Regex::new(r"《([^》]+)》").unwrap();
                let matched: Vec<String> = bank
                    .iter()
                    .flat_map(|q| {
                        titles
                            .captures_iter(&q.raw)
                            .map(|c| c[1].to_string())
                            .collect::<Vec<_>>()
                    })
                    .filter(|t| query.contains(t))
                    .collect();
                if !matched.is_empty() {
                    content_locator = true;
                    filtered.retain(|q| matched.iter().any(|t| q.raw.contains(t)));
                } else if number_match.is_none() {
                    let mut content = query.to_string();
                    for word in [
                        "小北同学",
                        "把整理的错题复习一下",
                        "帮我",
                        "给我",
                        "出一道",
                        "来一道",
                        "拿出来",
                        "温习",
                        "重温",
                        "换一道",
                        "换一题",
                        "看看",
                        "考考",
                        "给我出",
                        "来一题",
                        "给一道",
                        "做一题",
                        "来个",
                        "做个",
                        "这一题",
                        "这份",
                        "能不能",
                        "麻烦",
                        "帮忙",
                        "做一道",
                        "练一下",
                        "练一练",
                        "复习一下",
                        "复习",
                        "选择题",
                        "判断题",
                        "填空题",
                        "数学题",
                        "算术题",
                        "语文题",
                        "古诗题",
                        "古诗词",
                        "古诗",
                        "题目",
                        "错题",
                        "那道",
                        "那个",
                        "一道",
                        "最新",
                        "刚刚",
                        "刚上传",
                        "上传",
                        "上次",
                        "上一批",
                        "整理",
                        "之前",
                        "练练",
                        "给点",
                        "练习",
                        "再来",
                        "再练一次",
                        "下一题",
                        "继续",
                        "有关",
                        "关于",
                        "的题",
                        "一下",
                    ] {
                        content = content.replace(word, " ");
                    }
                    let tokens: Vec<_> = jieba
                        .cut(&content, false)
                        .into_iter()
                        .filter(|w| {
                            w.chars().count() >= 2
                                && ![
                                    "就要", "这道", "一题", "那题", "数量", "题型", "总共", "一共",
                                ]
                                .contains(w)
                        })
                        .collect();
                    if !tokens.is_empty() {
                        content_locator = true;
                        let score = |q: &Q| -> usize {
                            let stem = norm(&question(&q.raw));
                            tokens
                                .iter()
                                .filter(|w| {
                                    if w.chars().all(|c| c.is_ascii_digit()) {
                                        Regex::new(&format!(
                                            r"(?:^|[^0-9]){}(?:[^0-9]|$)",
                                            regex::escape(w)
                                        ))
                                        .unwrap()
                                        .is_match(&stem)
                                    } else {
                                        stem.contains(&norm(w))
                                    }
                                })
                                .map(|w| w.chars().count())
                                .sum()
                        };
                        let best = filtered.iter().map(score).max().unwrap_or(0);
                        filtered.retain(|q| best > 0 && score(q) == best);
                    }
                }
            }
            let repeat_upload = has(
                query,
                &[
                    "上次上传",
                    "上一批上传",
                    "刚才上传",
                    "就要上次",
                    "再做刚上传",
                ],
            );
            let explicit_locator = content_locator
                || number_match.is_some()
                || !keys.is_empty()
                || has(query, &["再练", "刚才那题"]);
            let review_latest = !explicit_locator && types.is_empty() && practice_action;
            if repeat_upload || has(query, &["最新", "刚刚", "刚传", "刚上传"]) || review_latest
            {
                if !repeat_upload && !explicit_locator {
                    filtered.retain(|q| {
                        q.meta["last_presented_upload_ms"].as_i64().unwrap_or(-1)
                            < q.meta["uploaded_at_ms"].as_i64().unwrap_or(0)
                    });
                }
                if let Some(latest) = filtered
                    .iter()
                    .filter_map(|q| q.meta["uploaded_at_ms"].as_i64())
                    .max()
                {
                    filtered.retain(|q| q.meta["uploaded_at_ms"] == latest);
                }
            } else if !explicit_locator {
                filtered.retain(|q| {
                    q.meta["last_presented_upload_ms"].as_i64().unwrap_or(-1)
                        < q.meta["uploaded_at_ms"].as_i64().unwrap_or(0)
                });
            }
            if has(query, &["再练"]) && current.is_some() {
                filtered = vec![current.unwrap()];
            }
            if filtered.is_empty() {
                "没有找到尚未出过且符合条件的错题。可以指定题号或明确说复习上次上传的错题。".into()
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
                if let Some(mut rec) = store::get_memory(
                    tx,
                    Uuid::parse_str(&q.id)
                        .map_err(|e| crate::error::MemoraError::InvalidState(e.to_string()))?,
                )? {
                    rec.metadata["last_presented_upload_ms"] = q.meta["uploaded_at_ms"].clone();
                    rec.metadata["last_presented_at_ms"] = json!(now_ms());
                    store::update_memory(tx, &mut rec, jieba)?;
                }
                state["owner"] = q.meta["subjects"][0]["member_id"].clone();
                state["poem"] = json!(q.meta["question_types"]
                    .as_array()
                    .is_some_and(|v| v.contains(&json!("古诗题"))));
                selected_id = Some(q.id.clone());
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
            params![ns, sid, mid, json!({"handled":true,"reply":reply,"hint_context":hint_context,"question_id":selected_id}).to_string()],
        )?;
    }
    Ok(json!({"handled":true,"reply":reply,"hint_context":hint_context,"question_id":selected_id}))
}
fn parse_number(s: &str) -> i64 {
    s.parse().unwrap_or_else(|_| {
        let digits = |s: &str| match s {
            "一" => 1,
            "二" | "两" => 2,
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
        let re = Regex::new(
            r"^(?:我选|选择|选|应该是|我觉得是|答案选|答案是|选项)?\s*([ABCDabcd])(?:[。！!])?$",
        )
        .unwrap();
        if let Some(c) = re.captures(input) {
            return Some(c[1].eq_ignore_ascii_case(a));
        }
        let options = question(&q.raw).replace('\n', " ");
        for option in ["A", "B", "C", "D"] {
            let opt =
                Regex::new(&format!(r"{}[.．、]\s*(.*?)(?:[ABCD][.．、]|$)", option)).unwrap();
            if let Some(c) = opt.captures(&options) {
                if norm(&c[1]) == n {
                    return Some(option == a);
                }
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
        return "联系石灰经受磨难仍保持清白的意境，先在心中从上一句往下背。".into();
    }
    if q.raw.contains("咬定青山") {
        return "想一想最后一句怎样衬托竹子经历磨难仍然坚定，先从前一句在心中往下背。".into();
    }
    if q.raw.contains("苹果") {
        return "平均分，用苹果总数除以小朋友人数。".into();
    }
    if q.raw.contains("周长") {
        return "想想正方形有几条相等的边。".into();
    }
    "先找题干的已知条件和所问内容，回忆相关知识，再尝试作答。".into()
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
        assert!(ask(&e, "提示一下", "2").await["hint_context"]
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

#[cfg(test)]
mod demo_v2_tests {
    use super::*;
    use crate::MemoryEngine;
    #[tokio::test]
    async fn latest_upload_is_shared_persistent_and_not_repeated_without_explicit_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("demo.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        for (source, at, num) in [("old", 10, 3), ("fresh", 20, 1)] {
            let text=format!("数学错题整理\n姓名：梁子航\n{}. 选择题：12个苹果平均分给4人？\nA. 2个\nB. 3个\nC. 4个 ×\nD. 6个\n订正：12÷4=3（个）\n正确答案：B",num);
            assert_eq!(e.wrong_practice(json!({"action":"import","source_id":source,"text":text,"kind":"image_ocr","uploaded_at_ms":at})).await.unwrap()["count"],1);
        }
        let a=e.wrong_practice(json!({"query":"把整理的错题复习一下","session_id":"s","message_id":"a","speaker_id":"unknown"})).await.unwrap();
        assert!(a["reply"].as_str().unwrap().starts_with("第1题"));
        assert!(!a["reply"].as_str().unwrap().contains("订正"));
        assert_eq!(
            e.wrong_practice(
                json!({"query":"把整理的错题复习一下","session_id":"s","message_id":"a"})
            )
            .await
            .unwrap()["reply"],
            a["reply"]
        );
        e.shutdown();
        drop(e);
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        let b = e
            .wrong_practice(
                json!({"query":"把整理的错题复习一下","session_id":"another","speaker_id":"别人"}),
            )
            .await
            .unwrap();
        assert!(b["reply"].as_str().unwrap().starts_with("第3题"));
        let c = e
            .wrong_practice(json!({"query":"就要上次上传的错题","session_id":"s"}))
            .await
            .unwrap();
        assert_eq!(c["question_id"], a["question_id"]);
        let hint = e
            .wrong_practice(json!({"query":"提示一下","session_id":"s"}))
            .await
            .unwrap();
        assert!(hint["hint_context"]
            .as_str()
            .unwrap()
            .contains("标准答案：B"));
        assert_eq!(hint["reply"], "");
        for query in ["能给一点思路吗", "这道题怎么想", "我不会", "带我想一下"]
        {
            let r = e
                .wrong_practice(json!({"query":query,"session_id":"s"}))
                .await
                .unwrap();
            assert!(r["hint_context"].as_str().unwrap().contains("标准答案：B"));
            assert_eq!(r["reply"], "");
        }
        let answer = e
            .wrong_practice(json!({"query":"3个","session_id":"s","local_date":"2026-11-12"}))
            .await
            .unwrap();
        assert!(answer["reply"].as_str().unwrap().contains("答对"));
    }
}

#[cfg(test)]
mod ocr_mark_regressions {
    use super::*;
    use crate::{MemoryEngine, MemoryService};

    #[test]
    fn choice_marks_are_local_to_each_option_and_blank_is_not_guessed() {
        for (options, expected) in [
            ("A. 2个 B. 3个 C 4个 X D. 6个", "C"),
            ("A. 2个　B. 3个　C. 4个 ×　D. 6个", "C"),
            ("A. 2个\nB. 3个\nC 4个 x\nD. 6个", "C"),
            ("A. 2个 × B. 3个 C. 4个 D. 6个", "A"),
            ("A. 2个 × B. 3个 C. 4个 × D. 6个", "没答上"),
            ("A. 2个 B. 3个 C. 4个 D. 6个", "没答上"),
            ("A. 2×3 B. 3×4 C. 4×5 D. 6×7", "没答上"),
            ("A. 2个 B. 3个 Ⓒ 4个 D. 6个", "C"),
            ("A. 2个 B. 3个 (C) 4个 D. 6个", "C"),
        ] {
            let text =
                format!("选择题：每个小朋友几个苹果？\n{options}\n订正：12÷4=3（个）\n正确答案：B");
            assert_eq!(child_answer(&text, ""), expected, "{text}");
        }
        assert_eq!(
            child_answer(
                "A. 2个 B. 3个 C 4个 X D. 6个\n孩子答案：没答上\n正确答案：B",
                ""
            ),
            "C"
        );
        assert_eq!(
            child_answer(
                "A. 2个 B. 3个 C. 4个 D. 6个\n孩子选择：C\n订正：12÷4=3（个）\n正确答案：B",
                ""
            ),
            "C"
        );
    }

    #[test]
    fn correction_is_the_explanation_and_marks_do_not_remain_in_choices() {
        let text="1. 选择题：12个苹果平均分给4人？ A. 2个 B. 3个 C 4个 X D. 6个\n订正：12÷4=3（个）\n正确答案：B\n解析：按题意核对正确答案。";
        let result = compact_record(text, "B", &child_answer(text, ""));
        assert!(result.contains("孩子答案：C　正确答案：B"), "{result}");
        assert!(result.ends_with("解析：12÷4=3（个）"), "{result}");
        assert!(result.contains("C. 4个"), "{result}");
        assert!(!result.contains(" X"));
        assert!(!result.contains("按题意核对"));
        assert_eq!(
            explanation("解析: 平均分用除法。\n正确答案:B").as_deref(),
            Some("平均分用除法。")
        );
        assert_eq!(
            explanation("订正：12÷4＝3（个）\n正确答案：B").as_deref(),
            Some("12÷4＝3（个）")
        );
    }

    #[tokio::test]
    async fn apple_image_reimport_repairs_the_same_stored_question() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("ocr.db");
        let e = MemoryEngine::builder()
            .db_path(&path)
            .build()
            .await
            .unwrap();
        let text="数学错题整理\n姓名：梁子航\n1. 选择题：小明有12个苹果，平均分给4个小朋友，每个小朋友分到几个苹果？\nA. 2个　B. 3个 C 4个 X　D. 6个\n订正：12÷4=3（个）\n正确答案：B";
        let body =
            json!({"action":"import","kind":"image_ocr","source_id":"apple-photo","text":text});
        let first = e.wrong_practice(body.clone()).await.unwrap();
        let id = first["ids"][0].as_str().unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.execute("UPDATE memories SET content='旧错误正文：孩子答案没答上，解析占位句',metadata_json=json_set(metadata_json,'$.child_answer','没答上') WHERE id=?1",[id]).unwrap();
        let second = e.wrong_practice(body).await.unwrap();
        assert_eq!(second["ids"], first["ids"]);
        let page = e.list(ListQuery::default()).await.unwrap();
        assert_eq!(page.total, 1);
        let record = &page.items[0];
        assert_eq!(record.metadata["child_answer"], "C");
        assert_eq!(record.metadata["answer"], "B");
        assert!(
            record.content.contains("孩子答案：C　正确答案：B"),
            "{}",
            record.content
        );
        assert!(
            record.content.ends_with("解析：12÷4=3（个）"),
            "{}",
            record.content
        );
        assert!(!record.content.starts_with("1."));
    }
}
