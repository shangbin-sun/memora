//! Original txt/md excerpts retained only for fact provenance.
//! They are not memories and never participate in recall.

use std::collections::{BTreeSet, HashSet};

use jieba_rs::Jieba;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::core::now_ms;
use crate::error::Result;

#[derive(Debug, Clone, Default)]
pub struct ChunkWrite {
    pub namespace: String,
    pub content: String,
    pub subject_ids: Vec<String>,
    pub tags: Vec<String>,
    pub source_kind: String,
    pub source_name: String,
    pub source_path: Option<String>,
    pub source_locator: Option<String>,
    pub session_id: Option<String>,
    pub message_id: Option<String>,
    pub start_char: i64,
    pub end_char: i64,
    pub memory_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: String,
    pub content: String,
    pub subject_ids: Vec<String>,
    pub tags: Vec<String>,
    pub source_kind: String,
    pub source_name: Option<String>,
    pub source_path: Option<String>,
    pub source_locator: Option<String>,
    pub session_id: Option<String>,
    pub message_id: Option<String>,
    pub start_char: i64,
    pub end_char: i64,
    pub memory_ids: Vec<String>,
}

/// A chunk returned to a host, clipped to the requested character budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledChunk {
    pub id: String,
    pub text: String,
    pub clipped: bool,
    pub subject_ids: Vec<String>,
    pub tags: Vec<String>,
    pub source_kind: String,
    pub source_name: Option<String>,
    pub source_path: Option<String>,
    pub source_locator: Option<String>,
    pub session_id: Option<String>,
    pub message_id: Option<String>,
    pub start_char: i64,
    pub end_char: i64,
    pub memory_ids: Vec<String>,
    /// `linked` = behind a recalled fact; `text` = direct query match.
    pub matched_by: String,
}

#[derive(Debug, Clone, Default)]
pub struct ChunkRecallQuery {
    pub text: String,
    pub namespace: Option<String>,
    pub hit_memory_ids: Vec<String>,
    /// Evidence sentence per recalled fact, used to center a clipped window.
    pub hit_evidence: Vec<(String, String)>,
    pub viewer_member_id: Option<String>,
    pub member_id: Option<String>,
    pub top_k: usize,
    pub max_chars: usize,
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn json_strings(raw: Option<String>) -> Vec<String> {
    raw.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn merge_sorted(existing: Vec<String>, new: &[String]) -> Vec<String> {
    let mut all: BTreeSet<String> = existing.into_iter().collect();
    all.extend(new.iter().cloned());
    all.into_iter().collect()
}

/// Insert or merge one chunk. Identical text (a re-upload, or the overlap of two
/// hard windows) stays a single row and gains the new links, people and tags.
pub fn save_chunk(conn: &Connection, _jieba: &Jieba, chunk: &ChunkWrite) -> Result<String> {
    let now = now_ms();
    let sha = hash_text(&chunk.content);
    let existing: Option<(String, Option<String>, Option<String>)> = conn
        .query_row(
            "SELECT id, subject_ids, tags FROM memory_chunks WHERE namespace = ?1 AND content_sha256 = ?2",
            params![chunk.namespace, sha],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let id = match existing {
        Some((id, subjects, tags)) => {
            let subjects = merge_sorted(json_strings(subjects), &chunk.subject_ids);
            let tags = merge_sorted(json_strings(tags), &chunk.tags);
            conn.execute(
                r#"UPDATE memory_chunks SET subject_ids = ?2, tags = ?3, source_name = ?4,
                   source_path = ?5, source_locator = ?6, session_id = ?7, message_id = ?8,
                   updated_at_ms = ?9 WHERE id = ?1"#,
                params![
                    id,
                    serde_json::to_string(&subjects)?,
                    serde_json::to_string(&tags)?,
                    chunk.source_name,
                    chunk.source_path,
                    chunk.source_locator,
                    chunk.session_id,
                    chunk.message_id,
                    now
                ],
            )?;
            id
        }
        None => {
            let id = Uuid::now_v7().to_string();
            let subjects = merge_sorted(Vec::new(), &chunk.subject_ids);
            let tags = merge_sorted(Vec::new(), &chunk.tags);
            conn.execute(
                r#"INSERT INTO memory_chunks(id, namespace, content_sha256, content, subject_ids, tags,
                   source_kind, source_name, source_path, source_locator, session_id, message_id,
                   start_char, end_char, created_at_ms, updated_at_ms)
                   VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?15)"#,
                params![
                    id,
                    chunk.namespace,
                    sha,
                    chunk.content,
                    serde_json::to_string(&subjects)?,
                    serde_json::to_string(&tags)?,
                    chunk.source_kind,
                    chunk.source_name,
                    chunk.source_path,
                    chunk.source_locator,
                    chunk.session_id,
                    chunk.message_id,
                    chunk.start_char,
                    chunk.end_char,
                    now
                ],
            )?;
            id
        }
    };
    for memory_id in &chunk.memory_ids {
        conn.execute(
            r#"INSERT OR IGNORE INTO memory_chunk_links(namespace, memory_id, chunk_id, created_at_ms)
               VALUES(?1, ?2, ?3, ?4)"#,
            params![chunk.namespace, memory_id, id, now],
        )?;
    }
    Ok(id)
}

fn row_to_chunk(r: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRecord> {
    Ok(ChunkRecord {
        id: r.get("id")?,
        content: r.get("content")?,
        subject_ids: json_strings(r.get("subject_ids")?),
        tags: json_strings(r.get("tags")?),
        source_kind: r.get("source_kind")?,
        source_name: r.get("source_name")?,
        source_path: r.get("source_path")?,
        source_locator: r.get("source_locator")?,
        session_id: r.get("session_id")?,
        message_id: r.get("message_id")?,
        start_char: r.get("start_char")?,
        end_char: r.get("end_char")?,
        memory_ids: Vec::new(),
    })
}

pub fn get_chunk(conn: &Connection, id: &str) -> Result<Option<ChunkRecord>> {
    let mut chunk = conn
        .query_row("SELECT * FROM memory_chunks WHERE id = ?1", params![id], row_to_chunk)
        .optional()?;
    if let Some(chunk) = chunk.as_mut() {
        chunk.memory_ids = linked_memory_ids(conn, &chunk.id)?;
    }
    Ok(chunk)
}

pub fn linked_memory_ids(conn: &Connection, chunk_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT memory_id FROM memory_chunk_links WHERE chunk_id = ?1")?;
    let rows = stmt.query_map(params![chunk_id], |r| r.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn chunks_for_memory(conn: &Connection, memory_id: &str) -> Result<Vec<ChunkRecord>> {
    let mut stmt =
        conn.prepare("SELECT chunk_id FROM memory_chunk_links WHERE memory_id = ?1 ORDER BY created_at_ms")?;
    let ids: Vec<String> = stmt
        .query_map(params![memory_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    let mut out = Vec::new();
    for id in ids {
        if let Some(chunk) = get_chunk(conn, &id)? {
            out.push(chunk);
        }
    }
    Ok(out)
}

const QUESTION_WORDS: &[&str] = &[
    "什么", "多少", "怎么", "怎样", "如何", "哪些", "哪个", "哪家", "哪里", "哪儿", "几个", "几次", "几节",
    "为什么", "是不是", "有没有", "吗", "呢", "吧", "呀", "的", "了", "和", "与", "是", "在", "有", "我", "你",
    "我们", "你们", "他们", "我家", "家里", "请问", "帮我", "告诉", "一下", "还", "要", "会", "能", "可以",
    "现在", "目前", "记得", "知道", "记忆", "一共", "总共", "来着", "最近", "哪块", "这个", "那个", "有什么",
    "什么时候", "几号", "多久", "怎么样",
];

/// One content word of a query with its acceptable spellings, separated by `|`:
/// 数学课 also counts as 数学, and 这周 as 本周.
const ALTERNATIVE: char = '|';

/// Content words of a query, so that question wording cannot satisfy a text match.
pub fn query_keywords(jieba: &Jieba, text: &str) -> Vec<String> {
    let stop: HashSet<&str> = QUESTION_WORDS.iter().copied().collect();
    let mut seen = HashSet::new();
    // Words missing from the dictionary (车贷) arrive as single characters. Join runs of
    // them back together, or the word that matters most never becomes a keyword.
    let mut words: Vec<String> = Vec::new();
    let mut run = String::new();
    for token in jieba.cut(text, false) {
        let token = token.trim();
        let single = token.chars().count() == 1
            && token.chars().all(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
            && !stop.contains(token);
        if single {
            run.push_str(token);
            continue;
        }
        if run.chars().count() >= 2 {
            words.push(std::mem::take(&mut run));
        }
        run.clear();
        words.push(token.to_string());
    }
    if run.chars().count() >= 2 {
        words.push(run);
    }
    let mut keywords: Vec<String> = words
        .iter()
        .map(String::as_str)
        .filter(|w| w.chars().count() >= 2 && !stop.contains(*w))
        .filter(|w| w.chars().any(char::is_alphanumeric))
        .filter(|w| seen.insert((*w).to_string()))
        .map(|word| {
            let mut spellings = vec![word.to_string()];
            let synonyms = crate::query_rewrite::expand_word(word).iter().copied();
            for other in synonyms.chain(jieba.cut_for_search(word, true)) {
                if other.chars().count() >= 2 && !spellings.iter().any(|s| s == other) {
                    spellings.push(other.to_string());
                }
            }
            // 点外卖 is one token here but the row says 外卖.
            let chars: Vec<char> = word.chars().collect();
            if chars.len() == 3 && chars.iter().all(|c| ('\u{3400}'..='\u{9fff}').contains(c)) {
                for part in [chars[..2].iter().collect::<String>(), chars[1..].iter().collect::<String>()] {
                    if !spellings.contains(&part) {
                        spellings.push(part);
                    }
                }
            }
            spellings.join(&ALTERNATIVE.to_string())
        })
        .collect();
    // A synonym the segmenter split apart must still count as something the user asked.
    for group in crate::query_rewrite::synonym_groups_in(text) {
        let covered = keywords.iter().any(|k| k.split(ALTERNATIVE).any(|s| group.contains(&s)));
        if !covered {
            keywords.push(group.join(&ALTERNATIVE.to_string()));
        }
    }
    keywords
}

/// Every acceptable spelling worth searching literally, at most `limit` of them.
pub fn keyword_spellings(keywords: &[String], limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    keywords
        .iter()
        .flat_map(|k| k.split(ALTERNATIVE))
        .filter(|s| s.chars().count() >= 2 && seen.insert(s.to_string()))
        .take(limit)
        .map(str::to_string)
        .collect()
}

/// One person is one keyword however they are called: asking about 爸爸 is answered by a
/// row that says 梁琰. Jieba may also split a name it does not know, so add it if missing.
pub fn merge_person_keyword(keywords: &mut Vec<String>, names: &[&str]) {
    let joined = names.join(&ALTERNATIVE.to_string());
    match keywords.iter_mut().find(|k| k.split(ALTERNATIVE).any(|s| names.contains(&s))) {
        Some(keyword) => {
            keyword.push(ALTERNATIVE);
            keyword.push_str(&joined);
        }
        None => keywords.push(joined),
    }
}

/// Share of content words present. A direct chunk match needs real coverage so
/// that an unrelated question cannot pull in a long excerpt and invite guessing.
pub fn keyword_coverage(keywords: &[String], content: &str) -> (usize, f64) {
    if keywords.is_empty() {
        return (0, 0.0);
    }
    let content = content.to_lowercase();
    let hits = keywords.iter().filter(|k| keyword_present(k, &content)).count();
    (hits, hits as f64 / keywords.len() as f64)
}

/// `content` must already be lowercase; 问 wifi 要能对上写着 WiFi 的记忆。
fn keyword_present(keyword: &str, content: &str) -> bool {
    keyword.split(ALTERNATIVE).any(|spelling| content.contains(&spelling.to_lowercase()))
}

/// How much each keyword says: a word on every row (the person's name) barely narrows
/// anything, while a rare one (来北京) is usually what the question is really about.
pub fn keyword_weights<'a>(keywords: &[String], bodies: impl Iterator<Item = &'a str>) -> Vec<f64> {
    let bodies: Vec<String> = bodies.map(str::to_lowercase).collect();
    let total = bodies.len().max(1) as f64;
    keywords
        .iter()
        .map(|keyword| {
            let found = bodies.iter().filter(|body| keyword_present(keyword, body)).count() as f64;
            (1.0 + total / (1.0 + found)).ln().max(0.1)
        })
        .collect()
}

pub fn weighted_coverage(keywords: &[String], weights: &[f64], content: &str) -> f64 {
    let total: f64 = weights.iter().sum();
    if keywords.is_empty() || total <= 0.0 {
        return 0.0;
    }
    let content = content.to_lowercase();
    let found: f64 = keywords
        .iter()
        .zip(weights)
        .filter(|(keyword, _)| keyword_present(keyword, &content))
        .map(|(_, weight)| *weight)
        .sum();
    found / total
}

/// Compatibility entry point for older hosts. Original excerpts are available
/// only through fact provenance (`chunks_for_memory` / `memory_trace`).
pub fn recall_chunks(
    _conn: &Connection,
    _jieba: &Jieba,
    _namespace: &str,
    _query: &ChunkRecallQuery,
) -> Result<Vec<RecalledChunk>> {
    Ok(Vec::new())
}

fn delete_chunk(conn: &Connection, chunk_id: &str) -> Result<()> {
    conn.execute("DELETE FROM memory_chunk_links WHERE chunk_id = ?1", params![chunk_id])?;
    conn.execute("DELETE FROM memory_chunks WHERE id = ?1", params![chunk_id])?;
    Ok(())
}

/// Explicit forgetting removes the excerpt too: sibling facts stay, but the
/// forgotten text must not keep coming back through the shared original.
pub fn forget_chunks_of_memory(conn: &Connection, memory_id: &str) -> Result<()> {
    let mut stmt = conn.prepare("SELECT chunk_id FROM memory_chunk_links WHERE memory_id = ?1")?;
    let ids: Vec<String> = stmt
        .query_map(params![memory_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    for id in ids {
        delete_chunk(conn, &id)?;
    }
    Ok(())
}

/// Decay removes only this fact's link; the excerpt lasts while any fact needs it.
pub fn unlink_memory(conn: &Connection, memory_id: &str) -> Result<()> {
    let mut stmt = conn.prepare("SELECT chunk_id FROM memory_chunk_links WHERE memory_id = ?1")?;
    let ids: Vec<String> = stmt
        .query_map(params![memory_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    conn.execute("DELETE FROM memory_chunk_links WHERE memory_id = ?1", params![memory_id])?;
    for id in ids {
        if linked_memory_ids(conn, &id)?.is_empty() {
            delete_chunk(conn, &id)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::migrate;

    fn memory(conn: &Connection, id: &Uuid, subject: &str, metadata: &str) {
        conn.execute(
            "INSERT INTO memories(id,kind,state,content,importance,confidence,salience,created_at_ms,updated_at_ms,namespace,subject_ids,metadata_json)
             VALUES(?1,'episodic','active','fact',0.5,0.8,0.4,1,1,'default',?2,?3)",
            params![id.to_string(), format!("[\"{subject}\"]"), metadata],
        )
        .unwrap();
    }

    fn write(content: &str, memory_ids: Vec<String>) -> ChunkWrite {
        ChunkWrite {
            namespace: "default".into(),
            content: content.into(),
            subject_ids: vec!["小航".into()],
            tags: vec!["医疗健康".into()],
            source_kind: "document".into(),
            source_name: "家庭.md".into(),
            source_path: Some("/data/家庭.md".into()),
            memory_ids,
            ..Default::default()
        }
    }

    fn query(text: &str) -> ChunkRecallQuery {
        ChunkRecallQuery { text: text.into(), top_k: 2, max_chars: 400, ..Default::default() }
    }

    #[test]
    fn identical_text_merges_links_people_and_tags() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let jieba = Jieba::new();
        let first = save_chunk(&conn, &jieba, &write("小航对花生过敏", vec!["a".into()])).unwrap();
        let mut second = write("小航对花生过敏", vec!["b".into()]);
        second.subject_ids = vec!["爸爸".into()];
        second.tags = vec!["饮食".into()];
        assert_eq!(first, save_chunk(&conn, &jieba, &second).unwrap());
        let chunk = get_chunk(&conn, &first).unwrap().unwrap();
        assert_eq!(chunk.subject_ids, vec!["小航", "爸爸"]);
        assert_eq!(chunk.tags, vec!["医疗健康", "饮食"]);
        assert_eq!(chunk.memory_ids.len(), 2);
    }

    #[test]
    fn excerpts_are_traceable_but_never_recalled() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let jieba = Jieba::new();
        let id = Uuid::now_v7();
        memory(&conn, &id, "小航", "{}");
        let content = "梁子航每周六有分数运算测试，主要巩固通分。";
        let chunk_id = save_chunk(&conn, &jieba, &write(content, vec![id.to_string()])).unwrap();
        let mut linked = query("分数运算测试");
        linked.hit_memory_ids = vec![id.to_string()];
        linked.hit_evidence = vec![(id.to_string(), "分数运算测试".into())];
        assert!(recall_chunks(&conn, &jieba, "default", &linked).unwrap().is_empty());
        assert!(recall_chunks(&conn, &jieba, "default", &query("分数运算测试")).unwrap().is_empty());
        let traced = chunks_for_memory(&conn, &id.to_string()).unwrap();
        assert_eq!(traced.len(), 1);
        assert_eq!(traced[0].id, chunk_id);
        assert_eq!(traced[0].content, content);
        assert_eq!(traced[0].source_path.as_deref(), Some("/data/家庭.md"));
    }

    #[test]
    fn v11_upgrade_removes_search_index_and_preserves_provenance() {
        let mut conn = Connection::open_in_memory().unwrap();
        for sql in &crate::store::MIGRATION_SQL[..11] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 11).unwrap();
        let jieba = Jieba::new();
        let id = Uuid::now_v7();
        memory(&conn, &id, "小航", "{}");
        let chunk = save_chunk(&conn, &jieba, &write("历史原文保留", vec![id.to_string()])).unwrap();
        conn.execute("INSERT INTO memory_chunk_fts(chunk_id,content) VALUES(?1,?2)", params![chunk, "历史 原文 保留"]).unwrap();
        migrate(&mut conn).unwrap();
        migrate(&mut conn).unwrap();
        let indexes: i64 = conn.query_row("SELECT count(*) FROM sqlite_master WHERE name = 'memory_chunk_fts' OR name GLOB 'memory_chunk_fts_*'", [], |r| r.get(0)).unwrap();
        assert_eq!(indexes, 0);
        assert_eq!(chunks_for_memory(&conn, &id.to_string()).unwrap()[0].content, "历史原文保留");
        assert_eq!(get_chunk(&conn, &chunk).unwrap().unwrap().memory_ids, vec![id.to_string()]);
        assert!(recall_chunks(&conn, &jieba, "default", &query("历史原文")).unwrap().is_empty());
        save_chunk(&conn, &jieba, &write("新增原文保留", vec![id.to_string()])).unwrap();
        assert_eq!(chunks_for_memory(&conn, &id.to_string()).unwrap().len(), 2);
    }

    #[test]
    fn forgetting_removes_excerpt_and_decay_only_when_last_link_goes() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let jieba = Jieba::new();
        let chunk = save_chunk(&conn, &jieba, &write("原文甲", vec!["a".into(), "b".into()])).unwrap();
        unlink_memory(&conn, "a").unwrap();
        assert!(get_chunk(&conn, &chunk).unwrap().is_some());
        unlink_memory(&conn, "b").unwrap();
        assert!(get_chunk(&conn, &chunk).unwrap().is_none());
        let chunk = save_chunk(&conn, &jieba, &write("原文乙", vec!["c".into(), "d".into()])).unwrap();
        forget_chunks_of_memory(&conn, "c").unwrap();
        assert!(get_chunk(&conn, &chunk).unwrap().is_none());
        let fts: i64 = conn.query_row("SELECT count(*) FROM memory_chunks", [], |r| r.get(0)).unwrap();
        assert_eq!(fts, 0);
    }

}
