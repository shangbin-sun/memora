use jieba_rs::Jieba;

pub fn tokenize_for_fts(jieba: &Jieba, text: &str) -> String {
    jieba
        .cut(text, true)
        .into_iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn fts_match_query(jieba: &Jieba, text: &str) -> String {
    fts_match_query_with(jieba, text, &[])
}

/// 单字虚词/代词黑名单。FTS5 unicode61 把中文按单字索引，这些字几乎
/// 出现在每条记忆里（如“是/的”），进入 MATCH 后 BM25 全库命中且被
/// 短文档长度归一化主导，彻底淹没实词信号。
const FTS_CHAR_STOPWORDS: &[&str] = &[
    "的", "是", "了", "在", "有", "和", "与", "或", "就", "都", "也", "还",
    "被", "把", "让", "对", "会", "能", "要", "去", "来", "说", "问", "个",
    "们", "很", "更", "最", "不", "没", "谁", "吗", "呢", "吧", "啊", "我",
    "你", "他", "她", "它", "这", "那", "等", "及", "以及", "之",
];

pub fn fts_match_query_with(jieba: &Jieba, text: &str, extra_words: &[String]) -> String {
    let mut parts: Vec<String> = jieba
        .cut(text, true)
        .into_iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|w| {
            let escaped = w.replace(['"', '\'', '*', '(', ')', ':'], " ");
            format!("\"{}\"", escaped.trim())
        })
        .filter(|w| {
            // 按字符数过滤：丢弃单字虚词，保留其余单字实词（如“药”“账”）。
            let bare = w.trim_matches('"');
            if bare.chars().count() == 1 {
                return !FTS_CHAR_STOPWORDS.contains(&bare);
            }
            !bare.is_empty()
        })
        .collect();
    // 语文课 is one query token but the index holds 语文 + 班: also search its parts.
    for token in jieba.cut(text, true) {
        if token.chars().count() >= 3 && token.chars().all(|c| ('\u{3400}'..='\u{9fff}').contains(&c)) {
            for part in jieba.cut_for_search(token, true) {
                let quoted = format!("\"{part}\"");
                if part.chars().count() >= 2 && !parts.contains(&quoted) {
                    parts.push(quoted);
                }
            }
        }
    }
    for w in extra_words {
        let w = w.trim();
        if w.is_empty() {
            continue;
        }
        let quoted = format!("\"{}\"", w.replace(['"', '\'', '*', '(', ')', ':'], " "));
        if !parts.contains(&quoted) {
            parts.push(quoted);
        }
    }
    parts.join(" OR ")
}

pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '\n') {
            let s = buf.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            buf.clear();
        }
    }
    let tail = buf.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jieba_keeps_chinese_tokens() {
        let jieba = Jieba::new();
        let tokens = tokenize_for_fts(&jieba, "用户偏好 SQLite 作为本地存储");
        assert!(tokens.contains("用户") || tokens.contains("SQLite") || tokens.contains("存储"));
    }

    #[test]
    fn split_chinese_sentences() {
        let s = split_sentences("你好。世界！再来一句");
        assert!(s.len() >= 2);
        assert!(s[0].contains("你好"));
    }

    #[test]
    fn fts_match_query_drops_single_char_stopwords() {
        let jieba = Jieba::new();
        let q = fts_match_query(&jieba, "航航的妈妈是谁");
        // “的/是/谁”不再进入 MATCH；实词保留。
        assert!(!q.contains("\"的\""), "q={q}");
        assert!(!q.contains("\"是\""), "q={q}");
        assert!(!q.contains("\"谁\""), "q={q}");
        assert!(q.contains("妈妈") || q.contains("妈"), "q={q}");
    }

    #[test]
    fn fts_match_query_merges_extra_words() {
        let jieba = Jieba::new();
        let extra = vec!["母亲".to_string(), "妈妈".to_string()];
        let q = fts_match_query_with(&jieba, "航航的妈妈是谁", &extra);
        assert!(q.contains("\"母亲\""), "q={q}");
        // 去重：妈妈已在原词列表，不重复。
        assert_eq!(q.matches("\"妈妈\"").count(), 1, "q={q}");
    }
}
