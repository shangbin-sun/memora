//! A question looks something up; it does not tell us anything new.
//!
//! Asked through the write path, "爷爷什么时候复诊" came back as the memory "梁建国需要复诊，
//! 时间尚未确定", which then contradicts the real record. This module only catches the clear-cut
//! lookups, which also saves their extraction call. Anything mixed or unsure goes to the model:
//! "小航下个月6号过生日送什么好" still has to yield the birthday.

/// Words that turn a question-shaped sentence into an instruction about memory itself.
const MEMORY_INTENT: &[&str] = &["记住", "记一下", "记下", "帮我记", "别忘", "忘掉", "忘记", "删掉", "不要记", "别记"];

/// "我女儿考上清华了你知道吗": the tail asks nothing about the fact in front of it.
const TAG_TAILS: &[&str] = &["你知道吗", "知道吗", "好吗", "对吗", "行吗", "是吗", "对吧", "是吧", "好吧", "行吧"];

const FILLERS: &[&str] = &["好的", "好", "嗯", "哦", "那", "对了", "请问", "你好", "行", "知道了", "另外", "还有", "然后", "麻烦", "麻烦你"];

/// Question words used as ordinary words. Masked before looking for a real question word.
const NOT_ASKING: &[&str] = &[
    "什么都", "什么也", "没什么", "啥都", "啥也", "没啥", "哪都", "哪儿都", "哪也", "哪怕", "哪吒", "谁都", "谁也",
    "不怎么", "没怎么", "怎么都", "怎么也", "多少有点", "多少还", "没多少", "好多少", "几天后", "几天前", "过几天", "这几", "那几",
    "前几", "好几", "没几", "十几", "几十", "几百", "几千", "几万", "几年前", "几年后", "几个月前", "几个月后", "我几个", "我们几个",
    "他们几个", "她们几个",
];

const QUESTION_WORDS: &[&str] = &[
    "多少", "几点", "几号", "几节", "几岁", "几次", "几个", "几天", "几年", "几月", "几口", "几门", "几家", "几张", "几笔",
    "什么", "啥", "哪", "谁", "怎么", "怎样", "如何", "为什么", "为啥", "是否", "有没有", "是不是", "能不能", "要不要",
    "会不会", "可不可以", "该不该", "算不算", "多久", "多大", "多高", "多远", "多贵", "多长时间",
];

/// Asking for advice often carries the news in the same breath ("明天去三亚带什么衣服好").
const ADVICE: &[&str] = &["好呢", "好吗", "怎么办", "推荐", "建议", "合适", "应该", "该"];
const SELF_NEWS: &[&str] = &[
    "明天", "后天", "今天", "昨天", "前天", "下周", "上周", "这周", "下个月", "上个月", "下月", "上月", "明年", "去年", "刚", "已经", "要去", "打算", "准备",
];

fn split_clauses(text: &str) -> Vec<(String, bool)> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if matches!(ch, '，' | ',' | '。' | '.' | '！' | '!' | '？' | '?' | '；' | ';' | '\n') {
            let clause = current.trim().to_string();
            if !clause.is_empty() {
                clauses.push((clause, matches!(ch, '？' | '?')));
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }
    let clause = current.trim().to_string();
    if !clause.is_empty() {
        clauses.push((clause, false));
    }
    clauses
}

fn asks_something(clause: &str, marked: bool) -> bool {
    let mut body = clause.to_string();
    let mut tagged = false;
    for tail in TAG_TAILS {
        if let Some(stripped) = body.strip_suffix(tail) {
            body = stripped.to_string();
            tagged = true;
            break;
        }
    }
    if tagged && !marked && body.trim().is_empty() {
        return false;
    }
    let mut masked = body.clone();
    for phrase in NOT_ASKING {
        masked = masked.replace(phrase, "");
    }
    if masked.contains("不管") || masked.contains("无论") {
        return false;
    }
    let has_word = QUESTION_WORDS.iter().any(|word| masked.contains(word));
    let particle = !tagged && masked.ends_with('吗')
        || !tagged && masked.ends_with('嘛')
        || masked.ends_with('呢')
            && (has_word || masked.starts_with('那') || masked.starts_with("还有") || masked.chars().count() <= 4);
    has_word || particle || (marked && !tagged)
}

fn carries_news(clause: &str) -> bool {
    if ADVICE.iter().any(|word| clause.contains(word)) || clause.ends_with('好') {
        return true;
    }
    clause.contains('我') && SELF_NEWS.iter().any(|word| clause.contains(word))
}

/// True only for a turn that is nothing but lookups.
pub fn is_pure_question(text: &str) -> bool {
    if MEMORY_INTENT.iter().any(|word| text.contains(word)) {
        return false;
    }
    let clauses = split_clauses(text);
    let mut asked = false;
    for (clause, marked) in &clauses {
        if FILLERS.contains(&clause.as_str()) {
            continue;
        }
        if !asks_something(clause, *marked) || carries_news(clause) {
            return false;
        }
        asked = true;
    }
    asked
}

/// The clause this evidence sits in is a question, so it told us nothing.
/// "我周三体检，爷爷什么时候复诊" keeps the first clause and drops the second.
pub fn evidence_in_question(text: &str, evidence: &str) -> bool {
    let evidence = evidence.trim();
    if evidence.is_empty() {
        return false;
    }
    let mut overlapping = split_clauses(text)
        .into_iter()
        .filter(|(clause, _)| clause.contains(evidence) || evidence.contains(clause.as_str()))
        .peekable();
    if overlapping.peek().is_none() {
        return false;
    }
    overlapping.all(|(clause, marked)| asks_something(&clause, marked) && !carries_news(&clause))
}

/// The model describing the conversation instead of the world.
pub fn describes_the_asking(content: &str) -> bool {
    ["用户在询问", "用户询问", "用户想知道", "用户在问", "用户问"].iter().any(|word| content.contains(word))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_are_pure_questions() {
        for text in [
            "房贷每月几号扣，扣多少",
            "给子航的200块零花钱算这周的消费吗",
            "子航最近哪块数学错得多",
            "爷爷什么时候复诊，去哪家医院",
            "范江楠最喜欢吃什么菜",
            "子航的数学课还剩几节",
            "那语文呢",
            "澄阅呢",
            "他吃什么药",
            "梁琰的血型是什么",
            "好的，那语文呢",
            "请问，这周一共花了多少钱？",
            "发票总额是全部消费吗",
        ] {
            assert!(is_pure_question(text), "{text}");
        }
    }

    #[test]
    fn statements_and_mixed_turns_go_to_the_model() {
        for text in [
            "我周末喜欢打羽毛球，也爱看《繁花》这种年代剧",
            "我不喝咖啡，喝了睡不着",
            "小航下个月6号就要过生日了，请问送什么礼物好呢",
            "小航下个月6号过生日送什么礼物好",
            "我明天去三亚带什么衣服",
            "我女儿考上清华了你知道吗",
            "他今年才五岁呢",
            "我不怎么吃辣",
            "我什么都吃",
            "过几天我要出差",
            "帮我记住房贷每月5号扣",
            "忘掉我喜欢什么颜色",
            "我们几个周末去爬山",
            "这几天我血压有点高",
            "不管什么时候我都不喝酒",
            "我最喜欢哪吒",
        ] {
            assert!(!is_pure_question(text), "{text}");
        }
    }

    #[test]
    fn only_the_question_clause_loses_its_evidence() {
        let text = "我周三去体检，爷爷什么时候复诊";
        assert!(!evidence_in_question(text, "我周三去体检"));
        assert!(evidence_in_question(text, "爷爷什么时候复诊"));
        assert!(evidence_in_question("给子航的200块零花钱算这周的消费吗", "给子航的200块零花钱"));
        assert!(!evidence_in_question("小航下个月6号就要过生日了，请问送什么礼物好呢", "小航下个月6号就要过生日了"));
    }

    #[test]
    fn meta_descriptions_are_recognised() {
        assert!(describes_the_asking("范江楠最喜欢吃什么菜尚不明确（用户在询问）"));
        assert!(!describes_the_asking("范江楠最喜欢吃川菜"));
    }
}
