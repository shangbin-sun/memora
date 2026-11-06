//! 查询改写与检索词扩展。
//!
//! 设计原则（何时改写 / 免改写）：
//! - **词面扩展**（[`expand_word`] / `RewritePlan::extra_words`）：对一切查询安全，
//!   几乎总是启用。只做同义词并集，不改变查询语义——BM25 与图谱词面打分从
//!   “妈妈”扩展到“母亲”，让书面表述（母亲/父亲）与口语查询（妈妈/爸爸）互通。
//! - **变体改写**（`RewritePlan::variants`）：仅当查询是**问句**时生成。长文本
//!   （用户贴原文）免改写——原文本身已是高质量检索输入，改写只会稀释信号。
//! - 变体只来自**同义词替换**（确定性、零成本、可测试），不引入 LLM 改写的
//!   延迟与不确定性。每个变体独立走语义召回，多路结果按最小距离合并。

/// 变体数量上限（含原句在内最多 1 + MAX_VARIANTS 路语义召回）。
pub const MAX_VARIANTS: usize = 3;

/// 超过该字符数的查询视为“贴原文”，免变体改写。
const LONG_QUERY_CHARS: usize = 48;

/// 同义词组：组内成员互为同义词（检索意图等价，不是严格定义）。
/// 保守收录称谓与高频生活词，避免引入跨域噪声。
const SYNONYM_GROUPS: &[&[&str]] = &[
    &["妈妈", "母亲", "老妈"],
    &["爸爸", "父亲", "老爸"],
    &["老婆", "妻子", "夫人", "爱人"],
    &["老公", "丈夫", "爱人"],
    &["爷爷", "祖父"],
    &["奶奶", "祖母"],
    &["外公", "外祖父"],
    &["外婆", "外祖母"],
    &["儿子", "孩子", "子女"],
    &["女儿", "孩子", "子女"],
    &["哥哥", "兄长"],
    &["生日", "出生日期", "出生"],
    &["学校", "就读学校", "小学", "中学", "上学", "就读"],
    // 家庭常问的说法和资料里的写法不一样；向量超时时全靠这里兜底。
    &["这周", "本周", "这星期", "一周"],
    &["花费", "消费", "支出", "花销", "开销", "花了"],
    &["付款", "支付", "付了", "付的钱"],
    &["职业", "工作", "工程师", "公务员"],
    &["诊断", "疾病", "什么病", "患有"],
    &["用药", "吃药", "什么药", "服用", "处方"],
    &["复诊", "复查"],
    &["错题", "做错", "错得", "错了", "错误"],
    &["数学", "分数", "通分", "运算", "方程", "几何", "应用题"],
    &["收入", "工资", "月薪", "挣多少", "挣钱"],
    &["持仓", "投资", "理财", "本金"],
    &["违章", "超速", "违法", "罚款"],
    &["保险", "保单", "续保"],
    &["保养", "维修", "换机油"],
    &["车位", "停车位"],
    &["课程", "上课", "上课时间", "什么时候上", "什么课"],
    &["学费", "课程费", "报班"],
    &["贷款", "房贷", "车贷", "还款", "月供"],
    &["过敏", "忌口"],
];

const QUESTION_WORDS: &[&str] = &[
    "谁", "什么", "哪些", "哪", "多少", "几", "怎么", "怎样", "如何", "吗", "呢",
    "是不是", "有没有", "为什么", "何时", "多久",
];

/// 问句判定：显式问号、疑问词。长文本（> [`LONG_QUERY_CHARS`]）恒为否——
/// 贴原文的场景不该改写。
pub fn looks_like_question(text: &str) -> bool {
    let trimmed = text.trim();
    let len = trimmed.chars().count();
    if len == 0 || len > LONG_QUERY_CHARS {
        return false;
    }
    if trimmed.ends_with(['？', '?']) {
        return true;
    }
    QUESTION_WORDS.iter().any(|w| trimmed.contains(w))
}

/// 返回包含 `word` 的同义词组（含自身）；未收录则空。
pub fn expand_word(word: &str) -> &'static [&'static str] {
    let w = word.trim();
    if w.is_empty() {
        return &[];
    }
    for group in SYNONYM_GROUPS {
        if group.iter().any(|g| *g == w) {
            return group;
        }
    }
    &[]
}

/// 句子里出现了哪些同义词组。按子串找，不依赖分词：jieba 可能把“这周”切成“这/周”。
pub fn synonym_groups_in(text: &str) -> Vec<&'static [&'static str]> {
    SYNONYM_GROUPS
        .iter()
        .copied()
        .filter(|group| group.iter().any(|word| text.contains(word)))
        .collect()
}

/// 多轮追问（“那语文呢”“他吃什么药”）单看本句查不到，要带上上一句。
/// 明确的新问题不带，否则上一轮的话题会污染这一轮的结果。
pub fn contextual_query(text: &str, previous: Option<&str>) -> String {
    let current = text.trim();
    let Some(previous) = previous.map(str::trim).filter(|p| !p.is_empty() && *p != current) else {
        return current.to_string();
    };
    let chars = current.chars().count();
    let starts_elliptical = ["那", "他", "她", "它", "这", "还有", "另外", "具体", "然后", "那么"]
        .iter()
        .any(|w| current.starts_with(w));
    let has_pronoun = ["他", "她", "它", "这个", "那个", "这家", "那家", "这门", "那门"]
        .iter()
        .any(|w| current.contains(w));
    if chars <= 6 || starts_elliptical || has_pronoun || current.ends_with('呢') {
        format!("{previous} {current}")
    } else {
        current.to_string()
    }
}

/// 改写计划。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RewritePlan {
    /// 语义召回变体（不含原句）。仅问句生成；每条是同义词替换后的完整查询。
    pub variants: Vec<String>,
    /// 词面扩展词（并入 FTS MATCH 的 OR 列表与图谱 search_words）。
    pub extra_words: Vec<String>,
}

impl RewritePlan {
    /// 语义召回的完整查询列表：原句 + 变体。
    pub fn query_texts<'a>(&'a self, original: &'a str) -> Vec<&'a str> {
        let mut out = vec![original];
        for v in &self.variants {
            if !out.contains(&v.as_str()) {
                out.push(v.as_str());
            }
        }
        out
    }
}

/// 生成改写计划。
///
/// - 非问句：只做词面扩展（`extra_words`），不改写变体。
/// - 问句：对文本中每个可替换词生成一个变体（整词最长匹配优先，
///   已被更长词覆盖的位置不重复匹配），变体数不超过 [`MAX_VARIANTS`]。
pub fn rewrite_plan(text: &str) -> RewritePlan {
    let trimmed = text.trim();
    let mut plan = RewritePlan::default();
    if trimmed.is_empty() {
        return plan;
    }

    // 词面扩展：同义词表中任何成员作为子串出现，就把整组并入扩展词。
    let mut extra = std::collections::BTreeSet::new();
    for group in SYNONYM_GROUPS {
        if group.iter().any(|w| trimmed.contains(w)) {
            for w in group.iter() {
                extra.insert((*w).to_string());
            }
        }
    }
    plan.extra_words = extra.into_iter().collect();

    // 变体改写：仅问句。
    if !looks_like_question(trimmed) {
        return plan;
    }

    // 收集可替换位置（最长匹配优先），同位置只认最长词。
    let chars: Vec<char> = trimmed.chars().collect();
    let mut words: Vec<&'static str> = SYNONYM_GROUPS
        .iter()
        .flat_map(|g| g.iter().copied())
        .collect();
    words.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    let mut matches: Vec<(usize, usize, &'static str)> = Vec::new(); // (start_char, len_chars, word)
    for word in words {
        let wlen = word.chars().count();
        // A question shorter than the synonym ("几点了" vs "出生日期") cannot contain it.
        if wlen > chars.len() {
            continue;
        }
        for start in 0..=(chars.len() - wlen) {
            let slice: String = chars[start..start + wlen].iter().collect();
            if slice != *word {
                continue;
            }
            let overlaps = matches
                .iter()
                .any(|(s, l, _)| start < s + l && *s < start + wlen);
            if !overlaps {
                matches.push((start, wlen, word));
            }
            break;
        }
    }
    matches.sort_by_key(|(s, _, _)| *s);

    for (start, len, word) in matches.iter().take(MAX_VARIANTS) {
        // 组内第一个不同于原词的成员；词组按顺序即“首选替身”。
        let Some(replacement) = expand_word(word).iter().find(|c| **c != *word).copied() else {
            continue;
        };
        let mut variant = String::new();
        variant.extend(chars[..*start].iter());
        variant.push_str(replacement);
        variant.extend(chars[*start + len..].iter());
        if variant != trimmed && !plan.variants.contains(&variant) {
            plan.variants.push(variant);
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_detection() {
        assert!(looks_like_question("航航的妈妈是谁"));
        assert!(looks_like_question("林子航的生日？"));
        assert!(looks_like_question("宋娜娜在哪工作"));
        assert!(!looks_like_question("航航的妈妈"));
        assert!(!looks_like_question("林子航是初中生"));
        // 长文本贴原文：即使含疑问词也不改写。
        let long: String = "今天聊了很多。".repeat(20) + "你们觉得吗";
        assert!(!looks_like_question(&long));
    }

    #[test]
    fn extra_words_cover_written_and_colloquial_forms() {
        let plan = rewrite_plan("航航的妈妈是谁");
        assert!(plan.extra_words.contains(&"母亲".to_string()));
        assert!(plan.extra_words.contains(&"妈妈".to_string()));

        let plan = rewrite_plan("林子航的生日");
        assert!(plan.extra_words.contains(&"出生".to_string()));

        let plan = rewrite_plan("林子航负责整理自己的书包与书桌");
        assert!(plan.extra_words.is_empty());
    }

    #[test]
    fn variants_replace_longest_match_first() {
        let plan = rewrite_plan("航航的妈妈是谁");
        // “妈妈”整体替换为“母亲”，而不是把“妈”单独替换。
        assert!(plan.variants.iter().any(|v| v == "航航的母亲是谁"));
        assert!(!plan
            .variants
            .iter()
            .any(|v| v.contains("妈母亲") || v.contains("母母亲")));
        assert!(plan.variants.len() <= MAX_VARIANTS);
    }

    #[test]
    fn long_pasted_text_gets_no_variants() {
        let text = "林子航是初中生，在大兴二中七年级3班，刚进入初中，成绩中等，妈妈很关心他的学习。";
        let plan = rewrite_plan(text);
        assert!(plan.variants.is_empty(), "长文本不做变体改写");
        assert!(!plan.extra_words.is_empty(), "但词面扩展仍然有效");
    }

    #[test]
    fn questions_shorter_than_any_synonym_do_not_panic() {
        for text in ["谁", "几点了", "子航呢", "多少钱"] {
            assert!(rewrite_plan(text).variants.is_empty());
        }
        assert_eq!(rewrite_plan("妈妈呢").variants, vec!["母亲呢".to_string()]);
    }

    #[test]
    fn follow_ups_carry_the_previous_question_and_new_questions_do_not() {
        let previous = Some("子航的数学课还剩几节");
        assert_eq!(contextual_query("那语文呢", previous), "子航的数学课还剩几节 那语文呢");
        assert_eq!(contextual_query("他吃什么药", Some("梁建国是什么病")), "梁建国是什么病 他吃什么药");
        assert_eq!(contextual_query("家里的车保险什么时候到期", previous), "家里的车保险什么时候到期");
        assert_eq!(contextual_query("那语文呢", None), "那语文呢");
    }

    #[test]
    fn query_texts_dedupes() {
        let plan = rewrite_plan("航航的妈妈是谁");
        let texts = plan.query_texts("航航的妈妈是谁");
        assert_eq!(texts.first().copied(), Some("航航的妈妈是谁"));
        let unique: std::collections::BTreeSet<_> = texts.iter().collect();
        assert_eq!(unique.len(), texts.len());
    }

    #[test]
    fn non_question_keeps_word_expansion_without_variants() {
        let plan = rewrite_plan("林子航的生日");
        assert!(plan.variants.is_empty());
        assert!(!plan.extra_words.is_empty());
    }
}
