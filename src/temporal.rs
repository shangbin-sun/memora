//! Relative → absolute date normalization for memory facts (Asia/Shanghai).
//!
//! Ported from Android `MemoraTemporal`. Event dates may be stamped as a
//! trailing `⌈event:YYYY-MM-DD⌋` marker for recall-time decay.

use chrono::{Datelike, Duration, FixedOffset, NaiveDate, TimeZone, Weekday};
use fancy_regex::Regex;
use std::sync::OnceLock;

const ZONE_SECS: i32 = 8 * 3600;

fn shanghai() -> FixedOffset {
    FixedOffset::east_opt(ZONE_SECS).expect("UTC+8")
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedFact {
    pub content: String,
    pub event_at_ms: Option<i64>,
    pub importance: Option<f64>,
}

fn event_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s*⌈event:(\d{4}-\d{2}-\d{2})⌋\s*$").unwrap())
}
fn iso_in_text() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?<!\d)(20\d{2})-(\d{1,2})-(\d{1,2})(?!\d)").unwrap())
}
fn cn_in_text() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日").unwrap())
}
fn non_schedule() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"出生|生日|纪念日|入职日期|毕业日期|每(?:周|星期|礼拜|天|日|月|年|逢)|每个(?:周|星期|礼拜)|通常|习惯|定期")
            .unwrap()
    })
}
fn historical() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"昨天|前天|上周|上个星期|去年|曾经|已经|去了|做过|发生于").unwrap()
    })
}
fn schedule() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Broad outing / dining / escort patterns — LLM event_at still wins first.
        Regex::new(
            r"将|计划|安排|预约|准备|要去|带.{0,12}去|去(?:医院|体检|看|游泳|上课|约会|吃|喝|玩|逛)|参加|出发|开会|体检|看病|约会|吃饭|用餐|聚餐",
        )
        .unwrap()
    })
}

/// Relative day cues: when present, a rewritten absolute date is treated as a
/// one-off schedule even if [schedule] verbs (去医院/…) did not match.
fn relative_schedule_cue() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"明天|后天|大后天|今晚|明早|今天|今日|下周|下星期|下个星期|本周|这周").unwrap()
    })
}

fn looks_like_schedule(text: &str) -> bool {
    schedule().is_match(text).unwrap_or(false)
        || relative_schedule_cue().is_match(text).unwrap_or(false)
}
fn absolute_date() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?<!\d)(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}-\d{1,2}-\d{1,2})(?:[（(](?:星期|周|礼拜)[一二三四五六日天][）)])?",
        )
        .unwrap()
    })
}
fn weekday_token() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(下下|下|本|这|上上|上)?个?(?:周|星期|礼拜)([一二三四五六日天末])$").unwrap()
    })
}
fn date_token() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"{}|(?<!每)(?:(?:下下|下|本|这|上上|上)个?)?(?:周|星期|礼拜)[一二三四五六日天末]|大后天|后天|明天|今天|今日|今晚|明早|昨天|前天|(?<![\d年])\d{{1,2}}\s*月\s*\d{{1,2}}\s*日|(?<![\d月年])\d{{1,2}}\s*号",
            absolute_date().as_str()
        ))
        .unwrap()
    })
}
fn digits() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d+").unwrap())
}

pub fn today(now_ms: i64) -> NaiveDate {
    shanghai()
        .timestamp_millis_opt(now_ms)
        .single()
        .map(|dt| dt.date_naive())
        .unwrap_or_else(|| chrono::Utc::now().with_timezone(&shanghai()).date_naive())
}

pub fn strip_event_marker(content: &str) -> String {
    event_marker_re()
        .replace_all(content.trim(), "")
        .trim()
        .to_string()
}

pub fn display_content(content: &str) -> String {
    strip_event_marker(content)
}

fn parse_iso_day(raw: Option<&str>) -> Option<NaiveDate> {
    let t = raw?.trim();
    if t.is_empty() {
        return None;
    }
    let head = if t.len() >= 10 { &t[..10] } else { t };
    NaiveDate::parse_from_str(head, "%Y-%m-%d").ok()
}

fn format_cn(date: NaiveDate) -> String {
    let week = match date.weekday() {
        Weekday::Mon => "星期一",
        Weekday::Tue => "星期二",
        Weekday::Wed => "星期三",
        Weekday::Thu => "星期四",
        Weekday::Fri => "星期五",
        Weekday::Sat => "星期六",
        Weekday::Sun => "星期日",
    };
    format!(
        "{}年{}月{}日（{}）",
        date.year(),
        date.month(),
        date.day(),
        week
    )
}

fn stamp_event_marker(content: &str, date: NaiveDate) -> String {
    format!(
        "{} ⌈event:{}⌋",
        strip_event_marker(content),
        date.format("%Y-%m-%d")
    )
}

fn weekday_from_token(token: &str) -> Option<Weekday> {
    let t = token.trim();
    if t.contains('一') {
        Some(Weekday::Mon)
    } else if t.contains('二') {
        Some(Weekday::Tue)
    } else if t.contains('三') {
        Some(Weekday::Wed)
    } else if t.contains('四') {
        Some(Weekday::Thu)
    } else if t.contains('五') {
        Some(Weekday::Fri)
    } else if t.contains('六') || t.contains('末') {
        Some(Weekday::Sat)
    } else if t.contains('日') || t.contains('天') {
        Some(Weekday::Sun)
    } else {
        None
    }
}

fn weekday_in_week(anchor: NaiveDate, dow: Weekday, week_offset: i64) -> NaiveDate {
    let monday = anchor - Duration::days(anchor.weekday().num_days_from_monday() as i64);
    let base = monday + Duration::weeks(week_offset);
    base + Duration::days(dow.num_days_from_monday() as i64)
}

fn next_or_same(anchor: NaiveDate, dow: Weekday) -> NaiveDate {
    let mut d = anchor;
    while d.weekday() != dow {
        d += Duration::days(1);
    }
    d
}

fn resolve_month_day(anchor: NaiveDate, month: u32, day: u32) -> Option<NaiveDate> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let this_year = NaiveDate::from_ymd_opt(anchor.year(), month, day);
    if let Some(d) = this_year {
        if d >= anchor {
            return Some(d);
        }
    }
    NaiveDate::from_ymd_opt(anchor.year() + 1, month, day).or(this_year)
}

fn contains_absolute_date(text: &str) -> bool {
    iso_in_text().is_match(text).unwrap_or(false) || cn_in_text().is_match(text).unwrap_or(false)
}

fn extract_event_date(text: &str) -> Option<NaiveDate> {
    if let Ok(Some(c)) = event_marker_re().captures(text) {
        if let Some(d) = parse_iso_day(c.get(1).map(|m| m.as_str())) {
            return Some(d);
        }
    }
    if let Ok(Some(c)) = cn_in_text().captures(text) {
        let y: i32 = c[1].parse().ok()?;
        let mo: u32 = c[2].parse().ok()?;
        let d: u32 = c[3].parse().ok()?;
        return NaiveDate::from_ymd_opt(y, mo, d);
    }
    if let Ok(Some(c)) = iso_in_text().captures(text) {
        let y: i32 = c[1].parse().ok()?;
        let mo: u32 = c[2].parse().ok()?;
        let d: u32 = c[3].parse().ok()?;
        return NaiveDate::from_ymd_opt(y, mo, d);
    }
    None
}

fn rewrite_relative_dates(text: &str, anchor: NaiveDate) -> String {
    // fancy-regex has no replace_all with Fn; walk matches left-to-right.
    let mut out = String::new();
    let mut last = 0usize;
    let re = date_token();
    let mut caps_iter = re.captures_iter(text);
    while let Some(Ok(caps)) = caps_iter.next() {
        let m = caps.get(0).expect("match");
        out.push_str(&text[last..m.start()]);
        let token = m.as_str();
        let abs = absolute_date().is_match(token).unwrap_or(false);
        let replacement = if abs {
            token.to_string()
        } else {
            let target = if let Ok(Some(week)) = weekday_token().captures(token) {
                weekday_from_token(&week[2]).map(|dow| {
                    let prefix = week.get(1).map(|m| m.as_str()).unwrap_or("");
                    if prefix.is_empty() {
                        next_or_same(anchor, dow)
                    } else {
                        let offset = match prefix {
                            "下下" => 2,
                            "下" => 1,
                            "上上" => -2,
                            "上" => -1,
                            _ => 0,
                        };
                        weekday_in_week(anchor, dow, offset)
                    }
                })
            } else {
                match token {
                    "大后天" => Some(anchor + Duration::days(3)),
                    "后天" => Some(anchor + Duration::days(2)),
                    "明天" | "明早" => Some(anchor + Duration::days(1)),
                    "昨天" => Some(anchor - Duration::days(1)),
                    "前天" => Some(anchor - Duration::days(2)),
                    "今天" | "今日" | "今晚" => Some(anchor),
                    t if t.contains('月') => {
                        let nums: Vec<u32> = digits()
                            .find_iter(t)
                            .filter_map(|m| m.ok().and_then(|mm| mm.as_str().parse().ok()))
                            .collect();
                        if nums.len() >= 2 {
                            if historical().is_match(text).unwrap_or(false) {
                                NaiveDate::from_ymd_opt(anchor.year(), nums[0], nums[1])
                            } else {
                                resolve_month_day(anchor, nums[0], nums[1])
                            }
                        } else {
                            None
                        }
                    }
                    t if t.ends_with('号') => {
                        let day = digits()
                            .find(t)
                            .ok()
                            .flatten()
                            .and_then(|m| m.as_str().parse::<u32>().ok());
                        day.and_then(|d| {
                            let current = anchor.with_day(d);
                            if let Some(c) = current {
                                if c < anchor && !historical().is_match(text).unwrap_or(false) {
                                    let next = anchor + Duration::days(32);
                                    let first =
                                        NaiveDate::from_ymd_opt(next.year(), next.month(), 1)?;
                                    first.with_day(d)
                                } else {
                                    Some(c)
                                }
                            } else {
                                None
                            }
                        })
                    }
                    _ => None,
                }
            };
            match target {
                Some(date) => {
                    let mut s = format_cn(date);
                    if token == "今晚" {
                        s.push_str("晚上");
                    } else if token == "明早" {
                        s.push_str("早上");
                    }
                    s
                }
                None => token.to_string(),
            }
        };
        out.push_str(&replacement);
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

pub fn normalize_fact(
    raw_content: &str,
    llm_event_at: Option<&str>,
    now_ms: i64,
) -> NormalizedFact {
    let base = strip_event_marker(raw_content);
    if base.is_empty() {
        return NormalizedFact {
            content: base,
            event_at_ms: None,
            importance: None,
        };
    }
    if non_schedule().is_match(&base).unwrap_or(false) {
        return NormalizedFact {
            content: base,
            event_at_ms: None,
            importance: None,
        };
    }
    let anchor = today(now_ms);
    let rewritten = rewrite_relative_dates(&base, anchor);
    let marked_date = event_marker_re()
        .captures(raw_content)
        .ok()
        .flatten()
        .and_then(|c| parse_iso_day(c.get(1).map(|m| m.as_str())));
    let event_date = if historical().is_match(&base).unwrap_or(false) {
        None
    } else {
        // Prefer LLM event_at; heuristic date only when schedule/relative cues hit.
        // Rule miss must not invent dates — callers keep LLM event_at instead of dropping.
        parse_iso_day(llm_event_at).or(marked_date).or_else(|| {
            extract_event_date(&rewritten).filter(|d| *d >= anchor && looks_like_schedule(&base))
        })
    };

    let content_with_abs = if let Some(event_date) = event_date {
        if !contains_absolute_date(&rewritten) {
            format!("{rewritten}（{}）", format_cn(event_date))
        } else {
            rewritten
        }
    } else {
        rewritten
    };

    let stamped = if let Some(event_date) = event_date {
        stamp_event_marker(&content_with_abs, event_date)
    } else {
        content_with_abs
    };

    let event_ms = event_date.map(|d| {
        shanghai()
            .from_local_datetime(&d.and_hms_opt(0, 0, 0).unwrap())
            .single()
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(0)
    });
    let importance = event_ms.map(|ms| importance_for_event(ms, now_ms));
    NormalizedFact {
        content: stamped,
        event_at_ms: event_ms,
        importance,
    }
}

pub fn extract_event_at_ms(content: &str) -> Option<i64> {
    let base = strip_event_marker(content);
    if non_schedule().is_match(&base).unwrap_or(false)
        || historical().is_match(&base).unwrap_or(false)
    {
        return None;
    }
    let marked = event_marker_re()
        .captures(content)
        .ok()
        .flatten()
        .and_then(|c| parse_iso_day(c.get(1).map(|m| m.as_str())))?;
    Some(
        shanghai()
            .from_local_datetime(&marked.and_hms_opt(0, 0, 0).unwrap())
            .single()?
            .timestamp_millis(),
    )
}

pub fn temporal_weight(event_at_ms: Option<i64>, now_ms: i64) -> f64 {
    let Some(event_at_ms) = event_at_ms else {
        return 1.0;
    };
    let event_day = today(event_at_ms);
    let today = today(now_ms);
    let days = (event_day - today).num_days();
    match days {
        d if d >= 0 => 1.0,
        d if d >= -1 => 0.55,
        d if d >= -3 => 0.35,
        d if d >= -7 => 0.18,
        d if d >= -30 => 0.08,
        _ => 0.03,
    }
}

pub fn importance_for_event(event_at_ms: i64, now_ms: i64) -> f64 {
    let event_day = today(event_at_ms);
    let today = today(now_ms);
    let days = (event_day - today).num_days();
    let v: f64 = match days {
        0..=2 => 0.9,
        3..=7 => 0.8,
        8..=30 => 0.65,
        d if d > 30 => 0.55,
        d if d >= -1 => 0.4,
        _ => 0.2,
    };
    v.clamp(0.05, 1.0)
}

pub fn adjusted_score(base_score: f64, event_at_ms: Option<i64>, now_ms: i64) -> f64 {
    let w = temporal_weight(event_at_ms, now_ms);
    let days_past = match event_at_ms {
        None => 0.0,
        Some(ms) => {
            let event_day = today(ms);
            let t = today(now_ms);
            (t - event_day).num_days().max(0) as f64
        }
    };
    let soft = if days_past <= 0.0 {
        1.0
    } else {
        (-days_past / 14.0).exp()
    };
    base_score * w * (0.65 + 0.35 * soft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> i64 {
        // 2026-10-21 12:00 Asia/Shanghai == 2026-10-21T04:00:00Z
        shanghai()
            .with_ymd_and_hms(2026, 10, 21, 12, 0, 0)
            .single()
            .unwrap()
            .timestamp_millis()
    }

    fn norm(text: &str, date: Option<&str>) -> NormalizedFact {
        normalize_fact(text, date, now())
    }

    #[test]
    fn week_phrases() {
        for text in ["下周二", "下个星期二", "下礼拜二", "星期二", "周二"] {
            let r = norm(&format!("张三{text}去医院"), None);
            assert!(r.content.contains("2026年10月27日"), "{}", r.content);
            assert!(!r.content.contains("星2026"), "{}", r.content);
            assert!(r.event_at_ms.is_some());
        }
        assert!(norm("张三下下周二去医院", None)
            .content
            .contains("2026年11月3日"));
        assert!(norm("张三本周二去医院", None)
            .content
            .contains("2026年10月20日"));
    }

    #[test]
    fn birthday_no_event() {
        for s in [
            "小航出生于2020年5月1日",
            "小航生日是5月1日",
            "张三每周二去游泳",
        ] {
            let r = norm(s, Some("2020-05-01"));
            assert_eq!(r.content, s);
            assert!(r.event_at_ms.is_none());
        }
    }

    #[test]
    fn tomorrow_schedules() {
        let r = norm("张三明天去医院", None);
        assert!(r.content.contains("2026年10月22日"), "{}", r.content);
        assert!(r.event_at_ms.is_some());
        assert!((r.importance.unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn expired_weight() {
        let r = norm("去医院", Some("2026-10-01"));
        assert!(temporal_weight(r.event_at_ms, now()) < 0.1);
        assert!((temporal_weight(None, now()) - 1.0).abs() < 1e-9);
    }
}
