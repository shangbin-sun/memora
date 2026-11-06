//! Extraction wire-protocol validation. This module checks shape and exact
//! provenance only; roster membership and interpretation belong to the caller.

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::Deserializer;
use serde_json::{json, Map, Value};
use std::fmt;

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ParsedExtraction {
    pub memories: Vec<Value>,
    pub registers: Vec<Value>,
    pub retractions: Vec<Value>,
    pub invalid_records: usize,
    pub complete: bool,
    /// The reply ended with its single completion marker (bad lines may remain).
    #[serde(default)]
    pub marker_ok: bool,
}

/// Parse compact JSONL, or a complete legacy extraction object.
///
/// A compact completion marker must be its own final nonempty physical line.
/// Invalid records never discard valid memory siblings. Controls are accepted
/// only when the entire source completed, including the transport stop state.
pub(crate) fn parse_extraction(
    text: &str,
    source_id: &str,
    source_text: &str,
    truncated: bool,
) -> ParsedExtraction {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(root)) if is_legacy_root(&root) => {
            return parse_legacy(&root, source_id, source_text, truncated);
        }
        Err(_) if first_root_is_legacy(text) => {
            // Never dig records out of a partial legacy object. In particular,
            // a transport-truncated legacy reply cannot be salvaged as JSONL.
            return ParsedExtraction {
                invalid_records: 1,
                ..ParsedExtraction::default()
            };
        }
        _ => {}
    }

    let mut parsed = ParsedExtraction::default();
    let mut marker_count = 0;
    let mut last_was_marker = false;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        last_was_marker = false;
        let Ok(Value::Object(record)) = serde_json::from_str::<Value>(line) else {
            parsed.invalid_records += 1;
            continue;
        };
        if record.len() == 1 && record.get("done") == Some(&Value::Bool(true)) {
            marker_count += 1;
            last_was_marker = true;
            continue;
        }
        if record.contains_key("done") {
            parsed.invalid_records += 1;
            continue;
        }
        let Some((kind, item)) = normalize_compact(&record, source_id, source_text) else {
            parsed.invalid_records += 1;
            continue;
        };
        push_item(&mut parsed, kind, item);
    }
    parsed.marker_ok = marker_count == 1 && last_was_marker;
    parsed.complete = !truncated && parsed.invalid_records == 0 && parsed.marker_ok;
    suppress_partial_controls(&mut parsed);
    parsed
}

#[derive(Clone, Copy)]
enum Kind {
    Memory,
    Register,
    Retraction,
}

fn push_item(parsed: &mut ParsedExtraction, kind: Kind, item: Value) {
    match kind {
        Kind::Memory => parsed.memories.push(item),
        Kind::Register => parsed.registers.push(item),
        Kind::Retraction => parsed.retractions.push(item),
    }
}

fn suppress_partial_controls(parsed: &mut ParsedExtraction) {
    if !parsed.complete {
        parsed.registers.clear();
        parsed.retractions.clear();
    }
}

fn required_string<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    object
        .get(key)?
        .as_str()
        .filter(|value| !value.trim().is_empty())
}

fn valid_role(role: &str) -> bool {
    matches!(role, "self" | "spouse" | "child" | "parent" | "other")
}

fn valid_aliases(aliases: &Value) -> bool {
    aliases.as_array().is_some_and(|aliases| {
        aliases
            .iter()
            .all(|alias| alias.as_str().is_some_and(|alias| !alias.trim().is_empty()))
    })
}

fn valid_date(value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    let Some(raw) = value.as_str() else {
        return false;
    };
    // chrono accepts some non-padded dates; require the exact wire syntax too.
    if raw.len() != 10
        || !raw.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 4 | 7) {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        })
    {
        return false;
    }
    chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .ok()
        .is_some_and(|date| date.format("%Y-%m-%d").to_string() == raw)
}

fn exact_evidence(evidence: &str, source_id: &str, source_text: &str) -> bool {
    !source_id.trim().is_empty() && source_text.contains(evidence)
}

fn folded(ch: char) -> Option<char> {
    match ch {
        c if c.is_whitespace() => None,
        '|' | '*' | '`' | ',' | '，' | '、' => None,
        '：' => Some(':'),
        '；' => Some(';'),
        '（' => Some('('),
        '）' => Some(')'),
        '。' => Some('.'),
        '－' | '−' | '–' | '—' => Some('-'),
        '×' => Some('x'),
        c => Some(c),
    }
}

/// Fold formatting that models normally smooth over, keeping the byte range of each
/// surviving character so a match can be mapped back onto the original text.
fn folded_with_offsets(text: &str) -> (Vec<char>, Vec<(usize, usize)>) {
    let cleaned = text;
    let mut chars = Vec::new();
    let mut spans = Vec::new();
    let mut iter = cleaned.char_indices().peekable();
    while let Some((start, ch)) = iter.next() {
        if cleaned[start..].starts_with("<br>") {
            for _ in 0..3 {
                iter.next();
            }
            continue;
        }
        if let Some(folded) = folded(ch) {
            chars.push(folded);
            spans.push((start, start + ch.len_utf8()));
        }
    }
    (chars, spans)
}

/// Models quote across table cells, or write a number or dash differently. That is
/// still the same original passage, so the evidence is restored to the exact slice.
/// A real paraphrase still fails, which is what the check is there to catch.
pub(crate) fn resolve_evidence(evidence: &str, source_text: &str) -> Option<String> {
    if source_text.contains(evidence) {
        return Some(evidence.to_string());
    }
    let (needle, _) = folded_with_offsets(evidence);
    if needle.len() < 2 {
        return None;
    }
    let (haystack, spans) = folded_with_offsets(source_text);
    let index = haystack.windows(needle.len()).position(|window| window == needle.as_slice())?;
    Some(source_text[spans[index].0..spans[index + needle.len() - 1].1].to_string())
}

fn forbidden_compact_field(key: &str) -> bool {
    key.starts_with("_host")
        || matches!(
            key,
            "source_id"
                | "subjects"
                | "owner"
                | "content"
                | "tag"
                | "event_at"
                | "evidence"
                | "display_name"
                | "role"
                | "aliases"
        )
}

fn normalize_compact(
    object: &Map<String, Value>,
    source_id: &str,
    source_text: &str,
) -> Option<(Kind, Value)> {
    if object.keys().any(|key| forbidden_compact_field(key)) {
        return None;
    }
    let kind = match object.get("k") {
        None => Kind::Memory,
        Some(Value::String(kind)) if kind == "m" => Kind::Memory,
        Some(Value::String(kind)) if kind == "r" => Kind::Register,
        Some(Value::String(kind)) if kind == "f" => Kind::Retraction,
        _ => return None,
    };
    let evidence = resolve_evidence(required_string(object, "e")?, source_text)?;
    let evidence = evidence.as_str();
    if !exact_evidence(evidence, source_id, source_text) {
        return None;
    }
    if matches!(kind, Kind::Register) {
        let name = required_string(object, "n")?;
        let role = required_string(object, "r")?;
        if !valid_role(role) {
            return None;
        }
        let aliases = object.get("a").cloned().unwrap_or_else(|| json!([]));
        if !valid_aliases(&aliases) {
            return None;
        }
        return Some((
            kind,
            json!({"display_name": name, "role": role, "aliases": aliases,
                "source_id": source_id, "evidence": evidence}),
        ));
    }
    let owner = required_string(object, "o")?;
    let content = required_string(object, "c")?;
    let mut item = json!({"owner": owner, "subjects": [owner], "content": content,
        "source_id": source_id, "evidence": evidence});
    if matches!(kind, Kind::Memory) {
        let tag = required_string(object, "t")?;
        let date = object.get("d").unwrap_or(&Value::Null);
        if !valid_date(date) {
            return None;
        }
        item["tag"] = json!(tag);
        item["event_at"] = date.clone();
    }
    Some((kind, item))
}

const LEGACY_ARRAYS: [(&str, Kind); 3] = [
    ("register", Kind::Register),
    ("memories", Kind::Memory),
    ("retractions", Kind::Retraction),
];

fn is_legacy_root(root: &Map<String, Value>) -> bool {
    // A compact row may carry harmless extra keys, including a legacy name.
    !root
        .keys()
        .any(|key| matches!(key.as_str(), "k" | "o" | "done"))
        && LEGACY_ARRAYS.iter().any(|(key, _)| root.contains_key(*key))
}

/// Identify actual root keys even when a legacy value is cut off. Parsing keys
/// with serde (rather than searching text) avoids mistaking quoted evidence or
/// nested fields for a legacy envelope. No partial values are ever extracted.
fn first_root_is_legacy(text: &str) -> bool {
    #[derive(Default)]
    struct RootKeys {
        legacy: bool,
        compact: bool,
    }
    struct RootKeysVisitor<'a>(&'a mut RootKeys);
    impl<'de> Visitor<'de> for RootKeysVisitor<'_> {
        type Value = ();

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an extraction object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            while let Some(key) = map.next_key::<String>()? {
                self.0.legacy |= LEGACY_ARRAYS.iter().any(|(name, _)| *name == key);
                self.0.compact |= matches!(key.as_str(), "k" | "o" | "done");
                map.next_value::<IgnoredAny>()?;
            }
            Ok(())
        }
    }
    let mut keys = RootKeys::default();
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let _ = deserializer.deserialize_map(RootKeysVisitor(&mut keys));
    keys.legacy && !keys.compact
}

fn parse_legacy(
    root: &Map<String, Value>,
    source_id: &str,
    source_text: &str,
    truncated: bool,
) -> ParsedExtraction {
    let mut parsed = ParsedExtraction::default();
    let mut all_arrays = true;
    for (key, kind) in LEGACY_ARRAYS {
        let Some(items) = root.get(key).and_then(Value::as_array) else {
            // Missing/wrong envelope arrays make the source partial, but are
            // not fabricated bad item records for the invalid-record counter.
            all_arrays = false;
            continue;
        };
        for item in items {
            if valid_legacy_item(kind, item, source_id, source_text) {
                push_item(&mut parsed, kind, item.clone());
            } else {
                parsed.invalid_records += 1;
            }
        }
    }
    parsed.complete = !truncated && all_arrays && parsed.invalid_records == 0;
    suppress_partial_controls(&mut parsed);
    parsed
}

fn valid_legacy_item(kind: Kind, item: &Value, source_id: &str, source_text: &str) -> bool {
    let Some(object) = item.as_object() else {
        return false;
    };
    let required: &[&str] = match kind {
        Kind::Register => &["display_name", "role", "aliases", "source_id", "evidence"],
        Kind::Memory => &[
            "owner",
            "subjects",
            "content",
            "tag",
            "event_at",
            "source_id",
            "evidence",
        ],
        Kind::Retraction => &["owner", "subjects", "content", "source_id", "evidence"],
    };
    if object.len() != required.len() || !required.iter().all(|key| object.contains_key(*key)) {
        return false;
    }
    let Some(evidence) = required_string(object, "evidence") else {
        return false;
    };
    if required_string(object, "source_id") != Some(source_id)
        || !exact_evidence(evidence, source_id, source_text)
    {
        return false;
    }
    if matches!(kind, Kind::Register) {
        return required_string(object, "display_name").is_some()
            && required_string(object, "role").is_some_and(valid_role)
            && object.get("aliases").is_some_and(valid_aliases);
    }
    let Some(owner) = required_string(object, "owner") else {
        return false;
    };
    if required_string(object, "content").is_none()
        || !object
            .get("subjects")
            .and_then(Value::as_array)
            .is_some_and(|subjects| subjects.len() == 1 && subjects[0].as_str() == Some(owner))
    {
        return false;
    }
    matches!(kind, Kind::Retraction)
        || (required_string(object, "tag").is_some()
            && object.get("event_at").is_some_and(valid_date))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "user-text:7";
    const EVIDENCE: &str = "小航喜欢面条";
    const DONE: &str = "{\"done\":true}";

    fn memory_row() -> Value {
        json!({"o":"roster:小航", "t":"饮食", "c":"喜欢面条", "e":EVIDENCE})
    }

    fn register_row() -> Value {
        json!({"k":"r", "n":"小航", "r":"child", "e":EVIDENCE})
    }

    fn retract_row() -> Value {
        json!({"k":"f", "o":"roster:小航", "c":"喜欢面条", "e":EVIDENCE})
    }

    fn compact(rows: &[Value], marker: bool, truncated: bool) -> ParsedExtraction {
        let mut lines = rows.iter().map(Value::to_string).collect::<Vec<_>>();
        if marker {
            lines.push(DONE.to_owned());
        }
        parse_extraction(&lines.join("\n"), SOURCE, EVIDENCE, truncated)
    }

    fn legacy_memory() -> Value {
        json!({"owner":"roster:小航", "subjects":["roster:小航"], "content":"喜欢面条",
            "tag":"饮食", "event_at":null, "source_id":SOURCE, "evidence":EVIDENCE})
    }

    fn legacy_register() -> Value {
        json!({"display_name":"小航", "role":"child", "aliases":[],
            "source_id":SOURCE, "evidence":EVIDENCE})
    }

    fn legacy_retraction() -> Value {
        json!({"owner":"roster:小航", "subjects":["roster:小航"], "content":"喜欢面条",
            "source_id":SOURCE, "evidence":EVIDENCE})
    }

    #[test]
    fn compact_defaults_bind_host_source_and_single_subject() {
        let mut row = memory_row();
        row["o"] = json!("unknown-roster-id"); // Membership belongs to the parent.
        row["c"] = json!("  不裁剪内容，也不按关键词推断  ");
        row["confidence"] = json!({"extra":"harmless"});
        let parsed = compact(&[row], true, false);
        assert!(parsed.complete);
        assert_eq!(parsed.invalid_records, 0);
        assert_eq!(
            parsed.memories,
            vec![
                json!({"owner":"unknown-roster-id", "subjects":["unknown-roster-id"],
                "content":"  不裁剪内容，也不按关键词推断  ", "tag":"饮食", "event_at":null,
                "source_id":SOURCE, "evidence":EVIDENCE})
            ]
        );
    }

    #[test]
    fn compact_normalizes_all_kinds_and_optional_aliases() {
        let mut memory = memory_row();
        memory["k"] = json!("m");
        memory["d"] = json!("2024-02-29");
        let mut register = register_row();
        register["a"] = json!(["航航", " 小航 "]);
        let parsed = compact(
            &[memory, register, register_row(), retract_row()],
            true,
            false,
        );
        assert!(parsed.complete);
        assert_eq!(parsed.memories[0]["event_at"], "2024-02-29");
        assert_eq!(parsed.registers[0]["aliases"], json!(["航航", " 小航 "]));
        assert_eq!(parsed.registers[1], legacy_register());
        assert_eq!(parsed.retractions, vec![legacy_retraction()]);
    }

    #[test]
    fn invalid_calendar_date_drops_only_bad_sibling() {
        let mut bad = memory_row();
        bad["d"] = json!("2026-02-30");
        let parsed = compact(
            &[
                memory_row(),
                bad,
                register_row(),
                retract_row(),
                memory_row(),
            ],
            true,
            false,
        );
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 1);
        assert_eq!(parsed.memories.len(), 2);
        assert!(parsed.registers.is_empty());
        assert!(parsed.retractions.is_empty());
    }

    #[test]
    fn dates_require_null_or_exact_iso_calendar_strings() {
        for date in [Value::Null, json!("2024-02-29"), json!("2026-12-31")] {
            let mut row = memory_row();
            row["d"] = date;
            assert!(compact(&[row], true, false).complete);
        }
        for date in [
            json!("2026-2-01"),
            json!("2026-02-1"),
            json!("2026-02-29"),
            json!("2026-00-01"),
            json!("2026-01-32"),
            json!("2026-01-01T00:00:00Z"),
            json!(" 2026-01-01"),
            json!("+2026-01-01"),
            json!(""),
            json!(20260101),
            json!(false),
            json!([]),
            json!({}),
        ] {
            let mut row = memory_row();
            row["d"] = date;
            let parsed = compact(&[row, memory_row()], true, false);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 1);
            assert_eq!(parsed.memories.len(), 1);
        }
    }

    #[test]
    fn bad_syntax_is_per_physical_line_without_joining_fragments() {
        let text = format!(
            "{}\n{{\"o\":\n\"roster:小航\"}}\nnot-json\n{}\n{DONE}",
            memory_row(),
            memory_row()
        );
        let parsed = parse_extraction(&text, SOURCE, EVIDENCE, false);
        assert_eq!(parsed.invalid_records, 3);
        assert_eq!(parsed.memories.len(), 2);
        assert!(!parsed.complete);
    }

    #[test]
    fn only_a_standalone_final_unique_marker_completes() {
        for text in [DONE.to_owned(), format!("\n \n{DONE}\n\n \r\n")] {
            let parsed = parse_extraction(&text, SOURCE, EVIDENCE, false);
            assert!(parsed.complete);
            assert_eq!(parsed.invalid_records, 0);
            assert!(parsed.memories.is_empty());
        }
        for text in ["", "\n \n"] {
            let parsed = parse_extraction(text, SOURCE, EVIDENCE, false);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 0);
        }
        for text in [
            format!("{DONE}\n{DONE}"),
            format!("{DONE}\n{}", memory_row()),
            format!("{DONE}\n{}\n{DONE}", memory_row()),
        ] {
            let parsed = parse_extraction(&text, SOURCE, EVIDENCE, false);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 0);
        }
        for row in [
            json!({"done":false}),
            json!({"done":null}),
            json!({"done":true, "extra":1}),
            json!({"done":true, "o":"roster:小航", "t":"饮食", "c":"喜欢面条", "e":EVIDENCE}),
        ] {
            let parsed = compact(&[row], true, false);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 1);
            assert!(parsed.memories.is_empty());
        }
        let joined = format!("{}{DONE}", memory_row());
        assert_eq!(
            parse_extraction(&joined, SOURCE, EVIDENCE, false).invalid_records,
            1
        );
    }

    #[test]
    fn missing_marker_and_truncation_suppress_controls_not_memories() {
        let rows = [memory_row(), register_row(), retract_row()];
        for (marker, truncated) in [(false, false), (true, true), (false, true)] {
            let parsed = compact(&rows, marker, truncated);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 0);
            assert_eq!(parsed.memories.len(), 1);
            assert!(parsed.registers.is_empty());
            assert!(parsed.retractions.is_empty());
        }
        let text = format!("{DONE}\n{}\n{}", register_row(), retract_row());
        let parsed = parse_extraction(&text, SOURCE, EVIDENCE, false);
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 0);
        assert!(parsed.registers.is_empty());
        assert!(parsed.retractions.is_empty());
    }

    #[test]
    fn escaped_unicode_quotes_and_newlines_match_contiguous_evidence() {
        let evidence = "他说：\"小航\"\n喜欢\\面条🍜";
        let text = concat!(
            "{\"o\":\"人一\",\"t\":\"饮食\",\"c\":\"第一行\\n第二行🍜\",",
            "\"e\":\"他说：\\\"\\u5c0f\\u822a\\\"\\n喜欢\\\\面条🍜\"}\n",
            "{\"done\":true}"
        );
        let parsed = parse_extraction(text, SOURCE, &format!("前{evidence}后"), false);
        assert!(parsed.complete);
        assert_eq!(parsed.memories[0]["evidence"], evidence);
        assert_eq!(parsed.memories[0]["content"], "第一行\n第二行🍜");
        let split_evidence = text.replace("\\n喜欢", "\n喜欢");
        let parsed = parse_extraction(&split_evidence, SOURCE, evidence, false);
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 2);
        assert!(parsed.memories.is_empty());
    }

    #[test]
    fn compact_rejects_forged_host_fields_even_if_they_match() {
        for field in [
            "source_id",
            "subjects",
            "owner",
            "content",
            "tag",
            "event_at",
            "evidence",
            "display_name",
            "role",
            "aliases",
            "_host_source_id",
            "_host_graph_already_ingested",
            "_host_future_field",
        ] {
            let mut row = memory_row();
            row[field] = json!(SOURCE);
            let parsed = compact(&[row, memory_row()], true, false);
            assert_eq!(parsed.invalid_records, 1, "field: {field}");
            assert_eq!(parsed.memories.len(), 1);
            assert!(!parsed.complete);
        }
        let mut row = memory_row();
        row["memories"] = json!([]); // Harmless unknown does not select legacy.
        assert!(compact(&[row], true, false).complete);
    }

    #[test]
    fn required_fields_and_exact_evidence_are_never_guessed() {
        let mut invalid = Vec::new();
        for field in ["o", "t", "c", "e"] {
            for value in [json!(" "), Value::Null, json!(1), json!([])] {
                let mut row = memory_row();
                row[field] = value;
                invalid.push(row);
            }
            let mut row = memory_row();
            row.as_object_mut().unwrap().remove(field);
            invalid.push(row);
        }
        for evidence in ["小航面条", "小航喜欢面条！"] {
            let mut row = memory_row();
            row["e"] = json!(evidence);
            invalid.push(row);
        }
        for kind in [Value::Null, json!(1), json!("unknown"), json!(" M ")] {
            let mut row = memory_row();
            row["k"] = kind;
            invalid.push(row);
        }
        for aliases in [Value::Null, json!("小航"), json!([""]), json!([2])] {
            let mut row = register_row();
            row["a"] = aliases;
            invalid.push(row);
        }
        let mut wrong_role = register_row();
        wrong_role["r"] = json!("friend");
        invalid.push(wrong_role);
        let count = invalid.len();
        invalid.push(memory_row());
        let parsed = compact(&invalid, true, false);
        assert_eq!(parsed.invalid_records, count);
        assert_eq!(parsed.memories.len(), 1);
        assert!(!parsed.complete);
        let parsed = parse_extraction(
            &format!("{}\n{DONE}", memory_row()),
            SOURCE,
            "小航喜欢别的；面条",
            false,
        );
        assert_eq!(parsed.invalid_records, 1);
        assert!(parsed.memories.is_empty());
    }

    #[test]
    fn evidence_tolerates_formatting_but_not_paraphrase() {
        let row = "| TX019 | 10-14<br>21:18 | 爸爸 → 爸爸 | 智能手表：个人自用<br>电子产品｜数码商城 | 899.00 |";
        // Across table cells and <br>: restored to the exact original slice.
        let found = resolve_evidence("10-14 21:18 爸爸 → 爸爸", row).unwrap();
        assert_eq!(found, "10-14<br>21:18 | 爸爸 → 爸爸");
        assert!(row.contains(&found));
        // Dash, full-width punctuation and thousands separators.
        assert_eq!(resolve_evidence("5/6-1/3", "题目 5/6 − 1/3 通分").unwrap(), "5/6 − 1/3");
        assert_eq!(resolve_evidence("8446.00元", "合计 8,446.00元。").unwrap(), "8,446.00元");
        // A rewrite or reorder is still not evidence.
        assert!(resolve_evidence("手表899元", row).is_none());
        assert!(resolve_evidence("爸爸买了智能手表", row).is_none());
        assert!(resolve_evidence("，", row).is_none());
    }

    #[test]
    fn legacy_complete_object_accepts_unknown_root_fields() {
        let root = json!({"register":[legacy_register()], "memories":[legacy_memory()],
            "retractions":[legacy_retraction()], "extra":{"harmless":true}});
        let parsed = parse_extraction(
            &serde_json::to_string_pretty(&root).unwrap(),
            SOURCE,
            EVIDENCE,
            false,
        );
        assert!(parsed.complete);
        assert_eq!(parsed.invalid_records, 0);
        assert_eq!(parsed.memories, vec![legacy_memory()]);
        assert_eq!(parsed.registers, vec![legacy_register()]);
        assert_eq!(parsed.retractions, vec![legacy_retraction()]);
    }

    #[test]
    fn legacy_invalid_siblings_preserve_valid_memory_without_source_repair() {
        let mut wrong_source = legacy_memory();
        wrong_source["source_id"] = json!("user-text");
        let mut wrong_subject = legacy_memory();
        wrong_subject["subjects"] = json!(["different-roster"]);
        let mut wrong_date = legacy_memory();
        wrong_date["event_at"] = json!("2026-02-30");
        let mut wrong_evidence = legacy_memory();
        wrong_evidence["evidence"] = json!("面条喜欢小航");
        let mut forged = legacy_memory();
        forged["_host_source_id"] = json!(SOURCE);
        let mut wrong_register = legacy_register();
        wrong_register["source_id"] = json!("elsewhere");
        let root = json!({"register":[legacy_register(), wrong_register],
            "memories":[legacy_memory(), wrong_source, wrong_subject, wrong_date,
                wrong_evidence, forged, 42, legacy_memory()],
            "retractions":[legacy_retraction(), null]});
        let parsed = parse_extraction(&root.to_string(), SOURCE, EVIDENCE, false);
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 8);
        assert_eq!(parsed.memories, vec![legacy_memory(), legacy_memory()]);
        assert!(parsed.registers.is_empty());
        assert!(parsed.retractions.is_empty());
    }

    #[test]
    fn legacy_missing_arrays_and_transport_truncation_are_partial() {
        let full = json!({"register":[legacy_register()], "memories":[legacy_memory()],
            "retractions":[legacy_retraction()]});
        for key in ["register", "memories", "retractions"] {
            let mut root = full.clone();
            root.as_object_mut().unwrap().remove(key);
            let parsed = parse_extraction(&root.to_string(), SOURCE, EVIDENCE, false);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 0);
            assert_eq!(parsed.memories.len(), usize::from(key != "memories"));
            assert!(parsed.registers.is_empty());
            assert!(parsed.retractions.is_empty());
        }
        let parsed = parse_extraction(&full.to_string(), SOURCE, EVIDENCE, true);
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 0);
        assert_eq!(parsed.memories, vec![legacy_memory()]);
        assert!(parsed.registers.is_empty());
        assert!(parsed.retractions.is_empty());
        let partial = format!("{{\n\"memories\":[\n{}\n", memory_row());
        for truncated in [false, true] {
            let parsed = parse_extraction(&partial, SOURCE, EVIDENCE, truncated);
            assert!(!parsed.complete);
            assert_eq!(parsed.invalid_records, 1);
            assert!(parsed.memories.is_empty());
        }
        let trailing = format!("{}{DONE}", full);
        let parsed = parse_extraction(&trailing, SOURCE, EVIDENCE, false);
        assert!(!parsed.complete);
        assert_eq!(parsed.invalid_records, 1);
        assert!(parsed.memories.is_empty());
    }
}
