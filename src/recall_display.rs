//! Recall-only presentation. Never rewrite durable facts, embeddings or event times.
use crate::household::{self, HouseholdMember};
use crate::types::MemoryRecord;
use chrono::{DateTime, FixedOffset};

/// Render a memory for a consumer: current member display names and original
/// creation time (UTC+8, minute precision), followed by unmodified display body.
/// The explicit `记录于` label prevents conflating storage time with event time.
pub fn render_memory_snippet(rec: &MemoryRecord, roster: &[HouseholdMember]) -> String {
    let subjects = if rec.subject_ids.is_empty() {
        household::subject_id_list(&rec.metadata)
    } else {
        rec.subject_ids.clone()
    };
    let body = rec
        .summary
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&rec.content);
    format_recall_snippet(
        &crate::temporal::display_content(body),
        rec.created_at_ms,
        &subjects,
        roster,
    )
}

fn single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn format_recall_snippet(
    body: &str,
    created_at_ms: i64,
    subjects: &[String],
    roster: &[HouseholdMember],
) -> String {
    let mut ids = Vec::new();
    let mut names = Vec::new();
    for id in subjects
        .iter()
        .map(|id| id.trim())
        .filter(|id| !id.is_empty())
    {
        if ids.contains(&id) {
            continue;
        }
        ids.push(id);
        let mut matches = roster.iter().filter(|member| member.member_id == id);
        let name = match (matches.next(), matches.next()) {
            (Some(member), None) if !member.display_name.trim().is_empty() => {
                single_line(&member.display_name)
            }
            _ => single_line(id), // Deleted/ambiguous roster entry: never guess a person.
        };
        names.push(name);
    }
    let members = if names.is_empty() {
        "未归属".to_string()
    } else {
        names.join("、")
    };
    let time = DateTime::from_timestamp_millis(created_at_ms).map(|time| {
        time.with_timezone(&FixedOffset::east_opt(8 * 3600).expect("UTC+8"))
            .format("%Y-%m-%d %H:%M")
            .to_string()
    });
    let stamp = time
        .map(|time| format!("记录于{time}"))
        .unwrap_or_else(|| "记录时间未知".into());
    format!("[{members} · {stamp}] {}", body.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::household::MemberRole;

    fn member(id: &str, name: &str) -> HouseholdMember {
        HouseholdMember {
            member_id: id.into(),
            display_name: name.into(),
            role: MemberRole::Other,
            aliases: vec![],
            age_band: None,
        }
    }
    fn timestamp() -> i64 {
        chrono::DateTime::parse_from_rfc3339("2026-10-26T11:20:59+08:00")
            .unwrap()
            .timestamp_millis()
    }
    #[test]
    fn rendering_uses_record_creation_and_display_body_without_mutating_record() {
        let record: MemoryRecord = serde_json::from_value(serde_json::json!({
            "id": "01900000-0000-7000-8000-000000000001", "kind": "episodic", "state": "active",
            "content": "我有点头疼", "summary": "觉得有点头疼 ⌈event:2030-01-01⌋",
            "importance": 0.8, "confidence": 1.0, "salience": 0.8,
            "created_at_ms": timestamp(), "updated_at_ms": 0, "last_accessed_at_ms": null,
            "next_review_at_ms": null, "review_count": 0, "source_hash": null,
            "embedding_model": null, "embedding_version": 1, "embedding_pending": false,
            "parent_id": null, "namespace": "default", "session_id": null, "entities": [],
            "metadata": {"subjects": [{"member_id": "different"}]}, "subject_ids": ["driver"]
        }))
        .unwrap();
        let before = serde_json::to_value(&record).unwrap();
        let roster = [member("driver", "主驾")];
        assert_eq!(
            render_memory_snippet(&record, &roster),
            "[主驾 · 记录于2026-10-26 11:20] 觉得有点头疼"
        );
        assert_eq!(serde_json::to_value(&record).unwrap(), before);
        let mut legacy = record.clone();
        legacy.subject_ids.clear();
        legacy.metadata =
            serde_json::json!({"subjects": [{"member_id": "driver", "confidence": 1.0}]});
        legacy.summary = Some(" \n".into());
        assert_eq!(
            render_memory_snippet(&legacy, &roster),
            "[主驾 · 记录于2026-10-26 11:20] 我有点头疼"
        );
    }

    #[test]
    fn minute_precision_and_fixed_china_timezone_preserve_body() {
        assert_eq!(
            format_recall_snippet(
                "我有点头疼",
                timestamp(),
                &["driver".into()],
                &[member("driver", "主驾")]
            ),
            "[主驾 · 记录于2026-10-26 11:20] 我有点头疼"
        );
        let midnight = chrono::DateTime::parse_from_rfc3339("2026-12-31T16:01:00Z")
            .unwrap()
            .timestamp_millis();
        assert_eq!(
            format_recall_snippet("新年计划", midnight, &[], &[]),
            "[未归属 · 记录于2027-01-01 00:01] 新年计划"
        );
    }
    #[test]
    fn multi_subjects_are_deduped_and_missing_members_are_not_guessed() {
        let roster = [member("driver", "主驾"), member("child", "小航")];
        let subjects = [
            "driver".into(),
            "child".into(),
            "driver".into(),
            "former".into(),
        ];
        assert_eq!(
            format_recall_snippet("共同出行", timestamp(), &subjects, &roster),
            "[主驾、小航、former · 记录于2026-10-26 11:20] 共同出行"
        );
    }
    #[test]
    fn shared_and_unassigned_are_distinct_and_invalid_time_never_uses_now() {
        assert!(
            format_recall_snippet("规则", timestamp(), &["公共".into()], &[])
                .starts_with("[公共 ·")
        );
        assert_eq!(
            format_recall_snippet("原文", i64::MAX, &[], &[]),
            "[未归属 · 记录时间未知] 原文"
        );
    }
    #[test]
    fn duplicate_roster_entries_and_blank_names_fall_back_to_id() {
        let roster = [member("x", "甲"), member("x", "乙"), member("y", " ")];
        assert!(
            format_recall_snippet("原文", timestamp(), &["x".into(), "y".into()], &roster)
                .starts_with("[x、y ·")
        );
        assert!(format_recall_snippet(
            "原文",
            timestamp(),
            &["x".into()],
            &[member("x", "主驾\n座位")]
        )
        .starts_with("[主驾 座位 ·"));
    }
}
