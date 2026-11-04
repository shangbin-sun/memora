use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Datelike, TimeZone, Timelike, Utc};

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn days_between(from_ms: i64, to_ms: i64) -> f64 {
    (to_ms - from_ms).max(0) as f64 / 86_400_000.0
}

pub fn weekday_and_hour(ms: i64) -> (i64, i64) {
    let dt = Utc
        .timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(Utc::now);
    (dt.weekday().num_days_from_monday() as i64, dt.hour() as i64)
}

pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(2).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_between_zero_and_one() {
        assert_eq!(days_between(0, 0), 0.0);
        assert!((days_between(0, 86_400_000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chinese_token_estimate() {
        assert_eq!(estimate_tokens("你好"), 1);
        assert_eq!(estimate_tokens("abcd"), 2);
    }
}
