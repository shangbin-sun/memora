pub fn stability(s0: f64, alpha: f64, beta: f64, importance: f64, review_count: i64) -> f64 {
    s0 * (1.0 + alpha * importance + beta * review_count as f64).max(0.01)
}

pub fn retention(t_days: f64, s: f64) -> f64 {
    (-t_days / s.max(1e-6)).exp()
}

pub fn rrf(rank: usize, k: f64) -> f64 {
    1.0 / (k + rank as f64 + 1.0)
}

pub fn next_interval_days(s: f64) -> f64 {
    let steps = [1.0, 3.0, 7.0, 16.0, 35.0];
    let idx = ((s / 2.0).floor() as usize).min(steps.len() - 1);
    steps[idx]
}

pub fn access_boost(review_count: i64) -> f64 {
    1.0 + (review_count.max(0) as f64 + 1.0).ln() * 0.1
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

pub fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_is_one_at_t0() {
        assert!((retention(0.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn retention_decays() {
        let r = retention(10.0, 1.0);
        assert!(r < 0.01);
    }

    #[test]
    fn rrf_prefers_better_rank() {
        assert!(rrf(0, 60.0) > rrf(5, 60.0));
    }

    #[test]
    fn stability_grows_with_reviews() {
        assert!(stability(1.0, 2.0, 0.15, 0.5, 10) > stability(1.0, 2.0, 0.15, 0.5, 0));
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_overlap() {
        let a = vec!["a".into(), "b".into()];
        let b = vec!["b".into(), "c".into()];
        assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn interval_steps() {
        assert_eq!(next_interval_days(0.5), 1.0);
        assert_eq!(next_interval_days(8.0), 35.0);
    }
}
