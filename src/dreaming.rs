use async_trait::async_trait;

use crate::core::split_sentences;
use crate::embedding::{SummarizeInput, SummarizeOutput, Summarizer};
use crate::error::Result;

pub struct ExtractiveSummarizer;

#[async_trait]
impl Summarizer for ExtractiveSummarizer {
    fn summarizer_id(&self) -> &str {
        "extractive-v1"
    }

    fn prompt_version(&self) -> &str {
        "extractive-1"
    }

    async fn summarize(&self, input: SummarizeInput) -> Result<SummarizeOutput> {
        Ok(SummarizeOutput {
            summary: extractive_summary(&input.texts, 4),
        })
    }
}

pub fn extractive_summary(texts: &[String], max_sentences: usize) -> String {
    let joined = texts.join("。");
    let sentences = split_sentences(&joined);
    if sentences.is_empty() {
        return texts.join(" ");
    }
    let mut df: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let tokenized: Vec<Vec<String>> = sentences
        .iter()
        .map(|s| {
            s.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .chars()
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| w.iter().collect::<String>())
                .collect()
        })
        .collect();
    for tokens in &tokenized {
        let uniq: std::collections::HashSet<&String> = tokens.iter().collect();
        for t in uniq {
            *df.entry(t.clone()).or_insert(0.0) += 1.0;
        }
    }
    let n = sentences.len() as f64;
    let mut scored: Vec<(usize, f64)> = tokenized
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let mut score = 0.0;
            for t in tokens {
                let d = *df.get(t).unwrap_or(&1.0);
                score += (n / d).ln().max(0.0);
            }
            if !tokens.is_empty() {
                score /= tokens.len() as f64;
            }
            let pos = 1.0 - (i as f64 / n.max(1.0));
            (i, score + 0.15 * pos)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: Vec<usize> = scored
        .into_iter()
        .take(max_sentences.min(sentences.len()))
        .map(|(i, _)| i)
        .collect();
    picked.sort_unstable();
    picked
        .into_iter()
        .map(|i| sentences[i].as_str())
        .collect::<Vec<_>>()
        .join("")
}

pub fn cluster_indices(
    time_buckets: &[i64],
    entities: &[Vec<String>],
    embeddings: &[Option<Vec<f32>>],
    jaccard_th: f64,
    cosine_th: f64,
) -> Vec<Vec<usize>> {
    let n = time_buckets.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            p[x] = find(p, p[x]);
        }
        p[x]
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let pa = find(p, a);
        let pb = find(p, b);
        if pa != pb {
            p[pa] = pb;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if time_buckets[i] != time_buckets[j] {
                continue;
            }
            let jac = crate::core::jaccard(&entities[i], &entities[j]);
            let cos = match (&embeddings[i], &embeddings[j]) {
                (Some(a), Some(b)) => crate::core::cosine(a, b),
                _ => 0.0,
            };
            if jac >= jaccard_th || cos >= cosine_th {
                union(&mut parent, i, j);
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Normalize text for exact-duplicate detection.
#[cfg(test)]
fn normalize_for_dedup(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .replace('℃', "度")
        .replace("°c", "度")
}

/// Near-duplicate grouping without time-bucket constraint.
/// Links when: exact normalized content, or high cosine, or (entity Jaccard + softer cosine).
#[cfg(test)]
fn dedup_indices(
    contents_norm: &[String],
    entities: &[Vec<String>],
    embeddings: &[Option<Vec<f32>>],
    jaccard_th: f64,
    cosine_th: f64,
) -> Vec<Vec<usize>> {
    let n = contents_norm.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            p[x] = find(p, p[x]);
        }
        p[x]
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let pa = find(p, a);
        let pb = find(p, b);
        if pa != pb {
            p[pa] = pb;
        }
    }
    let soft_cos = (cosine_th * 0.92).clamp(0.7, cosine_th);
    for i in 0..n {
        for j in (i + 1)..n {
            let exact = !contents_norm[i].is_empty() && contents_norm[i] == contents_norm[j];
            let cos = match (&embeddings[i], &embeddings[j]) {
                (Some(a), Some(b)) => crate::core::cosine(a, b),
                _ => 0.0,
            };
            let jac = crate::core::jaccard(&entities[i], &entities[j]);
            if exact || cos >= cosine_th || (jac >= jaccard_th && cos >= soft_cos) {
                union(&mut parent, i, j);
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().filter(|g| g.len() >= 2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractive_keeps_content() {
        let texts = vec![
            "用户偏好 SQLite。".into(),
            "本地优先，不要云数据库。".into(),
        ];
        let s = extractive_summary(&texts, 2);
        assert!(!s.is_empty());
    }

    #[test]
    fn cluster_same_bucket_and_entities() {
        let buckets = vec![1, 1, 2];
        let entities = vec![
            vec!["SQLite".into()],
            vec!["SQLite".into()],
            vec!["日报".into()],
        ];
        let embeddings = vec![None, None, None];
        let groups = cluster_indices(&buckets, &entities, &embeddings, 0.3, 0.75);
        let joined = groups.iter().any(|g| g.len() >= 2);
        assert!(joined);
        assert!(groups.len() >= 2);
    }

    #[test]
    fn dedup_exact_content() {
        let norms = vec![
            normalize_for_dedup("夏天空调 22 度"),
            normalize_for_dedup("夏天空调22度"),
            normalize_for_dedup("完全不同的事"),
        ];
        let entities = vec![vec![], vec![], vec![]];
        let embeddings = vec![None, None, None];
        let groups = dedup_indices(&norms, &entities, &embeddings, 0.3, 0.9);
        assert!(groups.iter().any(|g| g.len() == 2));
    }
}
