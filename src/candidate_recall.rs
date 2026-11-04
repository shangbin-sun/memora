//! Internal memory-only candidate search for reconciliation.
//! No STM, graph traversal, presentation, access touches or result-cache side effects.
use crate::config::Config;
use crate::core::{access_boost, days_between, now_ms, retention, rrf, stability};
use crate::types::{MemoryId, MemoryRecord, ScoreBreakdown};
use std::collections::HashMap;

pub(crate) fn fuse_candidates(
    fts: Vec<(MemoryId, f64)>,
    vectors: Vec<(MemoryId, f32)>,
    k: f64,
) -> Vec<(MemoryId, ScoreBreakdown)> {
    let mut fused = HashMap::new();
    for (rank, (id, score)) in fts.into_iter().enumerate() {
        let entry = fused.entry(id).or_insert_with(empty_scores);
        entry.bm25_rank = Some(rank);
        entry.bm25_score = Some(score);
        entry.rrf += rrf(rank, k);
    }
    for (rank, (id, distance)) in vectors.into_iter().enumerate() {
        let entry = fused.entry(id).or_insert_with(empty_scores);
        entry.vector_rank = Some(rank);
        entry.l2 = Some(distance);
        entry.rrf += rrf(rank, k);
    }
    let mut ranked: Vec<_> = fused.into_iter().collect();
    ranked.sort_by(|a, b| b.1.rrf.total_cmp(&a.1.rrf).then_with(|| a.0.cmp(&b.0)));
    ranked
}

fn empty_scores() -> ScoreBreakdown {
    ScoreBreakdown {
        bm25_rank: None,
        bm25_score: None,
        vector_rank: None,
        l2: None,
        rrf: 0.0,
        decay: 1.0,
        confidence: 1.0,
        importance: 1.0,
        access_boost: 1.0,
        final_score: 0.0,
    }
}

/// Input records must have passed the same RecallQuery filters as ordinary recall.
pub(crate) fn rank_candidates(
    ranked: Vec<(MemoryId, ScoreBreakdown)>,
    records: &HashMap<MemoryId, MemoryRecord>,
    config: &Config,
    top: usize,
) -> Vec<MemoryRecord> {
    let now = now_ms();
    let mut scored = Vec::new();
    for (id, mut scores) in ranked
        .into_iter()
        .filter(|(id, _)| records.contains_key(id))
        .take(config.rrf_merge_k.max(top))
    {
        let record = &records[&id];
        let last = record.last_accessed_at_ms.unwrap_or(record.created_at_ms);
        scores.decay = retention(
            days_between(last, now),
            stability(
                config.s0,
                config.alpha,
                config.beta,
                record.importance,
                record.review_count,
            ),
        );
        scores.final_score = scores.rrf
            * record.confidence
            * scores.decay
            * access_boost(record.review_count)
            * record.importance.max(0.01);
        scored.push((id, scores.final_score));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored
        .into_iter()
        .take(top)
        .map(|(id, _)| records[&id].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn record(id: u128, importance: f64) -> MemoryRecord {
        let now = now_ms();
        serde_json::from_value(serde_json::json!({
            "id": Uuid::from_u128(id), "kind": "episodic", "state": "active", "content": "hospital",
            "summary": null, "importance": importance, "confidence": 1.0, "salience": importance,
            "created_at_ms": now, "updated_at_ms": now, "last_accessed_at_ms": now,
            "next_review_at_ms": null, "review_count": 0, "source_hash": null, "embedding_model": null,
            "embedding_version": 1, "embedding_pending": false, "parent_id": null,
            "namespace": "fixture", "session_id": null, "entities": [], "metadata": {}
        })).unwrap()
    }

    #[test]
    fn fusion_keeps_one_candidate_and_adds_both_reciprocal_rank_contributions() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let ranked = fuse_candidates(vec![(a, -1.0), (b, -0.5)], vec![(b, 0.0), (a, 0.1)], 60.0);
        assert_eq!(ranked.len(), 2);
        assert_eq!(
            ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![a, b]
        );
        for (_, score) in ranked {
            assert!((score.rrf - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
            assert!(score.bm25_rank.is_some() && score.vector_rank.is_some());
        }
    }

    #[test]
    fn filtering_precedes_fusion_cap_and_empty_top_is_empty() {
        let eligible = record(100, 0.9);
        let ranked = fuse_candidates(
            (1..=100).map(|id| (Uuid::from_u128(id), 0.0)).collect(),
            vec![],
            60.0,
        );
        let records = HashMap::from([(eligible.id, eligible.clone())]);
        assert_eq!(
            rank_candidates(ranked.clone(), &records, &Config::default(), 1)[0].id,
            eligible.id
        );
        assert!(rank_candidates(ranked, &records, &Config::default(), 0).is_empty());
    }

    #[test]
    fn final_score_prefers_importance_and_does_not_mutate_input_records() {
        let low = record(1, 0.1);
        let high = record(2, 0.9);
        let records = HashMap::from([(low.id, low.clone()), (high.id, high.clone())]);
        let before: Vec<_> = [low.id, high.id]
            .into_iter()
            .map(|id| serde_json::to_value(&records[&id]).unwrap())
            .collect();
        let ranked = fuse_candidates(vec![(low.id, -1.0), (high.id, -0.5)], vec![], 60.0);
        let output = rank_candidates(ranked, &records, &Config::default(), 2);
        assert_eq!(
            output.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![high.id, low.id]
        );
        let after: Vec<_> = [low.id, high.id]
            .into_iter()
            .map(|id| serde_json::to_value(&records[&id]).unwrap())
            .collect();
        assert_eq!(before, after);
    }
}
