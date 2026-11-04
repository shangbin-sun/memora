use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub namespace: String,

    pub session_events_ttl: Duration,
    pub session_events_capacity: u64,
    pub working_facts_ttl: Duration,
    pub working_facts_capacity: u64,
    pub query_cache_ttl: Duration,
    pub query_cache_capacity: u64,
    /// Query vectors only; memory/ACL result invalidation does not evict these.
    pub query_embedding_cache_ttl: Duration,
    pub query_embedding_cache_capacity: u64,
    /// Maximum simultaneous provider batches; waiting callers share their own deadline.
    pub query_embedding_max_in_flight: usize,
    /// Provider batch size, hard-capped at 16 by the cache implementation.
    pub query_embedding_batch_size: usize,
    pub stm_promote_importance: f64,

    pub fts_top_k: usize,
    pub hnsw_top_k: usize,
    pub rrf_k: f64,
    pub rrf_merge_k: usize,
    pub recall_top_k: usize,
    pub token_budget: usize,
    /// A failed/slow semantic branch must not prevent local FTS recall.
    pub recall_embed_timeout: Duration,
    pub embedding_timeout: Duration,

    pub s0: f64,
    pub alpha: f64,
    pub beta: f64,
    pub r_archive: f64,
    pub forgotten_grace_days: i64,

    pub dream_candidate_limit: usize,
    pub dream_cluster_min: usize,
    pub dream_time_bucket_hours: i64,
    pub pattern_support: i64,
    pub entity_jaccard_threshold: f64,
    pub cosine_threshold: f64,
    /// Cosine threshold for dream near-duplicate merge (stricter than cluster).
    pub dream_dedup_cosine: f64,
    pub consolidation_batch_size: usize,
    pub consolidation_archive_importance: f64,

    pub hnsw_m: usize,
    pub hnsw_ef_construct: usize,
    pub hnsw_max_layer: usize,
    pub hnsw_ef: usize,
    pub hnsw_max_elements: usize,

    pub max_import_uncompressed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            namespace: "default".into(),
            session_events_ttl: Duration::from_secs(2 * 3600),
            session_events_capacity: 1024,
            working_facts_ttl: Duration::from_secs(3 * 24 * 3600),
            working_facts_capacity: 256,
            query_cache_ttl: Duration::from_secs(15),
            query_cache_capacity: 256,
            query_embedding_cache_ttl: Duration::from_secs(5 * 60),
            query_embedding_cache_capacity: 512,
            query_embedding_max_in_flight: 3,
            query_embedding_batch_size: 16,
            stm_promote_importance: 0.5,
            fts_top_k: 80,
            hnsw_top_k: 80,
            rrf_k: 60.0,
            rrf_merge_k: 20,
            recall_top_k: 10,
            token_budget: 2000,
            recall_embed_timeout: Duration::from_millis(700),
            embedding_timeout: Duration::from_secs(30),
            s0: 1.0,
            alpha: 2.0,
            beta: 0.15,
            r_archive: 0.15,
            forgotten_grace_days: 30,
            dream_candidate_limit: 500,
            dream_cluster_min: 2,
            dream_time_bucket_hours: 6,
            pattern_support: 5,
            entity_jaccard_threshold: 0.3,
            cosine_threshold: 0.75,
            dream_dedup_cosine: 0.90,
            consolidation_batch_size: 20,
            consolidation_archive_importance: 0.2,
            hnsw_m: 16,
            hnsw_ef_construct: 200,
            hnsw_max_layer: 16,
            hnsw_ef: 64,
            hnsw_max_elements: 100_000,
            max_import_uncompressed: 512 * 1024 * 1024,
        }
    }
}
