use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jieba_rs::Jieba;
use moka::future::Cache;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::classify::{ClassifyItem, HeuristicClassifier, MemoryClassifier};
use crate::config::Config;
use crate::core::{
    access_boost, days_between, estimate_tokens, next_interval_days, now_ms, retention, rrf,
    stability, weekday_and_hour,
};
use crate::dreaming::{cluster_indices, ExtractiveSummarizer};
use crate::embedding::{normalize_vector, vector_to_bytes, Embedder, SummarizeInput, Summarizer};
use crate::error::{MemoraError, Result};
use crate::exchange::{export_archive, import_archive};
use crate::forgetting::apply_forgetting;
use crate::graph::{
    EmptyGraphExtractor, GrantInput, GraphExtractInput, GraphExtractor, PrivacyLevel,
    RevokeGrantInput,
};
use crate::household::{self, HouseholdMember};
use crate::retrieval::VectorIndex;
use crate::short_term::{ShortTermMemory, StmItem};
use crate::store::{
    append_audit, bump_pattern, delete_embedding, delete_memory,
    find_exact_content_duplicate_groups, find_pattern, get_embedding, get_memory, has_relations,
    insert_dream_run, insert_memory, insert_relation, insert_version, invalidate_derivatives,
    list_audit, list_memories_page, list_relations, list_versions, load_all_embeddings,
    mature_patterns, next_embedding_label, open_pool, pending_consolidation, pending_embedding_ids,
    prepare_vector_space, remove_fts, retry_busy, touch_memories, update_memory, upsert_embedding,
    Pool,
};
use crate::types::{
    ContextItem, ContextPack, ContextRole, DreamOptions, DreamReport, ExportRequest, ForgetReason,
    ImportMode, ImportReport, ListQuery, MemoryExplanation, MemoryId, MemoryKind, MemoryPage,
    MemoryRecord, MemoryService, MemoryState, MemoryUpdate, RecallQuery, RecallResult,
    RecalledMemory, RelationKind, RememberInput, ScoreBreakdown, TickReport,
};

const DEFAULT_STM_SESSION: &str = "__default__";

struct Inner {
    pool: Pool,
    config: Config,
    jieba: Arc<Jieba>,
    embedder: Option<Arc<dyn Embedder>>,
    query_embeddings: crate::query_embedding::QueryEmbeddingCache,
    summarizer: Arc<dyn Summarizer>,
    classifier: Arc<dyn MemoryClassifier>,
    graph_extractor: Arc<dyn GraphExtractor>,
    graph_extractor_configured: bool,
    graph_wake: Arc<tokio::sync::Notify>,
    graph_retry_lock: tokio::sync::Mutex<()>,
    graph_job: Mutex<Option<tokio::task::JoinHandle<()>>>,
    stm: ShortTermMemory,
    index: Mutex<VectorIndex>,
    embed_tx: Option<mpsc::UnboundedSender<MemoryId>>,
    last_scores: Cache<MemoryId, ScoreBreakdown>,
    namespace: String,
    data_epoch: std::sync::atomic::AtomicU64,
    mutation_lock: tokio::sync::Mutex<()>,
    dream_lock: tokio::sync::Mutex<()>,
    embed_job: Mutex<Option<tokio::task::JoinHandle<()>>>,
    closed: std::sync::atomic::AtomicBool,
}

#[derive(Clone)]
pub struct MemoryEngine {
    inner: Arc<Inner>,
}

pub struct EngineBuilder {
    db_path: Option<PathBuf>,
    index_path: Option<PathBuf>,
    namespace: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    summarizer: Option<Arc<dyn Summarizer>>,
    classifier: Option<Arc<dyn MemoryClassifier>>,
    graph_extractor: Option<Arc<dyn GraphExtractor>>,
    graph_projection_worker: bool,
    config: Config,
}

impl MemoryEngine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder {
            db_path: None,
            index_path: None,
            namespace: None,
            embedder: None,
            summarizer: None,
            classifier: None,
            graph_extractor: None,
            graph_projection_worker: true,
            config: Config::default(),
        }
    }
}

impl EngineBuilder {
    pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    pub fn index_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.index_path = Some(path.into());
        self
    }

    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    pub fn classifier(mut self, classifier: Arc<dyn MemoryClassifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    pub fn graph_extractor(mut self, extractor: Arc<dyn GraphExtractor>) -> Self {
        self.graph_extractor = Some(extractor);
        self
    }

    /// Automatically drain durable projection jobs after writes (default true).
    /// Disable only when a host explicitly owns queue draining or in deterministic tests.
    pub fn graph_projection_worker(mut self, enabled: bool) -> Self {
        self.graph_projection_worker = enabled;
        self
    }

    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub async fn build(self) -> Result<MemoryEngine> {
        let db_path = self
            .db_path
            .ok_or_else(|| MemoraError::Other("db_path is required".into()))?;
        let index_path = self.index_path.unwrap_or_else(|| {
            let mut p = db_path.clone();
            p.set_extension("hnsw");
            p
        });
        let namespace = self
            .namespace
            .unwrap_or_else(|| self.config.namespace.clone());
        let embedder_for_init = self.embedder.clone();
        let config_for_init = self.config.clone();
        let (pool, jieba, index, pending) = tokio::task::spawn_blocking(move || {
            let pool = open_pool(&db_path)?;
            let jieba = Arc::new(Jieba::new());
            let (rows, pending) = if let Some(embedder) = embedder_for_init {
                let mut conn = pool.get()?;
                prepare_vector_space(
                    &mut conn,
                    &embedder.space_id(),
                    embedder.model_id(),
                    embedder.dimension(),
                )?;
                let existing =
                    load_all_embeddings(&conn, &embedder.space_id(), embedder.dimension())?;
                if let Some((_, _, vector)) = existing.first() {
                    embedder.bind_dimension(vector.len());
                }
                prepare_vector_space(
                    &mut conn,
                    &embedder.space_id(),
                    embedder.model_id(),
                    embedder.dimension(),
                )?;
                (
                    load_all_embeddings(&conn, &embedder.space_id(), embedder.dimension())?,
                    pending_embedding_ids(&conn)?,
                )
            } else {
                (Vec::new(), Vec::new())
            };
            let index = VectorIndex::rebuild(&config_for_init, index_path, rows);
            Ok::<_, MemoraError>((pool, jieba, index, pending))
        })
        .await??;
        let stm = ShortTermMemory::new(&self.config);
        let last_scores = Cache::builder()
            .max_capacity(2048)
            .time_to_live(Duration::from_secs(3600))
            .build();
        let (embed_tx, rx) = if self.embedder.is_some() {
            let (tx, rx) = mpsc::unbounded_channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let archive_threshold = self.config.consolidation_archive_importance;
        let graph_extractor_configured = self
            .graph_extractor
            .as_ref()
            .is_some_and(|extractor| extractor.extractor_id() != "empty-kg-v1");
        let start_graph_worker = self.graph_projection_worker && graph_extractor_configured;
        let query_embeddings = crate::query_embedding::QueryEmbeddingCache::new(&self.config);
        let engine = MemoryEngine {
            inner: Arc::new(Inner {
                pool,
                query_embeddings,
                config: self.config,
                jieba,
                embedder: self.embedder,
                summarizer: self
                    .summarizer
                    .unwrap_or_else(|| Arc::new(ExtractiveSummarizer)),
                classifier: self
                    .classifier
                    .unwrap_or_else(|| Arc::new(HeuristicClassifier::new(archive_threshold))),
                graph_extractor: self
                    .graph_extractor
                    .unwrap_or_else(|| Arc::new(EmptyGraphExtractor)),
                graph_extractor_configured,
                graph_wake: Arc::new(tokio::sync::Notify::new()),
                graph_retry_lock: tokio::sync::Mutex::new(()),
                graph_job: Mutex::new(None),
                stm,
                index: Mutex::new(index),
                embed_tx,
                last_scores,
                namespace,
                data_epoch: std::sync::atomic::AtomicU64::new(0),
                mutation_lock: tokio::sync::Mutex::new(()),
                dream_lock: tokio::sync::Mutex::new(()),
                embed_job: Mutex::new(None),
                closed: std::sync::atomic::AtomicBool::new(false),
            }),
        };
        if let Some(mut rx) = rx {
            // Inner owns the sender. Do not strongly own Inner while waiting
            // on the receiver, otherwise the channel can never close.
            let weak = Arc::downgrade(&engine.inner);
            let job = tokio::spawn(async move {
                while let Some(id) = rx.recv().await {
                    let Some(inner) = weak.upgrade() else {
                        break;
                    };
                    let worker = MemoryEngine { inner };
                    if let Err(e) = worker.embed_one(id).await {
                        tracing::warn!(error = %e, memory_id = %id, "embed worker failed");
                    }
                }
            });
            *engine.inner.embed_job.lock().expect("embedding job") = Some(job);
        }
        for id in pending {
            engine.enqueue_embed(id);
        }
        if start_graph_worker {
            engine.start_graph_projection_worker();
        } else if !graph_extractor_configured {
            tracing::warn!("graph extractor unavailable; durable projection jobs retained until a configured restart");
        }
        Ok(engine)
    }
}

impl MemoryEngine {
    /// Cancel internal embedding/projection work, including in-flight requests.
    /// Durable jobs remain recoverable; hosts also abort maintenance/dream handles.
    pub fn shutdown(&self) {
        self.inner
            .closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(job) = self.inner.embed_job.lock().expect("embedding job").take() {
            job.abort();
        }
        if let Some(job) = self.inner.graph_job.lock().expect("graph job").take() {
            job.abort();
        }
        self.inner.graph_wake.notify_waiters();
        self.inner.query_embeddings.close();
    }

    /// Projection runs independently of periodic forgetting/dream maintenance.
    pub fn graph_projection_worker_enabled(&self) -> bool {
        self.inner
            .graph_job
            .lock()
            .expect("graph job")
            .as_ref()
            .is_some_and(|job| !job.is_finished())
    }

    fn start_graph_projection_worker(&self) {
        let weak = Arc::downgrade(&self.inner);
        let wake = self.inner.graph_wake.clone();
        let job = tokio::spawn(async move {
            // Scan once at startup so existing durable jobs need no new write.
            loop {
                let Some(inner) = weak.upgrade() else {
                    break;
                };
                if inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                let engine = MemoryEngine { inner };
                let attempted = match engine.retry_graph_projections(8).await {
                    Ok(count) => count,
                    Err(error) => {
                        tracing::warn!(error = %error, "graph projection worker failed; durable jobs retained");
                        0
                    }
                };
                // Never retain Inner while idle: Notify stores one coalesced signal,
                // not one task/payload per write. Last owner drop closes the worker.
                drop(engine);
                if attempted == 8 {
                    tokio::task::yield_now().await;
                    continue;
                }
                tokio::select! {
                    _ = wake.notified() => {},
                    _ = tokio::time::sleep(Duration::from_secs(15)) => {},
                }
            }
        });
        *self.inner.graph_job.lock().expect("graph job") = Some(job);
    }

    fn wake_graph_projection_worker(&self) {
        if self.inner.graph_extractor_configured {
            self.inner.graph_wake.notify_one();
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(MemoraError::InvalidState(
                "memory engine is shut down".into(),
            ));
        }
        Ok(())
    }

    async fn invalidate_memories(&self, ids: &[MemoryId]) {
        self.inner
            .data_epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.stm.query_cache.invalidate_all();
        self.inner.stm.remove_memories(ids).await;
        for id in ids {
            self.inner.last_scores.invalidate(id).await;
        }
    }

    /// Graph / grant mutations must invalidate recall cache even without memory-id churn.
    async fn invalidate_graph_cache(&self) {
        self.inner
            .data_epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.stm.query_cache.invalidate_all();
    }

    async fn withdraw_graph_for_memory(
        &self,
        memory_id: MemoryId,
        namespace: &str,
    ) -> Result<usize> {
        let pool = self.inner.pool.clone();
        let ns = namespace.to_string();
        let n = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::store::withdraw_facts_by_source_memory(&conn, &ns, memory_id)
        })
        .await??;
        if n > 0 {
            self.invalidate_graph_cache().await;
        }
        Ok(n)
    }

    fn enqueue_embed(&self, id: MemoryId) {
        if self.ensure_open().is_err() {
            return;
        }
        if let Some(tx) = &self.inner.embed_tx {
            let _ = tx.send(id);
        }
    }

    /// Internal reconciliation candidates. Batch vectors, DB loads and filters;
    /// deliberately omit context construction, graph traversal and access touches.
    pub(crate) async fn recall_candidates_batch(
        &self,
        queries: Vec<RecallQuery>,
    ) -> Result<Vec<Vec<MemoryRecord>>> {
        let mut performance = crate::performance::StageTimer::new("candidate_recall");
        self.ensure_open()?;
        let mut output = Vec::with_capacity(queries.len());
        // Reconciliation uses one namespace; generic callers with mixed namespaces
        // are split, never sharing records or authorization snapshots.
        let mut offset = 0;
        let batch_size = self.inner.config.query_embedding_batch_size.clamp(1, 16);
        while offset < queries.len() {
            let namespace = queries[offset]
                .namespace
                .clone()
                .unwrap_or_else(|| self.inner.namespace.clone());
            let end = (offset..queries.len())
                .take(batch_size)
                .take_while(|index| {
                    queries[*index]
                        .namespace
                        .as_deref()
                        .unwrap_or(&self.inner.namespace)
                        == namespace
                })
                .last()
                .map(|index| index + 1)
                .unwrap_or(offset + 1);
            let batch = queries[offset..end].to_vec();
            let pool = self.inner.pool.clone();
            let jieba = self.inner.jieba.clone();
            let fts_batch = batch.clone();
            let ns = namespace.clone();
            let fts_top = self.inner.config.fts_top_k;
            let fts = tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                fts_batch
                    .iter()
                    .map(|query| {
                        let rewrite = crate::query_rewrite::rewrite_plan(&query.text);
                        let text = crate::core::tokenize::fts_match_query_with(
                            &jieba,
                            &query.text,
                            &rewrite.extra_words,
                        );
                        crate::store::fts_search(&conn, &text, &ns, fts_top, query)
                    })
                    .collect::<Result<Vec<_>>>()
            });
            let positions: Vec<_> = batch
                .iter()
                .enumerate()
                .filter(|(_, query)| {
                    !query.fts_only && query.query_embedding.is_none() && !query.text.is_empty()
                })
                .map(|(index, _)| index)
                .collect();
            let texts: Vec<_> = positions
                .iter()
                .map(|index| batch[*index].text.clone())
                .collect();
            let mut query_vectors: Vec<_> = batch
                .iter()
                .map(|query| {
                    if query.fts_only {
                        None
                    } else {
                        query.query_embedding.clone()
                    }
                })
                .collect();
            if let Some(embedder) = &self.inner.embedder {
                let vectors = self
                    .inner
                    .query_embeddings
                    .get_many(
                        embedder.clone(),
                        &namespace,
                        &texts,
                        self.inner.config.recall_embed_timeout,
                    )
                    .await;
                for (index, vector) in positions.into_iter().zip(vectors) {
                    query_vectors[index] = vector;
                }
            }
            performance.mark("query_embedding");
            let fts = fts.await??;
            let mut rankings = Vec::with_capacity(batch.len());
            {
                let index = self
                    .inner
                    .index
                    .lock()
                    .map_err(|_| MemoraError::Retrieval("vector index poisoned".into()))?;
                for ((query, vector), fts_hits) in batch.iter().zip(query_vectors).zip(fts) {
                    let filtered = query.member_id.is_some()
                        || query.tag.is_some()
                        || !query.kinds.is_empty()
                        || !query.domains.is_empty()
                        || query.time_from_ms.is_some()
                        || query.time_to_ms.is_some();
                    let top = if filtered {
                        self.inner.config.hnsw_top_k.saturating_mul(8)
                    } else {
                        self.inner.config.hnsw_top_k
                    };
                    let vector = vector
                        .and_then(|vector| normalize_vector(vector).ok())
                        .filter(|vector| {
                            self.inner
                                .embedder
                                .as_ref()
                                .map_or(true, |embedder| embedder.bind_dimension(vector.len()))
                        });
                    let vector_hits = vector
                        .map(|vector| index.search(&vector, top, self.inner.config.hnsw_ef))
                        .unwrap_or_default();
                    rankings.push(crate::candidate_recall::fuse_candidates(
                        fts_hits,
                        vector_hits,
                        self.inner.config.rrf_k,
                    ));
                }
            }
            performance.mark("local_search");
            let _mutation = self.inner.mutation_lock.lock().await;
            performance.mark("lock_wait");
            self.ensure_open()?;
            let mut ids: Vec<_> = rankings.iter().flatten().map(|(id, _)| *id).collect();
            ids.sort();
            ids.dedup();
            let pool = self.inner.pool.clone();
            let ns = namespace.clone();
            let records = tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                crate::store::get_memories_by_ids(&conn, &ns, &ids)
            })
            .await??;
            let records: HashMap<_, _> = records
                .into_iter()
                .map(|record| (record.id, record))
                .collect();
            for (query, ranking) in batch.iter().zip(rankings) {
                let eligible = records
                    .iter()
                    .filter(|(_, record)| recall_matches(record, query, &namespace))
                    .map(|(id, record)| (*id, record.clone()))
                    .collect();
                output.push(crate::candidate_recall::rank_candidates(
                    ranking,
                    &eligible,
                    &self.inner.config,
                    query.top_k.unwrap_or(self.inner.config.recall_top_k),
                ));
            }
            drop(_mutation);
            performance.mark("load_filter_rank");
            offset = end;
        }
        Ok(output)
    }

    async fn embed_one(&self, id: MemoryId) -> Result<()> {
        let embedder = match &self.inner.embedder {
            Some(e) => e.clone(),
            None => return Ok(()),
        };
        let pool = self.inner.pool.clone();
        let rec = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            get_memory(&conn, id)
        })
        .await??;
        let Some(rec) = rec else {
            return Ok(());
        };
        if rec.state != MemoryState::Active || !rec.embedding_pending {
            return Ok(());
        }
        let text = rec.summary.clone().unwrap_or_else(|| rec.content.clone());
        let vectors =
            tokio::time::timeout(self.inner.config.embedding_timeout, embedder.embed(&[text]))
                .await
                .map_err(|_| MemoraError::Embedding("embedding timed out".into()))??;
        let vector = vectors
            .into_iter()
            .next()
            .ok_or_else(|| MemoraError::Embedding("empty embedding".into()))?;
        let vector = normalize_vector(vector)?;
        if !embedder.bind_dimension(vector.len()) {
            return Err(MemoraError::Embedding(
                "embedding dimension mismatch".into(),
            ));
        }
        let space = embedder.space_id();
        let model = embedder.model_id().to_string();
        let dim = vector.len();
        let blob = vector_to_bytes(&vector);
        // Revalidate after the network wait, and publish DB+index under the
        // foreground mutation lock. An old request cannot overwrite a revision.
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let published = tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let Some(current) = get_memory(&tx, id)? else {
                    return Ok(false);
                };
                if current.state != MemoryState::Active
                    || current.revision != rec.revision
                    || current.embedding_version != rec.embedding_version
                    || !current.embedding_pending
                {
                    return Ok(false);
                }
                let label = next_embedding_label(&tx)?;
                upsert_embedding(&tx, id, label, &model, dim, rec.embedding_version, &blob)?;
                tx.execute(
                    "UPDATE embeddings SET space_id = ?2 WHERE memory_id = ?1",
                    rusqlite::params![id.to_string(), space],
                )?;
                tx.commit()?;
                Ok::<_, MemoraError>(true)
            })
        })
        .await??;
        if published {
            if let Ok(mut idx) = self.inner.index.lock() {
                idx.upsert(id, vector)?;
            }
            self.invalidate_memories(&[]).await;
        }
        Ok(())
    }

    pub async fn retry_graph_projections(&self, limit: usize) -> Result<usize> {
        let mut performance = crate::performance::StageTimer::new("graph_projection");
        if limit == 0 {
            return Ok(0);
        }
        self.ensure_open()?;
        if !self.inner.graph_extractor_configured {
            tracing::warn!("graph extractor unavailable; projection queue not acknowledged");
            return Ok(0);
        }
        // Drain admission acquires mutation before graph, without waiting for
        // graph while holding mutation. Another drain owns graph during its LLM
        // wait; try-lock keeps foreground writes and recalls unblocked.
        let _mutation = self.inner.mutation_lock.lock().await;
        performance.mark("lock_wait");
        self.ensure_open()?;
        let Ok(_drain) = self.inner.graph_retry_lock.try_lock() else {
            return Ok(0);
        };
        let jobs = {
            let pool = self.inner.pool.clone();
            tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                crate::graph::store::due_graph_projection_jobs(&conn, now_ms(), limit)
            })
            .await??
        };
        drop(_mutation);
        performance.mark("queue_load");
        let attempted = jobs.len();
        for job in jobs {
            let (record, roster) = {
                let pool = self.inner.pool.clone();
                let id = job.memory_id;
                let ns = job.namespace.clone();
                tokio::task::spawn_blocking(move || {
                    let conn = pool.get()?;
                    let record = crate::store::get_memory(&conn, id)?;
                    let roster = crate::household::list_members(&conn, &ns)?;
                    Ok::<_, MemoraError>((record, roster))
                })
                .await??
            };
            let Some(record) = record else {
                let pool = self.inner.pool.clone();
                let id = job.memory_id;
                let rev = job.revision;
                tokio::task::spawn_blocking(move || {
                    let conn = pool.get()?;
                    crate::graph::store::delete_graph_projection_job(&conn, id, rev)
                })
                .await??;
                continue;
            };
            if record.namespace != job.namespace
                || record.revision != job.revision
                || record.state != MemoryState::Active
            {
                let pool = self.inner.pool.clone();
                let id = job.memory_id;
                let rev = job.revision;
                tokio::task::spawn_blocking(move || {
                    let conn = pool.get()?;
                    crate::graph::store::delete_graph_projection_job(&conn, id, rev)
                })
                .await??;
                continue;
            }
            let speaker = job
                .speaker_member_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty() && roster.iter().any(|m| m.member_id == *id))
                .map(str::to_string);
            performance.mark("source_load");
            let extraction = self
                .inner
                .graph_extractor
                .extract(GraphExtractInput {
                    text: job.source_text.clone(),
                    roster: roster.clone(),
                    speaker_member_id: speaker,
                    message_at_ms: record
                        .metadata
                        .get("source")
                        .and_then(|source| source.get("message_at_ms"))
                        .and_then(serde_json::Value::as_i64),
                    roster_json: serde_json::to_string(&roster).unwrap_or_else(|_| "[]".into()),
                })
                .await;
            performance.mark("extract_model");
            match extraction {
                Ok(out) => {
                    let pool = self.inner.pool.clone();
                    let ns = job.namespace.clone();
                    let id = job.memory_id;
                    let rev = job.revision;
                    let mut triples = out.triples;
                    let mut facts = out.facts;
                    let entities = out.entities;
                    // Every deferred projection inherits source privacy; retrying
                    // a transient failure must not publish private evidence as shared.
                    if household::is_privacy(&record.metadata) {
                        for triple in &mut triples {
                            triple.privacy = Some(PrivacyLevel::OwnerPrivate);
                        }
                        for fact in &mut facts {
                            fact.privacy = PrivacyLevel::OwnerPrivate;
                        }
                    }
                    let extractor_id = out.extractor_id;
                    let pending = out.pending.len();
                    match tokio::task::spawn_blocking(move || {
                        let mut conn = pool.get()?;
                        let tx = conn
                            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                        // Validate and acknowledge the exact revision in one write
                        // transaction. A stale extraction must never resurrect facts
                        // after revise/forget won the race.
                        let current = get_memory(&tx, id)?;
                        if current.as_ref().is_none_or(|memory| {
                            memory.namespace != ns
                                || memory.revision != rev
                                || memory.state != MemoryState::Active
                        }) {
                            crate::graph::store::delete_graph_projection_job(&tx, id, rev)?;
                            tx.commit()?;
                            return Ok::<_, MemoraError>(());
                        }
                        if !entities.is_empty() || !triples.is_empty() {
                            crate::graph::ingest_graph_batch(
                                &tx,
                                &ns,
                                &roster,
                                &entities,
                                &triples,
                                Some(id),
                            )?;
                        } else {
                            crate::graph::ingest_extracted(
                                &tx,
                                &ns,
                                &roster,
                                &facts,
                                Some(id),
                                &extractor_id,
                                pending,
                            )?;
                        }
                        // Acknowledge the exact revision in the graph write transaction.
                        crate::graph::store::delete_graph_projection_job(&tx, id, rev)?;
                        tx.commit()?;
                        Ok::<_, MemoraError>(())
                    })
                    .await
                    {
                        Ok(Ok(())) => self.invalidate_graph_cache().await,
                        Ok(Err(error)) => {
                            let pool = self.inner.pool.clone();
                            let message = error.to_string();
                            let id = job.memory_id;
                            let rev = job.revision;
                            let _ = tokio::task::spawn_blocking(move || {
                                let conn = pool.get()?;
                                crate::graph::store::fail_graph_projection_job(
                                    &conn, id, rev, &message,
                                )
                            })
                            .await;
                        }
                        Err(error) => {
                            tracing::warn!(error = %error, "graph projection retry join failed");
                        }
                    }
                }
                Err(error) => {
                    let pool = self.inner.pool.clone();
                    let message = error.to_string();
                    let id = job.memory_id;
                    let rev = job.revision;
                    tokio::task::spawn_blocking(move || {
                        let conn = pool.get()?;
                        crate::graph::store::fail_graph_projection_job(&conn, id, rev, &message)
                    })
                    .await??;
                }
            }
            performance.mark("persist_or_retry");
        }
        Ok(attempted)
    }

    pub async fn tick(&self) -> Result<TickReport> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let jieba = self.inner.jieba.clone();
        let pool = self.inner.pool.clone();
        let config = self.inner.config.clone();
        let report =
            tokio::task::spawn_blocking(move || apply_forgetting(&pool, &config, jieba.as_ref()))
                .await??;

        let pool = self.inner.pool.clone();
        let forgotten: Vec<MemoryId> = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let mut stmt = conn.prepare("SELECT id FROM memories WHERE state != 'active'")?;
            let rows = stmt.query_map([], |r| {
                Ok(Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or(Uuid::nil()))
            })?;
            Ok::<_, MemoraError>(rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
        })
        .await??;
        if let Ok(mut idx) = self.inner.index.lock() {
            for id in &forgotten {
                idx.remove(*id);
            }
        }
        self.invalidate_memories(&forgotten).await;
        drop(_mutation);
        if self.inner.embedder.is_some() {
            let pool = self.inner.pool.clone();
            let pending = tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                pending_embedding_ids(&conn)
            })
            .await??;
            for id in pending {
                self.enqueue_embed(id);
            }
        }
        // Durable KG extract retries (network / parse failures).
        if let Err(e) = self.retry_graph_projections(32).await {
            tracing::warn!(error = %e, "graph projection retry failed");
        }
        Ok(report)
    }

    pub fn spawn_maintenance(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        let engine = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                if let Err(e) = engine.tick().await {
                    tracing::warn!(error = %e, "maintenance tick failed");
                }
            }
        })
    }

    /// Periodically run [`Self::dream`] with default options (consolidate + cluster).
    pub fn spawn_dream(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        let engine = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                match engine.dream(DreamOptions::default()).await {
                    Ok(report) => {
                        tracing::info!(
                            consolidated = report.consolidated,
                            archived = report.archived_low_value,
                            merged = report.merged,
                            purged = report.purged,
                            dedup_groups = report.dedup_groups,
                            clusters = report.clusters,
                            patterns = report.patterns,
                            outputs = report.output_ids.len(),
                            "background dream finished"
                        );
                    }
                    Err(e) => tracing::warn!(error = %e, "background dream failed"),
                }
            }
        })
    }

    pub async fn rebuild_index(&self) -> Result<()> {
        let pool = self.inner.pool.clone();
        let config = self.inner.config.clone();
        let embedder = self.inner.embedder.clone();
        let path = self.inner.index.lock().expect("vector index").path();
        let index = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let rows = match embedder {
                Some(e) => load_all_embeddings(&conn, &e.space_id(), e.dimension())?,
                None => Vec::new(),
            };
            Ok::<_, MemoraError>(VectorIndex::rebuild(&config, path, rows))
        })
        .await??;
        *self.inner.index.lock().expect("vector index") = index;
        Ok(())
    }

    pub async fn dream(&self, opts: DreamOptions) -> Result<DreamReport> {
        self.ensure_open()?;
        let _dream = self.inner.dream_lock.lock().await;
        let started = now_ms();
        let ns = self.inner.namespace.clone();
        let consolidate_limit = opts
            .consolidate_limit
            .unwrap_or(self.inner.config.consolidation_batch_size);
        let archive_threshold = self.inner.config.consolidation_archive_importance;

        // Phase 1: drain pending consolidations in batches (one classify call
        // and one transaction per batch) before clustering.
        let batch_size = consolidate_limit.max(1);
        let max_total = self.inner.config.dream_candidate_limit.max(batch_size);
        let mut consolidated = 0usize;
        let mut archived_low_value = 0usize;
        let mut pattern_extras: Vec<(MemoryRecord, String)> = Vec::new();
        let classifier_id = self.inner.classifier.classifier_id().to_string();

        loop {
            if consolidated >= max_total {
                break;
            }
            let take = batch_size.min(max_total - consolidated);
            let pool = self.inner.pool.clone();
            let ns_pending = ns.clone();
            let pending: Vec<MemoryRecord> = tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                pending_consolidation(&conn, &ns_pending, take)
            })
            .await??;
            if pending.is_empty() {
                break;
            }

            let pool_roster = self.inner.pool.clone();
            let ns_roster = ns.clone();
            let roster = tokio::task::spawn_blocking(move || {
                let conn = pool_roster.get()?;
                household::list_members(&conn, &ns_roster)
            })
            .await??;
            let roster_json = household::roster_prompt(&roster);
            let items: Vec<ClassifyItem> = pending
                .iter()
                .map(|rec| ClassifyItem {
                    id: rec.id,
                    content: rec.content.clone(),
                    summary: rec.summary.clone(),
                    kind: rec.kind,
                    entities: rec.entities.clone(),
                    importance: rec.importance,
                    confidence: rec.confidence,
                    roster_json: roster_json.clone(),
                })
                .collect();
            let outputs = self.inner.classifier.classify_batch(&items).await?;
            let by_id: HashMap<MemoryId, MemoryRecord> =
                pending.into_iter().map(|r| (r.id, r)).collect();

            if outputs.is_empty() {
                break;
            }
            // Classification ran outside the write lock. Apply only to the
            // exact revision it saw, and preserve access/embedding fields that
            // may have changed independently while the model was running.
            let _mutation = self.inner.mutation_lock.lock().await;
            self.ensure_open()?;
            let pool = self.inner.pool.clone();
            let jieba = self.inner.jieba.clone();
            let cid = classifier_id.clone();
            let (applied, remove_ids, embed_ids, patterns) = tokio::task::spawn_blocking(move || {
                retry_busy(|| {
                    let mut conn = pool.get()?;
                    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    let mut applied = Vec::new();
                    let mut removed = Vec::new();
                    let mut embeds = Vec::new();
                    let mut patterns = Vec::new();
                    for out in &outputs {
                        let Some(snapshot) = by_id.get(&out.id) else { continue; };
                        let Some(mut rec) = get_memory(&tx, out.id)? else { continue; };
                        if rec.state != MemoryState::Active || rec.revision != snapshot.revision ||
                            !rec.consolidation_pending { continue; }
                        let before = rec.clone();
                        let importance = out.importance.clamp(0.0, 1.0);
                        let archive = out.archive || importance < archive_threshold;
                        rec.kind = out.kind;
                        rec.importance = importance;
                        rec.confidence = out.confidence.clamp(0.0, 1.0);
                        rec.salience = rec.importance * rec.confidence;
                        if let Some(entities) = &out.entities { rec.entities = entities.clone(); }
                        rec.metadata = household::merge_classifier_facets(
                            &rec.metadata, &roster, out.subjects.as_ref(), out.domains.as_ref());
                        rec.subject_ids = household::subject_id_list(&rec.metadata);
                        if let Some(domains) = &out.domains {
                            let facets = serde_json::json!({"domains": domains});
                            if let Some(tag) = household::validated_metadata_tag(&[], &facets) {
                                rec.tag = tag;
                            }
                        }
                        rec.metadata = household::apply_tag(&rec.metadata, &rec.tag);
                        rec.consolidation_pending = false;
                        rec.updated_at_ms = now_ms();
                        if archive {
                            rec.state = MemoryState::Archived;
                            rec.embedding_pending = false;
                            delete_embedding(&tx, rec.id)?;
                            removed.push(rec.id);
                        } else if rec.embedding_pending { embeds.push(rec.id); }
                        update_memory(&tx, &mut rec, jieba.as_ref())?;
                        insert_version(&tx, rec.id, "consolidate", Some(&before), &rec, "dream_job", None)?;
                        append_audit(&tx, Some(rec.id), "consolidate", serde_json::json!({
                            "kind": rec.kind.as_str(), "importance": rec.importance,
                            "confidence": rec.confidence, "archived": archive, "classifier": cid,
                        }))?;
                        applied.push(rec.id);
                        // Do not turn a stale classification into a new pattern.
                        for p in &out.patterns {
                            // Classification may promote the source itself to a pattern.
                            // Repeating its text is not a new fact, regardless of source kind.
                            if !archive && !p.trim().is_empty() && p.trim() != rec.content.trim() {
                                patterns.push((rec.clone(), p.trim().to_string()));
                            }
                        }
                    }
                    tx.commit()?;
                    Ok::<_, MemoraError>((applied, removed, embeds, patterns))
                })
            }).await??;
            if applied.is_empty() {
                break;
            }
            consolidated += applied.len();
            archived_low_value += remove_ids.len();
            if let Ok(mut idx) = self.inner.index.lock() {
                for id in remove_ids {
                    idx.remove(id);
                }
            }
            self.invalidate_memories(&applied).await;
            for id in embed_ids {
                self.enqueue_embed(id);
            }
            for (source, pattern) in patterns {
                if !pattern_extras
                    .iter()
                    .any(|(s, p)| s.id == source.id && *p == pattern)
                {
                    pattern_extras.push((source, pattern));
                }
            }
        }

        for (snapshot, content) in pattern_extras {
            let _mutation = self.inner.mutation_lock.lock().await;
            self.ensure_open()?;
            let pool = self.inner.pool.clone();
            let jieba = self.inner.jieba.clone();
            let model = self
                .inner
                .embedder
                .as_ref()
                .map(|e| e.model_id().to_string());
            let id = Uuid::now_v7();
            let written = tokio::task::spawn_blocking(move || {
                retry_busy(|| {
                    let mut conn = pool.get()?;
                    let tx =
                        conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    let Some(source) = get_memory(&tx, snapshot.id)? else {
                        return Ok(false);
                    };
                    if source.state != MemoryState::Active
                        || source.revision != snapshot.revision
                        || content.trim() == source.content.trim()
                    {
                        return Ok(false);
                    }
                    // Check durable provenance inside the write transaction, not only the
                    // current batch's in-memory extras. Reclassification must be idempotent.
                    let mut expected = source.clone();
                    expected.kind = MemoryKind::Pattern;
                    let mut stmt = tx.prepare(
                        "SELECT m.id FROM memories m JOIN memory_relations r ON r.from_id=m.id
                         WHERE r.to_id=?1 AND r.relation='derived_from' AND m.namespace=?2
                           AND m.state='active' AND m.kind='pattern' AND m.content=?3",
                    )?;
                    let existing: Vec<String> = stmt
                        .query_map(
                            rusqlite::params![source.id.to_string(), source.namespace, content],
                            |row| row.get(0),
                        )?
                        .collect::<std::result::Result<_, _>>()?;
                    for existing_id in existing {
                        if let Some(existing) = get_memory(
                            &tx,
                            Uuid::parse_str(&existing_id)
                                .map_err(|e| MemoraError::InvalidState(e.to_string()))?,
                        )? {
                            if exact_dedup_bucket(&existing, true)
                                == exact_dedup_bucket(&expected, true)
                                && !has_independent_memory_policy(&tx, &existing)?
                                && !has_independent_memory_policy(&tx, &source)?
                            {
                                return Ok(false);
                            }
                        }
                    }
                    drop(stmt);
                    let now = now_ms();
                    let mut rec = source.clone();
                    rec.id = id;
                    rec.revision = 1;
                    rec.kind = MemoryKind::Pattern;
                    rec.content = content.clone();
                    rec.summary = None;
                    rec.importance = 0.7;
                    rec.confidence = 0.8;
                    rec.salience = 0.56;
                    rec.created_at_ms = now;
                    rec.updated_at_ms = now;
                    rec.last_accessed_at_ms = Some(now);
                    rec.next_review_at_ms = None;
                    rec.review_count = 0;
                    rec.source_hash = None;
                    rec.parent_id = None;
                    rec.embedding_model = model.clone();
                    rec.embedding_version = 1;
                    rec.embedding_pending = model.is_some();
                    rec.consolidation_pending = false;
                    rec.metadata["dream"] = serde_json::json!(true);
                    rec.metadata["from_classifier"] = serde_json::json!(true);
                    insert_memory(&tx, &rec, jieba.as_ref())?;
                    insert_relation(&tx, id, source.id, RelationKind::DerivedFrom, 1.0)?;
                    insert_version(&tx, id, "create", None, &rec, "dream_job", None)?;
                    append_audit(
                        &tx,
                        Some(id),
                        "dream_pattern",
                        serde_json::json!({"source": source.id}),
                    )?;
                    tx.commit()?;
                    Ok::<_, MemoraError>(true)
                })
            })
            .await??;
            if written {
                self.invalidate_memories(&[]).await;
                self.enqueue_embed(id);
            }
        }

        // Phase 1.5: exact duplicate links only, never physically delete sources.
        let (merged, dedup_groups, purged) = self.merge_near_duplicates(opts.lookback_ms).await?;

        let limit = opts
            .candidate_limit
            .unwrap_or(self.inner.config.dream_candidate_limit);
        let lookback = opts.lookback_ms;
        let pool = self.inner.pool.clone();
        let ns_cand = ns.clone();
        let candidates: Vec<MemoryRecord> = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let since = now_ms() - lookback;
            let mut stmt = conn.prepare(
                r#"SELECT * FROM memories
                   WHERE state = 'active' AND namespace = ?1 AND kind = 'episodic'
                     AND consolidation_pending = 0
                     AND NOT EXISTS (SELECT 1 FROM memory_relations r
                         JOIN memories derived ON derived.id = r.from_id
                         WHERE r.to_id = memories.id AND r.relation IN ('derived_from', 'summarizes')
                           AND derived.state = 'active')
                     AND (created_at_ms >= ?2 OR COALESCE(last_accessed_at_ms, 0) >= ?2)
                   ORDER BY importance DESC LIMIT ?3"#,
            )?;
            let rows = stmt.query_map(
                rusqlite::params![ns_cand, since, limit as i64],
                crate::store::row_to_memory,
            )?;
            Ok::<_, MemoraError>(rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
        })
        .await??;

        let input_ids: Vec<MemoryId> = candidates.iter().map(|m| m.id).collect();
        let embed_ids = input_ids.clone();
        let pool = self.inner.pool.clone();
        let embeddings: Vec<Option<Vec<f32>>> = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let mut out = Vec::with_capacity(embed_ids.len());
            for id in embed_ids {
                out.push(get_embedding(&conn, id)?.map(|(_, v)| v));
            }
            Ok::<_, MemoraError>(out)
        })
        .await??;
        let mut time_buckets = Vec::with_capacity(candidates.len());
        let mut entities = Vec::with_capacity(candidates.len());
        let bucket_ms = self.inner.config.dream_time_bucket_hours * 3_600_000;
        for rec in &candidates {
            time_buckets.push(if bucket_ms > 0 {
                rec.created_at_ms / bucket_ms
            } else {
                rec.created_at_ms
            });
            entities.push(rec.entities.clone());
        }

        let groups = {
            let mut by_key: std::collections::BTreeMap<String, Vec<usize>> =
                std::collections::BTreeMap::new();
            for (i, rec) in candidates.iter().enumerate() {
                by_key.entry(memory_bucket(rec)).or_default().push(i);
            }
            let mut groups = Vec::new();
            for idxs in by_key.values() {
                let time_buckets_g: Vec<i64> = idxs.iter().map(|i| time_buckets[*i]).collect();
                let entities_g: Vec<Vec<String>> =
                    idxs.iter().map(|i| entities[*i].clone()).collect();
                let embeddings_g: Vec<Option<Vec<f32>>> =
                    idxs.iter().map(|i| embeddings[*i].clone()).collect();
                for local in cluster_indices(
                    &time_buckets_g,
                    &entities_g,
                    &embeddings_g,
                    self.inner.config.entity_jaccard_threshold,
                    self.inner.config.cosine_threshold,
                ) {
                    groups.push(local.into_iter().map(|j| idxs[j]).collect::<Vec<_>>());
                }
            }
            groups
        };
        let min_cluster = self.inner.config.dream_cluster_min;
        let mut output_ids = Vec::new();
        let mut clusters_written = 0usize;

        for group in groups {
            if group.len() < min_cluster {
                continue;
            }
            let members: Vec<&MemoryRecord> = group.iter().map(|i| &candidates[*i]).collect();
            let texts: Vec<String> = members
                .iter()
                .map(|m| m.summary.clone().unwrap_or_else(|| m.content.clone()))
                .collect();
            let ents: Vec<String> = members.iter().flat_map(|m| m.entities.clone()).collect();
            let summary = self
                .inner
                .summarizer
                .summarize(SummarizeInput {
                    texts,
                    entities: ents.clone(),
                })
                .await?
                .summary;
            if summary.trim().is_empty() {
                continue;
            }
            let snapshots: Vec<MemoryRecord> = members.iter().map(|rec| (*rec).clone()).collect();
            let new_id = Uuid::now_v7();
            let archive_sources = opts.archive_sources;
            let model = self
                .inner
                .embedder
                .as_ref()
                .map(|e| e.model_id().to_string());
            let _mutation = self.inner.mutation_lock.lock().await;
            self.ensure_open()?;
            let pool = self.inner.pool.clone();
            let jieba = self.inner.jieba.clone();
            let (written, archived) = tokio::task::spawn_blocking(move || {
                retry_busy(|| {
                    let mut conn = pool.get()?;
                    let tx =
                        conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    let mut sources = Vec::new();
                    for snapshot in &snapshots {
                        let Some(rec) = get_memory(&tx, snapshot.id)? else {
                            return Ok((false, Vec::new()));
                        };
                        if rec.state != MemoryState::Active || rec.revision != snapshot.revision {
                            return Ok((false, Vec::new()));
                        }
                        sources.push(rec);
                    }
                    let now = now_ms();
                    let mut output = sources[0].clone();
                    output.id = new_id;
                    output.revision = 1;
                    output.kind = if sources.len() >= 3 {
                        MemoryKind::Semantic
                    } else {
                        MemoryKind::Summary
                    };
                    output.content = summary.clone();
                    output.summary = Some(summary.clone());
                    output.importance =
                        sources.iter().map(|m| m.importance).sum::<f64>() / sources.len() as f64;
                    output.confidence = 0.75;
                    output.salience = output.importance * output.confidence;
                    output.created_at_ms = now;
                    output.updated_at_ms = now;
                    output.last_accessed_at_ms = Some(now);
                    output.next_review_at_ms = None;
                    output.review_count = 0;
                    output.source_hash = None;
                    output.parent_id = None;
                    output.embedding_model = model.clone();
                    output.embedding_version = 1;
                    output.embedding_pending = model.is_some();
                    output.consolidation_pending = false;
                    output.entities = ents.clone();
                    if !output.metadata.is_object() {
                        output.metadata = serde_json::json!({});
                    }
                    output.metadata["dream"] = serde_json::json!(true);
                    output.metadata["support_count"] = serde_json::json!(sources.len());
                    insert_memory(&tx, &output, jieba.as_ref())?;
                    insert_version(&tx, new_id, "create", None, &output, "dream_job", None)?;
                    append_audit(
                        &tx,
                        Some(new_id),
                        "dream_summary",
                        serde_json::json!({
                            "sources": sources.iter().map(|r| r.id).collect::<Vec<_>>(),
                        }),
                    )?;
                    let mut archived = Vec::new();
                    for mut source in sources {
                        insert_relation(&tx, new_id, source.id, RelationKind::DerivedFrom, 1.0)?;
                        insert_relation(&tx, new_id, source.id, RelationKind::Summarizes, 1.0)?;
                        if archive_sources {
                            let before = source.clone();
                            source.state = MemoryState::Archived;
                            source.metadata["recall_archived"] = serde_json::json!(true);
                            source.embedding_pending = false;
                            source.updated_at_ms = now;
                            update_memory(&tx, &mut source, jieba.as_ref())?;
                            delete_embedding(&tx, source.id)?;
                            insert_version(
                                &tx,
                                source.id,
                                "archive",
                                Some(&before),
                                &source,
                                "dream_job",
                                None,
                            )?;
                            append_audit(
                                &tx,
                                Some(source.id),
                                "dream_archive",
                                serde_json::json!({"by": new_id}),
                            )?;
                            archived.push(source.id);
                        }
                    }
                    tx.commit()?;
                    Ok::<_, MemoraError>((true, archived))
                })
            })
            .await??;
            if written {
                if let Ok(mut idx) = self.inner.index.lock() {
                    for id in &archived {
                        idx.remove(*id);
                    }
                }
                self.invalidate_memories(&archived).await;
                self.enqueue_embed(new_id);
                output_ids.push(new_id);
                clusters_written += 1;
            }
        }

        let pool = self.inner.pool.clone();
        let ns = self.inner.namespace.clone();
        let support = self.inner.config.pattern_support;
        let patterns = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            mature_patterns(&conn, &ns, support)
        })
        .await??;
        let mut pattern_count = 0usize;
        for p in patterns {
            let content = format!(
                "规律：在 {} 周{} {}时经常出现「{}」（支持度 {}）",
                p.timezone,
                p.weekday + 1,
                p.hour_bucket,
                p.entity_key,
                p.count
            );
            let pool = self.inner.pool.clone();
            let ns = self.inner.namespace.clone();
            let entity_key = p.entity_key.clone();
            let timezone = p.timezone.clone();
            let weekday = p.weekday;
            let hour_bucket = p.hour_bucket;
            let existing = tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                find_pattern(&conn, &ns, &entity_key, &timezone, weekday, hour_bucket)
            })
            .await??;
            if let Some(rec) = existing {
                self.revise(
                    rec.id,
                    MemoryUpdate {
                        content: Some(content),
                        reason: Some("pattern_update".into()),
                        ..MemoryUpdate::default()
                    },
                )
                .await?;
                output_ids.push(rec.id);
                pattern_count += 1;
                continue;
            }
            let mut input = RememberInput::episodic(content);
            input.kind = MemoryKind::Pattern;
            input.importance = 0.7;
            input.confidence = 0.8;
            input.namespace = Some(self.inner.namespace.clone());
            input.entities = vec![p.entity_key.clone()];
            input.actor = Some("dream_job".into());
            input.metadata = serde_json::json!({
                "timezone": p.timezone,
                "weekday_mask": p.weekday,
                "time_bucket": p.hour_bucket,
                "periodicity": "weekly",
                "support_count": p.count,
                "last_observed_at_ms": now_ms(),
            });
            let id = self.remember(input).await?;
            output_ids.push(id);
            pattern_count += 1;
        }

        let finished = now_ms();
        let summarizer_id = self.inner.summarizer.summarizer_id().to_string();
        let prompt_version = self.inner.summarizer.prompt_version().to_string();
        let pool = self.inner.pool.clone();
        let in_ids = input_ids.clone();
        let out_ids = output_ids.clone();
        let sid = summarizer_id.clone();
        let stats = serde_json::json!({
            "clusters": clusters_written,
            "patterns": pattern_count,
            "consolidated": consolidated,
            "archived_low_value": archived_low_value,
            "merged": merged,
            "dedup_groups": dedup_groups,
            "purged": purged,
            "classifier_id": classifier_id.clone(),
        });
        let consolidated_audit = consolidated;
        let archived_audit = archived_low_value;
        let merged_audit = merged;
        let dedup_audit = dedup_groups;
        let purged_audit = purged;
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            insert_dream_run(
                &conn,
                started,
                finished,
                "ok",
                &in_ids,
                &out_ids,
                &sid,
                &prompt_version,
                stats,
            )?;
            append_audit(
                &conn,
                None,
                "dream",
                serde_json::json!({
                    "clusters": clusters_written,
                    "patterns": pattern_count,
                    "consolidated": consolidated_audit,
                    "archived_low_value": archived_audit,
                    "merged": merged_audit,
                    "dedup_groups": dedup_audit,
                    "purged": purged_audit,
                }),
            )?;
            Ok::<_, MemoraError>(())
        })
        .await??;

        Ok(DreamReport {
            input_ids,
            output_ids,
            clusters: clusters_written,
            patterns: pattern_count,
            summarizer_id,
            consolidated,
            archived_low_value,
            classifier_id,
            merged,
            dedup_groups,
            purged,
        })
    }

    /// Local-only maintenance: supersede safe exact duplicates, preserving all
    /// records, versions and provenance. Does not call a classifier or summarizer.
    pub async fn deduplicate_exact_memories(&self) -> Result<usize> {
        let (merged, _, _) = self.merge_near_duplicates(0).await?;
        Ok(merged)
    }

    async fn remove_redundant_classifier_derivations(&self) -> Result<Vec<MemoryId>> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let ns = self.inner.namespace.clone();
        let jieba = self.inner.jieba.clone();
        let removed = tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let mut stmt = tx.prepare(
                    "SELECT m.id, r.to_id FROM memories m JOIN memory_relations r ON r.from_id=m.id
                 WHERE m.namespace=?1 AND m.state='active' AND m.kind='pattern'
                   AND m.consolidation_pending=0 AND r.relation='derived_from'
                   AND (SELECT COUNT(*) FROM memory_relations x WHERE x.from_id=m.id
                        AND x.relation IN ('derived_from','summarizes'))=1",
                )?;
                let pairs: Vec<(String, String)> = stmt
                    .query_map([&ns], |r| Ok((r.get(0)?, r.get(1)?)))?
                    .collect::<std::result::Result<_, _>>()?;
                drop(stmt);
                let mut removed = Vec::new();
                for (child_id, source_id) in pairs {
                    let parse = |id: &str| {
                        Uuid::parse_str(id).map_err(|e| MemoraError::InvalidState(e.to_string()))
                    };
                    let Some(mut child) = get_memory(&tx, parse(&child_id)?)? else {
                        continue;
                    };
                    let Some(source) = get_memory(&tx, parse(&source_id)?)? else {
                        continue;
                    };
                    if !is_classifier_pattern(&child)
                        || child.id == source.id
                        || source.state != MemoryState::Active
                        || source.consolidation_pending
                        || child.state != MemoryState::Active
                        || child.content.trim().is_empty()
                        || child.content.trim() != source.content.trim()
                        || exact_dedup_bucket(&child, false) != exact_dedup_bucket(&source, false)
                        || has_independent_memory_policy(&tx, &child)?
                        || has_independent_memory_policy(&tx, &source)?
                        || protected_memory_source(&tx, &child)?
                    {
                        continue;
                    }
                    supersede_exact_duplicate(&tx, &mut child, source.id, false, jieba.as_ref())?;
                    removed.push(child.id);
                }
                tx.commit()?;
                Ok::<_, MemoraError>(removed)
            })
        })
        .await??;
        if !removed.is_empty() {
            if let Ok(mut index) = self.inner.index.lock() {
                for id in &removed {
                    index.remove(*id);
                }
            }
            self.invalidate_memories(&removed).await;
        }
        Ok(removed)
    }

    /// Only exact facts in the same ownership/tag/time/privacy bucket are
    /// deduplicated. Similarity alone cannot justify dropping dates or negations.
    async fn merge_near_duplicates(&self, _lookback_ms: i64) -> Result<(usize, usize, usize)> {
        let redundant = self.remove_redundant_classifier_derivations().await?;
        let ns = self.inner.namespace.clone();
        let pool = self.inner.pool.clone();
        let groups = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            find_exact_content_duplicate_groups(&conn, &ns, 5000)
        })
        .await??;
        let mut merged = redundant.len();
        let mut count = usize::from(!redundant.is_empty());
        for group in groups {
            let mut buckets: std::collections::BTreeMap<String, Vec<MemoryRecord>> =
                std::collections::BTreeMap::new();
            for rec in group {
                buckets
                    .entry(exact_dedup_bucket(&rec, true))
                    .or_default()
                    .push(rec);
            }
            for mut records in buckets.into_values() {
                if records.len() < 2 {
                    continue;
                }
                records.sort_by_key(|rec| std::cmp::Reverse((rec.updated_at_ms, rec.id)));
                let _mutation = self.inner.mutation_lock.lock().await;
                self.ensure_open()?;
                let pool = self.inner.pool.clone();
                let jieba = self.inner.jieba.clone();
                let removed = tokio::task::spawn_blocking(move || {
                    retry_busy(|| {
                        let mut conn = pool.get()?;
                        let tx = conn
                            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                        let mut current = Vec::new();
                        for snapshot in &records {
                            let Some(rec) = get_memory(&tx, snapshot.id)? else {
                                return Ok(Vec::new());
                            };
                            if rec.state != MemoryState::Active || rec.revision != snapshot.revision
                            {
                                return Ok(Vec::new());
                            }
                            if has_independent_memory_policy(&tx, &rec)? {
                                continue;
                            }
                            let protected = protected_memory_source(&tx, &rec)?;
                            current.push((rec, protected));
                        }
                        if current.len() < 2 {
                            return Ok(Vec::new());
                        }
                        // Keep a referenced source first, then the original record rather
                        // than a classifier clone. Never withdraw graph/derived support.
                        current.sort_by_key(|(r, protected)| {
                            (
                                !*protected,
                                is_classifier_pattern(r),
                                std::cmp::Reverse(r.updated_at_ms),
                                r.id,
                            )
                        });
                        let survivor = current[0].0.id;
                        let mut removed = Vec::new();
                        for (mut rec, protected) in current.into_iter().skip(1) {
                            if protected {
                                continue;
                            }
                            supersede_exact_duplicate(
                                &tx,
                                &mut rec,
                                survivor,
                                true,
                                jieba.as_ref(),
                            )?;
                            removed.push(rec.id);
                        }
                        tx.commit()?;
                        Ok::<_, MemoraError>(removed)
                    })
                })
                .await??;
                if !removed.is_empty() {
                    merged += removed.len();
                    count += 1;
                    if let Ok(mut index) = self.inner.index.lock() {
                        for id in &removed {
                            index.remove(*id);
                        }
                    }
                    self.invalidate_memories(&removed).await;
                }
            }
        }
        Ok((merged, count, 0))
    }

    pub fn namespace(&self) -> &str {
        &self.inner.namespace
    }

    pub fn embedder_model(&self) -> Option<&str> {
        self.inner.embedder.as_ref().map(|e| e.model_id())
    }

    pub fn summarizer_id(&self) -> &str {
        self.inner.summarizer.summarizer_id()
    }

    pub fn classifier_id(&self) -> &str {
        self.inner.classifier.classifier_id()
    }

    pub async fn list(&self, query: ListQuery) -> Result<MemoryPage> {
        let ns = query
            .namespace
            .unwrap_or_else(|| self.inner.namespace.clone());
        let state = query.state;
        let kind = query.kind;
        let keyword = query.keyword.clone();
        let member_id = query.member_id.clone();
        let tag = query.tag.clone();
        let limit = query.limit.unwrap_or(10) as i64;
        let offset = query.offset.unwrap_or(0) as i64;
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let (items, total) = list_memories_page(
                &conn,
                &ns,
                state,
                kind,
                keyword.as_deref(),
                member_id.as_deref(),
                tag.as_deref(),
                limit,
                offset,
            )?;
            Ok(MemoryPage { items, total })
        })
        .await?
    }

    pub async fn get(&self, id: MemoryId) -> Result<Option<MemoryRecord>> {
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            get_memory(&conn, id)
        })
        .await?
    }

    pub async fn restore(&self, id: MemoryId) -> Result<()> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let jieba = self.inner.jieba.clone();
        let pending = self.inner.embedder.is_some();
        tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let mut rec =
                    get_memory(&tx, id)?.ok_or_else(|| MemoraError::NotFound(id.to_string()))?;
                if rec.state != MemoryState::Forgotten && rec.state != MemoryState::Archived {
                    return Err(MemoraError::InvalidState(format!(
                        "cannot restore from {}",
                        rec.state.as_str()
                    )));
                }
                let before = rec.clone();
                rec.state = MemoryState::Active;
                rec.updated_at_ms = now_ms();
                rec.embedding_pending = pending;
                rec.embedding_version += 1;
                delete_embedding(&tx, id)?;
                update_memory(&tx, &mut rec, jieba.as_ref())?;
                insert_version(
                    &tx,
                    id,
                    "restore",
                    Some(&before),
                    &rec,
                    "user",
                    Some("restore"),
                )?;
                append_audit(&tx, Some(id), "restore", serde_json::json!({}))?;
                tx.commit()?;
                Ok::<_, MemoraError>(())
            })
        })
        .await??;
        self.invalidate_memories(&[id]).await;
        self.enqueue_embed(id);
        Ok(())
    }

    pub async fn upsert_member(
        &self,
        namespace: Option<&str>,
        member: HouseholdMember,
    ) -> Result<()> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let ns = namespace.unwrap_or(&self.inner.namespace).to_string();
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            household::upsert_member(&conn, &ns, &member)
        })
        .await??;
        // A rename changes recall presentation even when memory text is unchanged.
        self.invalidate_memories(&[]).await;
        Ok(())
    }

    pub async fn list_members(&self, namespace: Option<&str>) -> Result<Vec<HouseholdMember>> {
        let ns = namespace.unwrap_or(&self.inner.namespace).to_string();
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            household::list_members(&conn, &ns)
        })
        .await?
    }

    pub async fn remove_member(&self, namespace: Option<&str>, member_id: &str) -> Result<()> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let ns = namespace.unwrap_or(&self.inner.namespace).to_string();
        let member_id = member_id.to_string();
        let id_for_err = member_id.clone();
        let pool = self.inner.pool.clone();
        let removed = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            household::remove_member(&conn, &ns, &member_id)
        })
        .await??;
        if !removed {
            return Err(MemoraError::NotFound(format!(
                "household member {id_for_err}"
            )));
        }
        self.invalidate_memories(&[]).await;
        Ok(())
    }

    pub async fn list_tags(&self, namespace: Option<&str>) -> Result<Vec<household::MemoryTag>> {
        let ns = namespace.unwrap_or(&self.inner.namespace).to_string();
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            household::ensure_default_tags(&conn, &ns)?;
            household::list_tags(&conn, &ns)
        })
        .await?
    }

    pub async fn upsert_tag(
        &self,
        namespace: Option<&str>,
        tag: household::MemoryTag,
    ) -> Result<()> {
        let ns = namespace.unwrap_or(&self.inner.namespace).to_string();
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            household::upsert_tag(&conn, &ns, &tag)
        })
        .await?
    }

    /// Same as [`MemoryService::explain`], omitting high-sensitivity memories when asked.
    pub async fn explain_visible(
        &self,
        id: MemoryId,
        exclude_sensitive: bool,
    ) -> Result<crate::types::MemoryExplanation> {
        let explanation = MemoryService::explain(self, id).await?;
        if exclude_sensitive && explanation.memory.household_sensitive() {
            return Err(MemoraError::NotFound(id.to_string()));
        }
        Ok(explanation)
    }

    /// Return the knowledge-graph neighborhood for one household member / person.
    pub async fn query_person_graph(
        &self,
        query: crate::graph::PersonGraphQuery,
    ) -> Result<crate::graph::PersonGraph> {
        self.ensure_open()?;
        let ns = query
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::query_person_graph(&conn, &ns, &query)
        })
        .await?
    }

    /// Namespace-wide knowledge-graph snapshot (Web overview; no member required).
    pub async fn query_graph_overview(
        &self,
        query: crate::graph::GraphOverviewQuery,
    ) -> Result<crate::graph::GraphOverview> {
        self.ensure_open()?;
        let ns = query
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::query_graph_overview(&conn, &ns, &query)
        })
        .await?
    }

    /// Explicit graph-only ingest (does not create a memory row). Prefer `remember`
    /// when both memory and graph should be written together.
    pub async fn ingest_graph_text(
        &self,
        text: impl Into<String>,
        member_id: Option<String>,
        namespace: Option<String>,
    ) -> Result<crate::graph::IngestReport> {
        self.ensure_open()?;
        let text = text.into();
        if text.trim().is_empty() {
            return Err(MemoraError::InvalidState("text is required".into()));
        }
        if self.inner.graph_extractor.extractor_id() == "empty-kg-v1" {
            return Ok(crate::graph::IngestReport {
                extractor_id: Some("no-chat".into()),
                messages: vec!["graph extractor not configured; llm_chat_* required".into()],
                ..Default::default()
            });
        }
        let ns = namespace.unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        let ns_roster = ns.clone();
        let roster = tokio::task::spawn_blocking({
            let pool = pool.clone();
            move || {
                let conn = pool.get()?;
                household::list_members(&conn, &ns_roster)
            }
        })
        .await??;
        let member_id = member_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty() && roster.iter().any(|member| member.member_id == *id))
            .map(str::to_string);
        let roster_json = serde_json::to_string(&roster).unwrap_or_else(|_| "[]".into());
        let out = self
            .inner
            .graph_extractor
            .extract(GraphExtractInput {
                text: text.clone(),
                roster: roster.clone(),
                speaker_member_id: member_id.clone(),
                message_at_ms: None,
                roster_json,
            })
            .await?;
        let entities = out.entities;
        let triples = out.triples;
        let facts = out.facts;
        let pending = out.pending.len();
        let extractor_id = out.extractor_id;
        let report = tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let report = if !entities.is_empty() || !triples.is_empty() {
                    let mut report = crate::graph::ingest_graph_batch(
                        &tx, &ns, &roster, &entities, &triples, None,
                    )?;
                    report.pending = report.pending.saturating_add(pending);
                    report.extractor_id = Some(extractor_id.clone());
                    report
                } else {
                    crate::graph::ingest_extracted(
                        &tx,
                        &ns,
                        &roster,
                        &facts,
                        None,
                        &extractor_id,
                        pending,
                    )?
                };
                tx.commit()?;
                Ok::<_, MemoraError>(report)
            })
        })
        .await??;
        self.invalidate_graph_cache().await;
        Ok(report)
    }

    pub async fn grant_access(&self, input: GrantInput) -> Result<crate::graph::KgGrant> {
        self.ensure_open()?;
        let ns = input
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        let grant = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::grant_access(&conn, &ns, &input)
        })
        .await??;
        self.invalidate_graph_cache().await;
        Ok(grant)
    }

    pub async fn revoke_access(&self, input: RevokeGrantInput) -> Result<usize> {
        self.ensure_open()?;
        let ns = input
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        let n = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::revoke_access(&conn, &ns, &input)
        })
        .await??;
        self.invalidate_graph_cache().await;
        Ok(n)
    }

    pub async fn list_grants(
        &self,
        member_id: Option<String>,
        as_grantor: bool,
        namespace: Option<String>,
    ) -> Result<Vec<crate::graph::KgGrant>> {
        self.ensure_open()?;
        let ns = namespace.unwrap_or_else(|| self.inner.namespace.clone());
        let pool = self.inner.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::graph::list_grants(&conn, &ns, member_id.as_deref(), as_grantor)
        })
        .await?
    }
}

impl MemoryEngine {
    /// Persist a memory and its durable graph projection job without awaiting
    /// graph extraction. The job is processed by `retry_graph_projections` / tick.
    /// Compatibility alias for ordinary `remember`, which also defers graph
    /// extraction. `skip_graph` still disables projection altogether.
    pub async fn remember_deferred_graph(&self, input: RememberInput) -> Result<MemoryId> {
        self.remember_with_graph_projection(input).await
    }

    /// Revise and enqueue graph work without awaiting graph extraction. Obsolete
    /// graph support is still withdrawn before returning, so readers cannot use
    /// the previous revision while the new durable projection job is pending.
    /// Compatibility alias for ordinary `revise`, which uses the same deferred path.
    pub async fn revise_deferred_graph(&self, id: MemoryId, update: MemoryUpdate) -> Result<()> {
        self.revise_with_graph_projection(id, update).await
    }

    async fn remember_with_graph_projection(&self, input: RememberInput) -> Result<MemoryId> {
        let started = std::time::Instant::now();
        let _mutation = self.inner.mutation_lock.lock().await;
        let lock_wait_ms = started.elapsed().as_millis() as u64;
        let persist_started = std::time::Instant::now();
        self.ensure_open()?;
        let now = now_ms();
        let id = Uuid::now_v7();
        let ns = input
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let session = input
            .session_id
            .clone()
            .unwrap_or_else(|| DEFAULT_STM_SESSION.to_string());
        let actor = input.actor.clone().unwrap_or_else(|| "user".into());
        if let crate::write_filter::WriteCheck::Reject { reason } =
            crate::write_filter::check_write(&input.content, Some(actor.as_str()))
        {
            return Err(MemoraError::Rejected(reason.as_str().into()));
        }
        let skip_consolidation =
            input.skip_consolidation || matches!(actor.as_str(), "dream_job" | "stm_promote");
        let skip_graph = input
            .metadata
            .get("skip_graph")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || matches!(actor.as_str(), "dream_job" | "stm_promote");
        let pool_roster = self.inner.pool.clone();
        let ns_roster = ns.clone();
        let roster = tokio::task::spawn_blocking(move || {
            let conn = pool_roster.get()?;
            household::list_members(&conn, &ns_roster)
        })
        .await??;
        let metadata = household::annotate(
            &roster,
            &input.content,
            &input.entities,
            input.actor.as_deref(),
            &input.metadata,
            input.member_id.as_deref(),
            &input.domains,
        );
        let pool_tags = self.inner.pool.clone();
        let ns_tags = ns.clone();
        let tag_catalog = tokio::task::spawn_blocking(move || {
            let conn = pool_tags.get()?;
            household::ensure_default_tags(&conn, &ns_tags)?;
            household::list_tags(&conn, &ns_tags)
        })
        .await??;
        // Validate host-curated facets before text inference. Annotation's
        // inferred domains are not explicit host classification.
        let mut requested_facets = input.metadata.clone();
        if requested_facets.get("domains").is_none() && !input.domains.is_empty() {
            requested_facets["domains"] = serde_json::json!(input
                .domains
                .iter()
                .map(|id| serde_json::json!({"id": id, "confidence": 1.0}))
                .collect::<Vec<_>>());
        }
        let tag = household::select_memory_tag(
            &tag_catalog,
            &requested_facets,
            &input.content,
            &input.entities,
        );
        if !tag_catalog.iter().any(|t| t.tag_id == tag) {
            let pool_upsert = self.inner.pool.clone();
            let ns_upsert = ns.clone();
            let tag_row = household::MemoryTag {
                tag_id: tag.clone(),
                display_name: tag.clone(),
                aliases: vec![tag.clone()],
            };
            tokio::task::spawn_blocking(move || {
                let conn = pool_upsert.get()?;
                household::upsert_tag(&conn, &ns_upsert, &tag_row)
            })
            .await??;
        }
        let metadata = household::apply_tag(&metadata, &tag);
        let subject_ids = household::subject_id_list(&metadata);
        let s = stability(
            self.inner.config.s0,
            self.inner.config.alpha,
            self.inner.config.beta,
            input.importance,
            0,
        );
        let rec = MemoryRecord {
            id,
            revision: 1,
            kind: input.kind,
            state: MemoryState::Active,
            content: input.content.clone(),
            summary: input.summary.clone(),
            importance: input.importance,
            confidence: input.confidence,
            salience: input.importance * input.confidence,
            created_at_ms: now,
            updated_at_ms: now,
            last_accessed_at_ms: Some(now),
            next_review_at_ms: Some(now + (next_interval_days(s) * 86_400_000.0) as i64),
            review_count: 0,
            source_hash: input.source_hash.clone(),
            embedding_model: self
                .inner
                .embedder
                .as_ref()
                .map(|e| e.model_id().to_string()),
            embedding_version: 1,
            embedding_pending: self.inner.embedder.is_some(),
            consolidation_pending: !skip_consolidation,
            parent_id: input.parent_id,
            namespace: ns.clone(),
            session_id: Some(session.clone()),
            entities: input.entities.clone(),
            metadata,
            tag,
            subject_ids,
        };

        let pool = self.inner.pool.clone();
        let jieba = self.inner.jieba.clone();
        let rec_clone = rec.clone();
        let ents = rec.entities.clone();
        // Clone before spawn_blocking so `input` remains usable after await.
        let graph_job_speaker = input
            .member_id
            .clone()
            .or_else(|| household::subject_id_list(&rec.metadata).into_iter().next());
        let graph_job_text = input
            .metadata
            .get("graph_text")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| rec.content.clone());
        tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                insert_memory(&tx, &rec_clone, jieba.as_ref())?;
                insert_version(&tx, rec_clone.id, "create", None, &rec_clone, &actor, None)?;
                append_audit(
                    &tx,
                    Some(rec_clone.id),
                    "remember",
                    serde_json::json!({
                        "kind": rec_clone.kind.as_str(),
                        "consolidation_pending": rec_clone.consolidation_pending,
                    }),
                )?;
                let (wd, hour) = weekday_and_hour(now);
                for e in &ents {
                    bump_pattern(&tx, &rec_clone.namespace, e, "UTC", wd, hour)?;
                }
                if !skip_graph {
                    crate::graph::store::enqueue_graph_projection_job(
                        &tx,
                        rec_clone.id,
                        &rec_clone.namespace,
                        rec_clone.revision,
                        &graph_job_text,
                        graph_job_speaker.as_deref(),
                    )?;
                }
                tx.commit()?;
                Ok::<_, MemoraError>(())
            })
        })
        .await??;

        self.invalidate_memories(&[]).await;
        {
            let key = ShortTermMemory::cache_key(
                &ns,
                &session,
                input.stm_key.as_deref().unwrap_or(&id.to_string()),
            );
            let item = StmItem {
                key: key.clone(),
                memory_id: Some(id),
                revision: rec.revision,
                content: input.content.clone(),
                importance: input.importance,
                pinned: input.pin_to_stm,
                kind: input.kind.as_str().to_string(),
                session_id: session,
                namespace: ns.clone(),
                entities: input.entities.clone(),
                metadata: rec.metadata.clone(),
            };
            self.inner.stm.put_event(item.clone()).await;
            if input.pin_to_stm {
                self.inner.stm.put_fact(item).await;
            }
        }

        self.enqueue_embed(id);

        // The durable job was committed with the memory. Never run graph LLM
        // work in a foreground write or while holding the mutation lock.
        let persist_ms = persist_started.elapsed().as_millis() as u64;
        drop(_mutation);
        tracing::debug!(
            operation = "remember",
            lock_wait_ms,
            persist_ms,
            total_ms = started.elapsed().as_millis() as u64,
            graph_queued = !skip_graph,
            "memory foreground write finished"
        );
        if !skip_graph {
            self.wake_graph_projection_worker();
        }
        Ok(id)
    }

    async fn revise_with_graph_projection(&self, id: MemoryId, update: MemoryUpdate) -> Result<()> {
        let started = std::time::Instant::now();
        let _mutation = self.inner.mutation_lock.lock().await;
        let lock_wait_ms = started.elapsed().as_millis() as u64;
        let persist_started = std::time::Instant::now();
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let jieba = self.inner.jieba.clone();
        let pending = self.inner.embedder.is_some();
        let content_changed = update.content.is_some();
        let (affected, graph_inputs_changed, graph_work_queued) =
            tokio::task::spawn_blocking(move || {
                retry_busy(|| {
                    let mut conn = pool.get()?;
                    let tx =
                        conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    let mut rec = get_memory(&tx, id)?
                        .ok_or_else(|| MemoraError::NotFound(id.to_string()))?;
                    if rec.state != MemoryState::Active {
                        return Err(MemoraError::InvalidState(
                            "revise requires active memory".into(),
                        ));
                    }
                    let mut affected = invalidate_derivatives(&tx, id, jieba.as_ref())?;
                    affected.push(id);
                    if update.expected_revision.is_some_and(|r| r != rec.revision) {
                        return Err(MemoraError::InvalidState("stale revise request".into()));
                    }
                    let before = rec.clone();
                    if let Some(c) = update.content.clone() {
                        if let crate::write_filter::WriteCheck::Reject { reason } =
                            crate::write_filter::check_write(&c, Some("user"))
                        {
                            return Err(MemoraError::Rejected(reason.as_str().into()));
                        }
                        rec.content = c;
                        // Stale summary must not keep driving embeddings / context.
                        if update.summary.is_none() {
                            rec.summary = None;
                        }
                    }
                    if let Some(s) = update.summary.clone() {
                        rec.summary = Some(s);
                    }
                    if let Some(i) = update.importance {
                        rec.importance = i;
                    }
                    if let Some(c) = update.confidence {
                        rec.confidence = c;
                    }
                    if let Some(m) = update.metadata.clone() {
                        // Merge host metadata (e.g. manual privacy) onto existing keys.
                        let mut merged = rec.metadata.as_object().cloned().unwrap_or_default();
                        if let Some(obj) = m.as_object() {
                            for (k, v) in obj {
                                merged.insert(k.clone(), v.clone());
                            }
                        }
                        rec.metadata = serde_json::Value::Object(merged);
                        rec.subject_ids = household::subject_id_list(&rec.metadata);
                    }
                    if let Some(e) = update.entities.clone() {
                        rec.entities = e;
                    }
                    // Preserve validated curated tags and accept explicit reclassification.
                    if content_changed || rec.tag.trim().is_empty() || update.metadata.is_some() {
                        let ns_tag = rec.namespace.clone();
                        household::ensure_default_tags(&tx, &ns_tag)?;
                        let catalog = household::list_tags(&tx, &ns_tag)?;
                        let explicit_patch = update.metadata.as_ref().filter(|patch| {
                            patch.get("tag").is_some() || patch.get("domains").is_some()
                        });
                        let tag = household::select_memory_tag(
                            &catalog,
                            explicit_patch.unwrap_or(&rec.metadata),
                            &rec.content,
                            &rec.entities,
                        );
                        if !catalog.iter().any(|t| t.tag_id == tag) {
                            household::upsert_tag(
                                &tx,
                                &ns_tag,
                                &household::MemoryTag {
                                    tag_id: tag.clone(),
                                    display_name: tag.clone(),
                                    aliases: vec![tag.clone()],
                                },
                            )?;
                        }
                        rec.metadata = household::apply_tag(&rec.metadata, &tag);
                        rec.tag = tag;
                        rec.subject_ids = household::subject_id_list(&rec.metadata);
                    } else if update.metadata.is_some() {
                        // Privacy-only revise: keep tag, refresh sensitivity/privacy attrs.
                        let tag = rec.tag.clone();
                        rec.metadata = household::apply_tag(&rec.metadata, &tag);
                    }
                    rec.updated_at_ms = now_ms();
                    rec.embedding_pending = pending;
                    rec.embedding_version += 1;
                    delete_embedding(&tx, id)?;
                    update_memory(&tx, &mut rec, jieba.as_ref())?;
                    insert_version(
                        &tx,
                        id,
                        "update",
                        Some(&before),
                        &rec,
                        "user",
                        update.reason.as_deref(),
                    )?;
                    for cid in &update.contradicts {
                        insert_relation(&tx, id, *cid, RelationKind::Contradicts, 1.0)?;
                        if let Some(mut old) = get_memory(&tx, *cid)? {
                            old.confidence *= 0.5;
                            old.updated_at_ms = now_ms();
                            update_memory(&tx, &mut old, jieba.as_ref())?;
                        }
                    }
                    append_audit(
                        &tx,
                        Some(id),
                        "revise",
                        serde_json::json!({"reason": update.reason}),
                    )?;
                    let mut after_subjects = household::subject_id_list(&rec.metadata);
                    let mut before_subjects = before.subject_ids.clone();
                    before_subjects.sort();
                    after_subjects.sort();
                    let graph_inputs_changed = content_changed
                        || before.metadata.get("graph_text") != rec.metadata.get("graph_text")
                        || before.metadata.get("source") != rec.metadata.get("source")
                        || household::is_privacy(&before.metadata)
                            != household::is_privacy(&rec.metadata)
                        || before_subjects != after_subjects
                        || before.metadata.get("skip_graph") != rec.metadata.get("skip_graph");
                    // Even a metadata-only revise advances the memory revision. Rebase
                    // an existing pending job, otherwise its exact-revision fence would
                    // discard the only projection after the old extraction finishes.
                    let pending_graph_job: Option<(String, Option<String>)> = {
                        use rusqlite::OptionalExtension;
                        tx.query_row(
                            "SELECT source_text,speaker_member_id FROM graph_projection_jobs WHERE memory_id=?1",
                            [id.to_string()], |row| Ok((row.get(0)?, row.get(1)?)),
                        ).optional()?
                    };
                    let graph_work_queued = graph_inputs_changed || pending_graph_job.is_some();
                    let graph_job_revision = rec.revision;
                    if graph_inputs_changed {
                        // Atomic withdrawal: a new worker result cannot be removed by
                        // a later post-commit withdrawal from this foreground revise.
                        crate::graph::store::withdraw_facts_by_source_memory(
                            &tx,
                            &rec.namespace,
                            id,
                        )?;
                    }
                    if graph_work_queued {
                        crate::graph::store::delete_graph_projection_job_for_memory(&tx, id)?;
                        if !rec.content.trim().is_empty()
                            && rec
                                .metadata
                                .get("skip_graph")
                                .and_then(serde_json::Value::as_bool)
                                != Some(true)
                        {
                            let speaker =
                                household::subject_id_list(&rec.metadata).into_iter().next();
                            let source_text = rec
                                .metadata
                                .get("graph_text")
                                .and_then(|value| value.as_str())
                                .map(str::trim)
                                .filter(|text| !text.is_empty())
                                .unwrap_or(&rec.content);
                            // A score-only revision changes no graph input. Keep
                            // the original evidence and host speaker, not owner-as-speaker.
                            let (source_text, speaker) = if !graph_inputs_changed {
                                pending_graph_job.as_ref().map(|(text, speaker)|
                                    (text.as_str(), speaker.as_deref()))
                                    .unwrap_or((source_text, speaker.as_deref()))
                            } else {
                                (source_text, speaker.as_deref())
                            };
                            crate::graph::store::enqueue_graph_projection_job(
                                &tx,
                                id,
                                &rec.namespace,
                                graph_job_revision,
                                source_text,
                                speaker,
                            )?;
                        }
                    }
                    tx.commit()?;
                    Ok::<_, MemoraError>((affected, graph_inputs_changed, graph_work_queued))
                })
            })
            .await??;
        if let Ok(mut idx) = self.inner.index.lock() {
            for id in &affected {
                idx.remove(*id);
            }
        }
        self.invalidate_memories(&affected).await;
        self.enqueue_embed(id);

        // Withdrawal and requeue are atomic with the revised record. A worker
        // finishing after this return can only publish that exact new revision.
        let persist_ms = persist_started.elapsed().as_millis() as u64;
        drop(_mutation);
        tracing::debug!(
            operation = "revise",
            lock_wait_ms,
            persist_ms,
            total_ms = started.elapsed().as_millis() as u64,
            graph_changed = graph_inputs_changed,
            "memory foreground write finished"
        );
        if graph_work_queued {
            self.wake_graph_projection_worker();
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl MemoryService for MemoryEngine {
    async fn remember(&self, input: RememberInput) -> Result<MemoryId> {
        self.remember_with_graph_projection(input).await
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallResult> {
        let mut performance = crate::performance::StageTimer::new("recall");
        self.ensure_open()?;
        let ns = query
            .namespace
            .clone()
            .unwrap_or_else(|| self.inner.namespace.clone());
        let epoch = self
            .inner
            .data_epoch
            .load(std::sync::atomic::Ordering::SeqCst);
        let cache_key = format!("{}|{}|{}", ns, epoch, serde_json::to_string(&query)?);
        // Graph visibility can change with time alone (grant expiry), so
        // only memory-only queries use the result cache.
        if query.query_embedding.is_none() && query.skip_graph {
            if let Some(hit) = self.inner.stm.query_cache.get(&cache_key).await {
                let _mutation = self.inner.mutation_lock.lock().await;
                self.ensure_open()?;
                if self
                    .inner
                    .data_epoch
                    .load(std::sync::atomic::Ordering::SeqCst)
                    == epoch
                {
                    return Ok(hit);
                }
            }
        }

        let stm_items = self
            .inner
            .stm
            .matching_facts(&ns, query.session_id.as_deref(), &query.text)
            .await;
        // Linked STM must be resolved against the durable lifecycle state.
        let pool = self.inner.pool.clone();
        let stm_items = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let mut valid = Vec::new();
            for mut item in stm_items {
                if let Some(id) = item.memory_id {
                    let Some(rec) = get_memory(&conn, id)? else {
                        continue;
                    };
                    if rec.state != MemoryState::Active {
                        continue;
                    }
                    item.content = rec.content;
                    item.importance = rec.importance;
                    item.metadata = rec.metadata;
                    item.revision = rec.revision;
                }
                valid.push(item);
            }
            Ok::<_, MemoraError>(valid)
        })
        .await??;
        let mut stm_hits: Vec<MemoryRecord> = stm_items
            .into_iter()
            .map(|item| MemoryRecord {
                id: item.memory_id.unwrap_or_else(Uuid::now_v7),
                revision: item.revision,
                kind: MemoryKind::Episodic,
                state: MemoryState::Active,
                content: item.content,
                summary: None,
                importance: item.importance,
                confidence: 1.0,
                salience: item.importance,
                created_at_ms: now_ms(),
                updated_at_ms: now_ms(),
                last_accessed_at_ms: Some(now_ms()),
                next_review_at_ms: None,
                review_count: 0,
                source_hash: None,
                embedding_model: None,
                embedding_version: 1,
                embedding_pending: false,
                consolidation_pending: false,
                parent_id: None,
                namespace: item.namespace,
                session_id: Some(item.session_id),
                entities: item.entities,
                metadata: item.metadata.clone(),
                tag: household::memory_tag(&item.metadata).to_string(),
                subject_ids: household::subject_id_list(&item.metadata),
            })
            .filter(|rec| recall_matches(rec, &query, &ns))
            .collect();

        performance.mark("stm");
        let jieba = self.inner.jieba.clone();
        // 查询改写：词面扩展进 FTS；问句变体用于多路语义召回。
        let rewrite = crate::query_rewrite::rewrite_plan(&query.text);
        let fts_q = crate::core::tokenize::fts_match_query_with(
            jieba.as_ref(),
            &query.text,
            &rewrite.extra_words,
        );
        let pool = self.inner.pool.clone();
        let top_k = self.inner.config.fts_top_k;
        let ns_fts = ns.clone();
        let fts_query = query.clone();
        let fts_fut = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            crate::store::fts_search(&conn, &fts_q, &ns_fts, top_k, &fts_query)
        });

        let mut query_embs_raw: Vec<Vec<f32>> = Vec::new();
        let mut semantic_ok = query.fts_only || self.inner.embedder.is_none();
        if !query.fts_only {
            if let Some(embedder) = self.inner.embedder.clone() {
                // 调用方自带向量时语义路完全由调用方控制，不做改写。
                if let Some(emb) = query.query_embedding.clone() {
                    query_embs_raw.push(emb);
                    semantic_ok = true;
                } else if !query.text.is_empty() {
                    let timeout = self.inner.config.recall_embed_timeout;
                    let mut any = false;
                    // 原句 + 问句变体多路召回；QueryEmbeddingCache 按 text 缓存，
                    // 变体在重复查询时几乎零成本。
                    for text in rewrite.query_texts(&query.text) {
                        if let Some(emb) = self
                            .inner
                            .query_embeddings
                            .get(embedder.clone(), &ns, text, timeout)
                            .await
                        {
                            query_embs_raw.push(emb);
                            any = true;
                        }
                    }
                    semantic_ok = any;
                    if !any {
                        tracing::debug!(
                            "recall embedding unavailable within budget; use local FTS"
                        );
                    }
                }
            }
        }
        performance.mark("query_embedding");
        let filtered = query.member_id.is_some()
            || query.tag.is_some()
            || !query.kinds.is_empty()
            || !query.domains.is_empty()
            || query.time_from_ms.is_some()
            || query.time_to_ms.is_some();
        let hnsw_top = if filtered {
            self.inner.config.hnsw_top_k.saturating_mul(8)
        } else {
            self.inner.config.hnsw_top_k
        };
        let ef = self.inner.config.hnsw_ef;
        let mut query_embs: Vec<Vec<f32>> = Vec::new();
        for emb in query_embs_raw {
            match normalize_vector(emb) {
                Ok(vector) => query_embs.push(vector),
                Err(_) => {
                    semantic_ok = false;
                }
            }
        }
        query_embs.retain(|v| {
            let compatible = self
                .inner
                .embedder
                .as_ref()
                .map_or(true, |e| e.bind_dimension(v.len()));
            if !compatible {
                semantic_ok = false;
            }
            compatible
        });
        let vec_hits = if query_embs.is_empty() {
            Vec::new()
        } else {
            // 多路（原句 + 变体）召回合并：同一记忆取最小 L2 距离后统一排名。
            let mut best: HashMap<MemoryId, f32> = HashMap::new();
            if let Ok(idx) = self.inner.index.lock() {
                for emb in &query_embs {
                    for (id, dist) in idx.search(emb, hnsw_top, ef) {
                        best.entry(id)
                            .and_modify(|d| {
                                if dist < *d {
                                    *d = dist;
                                }
                            })
                            .or_insert(dist);
                    }
                }
            }
            let mut merged: Vec<(MemoryId, f32)> = best.into_iter().collect();
            merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            merged
        };

        let fts_hits = fts_fut.await??;
        performance.mark("local_search");
        let k = self.inner.config.rrf_k;
        let mut fused: HashMap<MemoryId, ScoreBreakdown> = HashMap::new();
        for (rank, (id, score)) in fts_hits.into_iter().enumerate() {
            let e = fused.entry(id).or_insert(ScoreBreakdown {
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
            });
            e.bm25_rank = Some(rank);
            e.bm25_score = Some(score);
            e.rrf += rrf(rank, k);
        }
        for (rank, (id, dist)) in vec_hits.into_iter().enumerate() {
            let e = fused.entry(id).or_insert(ScoreBreakdown {
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
            });
            e.vector_rank = Some(rank);
            e.l2 = Some(dist);
            e.rrf += rrf(rank, k);
        }

        let mut ranked: Vec<(MemoryId, ScoreBreakdown)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.rrf
                .partial_cmp(&a.1.rrf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Filter all retrieved candidates before applying the fusion cap.
        performance.mark("fusion");
        let _mutation = self.inner.mutation_lock.lock().await;
        performance.mark("lock_wait");
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let ids: Vec<MemoryId> = ranked.iter().map(|(id, _)| *id).collect();
        let records: HashMap<MemoryId, MemoryRecord> = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let mut map = HashMap::new();
            for id in ids {
                if let Some(rec) = get_memory(&conn, id)? {
                    map.insert(id, rec);
                }
            }
            Ok::<_, MemoraError>(map)
        })
        .await??;

        ranked.retain(|(id, _)| {
            records
                .get(id)
                .is_some_and(|r| recall_matches(r, &query, &ns))
        });
        ranked.truncate(self.inner.config.rrf_merge_k.max(query.top_k.unwrap_or(0)));
        let now = now_ms();
        let kinds = query.kinds.clone();
        let mut hits = Vec::new();
        for (id, mut scores) in ranked {
            let Some(rec) = records.get(&id) else {
                continue;
            };
            if !crate::store::memory_is_recallable(rec) {
                continue;
            }
            if rec.namespace != ns {
                continue;
            }
            if !recall_matches(rec, &query, &ns) {
                continue;
            }
            if !kinds.is_empty() && !kinds.contains(&rec.kind) {
                continue;
            }
            if let Some(from) = query.time_from_ms {
                if rec.created_at_ms < from {
                    continue;
                }
            }
            if let Some(to) = query.time_to_ms {
                if rec.created_at_ms > to {
                    continue;
                }
            }
            let last = rec.last_accessed_at_ms.unwrap_or(rec.created_at_ms);
            let s = stability(
                self.inner.config.s0,
                self.inner.config.alpha,
                self.inner.config.beta,
                rec.importance,
                rec.review_count,
            );
            scores.decay = retention(days_between(last, now), s);
            scores.confidence = rec.confidence;
            scores.importance = rec.importance;
            scores.access_boost = access_boost(rec.review_count);
            scores.final_score = scores.rrf
                * scores.confidence
                * scores.decay
                * scores.access_boost
                * rec.importance.max(0.01);
            hits.push(RecalledMemory {
                memory: rec.clone(),
                scores,
                from_stm: false,
                display_text: String::new(),
            });
        }
        hits.sort_by(|a, b| {
            b.scores
                .final_score
                .partial_cmp(&a.scores.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = query.top_k.unwrap_or(self.inner.config.recall_top_k);
        hits.truncate(top);

        for h in &hits {
            self.inner
                .last_scores
                .insert(h.memory.id, h.scores.clone())
                .await;
        }
        let touch_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory.id).collect();
        if !touch_ids.is_empty() {
            let pool = self.inner.pool.clone();
            let ts = now;
            tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                touch_memories(&conn, &touch_ids, ts)
            })
            .await??;
        }

        let pool = self.inner.pool.clone();
        let stm_query = query.clone();
        let stm_ns = ns.clone();
        stm_hits = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let mut valid = Vec::new();
            for old in stm_hits {
                if let Some(rec) = get_memory(&conn, old.id)? {
                    if recall_matches(&rec, &stm_query, &stm_ns) {
                        valid.push(rec);
                    }
                }
            }
            Ok::<_, MemoraError>(valid)
        })
        .await??;
        performance.mark("memory_rank_and_touch");
        // Use one namespace-local roster snapshot for person anchors and snippets.
        let roster = self.list_members(Some(&ns)).await?;
        // Second recall path: structured knowledge-graph facts.
        let graph_hits = if query.skip_graph {
            Vec::new()
        } else {
            let pool = self.inner.pool.clone();
            let ns_g = ns.clone();
            let graph_query = query.clone();
            let roster_g = roster.clone();
            let limit = query.top_k.unwrap_or(self.inner.config.recall_top_k);
            tokio::task::spawn_blocking(move || {
                let conn = pool.get()?;
                let candidates = crate::graph::service::collect_recall_graph_with_names(
                    &conn,
                    &ns_g,
                    &graph_query.text,
                    graph_query.member_id.as_deref(),
                    &graph_query.domains,
                    graph_query.viewer_member_id.as_deref(),
                    &roster_g,
                )?;
                let mut filtered = Vec::new();
                for mut hit in candidates {
                    if graph_matches_query(&conn, &ns_g, &mut hit.fact, &graph_query)? {
                        filtered.push(hit);
                    }
                }
                filtered.truncate(limit);
                Ok::<_, MemoraError>(filtered)
            })
            .await??
        };

        performance.mark("graph_search");
        for hit in &mut hits {
            hit.display_text = crate::recall_display::render_memory_snippet(&hit.memory, &roster);
        }
        let context = build_context_pack(
            &stm_hits,
            &hits,
            &graph_hits,
            &self.inner.pool,
            &query,
            &ns,
            &roster,
            self.inner.config.token_budget,
        )?;

        performance.mark("presentation");
        let result = RecallResult {
            hits,
            stm_hits,
            context,
            graph_hits,
        };
        if semantic_ok && query.skip_graph {
            self.inner
                .stm
                .query_cache
                .insert(cache_key, result.clone())
                .await;
        }
        Ok(result)
    }

    async fn revise(&self, id: MemoryId, update: MemoryUpdate) -> Result<()> {
        self.revise_with_graph_projection(id, update).await
    }

    async fn forget(&self, id: MemoryId, reason: ForgetReason) -> Result<()> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let jieba = self.inner.jieba.clone();
        let purge = matches!(reason, ForgetReason::Purge);
        let (affected, graph_ns) = tokio::task::spawn_blocking(move || {
            retry_busy(|| {
                let mut conn = pool.get()?;
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let mut rec =
                    get_memory(&tx, id)?.ok_or_else(|| MemoraError::NotFound(id.to_string()))?;
                let graph_ns = rec.namespace.clone();
                let mut affected = Vec::new();
                crate::graph::store::delete_graph_projection_job_for_memory(&tx, id)?;
                if purge {
                    if has_relations(&tx, id)? {
                        return Err(MemoraError::InvalidState(
                            "cannot purge memory that still has relations".into(),
                        ));
                    }
                    delete_memory(&tx, id)?;
                    append_audit(
                        &tx,
                        Some(id),
                        "purge",
                        serde_json::json!({"reason": format!("{reason:?}")}),
                    )?;
                } else {
                    affected = invalidate_derivatives(&tx, id, jieba.as_ref())?;
                    rec.state = MemoryState::Forgotten;
                    rec.embedding_pending = false;
                    rec.consolidation_pending = false;
                    rec.updated_at_ms = now_ms();
                    update_memory(&tx, &mut rec, jieba.as_ref())?;
                    remove_fts(&tx, id)?;
                    delete_embedding(&tx, id)?;
                    insert_version(&tx, id, "forget", None, &rec, "user", Some("forget"))?;
                    append_audit(
                        &tx,
                        Some(id),
                        "forget",
                        serde_json::json!({"reason": format!("{reason:?}")}),
                    )?;
                }
                affected.push(id);
                tx.commit()?;
                Ok::<_, MemoraError>((affected, graph_ns))
            })
        })
        .await??;
        if let Ok(mut idx) = self.inner.index.lock() {
            for id in &affected {
                idx.remove(*id);
            }
        }
        self.invalidate_memories(&affected).await;
        let _ = self.withdraw_graph_for_memory(id, &graph_ns).await?;
        Ok(())
    }

    async fn explain(&self, id: MemoryId) -> Result<MemoryExplanation> {
        let pool = self.inner.pool.clone();
        let last = self.inner.last_scores.get(&id).await;
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            let memory =
                get_memory(&conn, id)?.ok_or_else(|| MemoraError::NotFound(id.to_string()))?;
            Ok(MemoryExplanation {
                memory,
                versions: list_versions(&conn, id)?,
                relations: list_relations(&conn, id)?,
                recent_audit: list_audit(&conn, id, 50)?,
                last_scores: last,
            })
        })
        .await?
    }

    async fn export(&self, req: ExportRequest) -> Result<Vec<u8>> {
        let pool = self.inner.pool.clone();
        let ns = self.inner.namespace.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            export_archive(&conn, &req, &ns)
        })
        .await?
    }

    async fn import(&self, archive: &[u8], mode: ImportMode) -> Result<ImportReport> {
        let _mutation = self.inner.mutation_lock.lock().await;
        self.ensure_open()?;
        let pool = self.inner.pool.clone();
        let jieba = self.inner.jieba.clone();
        let config = self.inner.config.clone();
        let ns = self.inner.namespace.clone();
        let bytes = archive.to_vec();
        let report = tokio::task::spawn_blocking(move || {
            let conn = pool.get()?;
            import_archive(&conn, jieba.as_ref(), &bytes, mode, &config, &ns)
        })
        .await??;
        if mode != ImportMode::DryRun {
            self.inner.stm.session_events.invalidate_all();
            self.inner.stm.working_facts.invalidate_all();
            self.invalidate_memories(&[]).await;
            if let Some(embedder) = &self.inner.embedder {
                let pool = self.inner.pool.clone();
                let embedder = embedder.clone();
                tokio::task::spawn_blocking(move || {
                    let mut conn = pool.get()?;
                    prepare_vector_space(
                        &mut conn,
                        &embedder.space_id(),
                        embedder.model_id(),
                        embedder.dimension(),
                    )
                })
                .await??;
                let pool = self.inner.pool.clone();
                let pending: Vec<MemoryId> = tokio::task::spawn_blocking(move || {
                    let conn = pool.get()?;
                    pending_embedding_ids(&conn)
                })
                .await??;
                for id in pending {
                    self.enqueue_embed(id);
                }
            }
            self.rebuild_index().await?;
        }
        Ok(report)
    }
}

fn build_context_pack(
    stm_hits: &[MemoryRecord],
    hits: &[RecalledMemory],
    graph_hits: &[crate::graph::RecalledGraphFact],
    pool: &Pool,
    query: &RecallQuery,
    namespace: &str,
    roster: &[HouseholdMember],
    budget: usize,
) -> Result<ContextPack> {
    let mut items = Vec::new();
    let mut used = 0usize;
    let push =
        |items: &mut Vec<ContextItem>, used: &mut usize, rec: &MemoryRecord, role: ContextRole| {
            if *used >= budget {
                return;
            }
            let text = crate::recall_display::render_memory_snippet(rec, roster);
            let tokens = estimate_tokens(&text);
            if *used + tokens > budget {
                return;
            }
            *used += tokens;
            items.push(ContextItem {
                memory_id: rec.id,
                role,
                text,
                tokens,
            });
        };
    for rec in stm_hits {
        push(&mut items, &mut used, rec, ContextRole::StmFact);
    }
    for hit in hits {
        let role = match hit.memory.kind {
            MemoryKind::Summary | MemoryKind::Semantic | MemoryKind::Pattern => {
                ContextRole::Summary
            }
            MemoryKind::Episodic => ContextRole::Atomic,
        };
        push(&mut items, &mut used, &hit.memory, role);
    }
    for g in graph_hits {
        if used >= budget {
            break;
        }
        let tokens = estimate_tokens(&g.display_text);
        if used + tokens > budget {
            continue;
        }
        used += tokens;
        items.push(ContextItem {
            memory_id: g.fact.source_memory_id.unwrap_or(g.fact.id),
            role: ContextRole::GraphFact,
            text: g.display_text.clone(),
            tokens,
        });
    }
    if used < budget {
        if let Ok(conn) = pool.get() {
            for hit in hits {
                if used >= budget {
                    break;
                }
                if let Ok(rels) = list_relations(&conn, hit.memory.id) {
                    for rel in rels {
                        if rel.relation != RelationKind::DerivedFrom
                            && rel.relation != RelationKind::Summarizes
                        {
                            continue;
                        }
                        if let Ok(Some(src)) = get_memory(&conn, rel.to_id) {
                            // Expansion is not a privilege bypass. Archived
                            // supporting atoms are eligible here, but every
                            // other RecallQuery restriction remains in force.
                            let mut eligible = src.clone();
                            if eligible.state == MemoryState::Archived {
                                eligible.state = MemoryState::Active;
                            }
                            if recall_matches(&eligible, query, namespace) {
                                push(&mut items, &mut used, &src, ContextRole::ExpandedSource);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ContextPack {
        estimated_tokens: used,
        budget,
        items,
    })
}

fn is_classifier_pattern(rec: &MemoryRecord) -> bool {
    rec.kind == MemoryKind::Pattern
        && rec
            .metadata
            .get("dream")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        && rec
            .metadata
            .get("from_classifier")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
}

/// Conservatively compare complete metadata, not a hand-picked subset of ACL
/// fields. Only the two provenance markers written by this classifier are
/// ignored. Unknown reader/owner/validity fields therefore remain boundaries.
fn exact_dedup_bucket(rec: &MemoryRecord, include_kind: bool) -> String {
    let mut metadata = rec.metadata.clone();
    if let Some(obj) = metadata.as_object_mut() {
        obj.remove("dream");
        obj.remove("from_classifier");
    }
    let mut subjects = rec.subject_ids.clone();
    subjects.sort();
    subjects.dedup();
    serde_json::json!([
        rec.namespace,
        subjects,
        rec.tag,
        include_kind.then(|| rec.kind.as_str()),
        metadata,
    ])
    .to_string()
}

/// Graph authorization is attached to resources, not just memory metadata.
/// Until an explicit resource migration is implemented, do not merge records
/// with independently registered ownership/domains or grants tied to their ID.
fn has_independent_memory_policy(conn: &rusqlite::Connection, rec: &MemoryRecord) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM kg_resource_owners WHERE namespace=?1 AND resource_id=?2)
          OR EXISTS(SELECT 1 FROM kg_resource_domains WHERE namespace=?1 AND resource_id=?2)
          OR EXISTS(SELECT 1 FROM kg_grants WHERE namespace=?1 AND instr(scope_json, ?2)>0)",
        rusqlite::params![rec.namespace, rec.id.to_string()],
        |r| r.get(0),
    )?)
}

/// Superseded memories cannot support graph facts (graph::policy). Preserve
/// every referenced source instead of silently changing graph/grant evidence.
fn protected_memory_source(conn: &rusqlite::Connection, rec: &MemoryRecord) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM kg_fact_sources WHERE namespace=?1 AND source_memory_id=?2)
          OR EXISTS(SELECT 1 FROM kg_facts WHERE namespace=?1 AND source_memory_id=?2)
          OR EXISTS(SELECT 1 FROM memory_relations WHERE to_id=?2
              AND relation IN ('derived_from','summarizes','supersedes'))",
        rusqlite::params![rec.namespace, rec.id.to_string()],
        |r| r.get(0),
    )?)
}

fn supersede_exact_duplicate(
    conn: &rusqlite::Connection,
    rec: &mut MemoryRecord,
    survivor: MemoryId,
    add_relation: bool,
    jieba: &Jieba,
) -> Result<()> {
    let before = rec.clone();
    rec.state = MemoryState::Superseded;
    rec.embedding_pending = false;
    rec.updated_at_ms = now_ms();
    update_memory(conn, rec, jieba)?;
    delete_embedding(conn, rec.id)?;
    if add_relation {
        insert_relation(conn, survivor, rec.id, RelationKind::Supersedes, 1.0)?;
    }
    // A redundant child's existing child -> source DerivedFrom is retained.
    // Adding source -> child Supersedes would create a provenance cycle.
    insert_version(
        conn,
        rec.id,
        "deduplicate",
        Some(&before),
        rec,
        "exact_dedup",
        None,
    )?;
    append_audit(
        conn,
        Some(rec.id),
        "dream_supersede",
        serde_json::json!({
            "by": survivor,
            "reason": if add_relation { "same_bucket_exact_duplicate" } else { "redundant_classifier_derivation" },
        }),
    )?;
    Ok(())
}

/// Facts with different owners, tags, kinds or event dates never share a bucket.
fn memory_bucket(rec: &MemoryRecord) -> String {
    let mut subjects = rec.subject_ids.clone();
    subjects.extend(household::subject_id_list(&rec.metadata));
    subjects.sort();
    subjects.dedup();
    serde_json::json!([
        rec.namespace,
        subjects,
        rec.tag,
        rec.kind.as_str(),
        rec.metadata.get("event_at"),
        rec.metadata.get("event_at_ms"),
        rec.metadata.get("periodicity"),
        household::cluster_key(&rec.metadata),
    ])
    .to_string()
}

/// A sourced graph assertion must be independently supported by a source that
/// satisfies the complete text query. Do not filter after the graph top-k.
fn graph_matches_query(
    conn: &rusqlite::Connection,
    namespace: &str,
    fact: &mut crate::graph::KgFact,
    query: &RecallQuery,
) -> Result<bool> {
    let visible = crate::graph::policy::visible_source_memory_ids(
        conn,
        namespace,
        fact,
        query.viewer_member_id.as_deref(),
    )?;
    for id in visible {
        let Some(mut memory) = get_memory(conn, id)? else {
            continue;
        };
        // Archived atoms are valid graph supports, even when they aren't direct
        // text candidates. All ACL/member/tag/time/kind checks still apply.
        if memory.state == MemoryState::Archived {
            memory.state = MemoryState::Active;
        }
        if recall_matches(&memory, query, namespace) {
            let (text, summary) = crate::graph::store::fact_source_evidence(
                conn,
                namespace,
                &fact.id.to_string(),
                Some(id),
            )?;
            fact.source_memory_id = Some(id);
            fact.evidence_text = text;
            fact.evidence_summary = summary;
            return Ok(true);
        }
    }
    if !crate::graph::store::has_graph_only_source(conn, namespace, &fact.id.to_string())? {
        return Ok(false);
    }
    // Graph-only assertions have no memory kind or text tag. Fail closed for
    // filters that cannot be truthfully evaluated from their own fields.
    if query.tag.is_some() || !query.kinds.is_empty() {
        return Ok(false);
    }
    if query.time_from_ms.is_some_and(|t| fact.created_at_ms < t)
        || query.time_to_ms.is_some_and(|t| fact.created_at_ms > t)
    {
        return Ok(false);
    }
    if query
        .member_id
        .as_deref()
        .is_some_and(|id| fact.owner_member_id.as_deref() != Some(id))
    {
        return Ok(false);
    }
    let (text, summary) =
        crate::graph::store::fact_source_evidence(conn, namespace, &fact.id.to_string(), None)?;
    fact.source_memory_id = None;
    fact.evidence_text = text;
    fact.evidence_summary = summary;
    Ok(true)
}

fn recall_matches(rec: &MemoryRecord, query: &RecallQuery, namespace: &str) -> bool {
    crate::store::memory_is_recallable(rec)
        && rec.namespace == namespace
        && (query.kinds.is_empty() || query.kinds.contains(&rec.kind))
        && query.time_from_ms.map_or(true, |t| rec.created_at_ms >= t)
        && query.time_to_ms.map_or(true, |t| rec.created_at_ms <= t)
        && household::matches_member_and_tag(
            &rec.metadata,
            &rec.subject_ids,
            &rec.tag,
            query.member_id.as_deref(),
            query.tag.as_deref(),
            &query.domains,
        )
        && household::visible_to_viewer(
            &rec.metadata,
            &rec.subject_ids,
            query.viewer_member_id.as_deref(),
        )
}

#[cfg(test)]
#[path = "candidate_recall_tests.rs"]
mod candidate_recall_tests;
