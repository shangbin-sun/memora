//! Bounded, namespace/vector-space isolated query-vector cache and single-flight.
//! Only vectors are cached; exact query keys stay in RAM and are never logged.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use moka::future::Cache;
use tokio::sync::{watch, Mutex, Semaphore};
use tokio::time::Instant;

use crate::config::Config;
use crate::embedding::{normalize_vector, Embedder};

type Vector = Vec<f32>;
type Completion = Option<Option<Vector>>; // None=pending, Some(None)=failure.

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Key {
    namespace: String,
    space: String,
    model: String,
    text: String,
}
impl Key {
    fn new(namespace: &str, embedder: &dyn Embedder, text: &str) -> Self {
        Self {
            namespace: namespace.into(),
            space: embedder.space_id(),
            model: embedder.model_id().into(),
            text: text.into(),
        }
    }
}
struct Flight {
    result: watch::Receiver<Completion>,
    provider: usize,
    initial_dimension: usize,
}
struct Core {
    vectors: Cache<Key, Vector>,
    flights: Mutex<HashMap<Key, Flight>>,
    permits: Arc<Semaphore>,
    closed: watch::Sender<bool>,
    batch_size: usize,
    worker_timeout: Duration,
}

#[derive(Clone)]
pub(crate) struct QueryEmbeddingCache {
    core: Arc<Core>,
}

impl QueryEmbeddingCache {
    pub(crate) fn new(config: &Config) -> Self {
        let (closed, _) = watch::channel(false);
        let capacity = if config.query_embedding_cache_ttl.is_zero() {
            0
        } else {
            config.query_embedding_cache_capacity
        };
        Self {
            core: Arc::new(Core {
                vectors: Cache::builder()
                    .max_capacity(capacity)
                    .time_to_live(config.query_embedding_cache_ttl)
                    .build(),
                flights: Mutex::new(HashMap::new()),
                permits: Arc::new(Semaphore::new(
                    config.query_embedding_max_in_flight.clamp(1, 64),
                )),
                closed,
                batch_size: config.query_embedding_batch_size.clamp(1, 16),
                worker_timeout: config.embedding_timeout,
            }),
        }
    }

    /// Stop admission and cancel provider calls/waiters without retaining engine Inner.
    /// Idempotent and synchronous so the host can call it during shutdown/drop.
    pub(crate) fn close(&self) {
        self.core.closed.send_replace(true);
        self.core.permits.close();
        self.core.vectors.invalidate_all();
    }

    pub(crate) async fn get(
        &self,
        embedder: Arc<dyn Embedder>,
        namespace: &str,
        text: &str,
        timeout: Duration,
    ) -> Option<Vector> {
        self.get_many(embedder, namespace, &[text.to_string()], timeout)
            .await
            .into_iter()
            .next()
            .flatten()
    }

    /// Preserve input order and duplicates; provider misses are deduplicated and
    /// submitted in batches of at most 16. Every caller's deadline includes
    /// admission, permit wait and single-flight wait. Failures are never cached.
    pub(crate) async fn get_many(
        &self,
        embedder: Arc<dyn Embedder>,
        namespace: &str,
        texts: &[String],
        timeout: Duration,
    ) -> Vec<Option<Vector>> {
        if timeout.is_zero() || *self.core.closed.borrow() {
            return vec![None; texts.len()];
        }
        // Outer timeout also covers cache housekeeping/lock awaits: no stage
        // may accidentally reset or exceed the caller's total time budget.
        tokio::time::timeout(
            timeout,
            self.get_many_inner(embedder, namespace, texts, timeout),
        )
        .await
        .unwrap_or_else(|_| vec![None; texts.len()])
    }

    async fn get_many_inner(
        &self,
        embedder: Arc<dyn Embedder>,
        namespace: &str,
        texts: &[String],
        timeout: Duration,
    ) -> Vec<Option<Vector>> {
        let mut output = vec![None; texts.len()];
        if texts.is_empty() || timeout.is_zero() || *self.core.closed.borrow() {
            return output;
        }
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            return output;
        };
        let mut unique: Vec<(String, Vec<usize>)> = Vec::new();
        let mut positions = HashMap::new();
        for (index, text) in texts.iter().enumerate() {
            if text.is_empty() {
                continue;
            }
            let entry = *positions.entry(text.clone()).or_insert_with(|| {
                unique.push((text.clone(), Vec::new()));
                unique.len() - 1
            });
            unique[entry].1.push(index);
        }
        let mut receivers = Vec::new();
        for chunk in unique.chunks(self.core.batch_size) {
            let results = self
                .admit(embedder.clone(), namespace, chunk, deadline)
                .await;
            for (indices, value) in results {
                match value {
                    Pending::Ready(vector) => {
                        for index in indices {
                            output[index] = Some(vector.clone());
                        }
                    }
                    Pending::Waiting(receiver) => receivers.push((indices, receiver)),
                }
            }
        }
        for (indices, mut receiver) in receivers {
            let mut closed = self.core.closed.subscribe();
            let vector = loop {
                if *closed.borrow() {
                    break None;
                }
                let current = receiver.borrow().clone();
                if let Some(result) = current {
                    break result;
                }
                tokio::select! {
                    biased;
                    _ = closed.changed() => break None,
                    _ = tokio::time::sleep_until(deadline) => break None,
                    changed = receiver.changed() => if changed.is_err() { break None; },
                }
            };
            // Recheck dimensions for this caller too (shared vector-space identity
            // can belong to a different provider instance with auto dimensions).
            let vector = vector.filter(|v| embedder.bind_dimension(v.len()));
            for index in indices {
                output[index] = vector.clone();
            }
        }
        output
    }

    async fn admit(
        &self,
        embedder: Arc<dyn Embedder>,
        namespace: &str,
        chunk: &[(String, Vec<usize>)],
        deadline: Instant,
    ) -> Vec<(Vec<usize>, Pending)> {
        let mut results = Vec::new();
        let mut missing = Vec::new();
        let mut closed = self.core.closed.subscribe();
        if *closed.borrow() || Instant::now() >= deadline {
            return results;
        }
        let mut flights = tokio::select! {
            biased;
            _ = closed.changed() => return results,
            _ = tokio::time::sleep_until(deadline) => return results,
            guard = self.core.flights.lock() => guard,
        };
        // Gate lookup and completion cache insertion together to avoid a gap
        // between a finished single-flight and its cache becoming visible.
        for (text, indices) in chunk {
            if let Some(value) = lookup(&self.core, &flights, &embedder, namespace, text).await {
                results.push((indices.clone(), value));
            } else {
                missing.push((text.clone(), indices.clone()));
            }
        }
        drop(flights);
        if missing.is_empty() {
            return results;
        }
        // Acquire BEFORE spawning/registering: at most max_in_flight tasks and
        // max_in_flight*16 distinct keys, never an unbounded queue of orphan jobs.
        let permit = tokio::select! {
            biased;
            _ = closed.changed() => return results,
            _ = tokio::time::sleep_until(deadline) => return results,
            permit = self.core.permits.clone().acquire_owned() => match permit {
                Ok(permit) => permit, Err(_) => return results,
            },
        };
        flights = tokio::select! {
            biased;
            _ = closed.changed() => return results,
            _ = tokio::time::sleep_until(deadline) => return results,
            guard = self.core.flights.lock() => guard,
        };
        let provider = provider_id(&embedder);
        let mut new_requests = Vec::new();
        for (text, indices) in missing {
            if let Some(value) = lookup(&self.core, &flights, &embedder, namespace, &text).await {
                results.push((indices, value));
            } else {
                new_requests.push((text, indices));
            }
        }
        let mut requests = Vec::new();
        for (text, indices) in new_requests {
            let key = Key::new(namespace, embedder.as_ref(), &text);
            let (sender, receiver) = watch::channel(None);
            flights.insert(
                key.clone(),
                Flight {
                    result: receiver.clone(),
                    provider,
                    initial_dimension: embedder.dimension(),
                },
            );
            requests.push((key, sender));
            results.push((indices, Pending::Waiting(receiver)));
        }
        // No await after insertion until spawn: cancellation cannot leave keys
        // registered without a worker owning their senders and permit.
        if !requests.is_empty() {
            let core = self.core.clone();
            let namespace = namespace.to_string();
            let worker_deadline = Instant::now()
                .checked_add(core.worker_timeout)
                .map_or(deadline, |limit| limit.min(deadline));
            tokio::spawn(async move {
                let _permit = permit;
                run_batch(core, embedder, namespace, requests, worker_deadline).await;
            });
        }
        results
    }
}

enum Pending {
    Ready(Vector),
    Waiting(watch::Receiver<Completion>),
}
fn provider_id(embedder: &Arc<dyn Embedder>) -> usize {
    Arc::as_ptr(embedder) as *const () as usize
}
async fn lookup(
    core: &Core,
    flights: &HashMap<Key, Flight>,
    embedder: &Arc<dyn Embedder>,
    namespace: &str,
    text: &str,
) -> Option<Pending> {
    let key = Key::new(namespace, embedder.as_ref(), text);
    if let Some(vector) = core.vectors.get(&key).await {
        if embedder.bind_dimension(vector.len()) {
            return Some(Pending::Ready(vector));
        }
        core.vectors.invalidate(&key).await;
    }
    let flight = flights.get(&key).or_else(|| {
        // Default Embedder::space_id includes dimension; auto-bind 0→N may
        // change it while a batch finishes. Same-instance initial-auto flights
        // remain joinable, but cache insertion always uses the post-bind key.
        flights
            .iter()
            .find(|(old, flight)| {
                old.namespace == namespace
                    && old.text == text
                    && old.model == embedder.model_id()
                    && flight.provider == provider_id(embedder)
                    && flight.initial_dimension == 0
                    && old.space == format!("{}:0", embedder.model_id())
                    && key.space == format!("{}:{}", embedder.model_id(), embedder.dimension())
            })
            .map(|(_, flight)| flight)
    });
    flight.map(|flight| Pending::Waiting(flight.result.clone()))
}

// Provider panics/runtime task cancellation must not leave stale in-flight keys.
// The cleanup only retains this independent cache core, never engine Inner.
struct BatchCleanup {
    core: Arc<Core>,
    keys: Vec<Key>,
    armed: bool,
}
impl Drop for BatchCleanup {
    fn drop(&mut self) {
        if self.armed {
            let core = self.core.clone();
            let keys = std::mem::take(&mut self.keys);
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let mut flights = core.flights.lock().await;
                    for key in keys {
                        flights.remove(&key);
                    }
                });
            }
        }
    }
}

/// The gateway is bimodal: most replies take 200ms, the rest take seconds. A second
/// request after a short wait turns most of the slow cases into fast ones (measured:
/// 3/12 within 700ms alone, 8/12 hedged). The loser is dropped, which cancels it.
const HEDGE_AFTER: Option<Duration> = if cfg!(test) { None } else { Some(Duration::from_millis(150)) };

async fn hedged_embed(embedder: &Arc<dyn Embedder>, texts: &[String]) -> Option<Vec<Vector>> {
    let first = embedder.embed_query(texts);
    tokio::pin!(first);
    let Some(delay) = HEDGE_AFTER.filter(|_| embedder.hedge_queries()) else {
        return first.await.ok();
    };
    tokio::select! {
        biased;
        values = &mut first => return values.ok(),
        _ = tokio::time::sleep(delay) => {}
    }
    let second = embedder.embed_query(texts);
    tokio::pin!(second);
    tokio::select! {
        values = &mut first => match values {
            Ok(values) => Some(values),
            Err(_) => second.await.ok(),
        },
        values = &mut second => match values {
            Ok(values) => Some(values),
            Err(_) => first.await.ok(),
        },
    }
}

async fn run_batch(
    core: Arc<Core>,
    embedder: Arc<dyn Embedder>,
    namespace: String,
    requests: Vec<(Key, watch::Sender<Completion>)>,
    deadline: Instant,
) {
    let mut cleanup = BatchCleanup {
        core: core.clone(),
        keys: requests.iter().map(|(key, _)| key.clone()).collect(),
        armed: true,
    };
    let texts: Vec<String> = requests.iter().map(|(key, _)| key.text.clone()).collect();
    let mut closed = core.closed.subscribe();
    let values = if *closed.borrow() {
        None
    } else {
        tokio::select! {
            biased;
            _ = closed.changed() => None,
            _ = tokio::time::sleep_until(deadline) => None,
            values = hedged_embed(&embedder, &texts) => values.filter(|values| values.len() == texts.len()),
        }
    };
    let mut flights = core.flights.lock().await;
    for (index, (key, sender)) in requests.into_iter().enumerate() {
        let vector = if *core.closed.borrow() || Instant::now() >= deadline {
            None
        } else {
            values
                .as_ref()
                .and_then(|values| values.get(index))
                .cloned()
                .and_then(|vector| normalize_vector(vector).ok())
                .filter(|vector| embedder.bind_dimension(vector.len()))
        };
        if let Some(vector) = &vector {
            // Do not cache pre-bind default space :0; Qianfan's configured space
            // is already stable, other auto providers transition to :N here.
            core.vectors
                .insert(
                    Key::new(&namespace, embedder.as_ref(), &key.text),
                    vector.clone(),
                )
                .await;
        }
        sender.send_replace(Some(vector));
        flights.remove(&key);
    }
    cleanup.armed = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Mock {
        model: String,
        space: String,
        dimension: AtomicUsize,
        delay_ms: AtomicUsize,
        mode: AtomicUsize,
        calls: AtomicUsize,
        active: Arc<AtomicUsize>,
        max_active: AtomicUsize,
        batches: std::sync::Mutex<Vec<Vec<String>>>,
        default_space: bool,
    }
    struct Active(Arc<AtomicUsize>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl Mock {
        fn new(delay_ms: usize) -> Arc<Self> {
            Arc::new(Self {
                model: "model".into(),
                space: "endpoint|model|2".into(),
                dimension: AtomicUsize::new(2),
                delay_ms: AtomicUsize::new(delay_ms),
                mode: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                active: Arc::new(AtomicUsize::new(0)),
                max_active: AtomicUsize::new(0),
                batches: std::sync::Mutex::new(Vec::new()),
                default_space: false,
            })
        }
    }
    #[async_trait]
    impl Embedder for Mock {
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dimension(&self) -> usize {
            self.dimension.load(Ordering::SeqCst)
        }
        fn space_id(&self) -> String {
            if self.default_space {
                format!("{}:{}", self.model, self.dimension())
            } else {
                self.space.clone()
            }
        }
        fn bind_dimension(&self, dimension: usize) -> bool {
            let _ =
                self.dimension
                    .compare_exchange(0, dimension, Ordering::SeqCst, Ordering::SeqCst);
            self.dimension() == dimension && dimension != 0
        }
        async fn embed(&self, texts: &[String]) -> crate::error::Result<Vec<Vector>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.batches.lock().unwrap().push(texts.to_vec());
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            let _guard = Active(self.active.clone());
            tokio::time::sleep(Duration::from_millis(
                self.delay_ms.load(Ordering::SeqCst) as u64
            ))
            .await;
            match self.mode.load(Ordering::SeqCst) {
                1 => Err(crate::error::MemoraError::Embedding("mock failure".into())),
                2 => Ok(texts.iter().map(|_| vec![0.0, 0.0]).collect()),
                3 => Ok(texts.iter().map(|_| vec![f32::NAN, 1.0]).collect()),
                4 => Ok(texts.iter().map(|_| vec![1.0, 2.0, 3.0]).collect()),
                5 => Ok(vec![]),
                _ => Ok(texts.iter().map(|_| vec![3.0, 4.0]).collect()),
            }
        }
    }
    fn config() -> Config {
        Config {
            embedding_timeout: Duration::from_secs(2),
            ..Default::default()
        }
    }
    async fn calls_at_least(mock: &Mock, target: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while mock.calls.load(Ordering::SeqCst) < target {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    async fn drained(cache: &QueryEmbeddingCache, mock: &Mock) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if mock.active.load(Ordering::SeqCst) == 0
                    && cache.core.flights.lock().await.is_empty()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn ordinary_get_and_batch_share_single_flight_and_normalized_cache() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(50);
        let task = tokio::spawn({
            let cache = cache.clone();
            let mock = mock.clone();
            async move {
                cache
                    .get_many(
                        mock,
                        "home",
                        &["exact".into(), "another".into(), "exact".into()],
                        Duration::from_secs(1),
                    )
                    .await
            }
        });
        calls_at_least(&mock, 1).await;
        let shared = cache
            .get(mock.clone(), "home", "exact", Duration::from_secs(1))
            .await
            .unwrap();
        let batch = task.await.unwrap();
        assert_eq!(batch[0], batch[2]);
        assert_eq!(batch[0], Some(shared.clone()));
        assert!((shared[0] - 0.6).abs() < 1e-6 && (shared[1] - 0.8).abs() < 1e-6);
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        assert_eq!(mock.batches.lock().unwrap()[0], vec!["exact", "another"]);
        assert_eq!(
            cache
                .get(mock.clone(), "home", "exact", Duration::from_secs(1))
                .await,
            Some(shared)
        );
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn large_batches_are_chunked_at_16_and_duplicate_order_is_preserved() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(5);
        let mut texts: Vec<String> = (0..35).map(|index| format!("query{index}")).collect();
        texts.extend(["query2".into(), "query0".into()]);
        let values = cache
            .get_many(mock.clone(), "home", &texts, Duration::from_secs(1))
            .await;
        assert_eq!(values.len(), 37);
        assert!(values.iter().all(Option::is_some));
        assert_eq!(values[2], values[35]);
        assert_eq!(values[0], values[36]);
        assert_eq!(mock.calls.load(Ordering::SeqCst), 3);
        assert!(mock
            .batches
            .lock()
            .unwrap()
            .iter()
            .all(|batch| batch.len() <= 16));
    }
    #[tokio::test]
    async fn exact_query_namespace_model_and_space_isolation() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(1);
        for (namespace, text) in [
            ("A", "query"),
            ("B", "query"),
            ("A", " Query"),
            ("A", "QUERY"),
        ] {
            assert!(cache
                .get(mock.clone(), namespace, text, Duration::from_secs(1))
                .await
                .is_some());
        }
        assert_eq!(mock.calls.load(Ordering::SeqCst), 4);
        let mut other = Mock::new(1);
        Arc::get_mut(&mut other).unwrap().model = "different-model".into();
        assert!(cache
            .get(other.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(other.calls.load(Ordering::SeqCst), 1);
        let mut endpoint = Mock::new(1);
        Arc::get_mut(&mut endpoint).unwrap().space = "different-endpoint".into();
        assert!(cache
            .get(endpoint.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(endpoint.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn errors_zero_nan_mismatch_and_bad_batch_lengths_do_not_poison_cache() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(1);
        for mode in 1..=5 {
            mock.mode.store(mode, Ordering::SeqCst);
            assert!(cache
                .get(mock.clone(), "A", "query", Duration::from_secs(1))
                .await
                .is_none());
        }
        assert_eq!(mock.calls.load(Ordering::SeqCst), 5);
        mock.mode.store(0, Ordering::SeqCst);
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 6);
    }
    #[tokio::test]
    async fn caller_timeout_and_drop_do_not_leave_infinite_or_poisoned_work() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(120);
        let leader = tokio::spawn({
            let cache = cache.clone();
            let mock = mock.clone();
            async move {
                cache
                    .get(mock, "A", "query", Duration::from_millis(150))
                    .await
            }
        });
        calls_at_least(&mock, 1).await;
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_millis(10))
            .await
            .is_none());
        leader.abort();
        let _ = leader.await;
        drained(&cache, &mock).await;
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(
            mock.calls.load(Ordering::SeqCst),
            1,
            "successful bounded leader can warm cache after caller drops"
        );
        assert!(cache
            .get(mock.clone(), "A", "timeout", Duration::from_millis(10))
            .await
            .is_none());
        drained(&cache, &mock).await;
        mock.delay_ms.store(1, Ordering::SeqCst);
        assert!(cache
            .get(mock.clone(), "A", "timeout", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 3);
    }
    #[tokio::test]
    async fn distinct_inflight_is_bounded_and_queue_timeout_does_not_spawn_extra_work() {
        let cache = QueryEmbeddingCache::new(&Config {
            query_embedding_max_in_flight: 2,
            ..config()
        });
        let mock = Mock::new(100);
        let mut tasks = Vec::new();
        for index in 0..8 {
            let cache = cache.clone();
            let mock = mock.clone();
            tasks.push(tokio::spawn(async move {
                cache
                    .get(mock, "A", &format!("q{index}"), Duration::from_millis(50))
                    .await
            }));
        }
        for task in tasks {
            assert!(task.await.unwrap().is_none());
        }
        drained(&cache, &mock).await;
        assert!(mock.max_active.load(Ordering::SeqCst) <= 2);
        assert_eq!(cache.core.permits.available_permits(), 2);
        mock.delay_ms.store(1, Ordering::SeqCst);
        assert!(cache
            .get(mock.clone(), "A", "retry", Duration::from_secs(1))
            .await
            .is_some());
    }
    #[tokio::test]
    async fn close_cancels_provider_and_all_waiters_and_denies_future_admission() {
        let cache = QueryEmbeddingCache::new(&config());
        let mock = Mock::new(10_000);
        let pending = tokio::spawn({
            let cache = cache.clone();
            let mock = mock.clone();
            async move { cache.get(mock, "A", "query", Duration::from_secs(2)).await }
        });
        calls_at_least(&mock, 1).await;
        cache.close();
        cache.close();
        assert!(tokio::time::timeout(Duration::from_millis(200), pending)
            .await
            .unwrap()
            .unwrap()
            .is_none());
        drained(&cache, &mock).await;
        assert!(cache.core.permits.is_closed());
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_none());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn auto_dimension_default_space_transition_is_cached_under_bound_space() {
        let mut mock = Mock::new(1);
        let inner = Arc::get_mut(&mut mock).unwrap();
        inner.default_space = true;
        inner.dimension.store(0, Ordering::SeqCst);
        let cache = QueryEmbeddingCache::new(&config());
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(mock.dimension(), 2);
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn ttl_expiry_zero_timeout_and_zero_worker_budget_are_not_successes() {
        let cache = QueryEmbeddingCache::new(&Config {
            query_embedding_cache_ttl: Duration::from_millis(20),
            ..config()
        });
        let mock = Mock::new(1);
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::ZERO)
            .await
            .is_none());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 0);
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(cache
            .get(mock.clone(), "A", "query", Duration::from_secs(1))
            .await
            .is_some());
        assert_eq!(mock.calls.load(Ordering::SeqCst), 2);
        let no_worker = QueryEmbeddingCache::new(&Config {
            embedding_timeout: Duration::ZERO,
            ..config()
        });
        assert!(no_worker
            .get(mock, "A", "query", Duration::from_secs(1))
            .await
            .is_none());
    }
}
