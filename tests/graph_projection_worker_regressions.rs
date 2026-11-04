//! Durable graph worker lifecycle and write-triggered draining (no network).
use async_trait::async_trait;
use memora::graph::{GraphExtractInput, GraphExtractOutput, GraphExtractor};
use memora::{MemoryEngine, MemoryService, RememberInput};
use rusqlite::Connection;
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

struct Controlled {
    blocked: AtomicBool,
    active: AtomicUsize,
    maximum: AtomicUsize,
    calls: AtomicUsize,
    inputs: Mutex<Vec<GraphExtractInput>>,
}
impl Controlled {
    fn new(blocked: bool) -> Self {
        Self {
            blocked: AtomicBool::new(blocked),
            active: AtomicUsize::new(0),
            maximum: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            inputs: Mutex::new(Vec::new()),
        }
    }
}
struct Active<'a>(&'a AtomicUsize);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
#[async_trait]
impl GraphExtractor for Controlled {
    fn extractor_id(&self) -> &str {
        "worker-controlled"
    }
    async fn extract(&self, input: GraphExtractInput) -> memora::Result<GraphExtractOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inputs.lock().unwrap().push(input);
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(now, Ordering::SeqCst);
        let _active = Active(&self.active);
        if self.blocked.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
        Ok(GraphExtractOutput {
            extractor_id: self.extractor_id().into(),
            ..Default::default()
        })
    }
}
fn count(dir: &tempfile::TempDir) -> i64 {
    Connection::open(dir.path().join("worker.db"))
        .unwrap()
        .query_row("SELECT count(*) FROM graph_projection_jobs", [], |row| {
            row.get(0)
        })
        .unwrap()
}
async fn wait_until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(4), async {
        while !predicate() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("write-triggered worker should not wait for a maintenance interval");
}
fn source() -> RememberInput {
    let mut input = RememberInput::episodic("小航生日事实");
    input.skip_consolidation = true;
    input.metadata = json!({"graph_text":"小航下个月6号就要过生日了，请问送什么礼物好呢", "source":{"message_at_ms":1792890000000i64}});
    input
}

#[tokio::test]
async fn worker_drains_on_write_and_recovers_jobs_on_restart_without_tick() {
    let dir = tempfile::tempdir().unwrap();
    let blocked = Arc::new(Controlled::new(true));
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("worker.db"))
        .graph_extractor(blocked.clone())
        .build()
        .await
        .unwrap();
    assert!(engine.graph_projection_worker_enabled());
    tokio::time::timeout(
        Duration::from_secs(2),
        engine.remember_deferred_graph(source()),
    )
    .await
    .unwrap()
    .unwrap();
    wait_until(|| blocked.calls.load(Ordering::SeqCst) == 1).await;
    // An in-flight graph LLM does not retain the foreground mutation lock.
    tokio::time::timeout(
        Duration::from_secs(2),
        engine.remember_deferred_graph(source()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(count(&dir), 2);
    assert_eq!(blocked.maximum.load(Ordering::SeqCst), 1);
    engine.shutdown();
    wait_until(|| blocked.active.load(Ordering::SeqCst) == 0).await;
    drop(engine);
    let ready = Arc::new(Controlled::new(false));
    let recovered = MemoryEngine::builder()
        .db_path(dir.path().join("worker.db"))
        .graph_extractor(ready.clone())
        .build()
        .await
        .unwrap();
    wait_until(|| count(&dir) == 0).await;
    assert_eq!(ready.calls.load(Ordering::SeqCst), 2);
    assert!(ready.inputs.lock().unwrap().iter().all(|input| input.text
        == "小航下个月6号就要过生日了，请问送什么礼物好呢"
        && input.message_at_ms == Some(1792890000000)));
    assert_eq!(ready.maximum.load(Ordering::SeqCst), 1);
    recovered.shutdown();
}

#[tokio::test]
async fn unconfigured_extractor_never_acknowledges_or_discards_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("worker.db"))
        .build()
        .await
        .unwrap();
    assert!(!engine.graph_projection_worker_enabled());
    engine.remember_deferred_graph(source()).await.unwrap();
    engine.remember(source()).await.unwrap();
    assert_eq!(count(&dir), 2);
    assert_eq!(engine.retry_graph_projections(8).await.unwrap(), 0);
    assert_eq!(count(&dir), 2);
    engine.shutdown();
}

#[tokio::test]
async fn eager_and_background_paths_do_not_extract_the_same_job_twice() {
    let dir = tempfile::tempdir().unwrap();
    let extractor = Arc::new(Controlled::new(false));
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("worker.db"))
        .graph_extractor(extractor.clone())
        .build()
        .await
        .unwrap();
    engine.remember_deferred_graph(source()).await.unwrap();
    let engine_for_tick = engine.clone();
    let tick = tokio::spawn(async move {
        engine_for_tick.retry_graph_projections(8).await.unwrap();
    });
    engine.remember(source()).await.unwrap();
    tick.await.unwrap();
    wait_until(|| count(&dir) == 0).await;
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 2);
    assert_eq!(extractor.maximum.load(Ordering::SeqCst), 1);
    engine.shutdown();
}
