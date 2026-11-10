//! 应用可选择让全部模型请求共用一个进程级FIFO队列。
//! 许可必须持有到响应体读取完成；不在排队时占用工作线程。
use std::sync::{Arc, OnceLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Default)]
struct EmbeddingState { busy: bool, queries: usize }
fn embedding_state() -> &'static std::sync::Mutex<EmbeddingState> {
    static STATE: OnceLock<std::sync::Mutex<EmbeddingState>> = OnceLock::new();
    STATE.get_or_init(Default::default)
}
fn embedding_notify() -> &'static tokio::sync::Notify {
    static N: OnceLock<tokio::sync::Notify> = OnceLock::new();
    N.get_or_init(tokio::sync::Notify::new)
}
pub(crate) struct EmbeddingPermit;
impl Drop for EmbeddingPermit {
    fn drop(&mut self) {
        embedding_state().lock().unwrap().busy = false;
        embedding_notify().notify_waiters();
    }
}
struct QueryWaiter(bool);
impl Drop for QueryWaiter {
    fn drop(&mut self) {
        if self.0 { embedding_state().lock().unwrap().queries -= 1; embedding_notify().notify_waiters(); }
    }
}
/// 设备实测并发>1会拒绝：保留一个物理槽，检索跳过待执行后台任务。
pub(crate) async fn acquire_embedding(enabled: bool, query: bool) -> Option<EmbeddingPermit> {
    if !enabled { return None; }
    if query { embedding_state().lock().unwrap().queries += 1; }
    let mut waiter = QueryWaiter(query);
    loop {
        let notified = embedding_notify().notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        {
            let mut state = embedding_state().lock().unwrap();
            if !state.busy && (query || state.queries == 0) {
                state.busy = true;
                if query { state.queries -= 1; waiter.0 = false; }
                return Some(EmbeddingPermit);
            }
        }
        notified.await;
    }
}

fn queue() -> &'static Arc<Semaphore> {
    static QUEUE: OnceLock<Arc<Semaphore>> = OnceLock::new();
    QUEUE.get_or_init(|| Arc::new(Semaphore::new(1)))
}

pub(crate) async fn acquire(enabled: bool) -> Option<OwnedSemaphorePermit> {
    if enabled {
        // 进程级队列从不close；Tokio信号量按等待顺序公平放行。
        Some(queue().clone().acquire_owned().await.expect("model queue open"))
    } else {
        None
    }
}

/// OCR与原生LLM/embedding使用同一个队列，返回原始OpenAI JSON。
#[cfg(any(feature = "jni", test))]
pub(crate) async fn request(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    timeout: std::time::Duration,
) -> crate::Result<String> {
    let _permit = acquire(true).await;
    // HTTP超时从轮到该请求后开始；排队不占OCR推理时间预算。
    let client = reqwest::Client::builder()
        .http1_only()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(timeout)
        .build()
        .map_err(|e| crate::MemoraError::Other(format!("model client: {e}")))?;
    let response = client.post(url).bearer_auth(api_key).json(body).send().await
        .map_err(|e| crate::MemoraError::Other(format!("model request: {e}")))?;
    let status = response.status();
    let text = response.text().await
        .map_err(|e| crate::MemoraError::Other(format!("model response: {e}")))?;
    if !status.is_success() {
        return Err(crate::MemoraError::Other(format!("model http {status}")));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn query_embedding_overtakes_waiting_background() {
        let held = acquire_embedding(true, false).await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (ready, ready_rx) = tokio::sync::oneshot::channel();
        let bg_tx = tx.clone();
        let bg = tokio::spawn(async move {
            let _ = ready.send(());
            let _p = acquire_embedding(true, false).await;
            bg_tx.send("background").unwrap();
        });
        ready_rx.await.unwrap();
        let (ready, ready_rx) = tokio::sync::oneshot::channel();
        let query = tokio::spawn(async move {
            let _ = ready.send(());
            let _p = acquire_embedding(true, true).await;
            tx.send("query").unwrap();
        });
        ready_rx.await.unwrap();
        drop(held);
        assert_eq!(rx.recv().await, Some("query"));
        assert_eq!(rx.recv().await, Some("background"));
        bg.await.unwrap(); query.await.unwrap();
    }

    #[tokio::test]
    async fn embedding_does_not_wait_for_llm_slot() {
        let _model = acquire(true).await;
        assert!(tokio::time::timeout(std::time::Duration::from_secs(2),acquire_embedding(true,true)).await.is_ok());
    }

    #[tokio::test]
    async fn serial_queue_is_fifo_and_cancelled_waiter_does_not_block() {
        let held = acquire(true).await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut jobs = Vec::new();
        for i in 0..5 {
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let tx = tx.clone();
            jobs.push(tokio::spawn(async move {
                let _ = ready_tx.send(());
                let _permit = acquire(true).await;
                tx.send(i).unwrap();
            }));
            ready_rx.await.unwrap();
        }
        // current_thread运行时中，ready信号送达时该任务已等待在信号量上。
        jobs[2].abort();
        drop(held);
        for expected in [0, 1, 3, 4] {
            assert_eq!(rx.recv().await, Some(expected));
        }
        for job in jobs { let _ = job.await; }
        assert!(acquire(false).await.is_none());
        assert!(tokio::time::timeout(std::time::Duration::from_secs(1), acquire(true)).await.is_ok());
    }

    #[tokio::test]
    async fn serial_queue_serializes_chat_embedding_and_ocr_through_response_body() {
        use crate::{Embedder, QianfanChat, QianfanConfig, QianfanEmbedder};
        use std::io::{Read, Write};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{Duration, Instant};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let peak = maximum.clone();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut workers = Vec::new();
            while workers.len() < 3 && Instant::now() < deadline {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(2)); continue;
                };
                let active = active.clone();
                let maximum = maximum.clone();
                workers.push(std::thread::spawn(move || {
                    stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                    let mut input = Vec::new();
                    while !input.windows(4).any(|w| w == b"\r\n\r\n") {
                        let mut buf = [0; 4096];
                        let n = stream.read(&mut buf).unwrap();
                        assert!(n > 0); input.extend_from_slice(&buf[..n]);
                    }
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(count, Ordering::SeqCst);
                    let embedding = String::from_utf8_lossy(&input).starts_with("POST /embed ");
                    let body = if embedding { r#"{"data":[{"index":0,"embedding":[1,0]}]}"# }
                        else { r#"{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}"# };
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                    let split = body.len()/2;
                    stream.write_all(&body.as_bytes()[..split]).unwrap();
                    stream.flush().unwrap();
                    // 若只锁到HTTP响应头，此处第二个请求会进入，峰值将大于1。
                    std::thread::sleep(Duration::from_millis(80));
                    active.fetch_sub(1, Ordering::SeqCst);
                    stream.write_all(&body.as_bytes()[split..]).unwrap();
                }));
            }
            assert_eq!(workers.len(), 3);
            for worker in workers { worker.join().unwrap(); }
        });
        let mut cfg = QianfanConfig::for_embed("local".into(), Some(format!("{base}/embed")), Some("mock".into()), Some(2));
        cfg.chat_url = format!("{base}/chat");
        let embed = QianfanEmbedder::new(cfg.clone()).unwrap().with_serial_requests(true);
        let chat = QianfanChat::new(cfg).unwrap().with_serial_requests(true);
        let embedding = tokio::spawn(async move { embed.embed(&["text".into()]).await.unwrap() });
        let text = tokio::spawn(async move { chat.complete(None, "text", Some(10), None).await.unwrap() });
        let ocr = tokio::spawn(async move {
            request(&format!("{base}/ocr"), "local", &serde_json::json!({"messages":[]}), Duration::from_secs(2)).await.unwrap()
        });
        assert_eq!(embedding.await.unwrap().len(), 1);
        assert_eq!(text.await.unwrap(), "ok");
        assert!(ocr.await.unwrap().contains("ok"));
        server.join().unwrap();
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[tokio::test]
    async fn serial_queue_releases_after_http_failure() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0;4096]; let _ = stream.read(&mut buf);
            stream.write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        assert!(request(&url,"local",&serde_json::json!({}),std::time::Duration::from_secs(2)).await.is_err());
        server.join().unwrap();
        assert!(tokio::time::timeout(std::time::Duration::from_secs(1), acquire(true)).await.is_ok());
    }

    #[tokio::test]
    async fn serial_queue_wait_does_not_consume_chat_execution_budget() {
        let held = acquire(true).await;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut cfg = crate::QianfanConfig::for_embed("local".into(),None,None,None);
        cfg.chat_url = format!("http://{}",listener.local_addr().unwrap());
        drop(listener);
        let chat = crate::QianfanChat::new(cfg).unwrap().with_serial_requests(true);
        let waiting = tokio::spawn(async move {
            chat.complete_with_timeout(None,"text",Some(8),None,std::time::Duration::from_millis(20)).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        assert!(!waiting.is_finished(), "排队时不能耗尽20ms的执行预算");
        drop(held);
        // 不存在的测试端口使请求失败，仍须归还许可。
        assert!(tokio::time::timeout(std::time::Duration::from_secs(2),waiting).await.unwrap().unwrap().is_err());
    }
}
