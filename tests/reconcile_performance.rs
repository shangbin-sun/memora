//! Public-API regressions for bounded, lossless turn extraction.
//! The only HTTP traffic is to a std::net loopback mock; no live model is needed.

use async_trait::async_trait;
use memora::graph::{GraphExtractInput, GraphExtractOutput, GraphExtractor};
use memora::{
    reconcile_turn, HouseholdMember, MemberRole, MemoryEngine, MemoryService, PersonGraphQuery,
    QianfanChat, QianfanConfig, QianfanGraphExtractor, ReconcileReport, ReconcileRequest,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const NAMESPACE: &str = "reconcile-performance";
const OWNER: &str = "zhang-san";
const HEAD: &str = "张三喜欢喝龙井茶";
const MIDDLE: &str = "张三对花生过敏";
const TAIL: &str = "张三在实验小学工作";
const HISTORY_TAIL: &str = "历史尾标记仅用于消歧";
const SOURCE_MARKER: &str = "本批证据来源（JSON数据）：\n";
const HISTORY_MARKER: &str = "最近会话（仅消歧）：";

#[derive(Clone, Copy)]
enum ReplyMode {
    Normal,
    RetryOnce503,
    OverflowOnce,
    Deadline,
    InvalidMiddle,
    Gates,
    Birthday,
    SemanticFacts,
    MixedSchema,
    MissingMarker,
    Empty,
    SharedAndPersonal,
}

#[derive(Clone, Debug)]
struct Observed {
    source_id: Option<String>,
    source_text: String,
    history: String,
    attempt: usize,
    thinking_disabled: bool,
}

#[derive(Default)]
struct MockState {
    requests: Mutex<Vec<Observed>>,
    graph_inputs: Mutex<Vec<Value>>,
    attempts: Mutex<BTreeMap<String, usize>>,
    active: AtomicUsize,
    maximum_active: AtomicUsize,
    ops_calls: AtomicUsize,
    errors: Mutex<Vec<String>>,
}

struct MockLlm {
    address: SocketAddr,
    state: Arc<MockState>,
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
}

impl MockLlm {
    fn start(mode: ReplyMode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback mock");
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let state = Arc::new(MockState::default());
        let stop = Arc::new(AtomicBool::new(false));
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        let accept_thread = thread::spawn(move || {
            let mut connections = Vec::new();
            while !worker_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let state = Arc::clone(&worker_state);
                        connections.push(thread::spawn(move || {
                            if let Err(error) = serve_connection(stream, &state, mode) {
                                state.errors.lock().unwrap().push(error);
                            }
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => {
                        worker_state.errors.lock().unwrap().push(error.to_string());
                        break;
                    }
                }
            }
            for connection in connections {
                if connection.join().is_err() {
                    worker_state
                        .errors
                        .lock()
                        .unwrap()
                        .push("connection worker panicked".into());
                }
            }
        });
        Self {
            address,
            state,
            stop,
            accept_thread: Some(accept_thread),
        }
    }

    fn config(&self) -> QianfanConfig {
        let url = format!("http://{}/v1/chat/completions", self.address);
        QianfanConfig {
            api_key: "local-test-key".into(),
            chat_url: url.clone(),
            embed_url: url,
            chat_model: "local-mock-extractor".into(),
            embed_model: "unused".into(),
            embed_dim: 8,
            max_tokens: 6000,
            temperature: 0.2,
            top_p: 0.7,
            reasoning_split: false,
        }
    }

    fn chat(&self) -> QianfanChat {
        QianfanChat::new(self.config()).expect("construct local chat client")
    }

    fn requests(&self) -> Vec<Observed> {
        self.state.requests.lock().unwrap().clone()
    }

    fn assert_clean(&self) {
        let errors = self.state.errors.lock().unwrap().clone();
        assert!(errors.is_empty(), "mock errors: {errors:?}");
    }
}

impl Drop for MockLlm {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
    }
}

struct ActiveRequest<'a>(&'a MockState);

impl Drop for ActiveRequest<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

fn read_json_request(stream: &mut TcpStream) -> Result<Value, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 8192];
    let header_end = loop {
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        let count = stream.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("EOF before HTTP headers".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 1_000_000 {
            return Err("mock request exceeded safe bound".into());
        }
    };
    let headers = std::str::from_utf8(&bytes[..header_end]).map_err(|e| e.to_string())?;
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .ok_or_else(|| "HTTP request missing Content-Length".to_string())?;
    if content_length > 1_000_000 {
        return Err("mock Content-Length exceeded safe bound".into());
    }
    while bytes.len() < header_end + content_length {
        let count = stream.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("EOF before complete HTTP body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    serde_json::from_slice(&bytes[header_end..header_end + content_length])
        .map_err(|e| e.to_string())
}

fn send_response(stream: &mut TcpStream, status: u16, content: &str) -> Result<(), String> {
    send_response_with_finish_reason(stream, status, content, "stop")
}

fn send_response_with_finish_reason(
    stream: &mut TcpStream,
    status: u16,
    content: &str,
    finish_reason: &str,
) -> Result<(), String> {
    let body = if status == 200 {
        json!({
            "id": "mock-chat", "object": "chat.completion", "created": 0, "model": "local-mock-extractor",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }).to_string()
    } else {
        json!({"error": {"message": "deliberate transient failure"}}).to_string()
    };
    let reason = if status == 200 {
        "OK"
    } else {
        "Service Unavailable"
    };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn memory(source_id: &str, content: &str, tag: &str) -> Value {
    json!({"owner": OWNER, "content": content, "subjects": [OWNER], "tag": tag,
        "event_at": null, "source_id": source_id, "evidence": content})
}

fn encode_ndjson(memories: &[Value], done: bool) -> String {
    let mut lines = Vec::new();
    for memory in memories {
        // The host derives subjects and binds provenance to this request. Models
        // emit only the short record schema, never echo a source_id.
        let mut record = json!({
            "o": memory["owner"], "t": memory["tag"],
            "c": memory["content"], "e": memory["evidence"]
        });
        // Missing k defaults to m; missing d is an undated memory.
        if !memory["event_at"].is_null() {
            record["d"] = memory["event_at"].clone();
        }
        lines.push(record.to_string());
    }
    if done {
        lines.push(json!({"done": true}).to_string());
    }
    lines.join("\n")
}

fn memories_for_source(source_id: &str, text: &str, mode: ReplyMode) -> Vec<Value> {
    let mut memories = Vec::new();
    if matches!(mode, ReplyMode::Empty) {
        return memories;
    }
    if matches!(mode, ReplyMode::SharedAndPersonal) {
        return [("张三", "张三喜欢咖啡", "饮食"), ("家庭", "家里的滤芯每六个月更换", "生活习惯")]
            .into_iter().map(|(owner, content, tag)| json!({"owner":owner,"subjects":[owner],
                "content":content,"tag":tag,"evidence":content,"source_id":source_id,"event_at":null})).collect();
    }
    if matches!(mode, ReplyMode::SemanticFacts) {
        return vec![
            memory(source_id, "张三可能明年搬家，尚未确定", "生活习惯"),
            memory(source_id, "张三如果胃不舒服就不吃辣", "饮食"),
            memory(source_id, "张三喜欢《什么是数学》这本书", "工作学习"),
            memory(source_id, "张三讨厌杀人题材的暴力电影", "生活习惯"),
        ];
    }
    if matches!(mode, ReplyMode::Birthday) {
        return vec![json!({"owner":"xiaohang", "subjects":["xiaohang"],
            "content":"小航下个月6号就要过生日了", "tag":"生活习惯", "event_at":null,
            "source_id":source_id, "evidence":"小航下个月6号就要过生日了"})];
    }
    for (fact, tag) in [(HEAD, "饮食"), (MIDDLE, "医疗健康"), (TAIL, "工作学习")] {
        if text.contains(fact) {
            memories.push(memory(source_id, fact, tag));
        }
    }
    // Every valid chunk contains at least one explicitly grounded fact; returning
    // one anchor makes missed middle chunks observable even without a marker.
    if let Some(fact) = text
        .split('。')
        .find(|part| part.starts_with("张三记忆编号"))
    {
        memories.push(memory(source_id, fact, "生活习惯"));
    }
    if matches!(mode, ReplyMode::Gates) {
        memories.clear();
        // Semantic attribution comes from the model; the host only checks IDs
        // and owner/subjects consistency, not names/keywords in the source.
        memories.push(memory(source_id, HEAD, "饮食"));
        memories.push(memory(source_id, MIDDLE, "invented-private-tag"));
        let mut mismatched = memory("another-source", TAIL, "工作学习");
        mismatched["evidence"] = json!(TAIL);
        memories.push(mismatched);
        let mut unknown = memory(source_id, "喜欢吃苹果", "饮食");
        unknown["owner"] = json!("unknown-person");
        unknown["subjects"] = json!(["unknown-person"]);
        memories.push(unknown);
    }
    memories
}

fn serve_connection(
    mut stream: TcpStream,
    state: &MockState,
    mode: ReplyMode,
) -> Result<(), String> {
    let request = read_json_request(&mut stream)?;
    let prompt = request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.iter().find(|message| message["role"] == "user"))
        .and_then(|message| message["content"].as_str())
        .ok_or("missing user prompt")?;
    if matches!(mode, ReplyMode::Deadline) {
        state.requests.lock().unwrap().push(Observed {
            source_id: None,
            source_text: prompt.to_string(),
            history: String::new(),
            attempt: 1,
            thinking_disabled: request["enable_thinking"] == false,
        });
        let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        state.maximum_active.fetch_max(active, Ordering::SeqCst);
        let active_guard = ActiveRequest(state);
        thread::sleep(Duration::from_millis(90));
        drop(active_guard);
        // The cancelled client may close its socket before the delayed response;
        // that is expected for this deadline test, not a mock implementation error.
        let _ = send_response(&mut stream, 200, "{\"ok\":true}");
        return Ok(());
    }
    // A graph completion uses the same local endpoint but a different compact
    // prompt. Decode its DATA JSON, not source words, to emulate an LLM response.
    if matches!(mode, ReplyMode::Birthday) && !prompt.contains(SOURCE_MARKER) {
        if let Some(index) = prompt.rfind("\n{") {
            let mut jsons =
                serde_json::Deserializer::from_str(&prompt[index + 1..]).into_iter::<Value>();
            let data = jsons
                .next()
                .ok_or("missing graph DATA JSON")?
                .map_err(|e| e.to_string())?;
            state.graph_inputs.lock().unwrap().push(data);
            let graph = json!({"complete":true, "entities":[
                {"temp_id":"child","name":"小航","type":"person","member_id":"xiaohang"},
                {"temp_id":"birthday","name":"11月6日生日","type":"concept"}
            ], "triples":[{"subject":"child","relation":"有生日","object":"birthday",
                "evidence_text":"小航下个月6号就要过生日了", "privacy":"family_shared",
                "attrs":{"month":11,"day":6,"next_occurrence":"2026-11-06"}}], "pending":[]});
            return send_response(&mut stream, 200, &graph.to_string());
        }
    }
    let Some((_, source_json)) = prompt.rsplit_once(SOURCE_MARKER) else {
        state.ops_calls.fetch_add(1, Ordering::SeqCst);
        state.requests.lock().unwrap().push(Observed {
            source_id: None,
            source_text: String::new(),
            history: String::new(),
            attempt: 1,
            thinking_disabled: request["enable_thinking"] == false,
        });
        return send_response(&mut stream, 200, "{\"ops\":[]}");
    };
    let source: Value = serde_json::from_str(source_json).map_err(|e| e.to_string())?;
    let source_id = source["source_id"]
        .as_str()
        .ok_or("missing source_id")?
        .to_string();
    let text = source["text"]
        .as_str()
        .ok_or("missing source.text")?
        .to_string();
    let history_json = prompt
        .split_once(HISTORY_MARKER)
        .and_then(|(_, tail)| tail.split_once('\n').map(|(head, _)| head))
        .ok_or("missing history framing")?;
    let history: String = serde_json::from_str(history_json).map_err(|e| e.to_string())?;
    let attempt = {
        let mut attempts = state.attempts.lock().unwrap();
        let attempt = attempts.entry(source_id.clone()).or_default();
        *attempt += 1;
        *attempt
    };
    state.requests.lock().unwrap().push(Observed {
        source_id: Some(source_id.clone()),
        source_text: text.clone(),
        history,
        attempt,
        thinking_disabled: request["enable_thinking"] == false,
    });
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.maximum_active.fetch_max(active, Ordering::SeqCst);
    let active_guard = ActiveRequest(state);
    // Deliberately finish later chunks earlier, so stable output cannot depend on
    // network completion order. Long enough to expose both in-flight calls.
    let index = source_id
        .rsplit(':')
        .next()
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(0);
    thread::sleep(Duration::from_millis(if index == 0 { 160 } else { 90 }));
    if matches!(mode, ReplyMode::RetryOnce503) && source_id.ends_with(":1") && attempt == 1 {
        drop(active_guard);
        return send_response(&mut stream, 503, "");
    }
    if matches!(mode, ReplyMode::InvalidMiddle) && text.contains(MIDDLE) {
        drop(active_guard);
        return send_response(&mut stream, 200, "{not complete JSON");
    }
    let memories = memories_for_source(&source_id, &text, mode);
    let extraction = if matches!(mode, ReplyMode::Gates) {
        // Retain a legacy response to cover compatibility and its explicit
        // wrong-source/unknown-owner safety gates.
        json!({"register": [], "memories": memories, "retractions": []}).to_string()
    } else if matches!(mode, ReplyMode::MixedSchema) {
        let mut lines = encode_ndjson(&memories, false)
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        lines.insert(
            1,
            json!({"o": OWNER, "t": "饮食", "c": 42, "e": HEAD}).to_string(),
        );
        lines.push(json!({"done": true}).to_string());
        lines.join("\n")
    } else {
        encode_ndjson(&memories, !matches!(mode, ReplyMode::MissingMarker))
    };
    if matches!(mode, ReplyMode::OverflowOnce) && source_id == "user-text:1" && attempt == 1 {
        drop(active_guard);
        // This is one complete HTTP response, not transport streaming. Closed
        // memory lines survive a provider length stop; the unfinished tail does
        // not authorize completion, subdivision, or re-extraction.
        let truncated = format!("{}\n{{\"o\":", encode_ndjson(&memories, false));
        return send_response_with_finish_reason(&mut stream, 200, &truncated, "length");
    }
    // Completion is measured before releasing the response to the client. This
    // avoids counting a finished connection while its replacement is accepted.
    drop(active_guard);
    send_response(&mut stream, 200, &extraction)
}

fn exact_query() -> String {
    // Fixed character offsets (not UTF-8 bytes), with a complete sentence in
    // every 1000-character segment. Head/middle/tail must survive splitting.
    let mut chars = vec!['填'; 28_000];
    for offset in (1000..27_000).step_by(1000) {
        let anchor = format!("。张三记忆编号{offset}是固定值。 ");
        for (target, ch) in chars[offset..].iter_mut().zip(anchor.chars()) {
            *target = ch;
        }
    }
    for (offset, text) in [
        (0, format!("{HEAD}。")),
        (14_030, format!("。{MIDDLE}。")),
        (28_000 - TAIL.chars().count() - 2, format!("。{TAIL}。")),
    ] {
        for (target, ch) in chars[offset..].iter_mut().zip(text.chars()) {
            *target = ch;
        }
    }
    let query: String = chars.into_iter().collect();
    assert_eq!(query.chars().count(), 28_000);
    assert!(query.starts_with(HEAD));
    assert!(query.ends_with(&format!("{TAIL}。")));
    query
}

fn request(query: &str) -> ReconcileRequest {
    serde_json::from_value(json!({
        "query": query, "namespace": NAMESPACE, "speaker_id": OWNER,
        "message_at_ms": 1_792_780_800_000i64,
        "transcript": format!("过期历史前缀{}{}", "史".repeat(8000), HISTORY_TAIL)
    }))
    .expect("deserialize public ReconcileRequest")
}

async fn engine(dir: &tempfile::TempDir, graph: Option<Arc<NeverRunGraph>>) -> MemoryEngine {
    let mut builder = MemoryEngine::builder()
        .db_path(dir.path().join("performance.db"))
        .namespace(NAMESPACE)
        .graph_projection_worker(false);
    if let Some(graph) = graph {
        builder = builder.graph_extractor(graph);
    }
    let engine = builder.build().await.unwrap();
    for (id, name) in [(OWNER, "张三"), ("li-si", "李四")] {
        engine
            .upsert_member(
                Some(NAMESPACE),
                HouseholdMember {
                    member_id: id.into(),
                    display_name: name.into(),
                    role: MemberRole::Other,
                    aliases: vec![],
                    age_band: None,
                },
            )
            .await
            .unwrap();
    }
    engine
}

#[derive(Default)]
struct NeverRunGraph(AtomicUsize);

#[async_trait]
impl GraphExtractor for NeverRunGraph {
    fn extractor_id(&self) -> &str {
        "deferred-only-test"
    }
    async fn extract(&self, _input: GraphExtractInput) -> memora::Result<GraphExtractOutput> {
        self.0.fetch_add(1, Ordering::SeqCst);
        // Any foreground call would hang until the outer test deadline; this
        // detects accidental graph waits without timing real model latency.
        std::future::pending::<memora::Result<GraphExtractOutput>>().await
    }
}

fn expected_sources(query: &str) -> BTreeMap<String, String> {
    let chunks = memora::graph::split_source_text(query, 2500, 240);
    let multiple = chunks.len() > 1;
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            (
                if multiple {
                    format!("user-text:{index}")
                } else {
                    "user-text".into()
                },
                text,
            )
        })
        .collect()
}

fn assert_coverage(mock: &MockLlm, query: &str, total: usize) {
    let expected = expected_sources(query);
    assert_eq!(total, expected.len());
    assert!(total >= 5, "28k source should exercise rolling concurrency");
    let observed = mock.requests();
    let first_attempts: BTreeMap<_, _> = observed
        .iter()
        .filter(|call| call.attempt == 1)
        .filter_map(|call| {
            call.source_id
                .as_ref()
                .map(|id| (id.clone(), call.source_text.clone()))
        })
        .collect();
    assert_eq!(
        first_attempts, expected,
        "every full chunk, including the tail, must be sent exactly"
    );
    for call in observed.iter().filter(|call| call.source_id.is_some()) {
        assert!(call.source_text.chars().count() <= 2500);
        assert!(call.history.chars().count() <= 1200);
        assert!(call.history.ends_with(HISTORY_TAIL));
        assert!(!call.history.contains("过期历史前缀"));
        assert!(
            call.thinking_disabled,
            "structured extraction must disable reasoning output"
        );
    }
    let maximum = mock.state.maximum_active.load(Ordering::SeqCst);
    assert!(
        maximum <= 2,
        "observed {maximum} concurrent requests exceeds the hard limit"
    );
    assert!(
        maximum >= 2,
        "requests were serialized instead of bounded-concurrent"
    );
    assert_eq!(
        mock.state.ops_calls.load(Ordering::SeqCst),
        0,
        "fresh store should skip candidate-free ops"
    );
    mock.assert_clean();
}

fn fact_contents(report: &ReconcileReport) -> BTreeSet<String> {
    report
        .facts
        .iter()
        .map(|fact| fact["content"].as_str().unwrap().to_string())
        .collect()
}

fn assert_success(report: &ReconcileReport) {
    assert!(!report.skipped);
    assert!(!report.extraction.partial);
    assert!(report.extraction.failed_source_ids.is_empty());
    assert_eq!(report.extraction.invalid_records, 0);
    assert!(report.extraction.truncated_source_ids.is_empty());
    assert_eq!(
        report.extraction.chunks_completed,
        report.extraction.chunks_total
    );
    assert_eq!(report.written_ids.len(), report.memories_kept);
    assert!(report.revised_ids.is_empty());
    assert_eq!(report.forgotten, 0);
    assert!(!report.fallback_used);
    assert!(report.ops_raw.is_none());
    let contents = fact_contents(report);
    for fact in [HEAD, MIDDLE, TAIL] {
        assert!(
            contents.iter().any(|content| content.contains(fact)),
            "missing {fact}"
        );
    }
    assert!(report.facts.iter().all(|fact| fact["owner"] == OWNER));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exact_28000_char_query_is_lossless_bounded_and_stable_without_graph_waits() {
    let query = exact_query();
    let mock = MockLlm::start(ReplyMode::Normal);
    let graph = Arc::new(NeverRunGraph::default());
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, Some(Arc::clone(&graph))).await;
    let chat = mock.chat();
    let report = tokio::time::timeout(
        Duration::from_secs(20),
        reconcile_turn(&engine, &chat, request(&query)),
    )
    .await
    .expect("reconcile must not wait for graph projection")
    .unwrap();
    assert_success(&report);
    assert_coverage(&mock, &query, report.extraction.chunks_total);
    assert_eq!(
        graph.0.load(Ordering::SeqCst),
        0,
        "foreground reconcile must defer graph extraction"
    );
    // A second fresh database and reversed network latency must produce identical
    // ordered facts, not merely equal counts or an unordered set of survivors.
    let second_mock = MockLlm::start(ReplyMode::Normal);
    let second_dir = tempfile::tempdir().unwrap();
    let second_engine = self::engine(&second_dir, None).await;
    let second_chat = second_mock.chat();
    let second = reconcile_turn(&second_engine, &second_chat, request(&query))
        .await
        .unwrap();
    assert_success(&second);
    assert_eq!(
        report.facts, second.facts,
        "result order must follow source order"
    );
    assert_coverage(&second_mock, &query, second.extraction.chunks_total);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transient_503_retries_only_that_chunk_without_duplicate_or_lost_facts() {
    let query = exact_query();
    let mock = MockLlm::start(ReplyMode::RetryOnce503);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let chat = mock.chat();
    let report = reconcile_turn(&engine, &chat, request(&query))
        .await
        .unwrap();
    assert_success(&report);
    assert_coverage(&mock, &query, report.extraction.chunks_total);
    let attempts = mock.state.attempts.lock().unwrap();
    assert_eq!(attempts.get("user-text:1"), Some(&2));
    assert!(attempts
        .iter()
        .all(|(id, count)| *count == if id == "user-text:1" { 2 } else { 1 }));
    assert_eq!(
        mock.requests()
            .iter()
            .filter(|call| call.source_id.is_some())
            .count(),
        report.extraction.chunks_total + 1
    );
    assert_eq!(
        fact_contents(&report).len(),
        report.facts.len(),
        "retry must not duplicate records"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_middle_chunk_reports_partial_and_keeps_other_chunks_including_tail() {
    let query = exact_query();
    let mock = MockLlm::start(ReplyMode::InvalidMiddle);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let chat = mock.chat();
    let report = reconcile_turn(&engine, &chat, request(&query))
        .await
        .unwrap();
    assert!(!report.skipped);
    assert!(report.extraction.partial);
    let failed: BTreeSet<_> = expected_sources(&query)
        .into_iter()
        .filter(|(_, text)| text.contains(MIDDLE))
        .map(|(id, _)| id)
        .collect();
    assert!(!failed.is_empty());
    assert_eq!(
        report
            .extraction
            .failed_source_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>(),
        failed
    );
    assert_eq!(
        report.extraction.chunks_completed + failed.len(),
        report.extraction.chunks_total
    );
    assert_eq!(report.written_ids.len(), report.memories_kept);
    assert!(!report.written_ids.is_empty());
    assert!(
        !report.fallback_used,
        "partial source failure cannot authorize broad fallback"
    );
    let contents = fact_contents(&report);
    assert!(contents.iter().any(|fact| fact.contains(HEAD)));
    assert!(contents.iter().any(|fact| fact.contains(TAIL)));
    assert!(!contents.iter().any(|fact| fact.contains(MIDDLE)));
    assert_coverage(&mock, &query, report.extraction.chunks_total);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn evidence_owner_id_consistency_and_tag_schema_are_validated_without_semantic_rules() {
    let query = format!("{HEAD}。{MIDDLE}。{TAIL}。喜欢吃苹果。");
    let mock = MockLlm::start(ReplyMode::Gates);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let chat = mock.chat();
    let report = reconcile_turn(&engine, &chat, request(&query))
        .await
        .unwrap();
    assert!(report.extraction.partial);
    assert_eq!(report.extraction.chunks_total, 1);
    assert_eq!(report.extraction.chunks_completed, 0);
    assert_eq!(report.extraction.invalid_records, 1);
    assert_eq!(report.extraction.failed_source_ids, ["user-text"]);
    assert!(report.extraction.truncated_source_ids.is_empty());
    assert_eq!(
        report.facts.len(),
        2,
        "wrong source and unknown owner must be rejected"
    );
    assert!(report.facts.iter().all(|fact| fact["owner"] == OWNER));
    let head = report
        .facts
        .iter()
        .find(|fact| fact["content"].as_str().unwrap().contains(HEAD))
        .unwrap();
    assert_eq!(head["tag"], "饮食");
    let middle = report
        .facts
        .iter()
        .find(|fact| fact["content"].as_str().unwrap().contains(MIDDLE))
        .unwrap();
    assert_eq!(
        middle["tag"], "其他",
        "unrecognized model tag must not become a new tag"
    );
    assert_eq!(report.written_ids.len(), 2);
    assert_eq!(mock.state.ops_calls.load(Ordering::SeqCst), 0);
    assert!(engine
        .list_tags(Some(NAMESPACE))
        .await
        .unwrap()
        .iter()
        .all(|tag| tag.tag_id != "invented-private-tag"));
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn birthday_statement_followed_by_gift_question_is_written_to_named_member() {
    let mock = MockLlm::start(ReplyMode::Birthday);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    engine
        .upsert_member(
            Some(NAMESPACE),
            HouseholdMember {
                member_id: "xiaohang".into(),
                display_name: "小航".into(),
                role: MemberRole::Child,
                aliases: vec![],
                age_band: None,
            },
        )
        .await
        .unwrap();
    let query = "小航下个月6号就要过生日了，请问送什么礼物好呢";
    let report = reconcile_turn(&engine, &mock.chat(), request(query))
        .await
        .unwrap();
    assert_eq!(report.memories_extracted, 1);
    assert_eq!(report.memories_kept, 1);
    assert_eq!(report.written_ids.len(), 1);
    assert_eq!(report.facts[0]["owner"], "xiaohang");
    assert_eq!(report.facts[0]["content"], "小航下个月6号就要过生日了");
    assert!(
        report.facts[0]["event_at"].is_null(),
        "birthday is not a one-off expiring schedule"
    );
    assert!(!report.warnings.iter().any(|w| w.contains("non-assertion")));
    assert_eq!(mock.state.ops_calls.load(Ordering::SeqCst), 0);
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_turn_extracts_shared_and_new_person_without_a_speaker_or_separate_storage() {
    let mock = MockLlm::start(ReplyMode::SharedAndPersonal);
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder().db_path(dir.path().join("owners.db"))
        .namespace(NAMESPACE).graph_projection_worker(false).build().await.unwrap();
    let mut req = request("张三喜欢咖啡，家里的滤芯每六个月更换。");
    req.speaker_id = None;
    req.session_id = Some("conversation-1".into());
    let report = reconcile_turn(&engine, &mock.chat(), req).await.unwrap();
    assert_eq!(report.memories_kept, 2);
    assert_eq!(report.written_ids.len(), 2);
    let members = engine.list_members(Some(NAMESPACE)).await.unwrap();
    assert!(members.iter().any(|m| m.member_id == "张三"));
    assert!(members.iter().any(|m| m.member_id == "公共"));
    assert!(!members.iter().any(|m| m.member_id == "家庭"));
    for (owner, content) in [("张三","张三喜欢咖啡"),("公共","家里的滤芯每六个月更换")] {
        let page = engine.list(memora::ListQuery {namespace:Some(NAMESPACE.into()),member_id:Some(owner.into()),..Default::default()}).await.unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.items[0].content, content);
        assert_eq!(page.items[0].session_id.as_deref(), Some("conversation-1"));
        assert!(page.items[0].metadata["source"]["evidence"].is_string());
    }
    // The existing same-owner candidate/update path must not create another copy.
    let mut req = request("张三喜欢咖啡，家里的滤芯每六个月更换。");
    req.speaker_id = None;
    req.session_id = Some("conversation-2".into());
    let repeat = reconcile_turn(&engine, &mock.chat(), req).await.unwrap();
    assert!(repeat.written_ids.is_empty());
    let all = engine.list(memora::ListQuery {namespace:Some(NAMESPACE.into()),..Default::default()}).await.unwrap();
    assert_eq!(all.total, 2);
    mock.assert_clean();
    engine.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn birthday_with_unknown_member_is_not_guessed_or_written_to_public() {
    let mock = MockLlm::start(ReplyMode::Birthday);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let report = reconcile_turn(
        &engine,
        &mock.chat(),
        request("小航下个月6号就要过生日了，请问送什么礼物好呢"),
    )
    .await
    .unwrap();
    assert_eq!(report.memories_extracted, 1);
    assert_eq!(report.memories_kept, 0);
    assert!(report.written_ids.is_empty());
    assert!(
        !report.warnings.is_empty(),
        "invalid model owner must be diagnosed"
    );
    assert!(!engine
        .list_members(Some(NAMESPACE))
        .await
        .unwrap()
        .iter()
        .any(|member| member.member_id == "xiaohang"));
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn llm_birthday_memory_projects_graph_from_original_evidence_speaker_and_time() {
    let mock = MockLlm::start(ReplyMode::Birthday);
    let dir = tempfile::tempdir().unwrap();
    let graph_llm = Arc::new(QianfanGraphExtractor::new(mock.config()).unwrap());
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("birthday-graph.db"))
        .namespace(NAMESPACE)
        .graph_projection_worker(false)
        .graph_extractor(graph_llm)
        .build()
        .await
        .unwrap();
    for (id, name, role) in [
        (OWNER, "张三", MemberRole::Self_),
        ("xiaohang", "小航", MemberRole::Child),
    ] {
        engine
            .upsert_member(
                Some(NAMESPACE),
                HouseholdMember {
                    member_id: id.into(),
                    display_name: name.into(),
                    role,
                    aliases: vec![],
                    age_band: None,
                },
            )
            .await
            .unwrap();
    }
    let text = "小航下个月6号就要过生日了，请问送什么礼物好呢";
    let mut req = request(text);
    let anchor = chrono::DateTime::parse_from_rfc3339("2026-10-25T13:44:25+08:00")
        .unwrap()
        .timestamp_millis();
    req.message_at_ms = Some(anchor);
    let report = reconcile_turn(&engine, &mock.chat(), req).await.unwrap();
    assert_eq!(report.written_ids.len(), 1);
    assert_eq!(report.facts[0]["owner"], "xiaohang");
    assert!(
        mock.state.graph_inputs.lock().unwrap().is_empty(),
        "foreground may not block on graph LLM"
    );
    assert_eq!(engine.retry_graph_projections(8).await.unwrap(), 1);
    let inputs = mock.state.graph_inputs.lock().unwrap().clone();
    assert_eq!(inputs.len(), 1);
    assert_eq!(
        inputs[0]["source_text"], "小航下个月6号就要过生日了",
        "graph must see the model-selected original evidence, not normalized memory prose"
    );
    assert_eq!(
        inputs[0]["speaker_member_id"], OWNER,
        "speaker is not birthday fact owner"
    );
    assert_eq!(inputs[0]["message_at_ms"], anchor);
    let graph = engine
        .query_person_graph(PersonGraphQuery {
            member_id: "xiaohang".into(),
            namespace: Some(NAMESPACE.into()),
            viewer_member_id: Some("xiaohang".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let birthday = graph
        .facts
        .iter()
        .find(|fact| fact.relation == "有生日")
        .expect("LLM birthday edge must be persisted");
    assert_eq!(birthday.attrs["month"], 11);
    assert_eq!(birthday.attrs["day"], 6);
    assert_eq!(birthday.attrs["next_occurrence"], "2026-11-06");
    assert_eq!(
        birthday.evidence_text.as_deref(),
        Some("小航下个月6号就要过生日了")
    );
    assert_eq!(
        birthday.source_memory_id.unwrap().to_string(),
        report.written_ids[0]
    );
    assert_eq!(mock.state.ops_calls.load(Ordering::SeqCst), 0);
    engine.shutdown();
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn llm_semantic_facts_are_not_rejected_by_question_conditional_or_topic_keywords() {
    let mock = MockLlm::start(ReplyMode::SemanticFacts);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let text = "张三可能明年搬家，尚未确定。张三如果胃不舒服就不吃辣。张三喜欢《什么是数学》这本书。张三讨厌杀人题材的暴力电影。";
    let report = reconcile_turn(&engine, &mock.chat(), request(text))
        .await
        .unwrap();
    assert_eq!(report.memories_extracted, 4);
    assert_eq!(
        report.memories_kept, 4,
        "semantic decision belongs to LLM, not keyword filters"
    );
    assert_eq!(report.written_ids.len(), 4);
    assert_eq!(
        fact_contents(&report),
        memories_for_source("user-text", text, ReplyMode::SemanticFacts)
            .iter()
            .map(|fact| fact["content"].as_str().unwrap().to_string())
            .collect()
    );
    assert!(!report.fallback_used);
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_llm_extraction_does_not_trigger_rule_fallback_or_rule_roster_creation() {
    for text in [
        "请记住，爸爸喜欢吃肉。",
        "小航下个月几号过生日？",
        "这份说明中说不要记住日志文件名，爸爸在例子里。",
    ] {
        let mock = MockLlm::start(ReplyMode::Empty);
        let dir = tempfile::tempdir().unwrap();
        let engine = engine(&dir, None).await;
        let report = reconcile_turn(&engine, &mock.chat(), request(text))
            .await
            .unwrap();
        assert_eq!(
            mock.requests().len(),
            1,
            "no rule may skip LLM interpretation: {text}"
        );
        assert_eq!(report.memories_extracted, 0);
        assert!(!report.extraction.partial);
        assert_eq!(report.extraction.chunks_completed, 1);
        assert_eq!(report.extraction.invalid_records, 0);
        assert!(report.extraction.truncated_source_ids.is_empty());
        assert_eq!(report.extract_raw.as_deref(), Some("{\"done\":true}"));
        assert!(report.written_ids.is_empty());
        assert!(!report.fallback_used);
        assert!(!engine
            .list_members(Some(NAMESPACE))
            .await
            .unwrap()
            .iter()
            .any(|member| member.display_name == "爸爸" || member.display_name == "小航"));
        assert!(mock.state.graph_inputs.lock().unwrap().is_empty());
        mock.assert_clean();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn truncated_response_keeps_valid_lines_marks_partial_and_does_not_reextract_completed_facts()
{
    let query = exact_query();
    let initial = expected_sources(&query);
    let affected_id = "user-text:1";
    let affected_text = initial.get(affected_id).expect("second initial chunk");
    let affected_memories = memories_for_source(affected_id, affected_text, ReplyMode::Normal);
    assert!(
        !affected_memories.is_empty(),
        "truncation must preserve a closed memory line"
    );
    let mock = MockLlm::start(ReplyMode::OverflowOnce);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let chat = mock.chat();
    let report = tokio::time::timeout(
        Duration::from_secs(20),
        reconcile_turn(&engine, &chat, request(&query)),
    )
    .await
    .expect("truncated extraction must stay bounded and not wait for graph")
    .unwrap();
    assert!(!report.skipped);
    assert!(report.extraction.partial);
    assert_eq!(report.extraction.chunks_total, initial.len());
    assert_eq!(report.extraction.chunks_completed, initial.len() - 1);
    assert_eq!(report.extraction.failed_source_ids, [affected_id]);
    assert_eq!(report.extraction.truncated_source_ids, [affected_id]);
    assert_eq!(report.written_ids.len(), report.memories_kept);
    assert!(!report.fallback_used);
    assert!(report.ops_raw.is_none());

    // Every closed record is kept, even if all named facts happen to precede
    // the cut. Present facts must never turn a truncated source into complete.
    let expected_contents: BTreeSet<_> = initial
        .iter()
        .flat_map(|(id, text)| memories_for_source(id, text, ReplyMode::Normal))
        .map(|fact| fact["content"].as_str().unwrap().to_string())
        .collect();
    let contents = fact_contents(&report);
    assert_eq!(contents, expected_contents);
    for fact in [HEAD, MIDDLE, TAIL] {
        assert!(
            contents.contains(fact),
            "complete records must retain {fact}"
        );
    }
    for fact in affected_memories {
        assert!(contents.contains(fact["content"].as_str().unwrap()));
    }
    assert_eq!(
        contents.len(),
        report.facts.len(),
        "no duplicate re-extraction"
    );
    assert!(report.facts.iter().all(|fact| fact["owner"] == OWNER));

    let observed = mock.requests();
    assert_eq!(
        observed.len(),
        initial.len(),
        "length must not trigger child requests"
    );
    assert!(observed.iter().all(|call| call.attempt == 1));
    assert!(observed.iter().all(|call| call
        .source_id
        .as_ref()
        .is_some_and(|id| initial.contains_key(id))));
    assert_coverage(&mock, &query, report.extraction.chunks_total);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_good_bad_schema_good_done_keeps_valid_records_and_reports_one_invalid_record() {
    let query = format!("{HEAD}。{TAIL}。");
    let mock = MockLlm::start(ReplyMode::MixedSchema);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let report = reconcile_turn(&engine, &mock.chat(), request(&query))
        .await
        .unwrap();
    assert!(!report.skipped);
    assert!(report.extraction.partial);
    assert_eq!(report.extraction.chunks_total, 1);
    assert_eq!(report.extraction.chunks_completed, 0);
    assert_eq!(report.extraction.invalid_records, 1);
    assert_eq!(report.extraction.failed_source_ids, ["user-text"]);
    assert!(report.extraction.truncated_source_ids.is_empty());
    assert_eq!(
        fact_contents(&report),
        BTreeSet::from([HEAD.into(), TAIL.into()])
    );
    assert_eq!(report.memories_extracted, 2);
    assert_eq!(report.memories_kept, 2);
    assert_eq!(
        report.written_ids.len(),
        2,
        "both good sibling records must persist"
    );
    assert!(!report.fallback_used);
    assert_eq!(
        mock.requests().len(),
        1,
        "bad schema must not retry valid siblings"
    );
    assert_eq!(mock.state.ops_calls.load(Ordering::SeqCst), 0);
    mock.assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn missing_done_marker_keeps_valid_records_but_does_not_mark_source_complete() {
    let query = format!("{HEAD}。{TAIL}。");
    let mock = MockLlm::start(ReplyMode::MissingMarker);
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(&dir, None).await;
    let report = reconcile_turn(&engine, &mock.chat(), request(&query))
        .await
        .unwrap();
    assert!(!report.skipped);
    assert!(report.extraction.partial);
    assert_eq!(report.extraction.chunks_total, 1);
    assert_eq!(report.extraction.chunks_completed, 0);
    assert_eq!(report.extraction.invalid_records, 0);
    assert_eq!(report.extraction.failed_source_ids, ["user-text"]);
    assert!(report.extraction.truncated_source_ids.is_empty());
    assert_eq!(
        fact_contents(&report),
        BTreeSet::from([HEAD.into(), TAIL.into()])
    );
    assert_eq!(report.memories_extracted, 2);
    assert_eq!(report.memories_kept, 2);
    assert_eq!(
        report.written_ids.len(),
        2,
        "valid records survive a missing marker"
    );
    assert!(!report.fallback_used);
    assert_eq!(
        mock.requests().len(),
        1,
        "missing marker must not re-extract facts"
    );
    assert_eq!(mock.state.ops_calls.load(Ordering::SeqCst), 0);
    mock.assert_clean();
}

#[test]
fn short_record_encoder_omits_host_bound_fields_and_uses_done_only_for_empty_output() {
    let raw = encode_ndjson(&[memory("must-not-echo", HEAD, "饮食")], true);
    let lines: Vec<Value> = raw
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        lines,
        vec![
            json!({"o": OWNER, "t": "饮食", "c": HEAD, "e": HEAD}),
            json!({"done": true})
        ]
    );
    assert_eq!(encode_ndjson(&[], true), "{\"done\":true}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_total_deadline_cancels_without_retrying_and_releases_all_permits() {
    let mock = MockLlm::start(ReplyMode::Deadline);
    let chat = mock.chat();
    let error = chat
        .complete_with_timeout(
            None,
            "deadline-probe",
            Some(100),
            Some(0.2),
            Duration::from_millis(20),
        )
        .await
        .expect_err("90ms mock cannot finish within a 20ms total deadline");
    assert!(error.to_string().to_ascii_lowercase().contains("timeout"));
    // Let the cancelled socket settle and any wrongly scheduled retry appear.
    tokio::time::sleep(Duration::from_millis(140)).await;
    assert_eq!(
        mock.requests().len(),
        1,
        "total deadline must not cause three retries"
    );

    // Hold both permits in the follow-up batch. Merely proving one later
    // request succeeds would miss a leaked permit in a two-permit limiter.
    let mut follow_ups = tokio::task::JoinSet::new();
    for index in 0..2 {
        let chat = chat.clone();
        follow_ups.spawn(async move {
            chat.complete_with_timeout(
                None,
                &format!("permit-probe-{index}"),
                Some(100),
                Some(0.2),
                Duration::from_secs(2),
            )
            .await
        });
    }
    for _ in 0..100 {
        if mock.requests().len() >= 3 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(
        mock.requests().len(),
        3,
        "both permits must be available after cancellation"
    );
    let mut completed = 0;
    while let Some(joined) = follow_ups.join_next().await {
        assert_eq!(joined.unwrap().unwrap(), "{\"ok\":true}");
        completed += 1;
    }
    assert_eq!(completed, 2);
    assert_eq!(
        mock.requests().len(),
        3,
        "successful follow-ups must not retry"
    );
    assert_eq!(
        mock.state.maximum_active.load(Ordering::SeqCst),
        2,
        "follow-up calls must overlap on both released permits"
    );
    mock.assert_clean();
}
