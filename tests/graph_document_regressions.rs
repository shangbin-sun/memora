//! Offline document ingestion regressions: deterministic loopback HTTP + temp SQLite.
use memora::graph::{
    merge_graph_extract_outputs, parse_and_validate_llm_json, GraphExtractInput, GraphExtractor,
    GraphOverviewQuery,
};
use memora::{MemoryEngine, QianfanConfig, QianfanGraphExtractor};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn extract(
    text: &str,
    who: &str,
    what: &str,
    person_id: &str,
    object_id: &str,
) -> memora::graph::GraphExtractOutput {
    parse_and_validate_llm_json(
        &json!({
            "entities": [{"temp_id":person_id,"name":who,"type":"person"},
                         {"temp_id":object_id,"name":what,"type":"concept"}],
            "triples": [{"subject":person_id,"relation":"likes","object":object_id,
                         "evidence_text":text,"privacy":"family_shared"}], "pending":[]
        })
        .to_string(),
        text,
        &[],
    )
    .unwrap()
}

#[test]
fn merge_rebinds_local_ids_and_deduplicated_entities_without_losing_conditions() {
    let first = extract("张三喜欢茶", "张三", "茶", "e1", "e2");
    let mut duplicate = extract("张三喜欢茶", "张三", "茶", "p9", "o9");
    duplicate.triples[0].condition_text = Some("饭后".into());
    let second = extract("李四喜欢咖啡", "李四", "咖啡", "e1", "e2");
    let out = merge_graph_extract_outputs("test", [first.clone(), second, first, duplicate]);
    assert_eq!(out.entities.len(), 4);
    assert_eq!(out.triples.len(), 3);
    for triple in &out.triples {
        let subject = out
            .entities
            .iter()
            .find(|e| e.temp_id == triple.subject)
            .unwrap();
        let object = out
            .entities
            .iter()
            .find(|e| e.temp_id == triple.object)
            .unwrap();
        assert_eq!(
            format!("{}喜欢{}", subject.name, object.name),
            triple.evidence_text
        );
    }
    assert_eq!(
        out.triples
            .iter()
            .filter(|t| t.condition_text.is_some())
            .count(),
        1
    );
}

#[test]
fn conflicting_ids_inside_one_response_fail_before_ingest() {
    let payload = json!({"entities":[
        {"temp_id":"e1","name":"张三","type":"person"},
        {"temp_id":"e1","name":"李四","type":"person"},
        {"temp_id":"e2","name":"茶","type":"concept"}],
        "triples":[{"subject":"e1","relation":"likes","object":"e2","evidence_text":"张三喜欢茶"}],
        "pending":[]});
    let error = parse_and_validate_llm_json(&payload.to_string(), "张三喜欢茶", &[]).unwrap_err();
    assert!(error
        .to_string()
        .contains("ambiguous duplicate entity temp_id"));
    let mut duplicate = payload;
    duplicate["entities"][1]["name"] = json!("张三");
    assert!(parse_and_validate_llm_json(&duplicate.to_string(), "张三喜欢茶", &[]).is_ok());
}

fn read_body(stream: &mut TcpStream) -> Value {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 4096];
        let n = stream.read(&mut buf).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buf[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let len = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            if bytes.len() >= end + 4 + len {
                return serde_json::from_slice(&bytes[end + 4..end + 4 + len]).unwrap();
            }
        }
    }
}

fn mock(
    responses: Vec<(u16, String)>,
) -> (QianfanGraphExtractor, std::thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut requests = vec![];
        for (status, body) in responses {
            let until = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < until =>
                    {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            };
            requests.push(read_body(&mut stream));
            write!(stream, "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
        requests
    });
    let mut cfg = QianfanConfig::for_embed("test-placeholder".into(), None, None, Some(8));
    cfg.chat_url = format!("http://{address}/chat");
    (QianfanGraphExtractor::new(cfg).unwrap(), server)
}
fn empty_response(reason: &str) -> String {
    json!({"choices":[{"finish_reason":reason,"message":{"content":"{\"entities\":[],\"triples\":[],\"pending\":[]}"}}]}).to_string()
}
fn input(text: String) -> GraphExtractInput {
    GraphExtractInput {
        text,
        roster: vec![],
        speaker_member_id: None,
        roster_json: "[]".into(),
        message_at_ms: None,
    }
}

#[tokio::test]
async fn transient_http_retries_but_permanent_auth_does_not() {
    let (extractor, server) = mock(vec![
        (503, "must-not-leak".into()),
        (429, "must-not-leak".into()),
        (200, empty_response("stop")),
    ]);
    extractor.extract(input("普通文本".into())).await.unwrap();
    assert_eq!(server.join().unwrap().len(), 3);
    let (extractor, server) = mock(vec![(401, "secret-upstream".into())]);
    let error = extractor
        .extract(input("普通文本".into()))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("HTTP 401"));
    assert!(!error.contains("secret-upstream"));
    assert_eq!(server.join().unwrap().len(), 1);
}

#[tokio::test]
async fn truncated_and_context_limited_chunks_split_without_dropping_tail() {
    for first in [
        (200, empty_response("length")),
        (
            400,
            "{\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"private\"}}".into(),
        ),
    ] {
        // 1000 chars -> 512 + overlap + 512 + tail.
        let source = format!("{}尾", "甲".repeat(999));
        let (extractor, server) = mock(vec![
            first,
            (200, empty_response("stop")),
            (200, empty_response("stop")),
            (200, empty_response("stop")),
        ]);
        extractor.extract(input(source.clone())).await.unwrap();
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 4);
        let mut last = String::new();
        for request in &requests[1..] {
            assert_eq!(request["enable_thinking"], false);
            let prompt = request["messages"][0]["content"].as_str().unwrap();
            let data: Value = serde_json::from_str(prompt.rsplit_once('\n').unwrap().1).unwrap();
            last = data["source_text"].as_str().unwrap().into();
            assert!(source.contains(&last));
            assert!(last.chars().count() <= 512);
        }
        assert!(last.ends_with('尾'));
    }
}

#[tokio::test]
async fn graph_only_failure_does_not_commit_earlier_chunks_or_queue_jobs() {
    let valid = json!({"choices":[{"message":{"content":json!({
        "entities":[{"temp_id":"e1","name":"张三","type":"person"},{"temp_id":"e2","name":"茶","type":"concept"}],
        "triples":[{"subject":"e1","relation":"likes","object":"e2","evidence_text":"张三喜欢茶","privacy":"family_shared"}],"pending":[]
    }).to_string()}}]}).to_string();
    let (extractor, server) = mock(vec![(200, valid), (401, "private".into())]);
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("test.db");
    let engine = MemoryEngine::builder()
        .db_path(&db)
        .graph_extractor(Arc::new(extractor))
        .build()
        .await
        .unwrap();
    let source = format!("张三喜欢茶{}", "甲".repeat(3100));
    assert!(engine.ingest_graph_text(source, None, None).await.is_err());
    assert_eq!(server.join().unwrap().len(), 2);
    let graph = engine
        .query_graph_overview(GraphOverviewQuery::default())
        .await
        .unwrap();
    assert!(graph.entities.is_empty() && graph.facts.is_empty());
    let conn = rusqlite::Connection::open(db).unwrap();
    let jobs: i64 = conn
        .query_row("SELECT count(*) FROM graph_projection_jobs", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(jobs, 0);
    engine.shutdown();
}

#[tokio::test]
async fn graph_report_preserves_model_pending_and_extractor_id() {
    let response = json!({"choices":[{"message":{"content":json!({
        "entities":[{"temp_id":"p","name":"张三","type":"person"}],
        "triples":[],"pending":[{"reason":"unresolved","text":"其他人"}]
    }).to_string()}}]})
    .to_string();
    let (extractor, server) = mock(vec![(200, response)]);
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("report.db"))
        .graph_extractor(Arc::new(extractor))
        .build()
        .await
        .unwrap();
    let report = engine
        .ingest_graph_text("张三和其他人", None, None)
        .await
        .unwrap();
    assert_eq!(report.extractor_id.as_deref(), Some("qianfan-kg-v1"));
    assert_eq!(report.pending, 1);
    server.join().unwrap();
    engine.shutdown();
}

#[tokio::test]
async fn unconfigured_graph_ingest_is_not_empty_success() {
    let dir = tempfile::tempdir().unwrap();
    let engine = MemoryEngine::builder()
        .db_path(dir.path().join("test.db"))
        .build()
        .await
        .unwrap();
    let report = engine
        .ingest_graph_text("普通文本", None, None)
        .await
        .unwrap();
    assert_eq!(report.extractor_id.as_deref(), Some("no-chat"));
    assert!(!report.messages.is_empty());
    assert!(engine.ingest_graph_text("  ", None, None).await.is_err());
    engine.shutdown();
}
