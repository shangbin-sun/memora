//! JNI bridge for embedding Memora into JVM / Android hosts.
//!
//! Java package: `com.aiwf.memory.memora.MemoraNative`
//!
//! Build (example):
//! ```bash
//! cargo build -p memora --release --features jni
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use serde::Deserialize;
use serde_json::json;
use tokio::runtime::Runtime;

use crate::embedding::HashEmbedder;
use crate::engine::MemoryEngine;
use crate::household::{self, HouseholdMember, MemberRole};
use crate::qianfan::{
    QianfanChat, QianfanClassifier, QianfanConfig, QianfanEmbedder, QianfanGraphExtractor,
    QianfanSummarizer,
};
use crate::reconcile::{reconcile_turn_incremental, ReconcileRequest};
use crate::types::{
    ForgetReason, ListQuery, MemoryService, MemoryState, MemoryUpdate, RecallQuery, RememberInput,
};
use crate::Config;
use uuid::Uuid;

const JNI_TICK_INTERVAL: Duration = Duration::from_secs(600);
/// Background dream consolidation cadence (4 hours).
const JNI_DREAM_INTERVAL: Duration = Duration::from_secs(4 * 3600);

struct EngineSlot {
    engine: MemoryEngine,
    /// Chat client for extract/ops (`llm_chat_*`); same cfg as GraphExtractor.
    chat: Option<QianfanChat>,
    jobs: Vec<tokio::task::JoinHandle<()>>,
}

/// Inspect compatibility before opening any user database.
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_schemaVersion(
    _env: JNIEnv, _class: JClass,
) -> jni::sys::jint {
    crate::store::schema_version() as jni::sys::jint
}

/// 无数据库的OCR请求入口，与所有启用serial_inference的引擎共享队列。
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_modelRequest(
    mut env: JNIEnv, _class: JClass, body_json: JString,
) -> jstring {
    #[derive(Deserialize)]
    struct Request {
        url: String,
        api_key: String,
        request: serde_json::Value,
        timeout_ms: u64,
    }
    let result = (|| -> Result<String, String> {
        let raw = jstring_to_string(&mut env, body_json)?;
        let body: Request = serde_json::from_str(&raw).map_err(|_| "invalid model request JSON".to_string())?;
        runtime().block_on(crate::model_queue::request(
            &body.url, &body.api_key, &body.request,
            Duration::from_millis(body.timeout_ms.clamp(1, 600_000)),
        )).map_err(|e| e.to_string())
    })();
    match result {
        Ok(raw) => to_jstring(&mut env, raw),
        Err(error) => { throw_runtime(&mut env, error); std::ptr::null_mut() }
    }
}

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("memora-jni")
            .build()
            .expect("memora tokio runtime")
    })
}

fn engines() -> &'static Mutex<HashMap<u64, EngineSlot>> {
    static ENGINES: OnceLock<Mutex<HashMap<u64, EngineSlot>>> = OnceLock::new();
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn engine_of(handle: jlong) -> Result<MemoryEngine, String> {
    engines()
        .lock()
        .expect("engines")
        .get(&(handle as u64))
        .map(|slot| slot.engine.clone())
        .ok_or_else(|| format!("invalid memora handle: {handle}"))
}

fn chat_of(handle: jlong) -> Result<QianfanChat, String> {
    engines()
        .lock()
        .expect("engines")
        .get(&(handle as u64))
        .and_then(|slot| slot.chat.clone())
        .ok_or_else(|| {
            format!("reconcileTurn needs llm_chat_* on create (invalid/missing chat): {handle}")
        })
}

fn next_handle() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn throw_runtime(env: &mut JNIEnv, msg: impl AsRef<str>) {
    let _ = env.throw_new("java/lang/RuntimeException", msg.as_ref());
}

fn jstring_to_string(env: &mut JNIEnv, value: JString) -> Result<String, String> {
    env.get_string(&value)
        .map(|s| s.into())
        .map_err(|e| format!("invalid jstring: {e}"))
}

fn to_jstring(env: &mut JNIEnv, value: impl AsRef<str>) -> jstring {
    match env.new_string(value.as_ref()) {
        Ok(s) => s.into_raw(),
        Err(e) => {
            throw_runtime(env, format!("new_string failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

#[derive(Debug, Deserialize)]
struct CreateOptions {
    #[serde(default = "default_db")]
    db_path: String,
    #[serde(default = "default_namespace")]
    namespace: String,
    index_path: Option<String>,
    /// When true, force HashEmbedder even if LLM_API_KEY is set.
    #[serde(default)]
    hash_embedder: bool,
    /// Embedder backend: auto | hash | llm | qianfan
    #[serde(default = "default_embedder")]
    embedder: String,
    /// When set, embeddings use this key instead of `LLM_API_KEY`.
    #[serde(default)]
    llm_api_key: Option<String>,
    #[serde(default)]
    llm_embed_url: Option<String>,
    #[serde(default)]
    llm_embed_model: Option<String>,
    #[serde(default)]
    llm_embed_dim: Option<usize>,
    /// Chat maintenance is opt-in with a separate explicit configuration.
    llm_chat_api_key: Option<String>,
    llm_chat_url: Option<String>,
    llm_chat_model: Option<String>,
    #[serde(default)]
    serial_inference: bool,
}

fn qianfan_config(opts: &CreateOptions) -> Result<QianfanConfig, String> {
    if let Some(key) = opts
        .llm_api_key
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        return Ok(QianfanConfig::for_embed(
            key.to_string(),
            opts.llm_embed_url.clone(),
            opts.llm_embed_model.clone(),
            opts.llm_embed_dim,
        ));
    }
    QianfanConfig::from_env().map_err(|e| e.to_string())
}

fn chat_config(opts: &CreateOptions) -> Result<Option<QianfanConfig>, String> {
    let key = opts
        .llm_chat_api_key
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let url = opts
        .llm_chat_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let model = opts
        .llm_chat_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if key.is_none() && url.is_none() && model.is_none() {
        return Ok(None);
    }
    let (Some(key), Some(url), Some(model)) = (key, url, model) else {
        return Err("chat maintenance needs an explicit chat key, URL and model".into());
    };
    let mut cfg = QianfanConfig::for_embed(key.into(), None, None, None);
    cfg.chat_url = url.into();
    cfg.chat_model = model.into();
    Ok(Some(cfg))
}

fn default_db() -> String {
    "memora.db".into()
}

fn default_namespace() -> String {
    "default".into()
}

fn default_embedder() -> String {
    "auto".into()
}

#[derive(Debug, Deserialize)]
struct RememberBody {
    content: String,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    importance: Option<f64>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    pin_to_stm: bool,
    #[serde(default)]
    actor: Option<String>,
    /// When true, block until the row is committed. Default returns immediately.
    #[serde(default)]
    wait: bool,
    /// Explicit household subjects. `Some([])` = public/shared memory.
    /// `None` lets the engine infer from roster aliases.
    #[serde(default)]
    subjects: Option<Vec<String>>,
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    domains: Vec<String>,
    #[serde(default)]
    metadata: serde_json::Value,
    event_at: Option<String>,
    event_at_ms: Option<i64>,
    /// Anchor for relative-date normalize when `event_at_ms` is absent.
    #[serde(default)]
    now_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ForgetBody {
    id: String,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReviseBody {
    id: String,
    content: String,
    #[serde(default)]
    summary: Option<String>,
    /// When present, overwrite household subjects (`[]` = public).
    #[serde(default)]
    subjects: Option<Vec<String>>,
    #[serde(default)]
    tag: Option<String>,
    importance: Option<f64>,
    #[serde(default)]
    metadata: serde_json::Value,
    event_at: Option<serde_json::Value>,
    event_at_ms: Option<serde_json::Value>,
    #[serde(default)]
    clear_event: bool,
    /// Anchor for relative-date normalize when `event_at_ms` is absent.
    #[serde(default)]
    now_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct UpsertTagBody {
    #[serde(default)]
    tag_id: Option<String>,
    display_name: String,
    #[serde(default)]
    aliases: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ListBody {
    #[serde(default)]
    facts_only: bool,
    #[serde(default)]
    keyword: Option<String>,
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    page: Option<usize>,
    #[serde(default)]
    page_size: Option<usize>,
    #[serde(default)]
    namespace: Option<String>,
}

fn subjects_metadata(subjects: &[String], tag: Option<&str>) -> crate::Result<serde_json::Value> {
    let mut meta = household::with_ownership(&json!({}), subjects)?;
    // Tag overlay is only for revise; remember always infers tag inside the engine.
    if let Some(tag) = tag.map(str::trim).filter(|s| !s.is_empty()) {
        meta = household::apply_tag(&meta, tag);
    }
    Ok(meta)
}

#[derive(Debug, Deserialize)]
struct UpsertMemberBody {
    #[serde(default)]
    member_id: Option<String>,
    display_name: String,
    #[serde(default = "default_member_role")]
    role: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    age_band: Option<String>,
}

fn default_member_role() -> String {
    "other".into()
}

#[derive(Debug, Deserialize)]
struct RecallBody {
    text: String,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    /// Drop others' private memories when set.
    #[serde(default)]
    viewer_member_id: Option<String>,
    #[serde(default)]
    fts_only: bool,
    timeout_ms: Option<u64>,
    /// Also return original excerpts behind the hits (third memory type).
    #[serde(default)]
    include_chunks: bool,
    #[serde(default)]
    chunk_top_k: Option<usize>,
    #[serde(default)]
    chunk_max_chars: Option<usize>,
    /// Previous user utterance, used only to resolve follow-ups like "那语文呢".
    #[serde(default)]
    previous_query: Option<String>,
}

fn build_engine(opts: CreateOptions) -> Result<(MemoryEngine, Option<QianfanChat>), String> {
    let db_path = PathBuf::from(&opts.db_path);
    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut builder = MemoryEngine::builder()
        .db_path(db_path)
        .namespace(opts.namespace.clone())
        .config(Config::default());
    if let Some(index) = opts.index_path.clone() {
        builder = builder.index_path(index);
    }

    let mode = if opts.hash_embedder {
        "hash"
    } else {
        opts.embedder.trim()
    };

    match mode {
        "none" | "fts" => {}
        "hash" => {
            builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
        }
        "llm" | "qianfan" => {
            let cfg = qianfan_config(&opts)
                .map_err(|e| format!("embedder={mode} needs an API key: {e}"))?;
            builder = builder.embedder(Arc::new(
                QianfanEmbedder::new(cfg).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference),
            ));
        }
        "auto" => {
            if let Ok(cfg) = qianfan_config(&opts) {
                builder = builder.embedder(Arc::new(
                    QianfanEmbedder::new(cfg).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference),
                ));
            } else {
                builder = builder.embedder(Arc::new(HashEmbedder::new(64)));
            }
        }
        other => {
            return Err(format!(
                "unknown embedder={other}, expected none|auto|hash|llm|qianfan"
            ))
        }
    }
    // An embedding key never authorises calls to an implicit chat endpoint.
    let mut chat = None;
    if let Some(cfg) = chat_config(&opts)? {
        let archive = Config::default().consolidation_archive_importance;
        builder = builder
            .summarizer(Arc::new(
                QianfanSummarizer::new(cfg.clone()).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference),
            ))
            .classifier(Arc::new(
                QianfanClassifier::new(cfg.clone(), archive).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference),
            ))
            .graph_extractor(Arc::new(
                QianfanGraphExtractor::new(cfg.clone()).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference),
            ));
        chat = Some(QianfanChat::new(cfg).map_err(|e| e.to_string())?.with_serial_requests(opts.serial_inference));
    }

    let engine = runtime()
        .block_on(builder.build())
        .map_err(|e| e.to_string())?;
    Ok((engine, chat))
}

/// `MemoraNative.create(String optionsJson) -> long`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_create(
    mut env: JNIEnv,
    _class: JClass,
    options_json: JString,
) -> jlong {
    let raw = match jstring_to_string(&mut env, options_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return 0;
        }
    };
    let opts: CreateOptions = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("create options json: {e}"));
            return 0;
        }
    };
    match build_engine(opts) {
        Ok((engine, chat)) => {
            if let Err(e) = runtime().block_on(engine.list_tags(None)) {
                throw_runtime(&mut env, format!("seed tags: {e}"));
                return 0;
            }
            let handle = next_handle();
            // tokio::spawn only works on a runtime thread. JNI calls arrive on
            // an Android thread, so enter the process runtime before scheduling.
            let jobs = runtime().block_on(async {
                vec![
                    engine.spawn_maintenance(JNI_TICK_INTERVAL),
                    engine.spawn_dream(JNI_DREAM_INTERVAL),
                ]
            });
            engines()
                .lock()
                .expect("engines")
                .insert(handle, EngineSlot { engine, chat, jobs });
            handle as jlong
        }
        Err(e) => {
            throw_runtime(&mut env, e);
            0
        }
    }
}

/// `MemoraNative.destroy(long handle)`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_destroy(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if let Some(slot) = engines().lock().expect("engines").remove(&(handle as u64)) {
        for job in slot.jobs {
            job.abort();
        }
        slot.engine.shutdown();
    }
}

/// `MemoraNative.remember(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_remember(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: RememberBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("remember json: {e}"));
            return std::ptr::null_mut();
        }
    };
    if body.content.trim().is_empty() {
        throw_runtime(&mut env, "remember content is required");
        return std::ptr::null_mut();
    }

    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let wait = body.wait;
    let input = match remember_input(body) {
        Ok(input) => input,
        Err(e) => { throw_runtime(&mut env, e.to_string()); return std::ptr::null_mut(); }
    };

    if wait {
        return match runtime().block_on(engine.remember(input)) {
            Ok(id) => to_jstring(
                &mut env,
                json!({ "ok": true, "id": id.to_string() }).to_string(),
            ),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        };
    }

    runtime().spawn(async move {
        match engine.remember(input).await {
            Ok(id) => tracing::info!(%id, "async remember finished"),
            Err(e) => tracing::warn!(error = %e, "async remember failed"),
        }
    });
    to_jstring(&mut env, json!({ "accepted": true }).to_string())
}

fn overlay_metadata(mut base: serde_json::Value, overlay: &serde_json::Value) -> serde_json::Value {
    if !base.is_object() {
        base = json!({});
    }
    if let Some(fields) = overlay.as_object() {
        for (key, value) in fields {
            if value.is_null() {
                base.as_object_mut().unwrap().remove(key);
            } else {
                base[key] = value.clone();
            }
        }
    }
    base
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn remember_input(body: RememberBody) -> crate::Result<RememberInput> {
    // Temporal normalize is native-only: relative dates → event_at_ms / absolute content.
    let anchor = body.event_at_ms.or(body.now_ms).unwrap_or_else(now_ms);
    let normalized = crate::normalize_fact(&body.content, body.event_at.as_deref(), anchor);
    let mut input = RememberInput::episodic(normalized.content);
    input.summary = body.summary;
    input.namespace = body.namespace;
    input.session_id = body.session_id;
    input.importance = body
        .importance
        .or(normalized.importance)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    input.confidence = body.confidence.unwrap_or(0.8).clamp(0.0, 1.0);
    input.entities = body.entities;
    input.pin_to_stm = body.pin_to_stm;
    input.actor = body.actor;
    input.domains = body.domains;
    input.member_id = body.member_id;
    input.metadata = overlay_metadata(json!({}), &body.metadata);
    if let Some(subjects) = body.subjects {
        // Subjects override only; tag is inferred by the engine on remember.
        input.metadata = overlay_metadata(input.metadata, &subjects_metadata(&subjects, None)?);
    }
    if let Some(date) = body.event_at {
        input.metadata["event_at"] = json!(date);
    }
    if let Some(ms) = body.event_at_ms.or(normalized.event_at_ms) {
        input.metadata["event_at_ms"] = json!(ms);
    }
    Ok(input)
}

fn revise_update(body: ReviseBody, record: crate::types::MemoryRecord) -> crate::Result<MemoryUpdate> {
    let llm_event_at = body
        .event_at
        .as_ref()
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let host_event_ms = body.event_at_ms.as_ref().and_then(|v| v.as_i64());
    let anchor = host_event_ms.or(body.now_ms).unwrap_or_else(now_ms);
    let normalized = crate::normalize_fact(&body.content, llm_event_at, anchor);
    let mut meta = overlay_metadata(record.metadata, &body.metadata);
    if let Some(subjects) = body.subjects {
        meta = overlay_metadata(meta, &subjects_metadata(&subjects, body.tag.as_deref())?);
    } else if let Some(tag) = body.tag {
        meta = household::apply_tag(&meta, &tag);
    }
    if body.clear_event {
        // revise merges metadata; null clears a stored value, omission preserves it.
        meta["event_at"] = serde_json::Value::Null;
        meta["event_at_ms"] = serde_json::Value::Null;
    }
    for (key, value) in [
        ("event_at", body.event_at),
        ("event_at_ms", body.event_at_ms),
    ] {
        if let Some(value) = value {
            meta = overlay_metadata(meta, &json!({key: value}));
        }
    }
    if !body.clear_event {
        if meta.get("event_at_ms").and_then(|v| v.as_i64()).is_none() {
            if let Some(ms) = normalized.event_at_ms {
                meta["event_at_ms"] = json!(ms);
            }
        }
    }
    Ok(MemoryUpdate {
        expected_revision: Some(record.revision),
        content: Some(normalized.content.clone()),
        summary: body.summary.or(Some(normalized.content)),
        metadata: Some(meta),
        importance: body
            .importance
            .or(normalized.importance)
            .map(|i| i.clamp(0.0, 1.0)),
        reason: Some("reconcile".into()),
        ..MemoryUpdate::default()
    })
}

fn parse_memory_id(raw: &str) -> Result<Uuid, String> {
    Uuid::parse_str(raw.trim()).map_err(|e| format!("memory id: {e}"))
}

/// `MemoraNative.forget(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_forget(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: ForgetBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("forget json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let id = match parse_memory_id(&body.id) {
        Ok(id) => id,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let reason = ForgetReason::User {
        reason: body.reason.unwrap_or_else(|| "reconcile".into()),
    };
    match runtime().block_on(engine.forget(id, reason)) {
        Ok(()) => to_jstring(&mut env, json!({ "ok": true }).to_string()),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.revise(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_revise(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: ReviseBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("revise json: {e}"));
            return std::ptr::null_mut();
        }
    };
    if body.content.trim().is_empty() {
        throw_runtime(&mut env, "revise content is required");
        return std::ptr::null_mut();
    }
    let id = match parse_memory_id(&body.id) {
        Ok(id) => id,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let result = runtime().block_on(async {
        let record = engine
            .get(id)
            .await?
            .ok_or_else(|| crate::MemoraError::NotFound(id.to_string()))?;
        let update = revise_update(body, record)?;
        engine.revise(id, update).await
    });
    match result {
        Ok(()) => to_jstring(&mut env, json!({ "ok": true }).to_string()),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.recall(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_recall(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: RecallBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("recall json: {e}"));
            return std::ptr::null_mut();
        }
    };
    if body.text.trim().is_empty() {
        throw_runtime(&mut env, "recall text is required");
        return std::ptr::null_mut();
    }

    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let timeout = Duration::from_millis(body.timeout_ms.unwrap_or(2000).clamp(100, 5000));
    let chunk_options = body
        .include_chunks
        .then(|| (body.chunk_top_k.unwrap_or(2), body.chunk_max_chars.unwrap_or(2_000)));
    let contextual = crate::contextual_query(&body.text, body.previous_query.as_deref());
    let chunk_text = contextual.clone();
    let chunk_namespace = body.namespace.clone();
    let chunk_viewer = body.viewer_member_id.clone();
    let chunk_member = body.member_id.clone();
    let query = RecallQuery {
        text: contextual,
        namespace: body.namespace,
        session_id: body.session_id,
        top_k: body.top_k,
        member_id: body.member_id,
        tag: body.tag,
        viewer_member_id: body.viewer_member_id,
        fts_only: body.fts_only,
        ..Default::default()
    };

    let result = runtime().block_on(async {
        match tokio::time::timeout(timeout, engine.recall(query.clone())).await {
            Ok(result) => result,
            Err(_) => {
                let local = RecallQuery {
                    fts_only: true,
                    query_embedding: None,
                    ..query
                };
                tokio::time::timeout(Duration::from_millis(500), engine.recall(local))
                    .await
                    .map_err(|_| crate::MemoraError::Retrieval("local recall timed out".into()))?
            }
        }
    });
    match result {
        Ok(result) => {
            let encoded = match chunk_options {
                None => serde_json::to_string(&result),
                Some((top_k, max_chars)) => {
                    let chunk_query = crate::chunks::ChunkRecallQuery {
                        text: chunk_text,
                        namespace: chunk_namespace,
                        hit_memory_ids: result.hits.iter().map(|h| h.memory.id.to_string()).collect(),
                        hit_evidence: result
                            .hits
                            .iter()
                            .filter_map(|h| {
                                let evidence = h.memory.metadata.get("source")?.get("evidence")?.as_str()?;
                                Some((h.memory.id.to_string(), evidence.to_string()))
                            })
                            .collect(),
                        viewer_member_id: chunk_viewer,
                        member_id: chunk_member,
                        top_k,
                        max_chars,
                    };
                    // An excerpt failure must never cost the recalled facts.
                    let chunks = runtime().block_on(engine.recall_chunks(chunk_query)).unwrap_or_default();
                    serde_json::to_value(&result).and_then(|mut value| {
                        value["chunks"] = serde_json::to_value(&chunks)?;
                        serde_json::to_string(&value)
                    })
                }
            };
            match encoded {
                Ok(s) => to_jstring(&mut env, s),
                Err(e) => {
                    throw_runtime(&mut env, e.to_string());
                    std::ptr::null_mut()
                }
            }
        }
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Backend debug lookup: `{ "id": "<memory uuid>" }` -> origin plus original excerpts.
/// `MemoraNative.memoryTrace(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_memoryTrace(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let id = serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| v.get("id").and_then(|id| id.as_str()).map(str::to_string))
        .and_then(|id| uuid::Uuid::parse_str(id.trim()).ok());
    let Some(id) = id else {
        throw_runtime(&mut env, "memoryTrace needs a memory id");
        return std::ptr::null_mut();
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.memory_trace(id)) {
        Ok(found) => to_jstring(&mut env, found.unwrap_or(serde_json::Value::Null).to_string()),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.listMembers(long handle) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_listMembers(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.list_members(None)) {
        Ok(members) => match serde_json::to_string(&members) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.upsertMember(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_upsertMember(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: UpsertMemberBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("upsertMember json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let display_name = body.display_name.trim().to_string();
    if display_name.is_empty() {
        throw_runtime(&mut env, "display_name is required");
        return std::ptr::null_mut();
    }
    let role = match MemberRole::parse(&body.role) {
        Ok(role) => role,
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let result = runtime().block_on(async {
        let existing = engine.list_members(None).await?;
        let member_id = if let Some(raw_id) = body
            .member_id
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            household::validate_member_id(raw_id)?
        } else if role == MemberRole::Self_ {
            existing
                .iter()
                .find(|m| m.role == MemberRole::Self_)
                .map(|m| m.member_id.clone())
                .unwrap_or_else(|| display_name.clone())
        } else {
            existing
                .iter()
                .find(|m| {
                    m.display_name == display_name
                        || m.aliases.iter().any(|a| a == &display_name)
                        || body.aliases.iter().any(|a| {
                            let a = a.trim();
                            !a.is_empty()
                                && (m.display_name == a || m.aliases.iter().any(|x| x == a))
                        })
                })
                .map(|m| m.member_id.clone())
                .unwrap_or_else(|| display_name.clone())
        };
        let prior = existing.iter().find(|m| m.member_id == member_id);
        let mut aliases = body.aliases;
        if let Some(old) = prior {
            for a in &old.aliases {
                if !aliases.iter().any(|x| x == a) {
                    aliases.push(a.clone());
                }
            }
        }
        if !aliases.iter().any(|a| a == &display_name) {
            aliases.push(display_name.clone());
        }
        if role == MemberRole::Self_ {
            for alias in ["我", "自己", "本人"] {
                if !aliases.iter().any(|a| a == alias) {
                    aliases.push(alias.into());
                }
            }
        }
        if role == MemberRole::Child {
            for alias in ["孩子", "小孩"] {
                if !aliases.iter().any(|a| a == alias) {
                    aliases.push(alias.into());
                }
            }
        }
        let member = HouseholdMember {
            member_id: member_id.clone(),
            display_name: display_name.clone(),
            role,
            aliases,
            age_band: body.age_band,
        };
        engine.upsert_member(None, member).await?;
        Ok::<_, crate::error::MemoraError>(member_id)
    });
    match result {
        Ok(member_id) => to_jstring(
            &mut env,
            json!({ "ok": true, "member_id": member_id }).to_string(),
        ),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.listTags(long handle) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_listTags(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.list_tags(None)) {
        Ok(tags) => match serde_json::to_string(&tags) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.upsertTag(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_upsertTag(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: UpsertTagBody = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("upsertTag json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let display_name = body.display_name.trim().to_string();
    if display_name.is_empty() {
        throw_runtime(&mut env, "display_name is required");
        return std::ptr::null_mut();
    }
    let tag_id = body
        .tag_id
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(display_name.as_str())
        .to_string();
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let tag = household::MemoryTag {
        tag_id: tag_id.clone(),
        display_name,
        aliases: body.aliases,
    };
    match runtime().block_on(engine.upsert_tag(None, tag)) {
        Ok(()) => to_jstring(
            &mut env,
            json!({ "ok": true, "tag_id": tag_id }).to_string(),
        ),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.list(long handle, String bodyJson) -> String`
///
/// Optional filters: keyword / member_id / tag. Pagination defaults to the
/// newest 10 rows (`page=0`, `page_size=10`).
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_list(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: ListBody = if raw.trim().is_empty() {
        ListBody {
            facts_only: false,
            keyword: None,
            member_id: None,
            tag: None,
            page: None,
            page_size: None,
            namespace: None,
        }
    } else {
        match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                throw_runtime(&mut env, format!("list json: {e}"));
                return std::ptr::null_mut();
            }
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let page_size = body.page_size.unwrap_or(10).clamp(1, 100);
    let page = body.page.unwrap_or(0);
    let offset = page.saturating_mul(page_size);
    let query = ListQuery {
        facts_only: body.facts_only,
        namespace: body.namespace,
        state: Some(MemoryState::Active),
        kind: None,
        keyword: body.keyword,
        member_id: body.member_id,
        tag: body.tag,
        limit: Some(page_size),
        offset: Some(offset),
    };
    match runtime().block_on(engine.list(query)) {
        Ok(result) => {
            let page_count = if result.total <= 0 {
                0usize
            } else {
                ((result.total as usize) + page_size - 1) / page_size
            };
            let payload = json!({
                "ok": true,
                "items": result.items,
                "total": result.total,
                "page": page,
                "page_size": page_size,
                "page_count": page_count,
            });
            to_jstring(&mut env, payload.to_string())
        }
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.reconcileTurn(long handle, String bodyJson) -> String`
///
/// Body: [`ReconcileRequest`] JSON (`query`, optional `answer`/`transcript`/
/// `speaker_id`/`namespace`/`session_id`/`message_at_ms`/`evidence[]`).
/// Returns [`crate::ReconcileReport`] JSON.
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_reconcileTurn(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let req: ReconcileRequest = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("reconcileTurn json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let chat = match chat_of(handle) {
        Ok(chat) => chat,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(reconcile_turn_incremental(&engine, &chat, req)) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, format!("reconcileTurn encode: {e}"));
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Stateless helpers for hosts (no engine handle).
/// `MemoraNative.normalizeFact(String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_normalizeFact(
    mut env: JNIEnv,
    _class: JClass,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    #[derive(Deserialize)]
    struct Body {
        content: String,
        #[serde(default)]
        event_at: Option<String>,
        #[serde(default)]
        now_ms: Option<i64>,
    }
    let body: Body = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("normalizeFact json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let now = body.now_ms.unwrap_or_else(crate::core::now_ms);
    let fact = crate::temporal::normalize_fact(&body.content, body.event_at.as_deref(), now);
    to_jstring(
        &mut env,
        json!({
            "content": fact.content,
            "event_at_ms": fact.event_at_ms,
            "importance": fact.importance,
        })
        .to_string(),
    )
}

/// `MemoraNative.checkWrite(String content, String actorOrEmpty) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_checkWrite(
    mut env: JNIEnv,
    _class: JClass,
    content: JString,
    actor: JString,
) -> jstring {
    let text = match jstring_to_string(&mut env, content) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let actor_s = jstring_to_string(&mut env, actor).unwrap_or_default();
    let actor_opt = {
        let t = actor_s.trim();
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    };
    let check = crate::write_filter::check_write(&text, actor_opt);
    to_jstring(
        &mut env,
        json!({
            "ok": check.ok(),
            "reason": check.reason_code(),
        })
        .to_string(),
    )
}

/// `MemoraNative.grantAccess(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_grantAccess(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let input: crate::graph::GrantInput = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("grantAccess json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.grant_access(input)) {
        Ok(grant) => match serde_json::to_string(&grant) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.revokeAccess(long handle, String bodyJson) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_revokeAccess(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let input: crate::graph::RevokeGrantInput = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("revokeAccess json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.revoke_access(input)) {
        Ok(n) => to_jstring(&mut env, json!({ "revoked": n }).to_string()),
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.listGrants(long handle, String bodyJson) -> String`
/// Body: `{ "member_id"?, "as_grantor"?, "namespace"? }`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_listGrants(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    #[derive(Deserialize)]
    struct Body {
        #[serde(default)]
        member_id: Option<String>,
        #[serde(default)]
        as_grantor: bool,
        #[serde(default)]
        namespace: Option<String>,
    }
    let body: Body = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("listGrants json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.list_grants(body.member_id, body.as_grantor, body.namespace)) {
        Ok(grants) => match serde_json::to_string(&grants) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.ingestGraph(long handle, String bodyJson) -> String`
///
/// Body: `{ "text": "...", "member_id"?, "namespace"? }` — graph-only ingest
/// (no memory row). Used for OCR / document parse → KG.
/// Synchronous: all internal chunks succeed before one atomic graph commit.
/// HTTP/transport/invalid-output failures throw RuntimeException; no durable
/// retry is queued. Retry failed App chunks, not already completed chunks.
/// Zero facts can be valid; extractor_id="no-chat" means NOT configured/completed.
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_ingestGraph(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    #[derive(serde::Deserialize)]
    struct Body {
        text: String,
        #[serde(default)]
        member_id: Option<String>,
        #[serde(default)]
        namespace: Option<String>,
    }
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let body: Body = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("ingestGraph json: {e}"));
            return std::ptr::null_mut();
        }
    };
    if body.text.trim().is_empty() {
        throw_runtime(&mut env, "text is required");
        return std::ptr::null_mut();
    }
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.ingest_graph_text(body.text, body.member_id, body.namespace)) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.queryPersonGraph(long handle, String bodyJson) -> String`
///
/// Body: `{ "member_id": "...", "namespace"?, "domains"?, "include_ended"?, "viewer_member_id"? }`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_queryPersonGraph(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    body_json: JString,
) -> jstring {
    let raw = match jstring_to_string(&mut env, body_json) {
        Ok(s) => s,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    let query: crate::graph::PersonGraphQuery = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            throw_runtime(&mut env, format!("queryPersonGraph json: {e}"));
            return std::ptr::null_mut();
        }
    };
    if query.member_id.trim().is_empty() {
        throw_runtime(&mut env, "member_id is required");
        return std::ptr::null_mut();
    }
    let engine = match engine_of(handle) {
        Ok(engine) => engine,
        Err(e) => {
            throw_runtime(&mut env, e);
            return std::ptr::null_mut();
        }
    };
    match runtime().block_on(engine.query_person_graph(query)) {
        Ok(graph) => match serde_json::to_string(&graph) {
            Ok(s) => to_jstring(&mut env, s),
            Err(e) => {
                throw_runtime(&mut env, e.to_string());
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            throw_runtime(&mut env, e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// `MemoraNative.promptBundle(String versionOrEmpty) -> String`
#[no_mangle]
pub extern "system" fn Java_com_aiwf_memory_memora_MemoraNative_promptBundle(
    mut env: JNIEnv,
    _class: JClass,
    version: JString,
) -> jstring {
    let ver = jstring_to_string(&mut env, version).unwrap_or_default();
    let Some(bundle) = crate::prompts::bundle(Some(ver.trim())) else {
        throw_runtime(&mut env, format!("unknown prompt version: {ver}"));
        return std::ptr::null_mut();
    };
    to_jstring(
        &mut env,
        json!({
            "version": bundle.version.trim(),
            "extract_system": bundle.extract_system,
            "extract_rules": bundle.extract_rules,
            "ocr_system": bundle.ocr_system,
            "ops_system": bundle.ops_system,
            "ops_rules": bundle.ops_rules,
            "kg_extract_system": bundle.kg_extract_system,
            "kg_extract_rules": bundle.kg_extract_rules,
            "chunk_chars": crate::graph::SOURCE_CHUNK_CHARS,
            "chunk_overlap": crate::graph::SOURCE_CHUNK_OVERLAP,
        })
        .to_string(),
    )
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn short_cjk_recall_crosses_word_boundaries_without_bypassing_filters() {
        runtime().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let engine = MemoryEngine::builder().db_path(dir.path().join("short.db"))
                .graph_projection_worker(false).build().await.unwrap();
            let mut public = Vec::new();
            for text in ["全家每周六上午清洁客厅", "全家每周日上午九点整理阳台"] {
                let body: RememberBody = serde_json::from_value(json!({"content":text,"subjects":["家庭"]})).unwrap();
                public.push(engine.remember(remember_input(body).unwrap()).await.unwrap());
            }
            for (owner, namespace, metadata) in [
                ("张三","default",json!({})),
                ("公共","other-family",json!({})),
                ("公共","default",json!({"privacy":true})),
                ("公共","default",json!({"graph_only":true}))
            ] {
                let body: RememberBody = serde_json::from_value(json!({"content":"全家每周六清洁客厅",
                    "subjects":[owner],"namespace":namespace,"metadata":metadata})).unwrap();
                engine.remember(remember_input(body).unwrap()).await.unwrap();
            }
            for (word, id) in ["周六", "周日"].into_iter().zip(public.iter()) {
                let result = engine.recall(RecallQuery {text:word.into(),member_id:Some("家庭".into()),
                    viewer_member_id:Some("李四".into()),fts_only:true,skip_graph:true,top_k:Some(1),..Default::default()}).await.unwrap();
                assert_eq!(result.hits.len(),1,"{word}");
                assert_eq!(result.hits[0].memory.id,*id);
            }
            engine.forget(public[0],ForgetReason::User {reason:"test deletion".into()}).await.unwrap();
            let result = engine.recall(RecallQuery {text:"周六".into(),member_id:Some("公共".into()),
                viewer_member_id:Some("李四".into()),fts_only:true,skip_graph:true,..Default::default()}).await.unwrap();
            assert!(result.hits.is_empty());
            engine.shutdown();
        });
    }

    #[test]
    fn invalid_explicit_subjects_never_fall_back_to_public() {
        for owners in [json!([" "]), json!(["bad|id"]), json!(["家庭","张三"])] {
            let body: RememberBody = serde_json::from_value(json!({"content":"测试事实","subjects":owners})).unwrap();
            assert!(remember_input(body).is_err());
        }
        let inferred: RememberBody = serde_json::from_value(json!({"content":"张三喜欢咖啡"})).unwrap();
        assert!(remember_input(inferred).unwrap().metadata.get("subjects").is_none());
    }

    #[test]
    fn shared_and_personal_jni_writes_reuse_storage_revision_and_filtered_recall() {
        runtime().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let db = dir.path().join("owners.db");
            let engine = MemoryEngine::builder().db_path(&db).build().await.unwrap();
            let private: RememberBody = serde_json::from_value(json!({
                "content":"张三的滤芯预算是318元","subjects":["张三"],
                "metadata":{"privacy":true,"source":{"message_id":"source-1"}}
            })).unwrap();
            let personal_id = engine.remember(remember_input(private).unwrap()).await.unwrap();
            let mut public_ids = Vec::new();
            for (i, owners) in [json!([]),json!(["家庭"]),json!(["PUBLIC"]),json!(["公共"]),json!(["全家"])].into_iter().enumerate() {
                let body: RememberBody = serde_json::from_value(json!({
                    "content":format!("全家滤芯维护项目编号{i}固定保存"),"subjects":owners,
                    "metadata":{"source":{"message_id":format!("shared-{i}")}}
                })).unwrap();
                let id = engine.remember(remember_input(body).unwrap()).await.unwrap();
                let rec = engine.get(id).await.unwrap().unwrap();
                assert_eq!(rec.subject_ids, [household::PUBLIC_MEMBER_ID]);
                assert_eq!(rec.metadata["scope"], "public");
                public_ids.push(id);
            }
            // Old records are readable without rewriting them or creating a second database.
            for owners in [json!([]),json!([{"member_id":"家庭","confidence":1.0}])] {
                let mut old = RememberInput::episodic("旧版全家滤芯维护记录");
                old.metadata = json!({"subjects":owners,"scope":"public","attribution":"manual"});
                engine.remember(old).await.unwrap();
            }
            for owner in ["公共","家庭","PUBLIC"] {
                let page = engine.list(ListQuery { member_id:Some(owner.into()),limit:Some(1),..Default::default() }).await.unwrap();
                assert_eq!(page.total, 7);
                assert_eq!(page.items.len(), 1);
                assert_ne!(page.items[0].id, personal_id);
                let hits = engine.recall(RecallQuery { text:"滤芯".into(),member_id:Some(owner.into()),fts_only:true,skip_graph:true,top_k:Some(20),..Default::default() }).await.unwrap();
                assert_eq!(hits.hits.len(), 7);
                assert!(hits.hits.iter().all(|h| h.memory.id != personal_id));
            }
            let own = engine.list(ListQuery {member_id:Some("张三".into()),..Default::default()}).await.unwrap();
            assert_eq!(own.total, 1);
            assert_eq!(own.items[0].id, personal_id);
            assert_eq!(own.items[0].metadata["privacy"], true);
            let rec = engine.get(public_ids[0]).await.unwrap().unwrap();
            let update: ReviseBody = serde_json::from_value(json!({
                "id":public_ids[0],"content":"全家滤芯每十二个月更换","subjects":["共享"]
            })).unwrap();
            engine.revise(public_ids[0], revise_update(update, rec).unwrap()).await.unwrap();
            let rec = engine.get(public_ids[0]).await.unwrap().unwrap();
            assert_eq!(rec.subject_ids, [household::PUBLIC_MEMBER_ID]);
            assert_eq!(rec.metadata["source"]["message_id"], "shared-0");
            let other = engine.list(ListQuery {namespace:Some("other-family".into()),member_id:Some("公共".into()),..Default::default()}).await.unwrap();
            assert_eq!(other.total, 0);
            engine.shutdown();
            drop(engine);
            let restored = MemoryEngine::builder().db_path(&db).build().await.unwrap();
            let page = restored.list(ListQuery {member_id:Some("家庭".into()),..Default::default()}).await.unwrap();
            assert_eq!(page.total, 7);
            assert!(page.items.iter().any(|r| r.id == public_ids[0] && r.content.contains("十二个月")));
            restored.shutdown();
        });
    }

    #[test]
    fn embedding_config_never_implies_chat_config() {
        let opts:CreateOptions=serde_json::from_value(json!({"embedder":"llm","llm_api_key":"embed-secret",
            "llm_embed_url":"https://embed.example/embeddings","llm_embed_model":"custom","llm_embed_dim":0})).unwrap();
        assert!(chat_config(&opts).unwrap().is_none());
        let cfg = qianfan_config(&opts).unwrap();
        assert_eq!(cfg.api_key, "embed-secret");
        assert_eq!(cfg.embed_dim, 0);
    }
    #[test]
    fn chat_requires_separate_complete_explicit_configuration() {
        let incomplete: CreateOptions =
            serde_json::from_value(json!({"llm_chat_api_key":"chat-secret"})).unwrap();
        assert!(chat_config(&incomplete).is_err());
        let opts: CreateOptions = serde_json::from_value(
            json!({"llm_api_key":"embed-secret","llm_chat_api_key":"chat-secret",
            "llm_chat_url":"https://chat.example/completions","llm_chat_model":"chat-model"}),
        )
        .unwrap();
        let cfg = chat_config(&opts).unwrap().unwrap();
        assert_eq!(cfg.api_key, "chat-secret");
        assert_eq!(cfg.chat_url, "https://chat.example/completions");
        assert_eq!(cfg.chat_model, "chat-model");
    }
    #[test]
    fn jni_remember_and_revise_persist_time_and_importance_without_losing_facets() {
        let dir = tempfile::tempdir().unwrap();
        let engine = runtime()
            .block_on(
                MemoryEngine::builder()
                    .db_path(dir.path().join("m.db"))
                    .build(),
            )
            .unwrap();
        let body: RememberBody = serde_json::from_value(json!({"content":"原安排去医院复查血压",
            "subjects":["张三"],"importance":0.9,"event_at":"2026-10-22",
            "event_at_ms":1792598400000i64,"metadata":{"custom":"kept"}}))
        .unwrap();
        let id = runtime()
            .block_on(engine.remember(remember_input(body).unwrap()))
            .unwrap();
        let rec = runtime().block_on(engine.get(id)).unwrap().unwrap();
        assert_eq!(rec.metadata["event_at"], "2026-10-22");
        assert_eq!(rec.metadata["custom"], "kept");
        assert_eq!(rec.tag, "医疗健康");
        let body: ReviseBody = serde_json::from_value(
            json!({"id":id,"content":"新安排去医院复查血压","importance":0.4,
            "tag":"医疗健康","event_at":"2026-10-23","event_at_ms":1792684800000i64}),
        )
        .unwrap();
        runtime()
            .block_on(engine.revise(id, revise_update(body, rec).unwrap()))
            .unwrap();
        let rec = runtime().block_on(engine.get(id)).unwrap().unwrap();
        assert_eq!(rec.importance, 0.4);
        assert_eq!(rec.subject_ids, vec!["张三"]);
        assert_eq!(rec.tag, "医疗健康");
        assert_eq!(rec.metadata["event_at"], "2026-10-23");
        assert_eq!(rec.metadata["custom"], "kept");
        let body: ReviseBody =
            serde_json::from_value(json!({"id":id,"content":"稳定事实","clear_event":true}))
                .unwrap();
        runtime()
            .block_on(engine.revise(id, revise_update(body, rec).unwrap()))
            .unwrap();
        let rec = runtime().block_on(engine.get(id)).unwrap().unwrap();
        assert!(rec.metadata.get("event_at").map_or(true, serde_json::Value::is_null));
        assert!(rec.metadata.get("event_at_ms").map_or(true, serde_json::Value::is_null));
    }
    #[test]
    fn null_metadata_explicitly_removes_an_event() {
        let meta = overlay_metadata(
            json!({"event_at":"2026-10-22","custom":1}),
            &json!({"event_at":null}),
        );
        assert!(meta.get("event_at").is_none());
        assert_eq!(meta["custom"], 1);
    }
}
