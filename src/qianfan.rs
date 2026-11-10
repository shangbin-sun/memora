use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::classify::{
    parse_classify_json, ClassifyItem, ClassifyOutput, HeuristicClassifier, MemoryClassifier,
};
use crate::embedding::{Embedder, SummarizeInput, SummarizeOutput, Summarizer};
use crate::error::{MemoraError, Result};
use crate::graph::{
    kg_prompt_with_context, merge_graph_extract_outputs, parse_and_validate_llm_json,
    split_source_text, GraphExtractInput, GraphExtractOutput, GraphExtractor, SOURCE_CHUNK_CHARS,
    SOURCE_CHUNK_OVERLAP,
};

const DEFAULT_CHAT_URL: &str = "https://qianfan.baidubce.com/v2/chat/completions";
const DEFAULT_EMBED_URL: &str = "https://qianfan.baidubce.com/v2/embeddings";
const DEFAULT_CHAT_MODEL: &str = "qwen3.5-27b";
const DEFAULT_EMBED_MODEL: &str = "qwen3-embedding-8b";

/// Per extraction source/part, including retries, subdivision and permit wait.
/// Not a whole-document SLA: hosts may submit several parts independently.
pub(crate) const EXTRACTION_TIMEOUT_SECS: u64 = 600;

/// Shared HTTP client for LLM gateways.
///
/// Many corporate token proxies hang or black-hole HTTP/2 (ALPN). curl on
/// Windows often speaks HTTP/1.1 and succeeds while reqwest's default HTTP/2
/// path times out — force HTTP/1.1.
fn build_llm_client(connect_secs: u64, timeout_secs: u64) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .gzip(true)
        .http1_only()
        .tcp_nodelay(true)
        .connect_timeout(std::time::Duration::from_secs(connect_secs))
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .pool_max_idle_per_host(2)
        .build()
        .map_err(|e| MemoraError::Other(format!("llm http client: {e}")))
}

#[derive(Debug, Clone)]
pub struct QianfanConfig {
    pub api_key: String,
    pub chat_url: String,
    pub embed_url: String,
    pub chat_model: String,
    pub embed_model: String,
    pub embed_dim: usize,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub reasoning_split: bool,
}

impl QianfanConfig {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("LLM_API_KEY")
            .map_err(|_| MemoraError::Embedding("missing LLM_API_KEY".into()))?;
        Ok(Self {
            api_key,
            chat_url: std::env::var("LLM_CHAT_URL").unwrap_or_else(|_| DEFAULT_CHAT_URL.into()),
            embed_url: std::env::var("LLM_EMBED_URL").unwrap_or_else(|_| DEFAULT_EMBED_URL.into()),
            chat_model: std::env::var("LLM_CHAT_MODEL")
                .unwrap_or_else(|_| DEFAULT_CHAT_MODEL.into()),
            embed_model: std::env::var("LLM_EMBED_MODEL")
                .unwrap_or_else(|_| DEFAULT_EMBED_MODEL.into()),
            embed_dim: std::env::var("LLM_EMBED_DIM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4096),
            max_tokens: 2048,
            temperature: 0.5,
            top_p: 0.7,
            reasoning_split: true,
        })
    }

    /// Embeddings from an explicit key and endpoint. Chat fields stay at the
    /// defaults; recall and remember only use the embedder.
    pub fn for_embed(
        api_key: String,
        embed_url: Option<String>,
        embed_model: Option<String>,
        embed_dim: Option<usize>,
    ) -> Self {
        Self {
            api_key,
            chat_url: DEFAULT_CHAT_URL.into(),
            embed_url: embed_url
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_EMBED_URL.into()),
            chat_model: DEFAULT_CHAT_MODEL.into(),
            embed_model: embed_model
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_EMBED_MODEL.into()),
            embed_dim: embed_dim.unwrap_or(4096),
            max_tokens: 2048,
            temperature: 0.5,
            top_p: 0.7,
            reasoning_split: true,
        }
    }
}

#[derive(Clone)]
pub struct QianfanEmbedder {
    serial_requests: bool,
    client: reqwest::Client,
    config: QianfanConfig,
    dim: std::sync::Arc<AtomicUsize>,
}

impl QianfanEmbedder {
    /// 与其他启用串行的Memora模型客户端共用FIFO请求队列。
    pub fn with_serial_requests(mut self, enabled: bool) -> Self {
        self.serial_requests = enabled;
        self
    }

    pub fn new(config: QianfanConfig) -> Result<Self> {
        let dim = config.embed_dim;
        Ok(Self {
            client: build_llm_client(10, 45)?,
            serial_requests: false,
            config,
            dim: std::sync::Arc::new(AtomicUsize::new(dim)),
        })
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    #[serde(default)]
    index: Option<usize>,
    embedding: Vec<f32>,
}

pub fn parse_embed_response(body: &str) -> Result<Vec<Vec<f32>>> {
    let parsed: EmbedResponse = serde_json::from_str(body)
        .map_err(|e| MemoraError::Embedding(format!("embed json: {e}")))?;
    if parsed.data.is_empty() {
        return Err(MemoraError::Embedding("empty embedding data".into()));
    }
    let mut data = parsed.data;
    if data.iter().any(|d| d.index.is_some()) {
        if data.iter().any(|d| d.index.is_none()) {
            return Err(MemoraError::Embedding("mixed embedding indices".into()));
        }
        data.sort_by_key(|d| d.index.unwrap());
        if data.iter().enumerate().any(|(i, d)| d.index != Some(i)) {
            return Err(MemoraError::Embedding("invalid embedding indices".into()));
        }
    }
    data.into_iter()
        .map(|d| crate::embedding::normalize_vector(d.embedding))
        .collect()
}

#[async_trait]
impl Embedder for QianfanEmbedder {
    fn model_id(&self) -> &str {
        &self.config.embed_model
    }

    fn dimension(&self) -> usize {
        self.dim.load(Ordering::Relaxed)
    }

    fn bind_dimension(&self, dim: usize) -> bool {
        if dim == 0 {
            return false;
        }
        let _ = self
            .dim
            .compare_exchange(0, dim, Ordering::SeqCst, Ordering::SeqCst);
        self.dimension() == dim
    }

    fn space_id(&self) -> String {
        format!(
            "{}|{}|{}",
            self.config.embed_url.trim(),
            self.config.embed_model,
            self.config.embed_dim
        )
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let _model_permit = crate::model_queue::acquire(self.serial_requests).await;
        let resp = self
            .client
            .post(&self.config.embed_url)
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&EmbedRequest {
                input: texts,
                model: &self.config.embed_model,
            })
            .send()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        if !status.is_success() {
            return Err(MemoraError::Embedding(format!(
                "embed http {status}: {body}"
            )));
        }
        let vectors = parse_embed_response(&body)?;
        if vectors.len() != texts.len() {
            return Err(MemoraError::Embedding("embedding count mismatch".into()));
        }
        let dim = vectors[0].len();
        if !self.bind_dimension(dim) || vectors.iter().any(|v| v.len() != dim) {
            return Err(MemoraError::Embedding(
                "embedding dimension mismatch".into(),
            ));
        }
        Ok(vectors)
    }
}

#[derive(Clone)]
pub struct QianfanSummarizer {
    serial_requests: bool,
    client: reqwest::Client,
    config: QianfanConfig,
}

impl QianfanSummarizer {
    /// 与其他启用串行的Memora模型客户端共用FIFO请求队列。
    pub fn with_serial_requests(mut self, enabled: bool) -> Self {
        self.serial_requests = enabled;
        self
    }

    pub fn new(config: QianfanConfig) -> Result<Self> {
        Ok(Self {
            client: build_llm_client(10, 60)?,
            serial_requests: false,
            config,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    messages: Vec<ChatMessage<'a>>,
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    reasoning_split: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatReply,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatReply {
    content: Option<String>,
}

pub fn parse_chat_response(body: &str) -> Result<String> {
    let parsed: ChatResponse =
        serde_json::from_str(body).map_err(|e| MemoraError::Other(format!("chat json: {e}")))?;
    if let Some(reason) = parsed
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
    {
        match reason {
            "length" | "max_tokens" => {
                return Err(MemoraError::Other(
                    "chat response truncated (length)".into(),
                ))
            }
            "content_filter" => {
                return Err(MemoraError::Other(
                    "chat response filtered (content_filter)".into(),
                ))
            }
            _ => {}
        }
    }
    parsed
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| MemoraError::Other("empty chat content".into()))
}

/// Text-memory extraction may retain complete NDJSON rows from a length-limited reply.
/// Callers must still validate each row; this is not a relaxed graph/ops JSON parser.
#[derive(Debug, Clone)]
pub(crate) struct ExtractionCompletion {
    pub text: String,
    pub truncated: bool,
}

pub(crate) fn parse_extraction_response(body: &str) -> Result<ExtractionCompletion> {
    // Do not echo malformed fields or unknown gateway finish reasons into logs.
    let parsed: ChatResponse = serde_json::from_str(body)
        .map_err(|_| MemoraError::Other("chat extraction response json invalid".into()))?;
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| MemoraError::Other("empty chat content".into()))?;
    let truncated = match choice.finish_reason.as_deref() {
        Some("length" | "max_tokens") => true,
        Some("stop") | None => false,
        Some("content_filter") => {
            return Err(MemoraError::Other(
                "chat response filtered (content_filter)".into(),
            ));
        }
        Some(_) => {
            return Err(MemoraError::Other(
                "chat extraction response unsupported finish reason".into(),
            ));
        }
    };
    let text = choice.message.content.unwrap_or_default();
    if !truncated && text.trim().is_empty() {
        return Err(MemoraError::Other("empty chat content".into()));
    }
    Ok(ExtractionCompletion { text, truncated })
}

#[derive(Clone, Copy)]
enum ChatResponseMode {
    Complete,
    Extraction,
}

/// Thin chat client for host curators (extract / ops), same endpoint as summarizer.
#[derive(Clone)]
pub struct QianfanChat {
    serial_requests: bool,
    client: reqwest::Client,
    config: QianfanConfig,
    permits: std::sync::Arc<tokio::sync::Semaphore>,
}

impl QianfanChat {
    /// 与其他启用串行的Memora模型客户端共用FIFO请求队列。
    pub fn with_serial_requests(mut self, enabled: bool) -> Self {
        self.serial_requests = enabled;
        self
    }

    pub fn new(config: QianfanConfig) -> Result<Self> {
        Ok(Self {
            // Extraction explicitly receives a shared ten-minute deadline.
            // Short ordinary calls/ops still override this with their own budget.
            client: build_llm_client(10, EXTRACTION_TIMEOUT_SECS)?,
            serial_requests: false,
            config,
            permits: std::sync::Arc::new(tokio::sync::Semaphore::new(2)),
        })
    }

    pub fn model(&self) -> &str {
        &self.config.chat_model
    }

    /// OpenAI-compatible chat.completions with optional system message.
    pub async fn complete(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        self.complete_with_timeout(
            system,
            user,
            max_tokens,
            temperature,
            std::time::Duration::from_secs(30),
        )
        .await
    }

    /// Bound the ENTIRE completion, including permit wait, retries and backoff.
    /// Suitable for structured extraction; never multiply this deadline by attempts.
    pub async fn complete_with_timeout(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        request_timeout: std::time::Duration,
    ) -> Result<String> {
        self.complete_in_mode(
            system,
            user,
            max_tokens,
            temperature,
            request_timeout,
            ChatResponseMode::Complete,
        )
        .await
        .map(|completion| completion.text)
    }

    /// Retain length-limited text for per-row NDJSON validation, never for graph/ops JSON.
    pub(crate) async fn extract_with_timeout(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        request_timeout: std::time::Duration,
    ) -> Result<ExtractionCompletion> {
        self.complete_in_mode(
            system,
            user,
            max_tokens,
            temperature,
            request_timeout,
            ChatResponseMode::Extraction,
        )
        .await
    }

    async fn complete_in_mode(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        request_timeout: std::time::Duration,
        mode: ChatResponseMode,
    ) -> Result<ExtractionCompletion> {
        tokio::time::timeout(request_timeout, async {
            let _permit = self
                .permits
                .acquire()
                .await
                .map_err(|_| MemoraError::Other("chat limiter closed".into()))?;
            self.complete_inner(system, user, max_tokens, temperature, request_timeout, mode)
                .await
        })
        .await
        .map_err(|_| MemoraError::Other("chat request timeout (total deadline)".into()))?
    }

    async fn complete_inner(
        &self,
        system: Option<&str>,
        user: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        request_timeout: std::time::Duration,
        mode: ChatResponseMode,
    ) -> Result<ExtractionCompletion> {
        let mut messages = Vec::new();
        if let Some(sys) = system.map(str::trim).filter(|s| !s.is_empty()) {
            messages.push(serde_json::json!({
                "role": "system",
                "content": sys,
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": user,
        }));
        let body = serde_json::json!({
            "model": self.config.chat_model,
            "messages": messages,
            "max_tokens": max_tokens.unwrap_or(self.config.max_tokens),
            "temperature": temperature.unwrap_or(0.2),
            "top_p": self.config.top_p,
            // Ark and DeepSeek spell it this way; other gateways ignore it.
            "thinking": {"type": "disabled"},
            "reasoning_split": self.config.reasoning_split,
            // This client is used for extract/ops, not open-ended reasoning.
            "enable_thinking": false,
        });
        let client = self.client.clone();
        let chat_url = self.config.chat_url.clone();
        let api_key = self.config.api_key.clone();
        with_transient_retries("chat-complete", || {
            let request_timeout = request_timeout;
            let client = client.clone();
            let chat_url = chat_url.clone();
            let api_key = api_key.clone();
            let body = body.clone();
            async move {
                let _model_permit = crate::model_queue::acquire(self.serial_requests).await;
                let resp = client
                    .post(&chat_url)
                    .timeout(request_timeout)
                    .bearer_auth(&api_key)
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| {
                        MemoraError::Other(format!(
                            "chat request failed: {}",
                            format_transport_error(e)
                        ))
                    })?;
                let status = resp.status();
                let text = resp.text().await.map_err(|e| {
                    MemoraError::Other(format!(
                        "chat response read failed: {}",
                        format_transport_error(e)
                    ))
                })?;
                if !status.is_success() {
                    // Inspect known overflow codes locally, never expose gateway body.
                    let lower = text.to_ascii_lowercase();
                    if matches!(status.as_u16(), 400 | 413 | 422)
                        && (lower.contains("context_length_exceeded")
                            || lower.contains("maximum context length")
                            || lower.contains("context window")
                            || lower.contains("too many tokens"))
                    {
                        return Err(MemoraError::Other("chat context limit exceeded".into()));
                    }
                    return Err(MemoraError::Other(format!("chat http {status}")));
                }
                match mode {
                    ChatResponseMode::Complete => {
                        parse_chat_response(&text).map(|text| ExtractionCompletion {
                            text,
                            truncated: false,
                        })
                    }
                    ChatResponseMode::Extraction => parse_extraction_response(&text),
                }
                .map_err(|e| MemoraError::Other(format!("chat response invalid: {e}")))
            }
        })
        .await
    }
}

#[async_trait]
impl Summarizer for QianfanSummarizer {
    fn summarizer_id(&self) -> &str {
        &self.config.chat_model
    }

    fn prompt_version(&self) -> &str {
        "qianfan-summarize-v1"
    }

    async fn summarize(&self, input: SummarizeInput) -> Result<SummarizeOutput> {
        let joined = input.texts.join("\n");
        let entities = if input.entities.is_empty() {
            String::new()
        } else {
            format!("相关实体：{}\n", input.entities.join("、"))
        };
        let prompt = format!(
            "请将下列记忆压缩为简洁中文摘要，保留关键事实与实体，不要编造。\n{entities}\n{joined}"
        );
        let _model_permit = crate::model_queue::acquire(self.serial_requests).await;
        let resp = self
            .client
            .post(&self.config.chat_url)
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&ChatRequest {
                messages: vec![ChatMessage {
                    role: "user",
                    content: &prompt,
                }],
                model: &self.config.chat_model,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                top_p: self.config.top_p,
                reasoning_split: self.config.reasoning_split,
            })
            .send()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        if !status.is_success() {
            return Err(MemoraError::Embedding(format!(
                "chat http {status}: {body}"
            )));
        }
        Ok(SummarizeOutput {
            summary: parse_chat_response(&body)?,
        })
    }
}

#[derive(Clone)]
pub struct QianfanClassifier {
    serial_requests: bool,
    client: reqwest::Client,
    config: QianfanConfig,
    fallback: HeuristicClassifier,
}

impl QianfanClassifier {
    /// 与其他启用串行的Memora模型客户端共用FIFO请求队列。
    pub fn with_serial_requests(mut self, enabled: bool) -> Self {
        self.serial_requests = enabled;
        self
    }

    pub fn new(config: QianfanConfig, archive_importance: f64) -> Result<Self> {
        Ok(Self {
            client: build_llm_client(10, 45)?,
            serial_requests: false,
            config,
            fallback: HeuristicClassifier::new(archive_importance),
        })
    }
}

#[async_trait]
impl MemoryClassifier for QianfanClassifier {
    fn classifier_id(&self) -> &str {
        "qianfan-classify-v1"
    }

    async fn classify_batch(&self, items: &[ClassifyItem]) -> Result<Vec<ClassifyOutput>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let payload: Vec<serde_json::Value> = items
            .iter()
            .map(|item| {
                serde_json::json!({
                    "id": item.id.to_string(),
                    "content": item.content,
                    "summary": item.summary,
                    "kind": item.kind.as_str(),
                    "entities": item.entities,
                    "importance": item.importance,
                    "confidence": item.confidence,
                })
            })
            .collect();
        let prompt = format!(
            r#"你是记忆巩固分类器。根据下列 JSON 数组中的记忆，为每条输出分类结果。
只返回 JSON 数组，不要 markdown，不要解释。每项字段：
- id: 原样返回
- kind: episodic|semantic|pattern|summary 之一
- importance: 0~1
- confidence: 0~1
- archive: 是否应归档（噪声/无价值为 true）
- entities: 可选，字符串数组
- patterns: 可选，从本条抽到的规律句子数组（没有则 []）
- subjects: [{{"member_id","confidence"}}]，member_id 只能来自名册；无法判断则 []
- domains: [{{"id","confidence"}}]，id 只能是 medical|education|travel|lifestyle|diet|finance|social|household

家庭名册：
{}

输入：
{}"#,
            items
                .first()
                .map(|i| i.roster_json.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("[]"),
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "[]".into())
        );
        let _model_permit = crate::model_queue::acquire(self.serial_requests).await;
        let resp = self
            .client
            .post(&self.config.chat_url)
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&ChatRequest {
                messages: vec![ChatMessage {
                    role: "user",
                    content: &prompt,
                }],
                model: &self.config.chat_model,
                max_tokens: self.config.max_tokens.max(2048),
                temperature: 0.2,
                top_p: self.config.top_p,
                reasoning_split: self.config.reasoning_split,
            })
            .send()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| MemoraError::Embedding(e.to_string()))?;
        if !status.is_success() {
            tracing::warn!(%status, "qianfan classify http failed; heuristic fallback");
            return Ok(self.fallback.classify_items(items));
        }
        let text = match parse_chat_response(&body) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "qianfan classify empty; heuristic fallback");
                return Ok(self.fallback.classify_items(items));
            }
        };
        match parse_classify_json(&text, items) {
            Ok(out) => Ok(out),
            Err(e) => {
                tracing::warn!(error = %e, "qianfan classify parse failed; heuristic fallback");
                Ok(self.fallback.classify_items(items))
            }
        }
    }
}

/// LLM-only knowledge-graph extractor. Failures never become empty success.
#[derive(Clone)]
pub struct QianfanGraphExtractor {
    serial_requests: bool,
    client: reqwest::Client,
    config: QianfanConfig,
    /// Shared by clones/callers; bound outbound requests, not just each document.
    permits: std::sync::Arc<tokio::sync::Semaphore>,
    limits: GraphExtractLimits,
}

/// Immediate retries for flaky networks; durable projection jobs still back off.
const LLM_TRANSIENT_ATTEMPTS: u32 = 3;

fn format_transport_error(err: reqwest::Error) -> String {
    // URLs (including query/userinfo credentials) and gateway causes are not
    // safe to surface in Android logs. Keep only actionable transport categories.
    if err.is_timeout() {
        "timeout".into()
    } else if err.is_connect() {
        "connection failed".into()
    } else if err.is_body() || err.is_decode() {
        "response read failed".into()
    } else {
        "transport failed".into()
    }
}

fn is_transient_llm_error(error: &MemoraError) -> bool {
    let msg = error.to_string().to_ascii_lowercase();
    // reqwest: "error sending request for url (...)" covers DNS/TLS/connect/proxy.
    msg.contains("error sending request")
        || msg.contains("chat request failed")
        || msg.contains("chat response read failed")
        || msg.contains("kg request failed")
        || msg.contains("kg response read failed")
        || msg.contains("timed out")
        || msg.contains("timeout")
        || msg.contains("connection")
        || msg.contains("connect")
        || msg.contains("dns")
        || msg.contains("tls")
        || msg.contains("ssl")
        || msg.contains("certificate")
        || msg.contains("reset")
        || msg.contains("broken pipe")
        || msg.contains("http 408")
        || msg.contains("http 425")
        || msg.contains("http 429")
        || msg.contains("http 500")
        || msg.contains("http 502")
        || msg.contains("http 503")
        || msg.contains("http 504")
        || msg.contains("chat http 408")
        || msg.contains("chat http 425")
        || msg.contains("chat http 429")
        || msg.contains("chat http 500")
        || msg.contains("chat http 502")
        || msg.contains("chat http 503")
        || msg.contains("chat http 504")
}

async fn with_transient_retries<T, F, Fut>(label: &str, op: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last = None;
    for attempt in 0..LLM_TRANSIENT_ATTEMPTS {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error)
                if attempt + 1 < LLM_TRANSIENT_ATTEMPTS && is_transient_llm_error(&error) =>
            {
                let delay_ms = 500u64.saturating_mul(1u64 << attempt.min(3));
                tracing::warn!(
                    %label,
                    attempt = attempt + 1,
                    delay_ms,
                    error = %error,
                    "llm transient failure; retrying"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                last = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last.unwrap_or_else(|| MemoraError::Other(format!("{label} retries exhausted"))))
}

/// Preserve the original source input cap independently of smaller chunks/request limits.
/// Previously (3000 - 240) * 12; larger documents must still be submitted as bounded sources.
const GRAPH_SOURCE_MAX_CHARS: usize = 33_120;

/// One source has ONE wall-clock deadline, not a fresh deadline per retry or child.
#[derive(Clone, Copy)]
struct GraphExtractLimits {
    source_timeout: std::time::Duration,
    request_timeout: std::time::Duration,
    max_requests: usize,
    max_splits: usize,
    max_depth: u8,
    concurrency: usize,
}

impl Default for GraphExtractLimits {
    fn default() -> Self {
        Self {
            source_timeout: std::time::Duration::from_secs(EXTRACTION_TIMEOUT_SECS),
            request_timeout: std::time::Duration::from_secs(EXTRACTION_TIMEOUT_SECS),
            // Boundary-aware chunks advance at least 750 - 240 = 510 chars.
            // 72 covers the preserved source cap plus up to three binary splits.
            max_requests: 72,
            max_splits: 3,
            max_depth: 2,
            concurrency: 1,
        }
    }
}

struct GraphRequestBudget {
    issued: AtomicUsize,
    maximum: usize,
}

impl GraphRequestBudget {
    fn claim(&self) -> Result<()> {
        self.issued
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < self.maximum).then_some(n + 1)
            })
            .map(|_| ())
            .map_err(|_| MemoraError::Other("kg source request budget exhausted".into()))
    }
}

#[derive(Clone)]
struct GraphWork {
    // Lexicographic tree path preserves source order even if siblings complete out of order.
    order: Vec<usize>,
    text: String,
    depth: u8,
}

/// Exactly TWO smaller children. The old general chunker could create three children
/// at every depth because overlap left a tiny tail, exponentially multiplying requests.
fn bisect_graph_chunk(text: &str) -> Option<[String; 2]> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= 512 {
        return None;
    }
    let mid = chars.len() / 2;
    let low = chars.len() * 2 / 5;
    let high = chars.len() * 3 / 5;
    let cut = (low..high)
        .filter(|&i| matches!(chars[i], '\n' | '。' | '！' | '？'))
        .min_by_key(|&i| i.abs_diff(mid))
        .map(|i| i + 1)
        .unwrap_or(mid);
    let overlap = 40.min(chars.len() / 16);
    let left: String = chars[..(cut + overlap).min(chars.len())].iter().collect();
    let right: String = chars[cut.saturating_sub(overlap)..].iter().collect();
    Some([left, right])
}

fn graph_error_allows_split(error: &MemoraError) -> bool {
    let message = error.to_string();
    // Timeout/network/refusal/malformed schema are NOT evidence of excessive input size.
    message.contains("truncated (length)")
        || message.contains("kg context limit exceeded")
        || message.contains("kg extraction incomplete (output budget)")
}

impl QianfanGraphExtractor {
    /// 与其他启用串行的Memora模型客户端共用FIFO请求队列。
    pub fn with_serial_requests(mut self, enabled: bool) -> Self {
        self.serial_requests = enabled;
        self
    }

    pub fn new(config: QianfanConfig) -> Result<Self> {
        Ok(Self {
            client: build_llm_client(10, EXTRACTION_TIMEOUT_SECS)?,
            serial_requests: false,
            config,
            permits: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
            limits: GraphExtractLimits::default(),
        })
    }

    async fn extract_chunk(
        &self,
        chunk: &str,
        roster: &[crate::household::HouseholdMember],
        speaker: Option<&str>,
        message_at_ms: Option<i64>,
        budget: &GraphRequestBudget,
    ) -> Result<GraphExtractOutput> {
        let roster_json = serde_json::to_string(roster)?;
        let prompt = kg_prompt_with_context(chunk, &roster_json, speaker, message_at_ms);
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| MemoraError::Other("kg request limiter closed".into()))?;
        budget.claim()?;
        let _model_permit = crate::model_queue::acquire(self.serial_requests).await;
        let resp = self
            .client
            .post(&self.config.chat_url)
            // Match the outer graph request budget; no hidden 30s HTTP cutoff.
            .timeout(self.limits.request_timeout)
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": prompt}],
                "model": self.config.chat_model,
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": self.config.top_p,
                // Ark and DeepSeek spell it this way; other gateways ignore it.
                "thinking": {"type": "disabled"},
                "reasoning_split": false,
                // Structured extraction should spend its output budget on JSON,
                // not a hidden reasoning trace. Compatible gateways ignore extras.
                "enable_thinking": false,
            }))
            .send()
            .await
            .map_err(|error| {
                MemoraError::Other(format!(
                    "kg request failed: {}",
                    format_transport_error(error)
                ))
            })?;
        let status = resp.status();
        let body = resp.text().await.map_err(|error| {
            MemoraError::Other(format!(
                "kg response read failed: {}",
                format_transport_error(error)
            ))
        })?;
        if !status.is_success() {
            // Classify context errors locally, but NEVER surface the response body.
            let lower = body.to_ascii_lowercase();
            if matches!(status.as_u16(), 400 | 413 | 422)
                && (lower.contains("context_length_exceeded")
                    || lower.contains("maximum context length")
                    || lower.contains("context window")
                    || lower.contains("too many tokens"))
            {
                return Err(MemoraError::Other("kg context limit exceeded".into()));
            }
            return Err(MemoraError::Other(format!(
                "kg request returned HTTP {status}"
            )));
        }
        let text = parse_chat_response(&body)
            .map_err(|error| MemoraError::Other(format!("kg response invalid: {error}")))?;
        let mut llm = parse_and_validate_llm_json(&text, chunk, roster)?;
        if llm.entities.is_empty() && llm.triples.is_empty() && !llm.facts.is_empty() {
            let (entities, triples) = crate::graph::ingest::facts_to_graph(&llm.facts);
            llm.entities = entities;
            llm.triples = triples;
        }
        llm.extractor_id = self.extractor_id().into();
        if !llm.pending.is_empty() {
            tracing::debug!(pending = llm.pending.len(), "kg llm left pending items");
        }
        Ok(llm)
    }
}

impl QianfanGraphExtractor {
    async fn extract_source(
        &self,
        input: GraphExtractInput,
        budget: std::sync::Arc<GraphRequestBudget>,
    ) -> Result<GraphExtractOutput> {
        // Bound CPU/memory preprocessing too. A large document must be submitted as bounded sources.
        if input.text.chars().take(GRAPH_SOURCE_MAX_CHARS + 1).count() > GRAPH_SOURCE_MAX_CHARS {
            return Err(MemoraError::Other(
                "kg source too large for request budget; split source before extraction".into(),
            ));
        }
        let chunks = split_source_text(&input.text, SOURCE_CHUNK_CHARS, SOURCE_CHUNK_OVERLAP);
        if chunks.len() > self.limits.max_requests {
            return Err(MemoraError::Other(
                "kg source has too many initial chunks for request budget".into(),
            ));
        }
        let input = std::sync::Arc::new(input);
        let mut queue: std::collections::VecDeque<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(index, text)| GraphWork {
                order: vec![index],
                text,
                depth: 0,
            })
            .collect();
        let mut jobs = tokio::task::JoinSet::new();
        let mut parts = Vec::new();
        let mut splits = 0;
        loop {
            while jobs.len() < self.limits.concurrency.max(1) {
                let Some(work) = queue.pop_front() else {
                    break;
                };
                let extractor = self.clone();
                let input = input.clone();
                let budget = budget.clone();
                jobs.spawn(async move {
                    let outcome = with_transient_retries("kg-chunk", || async {
                        // Includes semaphore wait, body read and parse. Outer source deadline also
                        // includes every retry/backoff and drops the JoinSet on timeout/failure.
                        tokio::time::timeout(
                            extractor.limits.request_timeout,
                            extractor.extract_chunk(
                                &work.text,
                                &input.roster,
                                input.speaker_member_id.as_deref(),
                                input.message_at_ms,
                                &budget,
                            ),
                        )
                        .await
                        .map_err(|_| MemoraError::Other("kg request timeout".into()))?
                    })
                    .await;
                    (work, outcome)
                });
            }
            let Some(joined) = jobs.join_next().await else {
                break;
            };
            let (work, result) = joined.map_err(|_| {
                MemoraError::Other("kg source worker failed; no partial output committed".into())
            })?;
            match result {
                Ok(part) => parts.push((work.order, part)),
                Err(error) => {
                    if graph_error_allows_split(&error)
                        && splits < self.limits.max_splits
                        && work.depth < self.limits.max_depth
                    {
                        if let Some(children) = bisect_graph_chunk(&work.text) {
                            // Every unresolved work item needs at least one request. Do not create
                            // children which can never fit even without transient retries.
                            let required = budget.issued.load(Ordering::SeqCst)
                                + queue.len()
                                + jobs.len()
                                + children.len();
                            if required > self.limits.max_requests {
                                return Err(MemoraError::Other("kg source request budget insufficient for subdivision; no partial output committed".into()));
                            }
                            splits += 1;
                            for (index, text) in children.into_iter().enumerate() {
                                let mut order = work.order.clone();
                                order.push(index);
                                queue.push_back(GraphWork {
                                    order,
                                    text,
                                    depth: work.depth + 1,
                                });
                            }
                            continue;
                        }
                    }
                    return Err(MemoraError::Other(format!(
                        "kg source failed; no partial output committed (requests={}, splits={}, completed_parts={}): {error}",
                        budget.issued.load(Ordering::SeqCst), splits, parts.len())));
                }
            }
        }
        parts.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(merge_graph_extract_outputs(
            self.extractor_id(),
            parts.into_iter().map(|(_, part)| part),
        ))
    }
}

#[async_trait]
impl GraphExtractor for QianfanGraphExtractor {
    fn extractor_id(&self) -> &str {
        "qianfan-kg-v2-bounded"
    }

    async fn extract(&self, input: GraphExtractInput) -> Result<GraphExtractOutput> {
        let budget = std::sync::Arc::new(GraphRequestBudget {
            issued: AtomicUsize::new(0),
            maximum: self.limits.max_requests,
        });
        // Encloses initial split, all spawned children, permits, retries and their sleeps.
        // This function returns no output unless ALL leaves finish successfully.
        tokio::time::timeout(self.limits.source_timeout, self.extract_source(input, budget.clone()))
            .await.map_err(|_| MemoraError::Other(format!(
                "kg source total deadline exceeded ({}ms, requests={}); no partial output committed",
                self.limits.source_timeout.as_millis(), budget.issued.load(Ordering::SeqCst))))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_embedding_payload() {
        let body = r#"{"data":[{"embedding":[0.1,0.2,0.3]},{"embedding":[0.4,0.5,0.6]}]}"#;
        let vecs = parse_embed_response(body).unwrap();
        assert_eq!(vecs.len(), 2);
        let norm: f32 = vecs[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parse_chat_payload() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"用户偏好 SQLite"}}]}"#;
        assert_eq!(parse_chat_response(body).unwrap(), "用户偏好 SQLite");
    }

    #[test]
    fn parse_rejects_truncated_and_filtered_even_if_json_looks_complete() {
        for reason in ["length", "max_tokens", "content_filter"] {
            let body = serde_json::json!({"choices":[{"finish_reason":reason,
                "message":{"content":"{\"entities\":[],\"triples\":[]}"}}]})
            .to_string();
            assert!(parse_chat_response(&body).is_err());
        }
    }

    #[test]
    fn extraction_preserves_length_limited_ndjson_without_relaxing_ordinary_json() {
        let ndjson =
            "{\"content\":\"完整第一行\"}\n{\"content\":\"完整第二行\"}\n{\"content\":\"未完成";
        for reason in ["length", "max_tokens"] {
            let body = serde_json::json!({"choices":[{"finish_reason":reason,
                "message":{"content":ndjson}}]})
            .to_string();
            let error = parse_chat_response(&body).unwrap_err().to_string();
            assert_eq!(error, "chat response truncated (length)");
            let completion = parse_extraction_response(&body).unwrap();
            assert_eq!(completion.text, ndjson);
            assert!(completion.truncated);
            assert!(completion
                .text
                .lines()
                .take(2)
                .all(|line| { serde_json::from_str::<serde_json::Value>(line).is_ok() }));
        }
    }

    #[test]
    fn extraction_accepts_empty_only_when_length_limited_and_supports_legacy_stop() {
        for content in [
            serde_json::json!(""),
            serde_json::json!("  "),
            serde_json::Value::Null,
        ] {
            for reason in ["length", "max_tokens"] {
                let body = serde_json::json!({"choices":[{"finish_reason":reason,
                    "message":{"content":content}}]})
                .to_string();
                let completion = parse_extraction_response(&body).unwrap();
                assert!(completion.truncated);
                assert!(completion.text.trim().is_empty());
            }
            for reason in [serde_json::json!("stop"), serde_json::Value::Null] {
                let body = serde_json::json!({"choices":[{"finish_reason":reason,
                    "message":{"content":content}}]})
                .to_string();
                assert!(parse_extraction_response(&body).is_err());
            }
        }
        for body in [
            r#"{"choices":[{"finish_reason":"stop","message":{"content":"完整行"}}]}"#,
            r#"{"choices":[{"message":{"content":"完整行"}}]}"#,
        ] {
            let completion = parse_extraction_response(body).unwrap();
            assert_eq!(completion.text, "完整行");
            assert!(!completion.truncated);
        }
        assert!(parse_extraction_response(r#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn extraction_rejects_filtered_unknown_and_error_finish_without_raw_content() {
        for reason in [
            "content_filter",
            "error",
            "tool_calls",
            "private-upstream-timeout",
        ] {
            let body = serde_json::json!({"choices":[{"finish_reason":reason,
                "message":{"content":"private-complete-ndjson-row"}}]})
            .to_string();
            let error = parse_extraction_response(&body).unwrap_err().to_string();
            assert!(!error.contains("private-complete-ndjson-row"));
            assert!(!error.contains("private-upstream-timeout"));
            assert!(!is_transient_llm_error(&MemoraError::Other(error)));
        }
        // serde's detailed invalid-type errors can contain raw gateway fields.
        let error = parse_extraction_response(
            r#"{"choices":[{"finish_reason":"stop","message":{"content":{"private-field":"private-value"}}}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(!error.contains("private-field") && !error.contains("private-value"));
    }

    #[test]
    fn parse_rejects_empty_chat() {
        let body = r#"{"choices":[]}"#;
        assert!(parse_chat_response(body).is_err());
    }

    #[test]
    fn classifies_transient_network_and_http_errors() {
        assert!(is_transient_llm_error(&MemoraError::Other(
            "chat request failed: error sending request for url (https://token.example/v1/chat/completions)".into()
        )));
        assert!(is_transient_llm_error(&MemoraError::Other(
            "kg request failed: error sending request for url".into()
        )));
        assert!(is_transient_llm_error(&MemoraError::Other(
            "kg request returned HTTP 503".into()
        )));
        assert!(is_transient_llm_error(&MemoraError::Other(
            "chat http 429".into()
        )));
        assert!(!is_transient_llm_error(&MemoraError::InvalidState(
            "kg extract json: invalid type".into()
        )));
        assert!(!is_transient_llm_error(&MemoraError::Other(
            "kg response was truncated or filtered".into()
        )));
    }
}

#[cfg(test)]
mod reliability_tests {
    use super::*;
    #[test]
    fn embedding_indices_are_sorted_and_invalid_data_rejected() {
        let vectors = parse_embed_response(
            r#"{"data":[{"index":1,"embedding":[0,2]},{"index":0,"embedding":[2,0]}]}"#,
        )
        .unwrap();
        assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(parse_embed_response(r#"{"data":[{"index":1,"embedding":[1,0]}]}"#).is_err());
        assert!(parse_embed_response(r#"{"data":[{"embedding":[0,0]}]}"#).is_err());
    }
    #[test]
    fn auto_dimension_binds_once_and_provider_identity_is_stable() {
        let cfg = QianfanConfig::for_embed(
            "key".into(),
            Some("https://embed.example/embeddings".into()),
            Some("model".into()),
            Some(0),
        );
        let e = QianfanEmbedder::new(cfg).unwrap();
        let space = e.space_id();
        assert!(e.bind_dimension(8));
        assert!(!e.bind_dimension(16));
        assert_eq!(e.dimension(), 8);
        assert_eq!(space, e.space_id());
    }
}

#[cfg(test)]
mod graph_budget_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    use std::time::{Duration, Instant};

    struct Server {
        url: String,
        stop: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
        thread: Option<std::thread::JoinHandle<()>>,
    }
    impl Server {
        fn start(
            handler: impl Fn(serde_json::Value) -> (u16, String) + Send + Sync + 'static,
        ) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let url = format!("http://{}/chat", listener.local_addr().unwrap());
            let stop = Arc::new(AtomicBool::new(false));
            let calls = Arc::new(AtomicUsize::new(0));
            let stopped = stop.clone();
            let counted = calls.clone();
            let handler = Arc::new(handler);
            let thread = std::thread::spawn(move || {
                let mut workers = Vec::new();
                while !stopped.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let handler = handler.clone();
                            let counted = counted.clone();
                            workers.push(std::thread::spawn(move || {
                                stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                                let mut bytes = Vec::new();
                                let body = loop {
                                    let mut buf = [0u8; 4096];
                                    let Ok(n) = stream.read(&mut buf) else { return; };
                                    if n == 0 { return; } bytes.extend_from_slice(&buf[..n]);
                                    if let Some(end) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                                        let header = String::from_utf8_lossy(&bytes[..end]);
                                        let len = header.lines().find_map(|s| {
                                            let (k,v) = s.split_once(':')?;
                                            k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().unwrap())
                                        }).unwrap_or(0);
                                        if bytes.len() >= end + 4 + len {
                                            break serde_json::from_slice(&bytes[end+4..end+4+len]).unwrap();
                                        }
                                    }
                                };
                                counted.fetch_add(1, Ordering::SeqCst);
                                let (status, body) = handler(body);
                                let response = format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                                let _ = stream.write_all(response.as_bytes());
                            }));
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(1))
                        }
                        Err(_) => break,
                    }
                }
                for worker in workers {
                    let _ = worker.join();
                }
            });
            Self {
                url,
                stop,
                calls,
                thread: Some(thread),
            }
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let _ = self.thread.take().unwrap().join();
        }
    }
    fn envelope(content: serde_json::Value, finish: &str) -> String {
        serde_json::json!({"choices":[{"finish_reason":finish,"message":{"content":content.to_string()}}]}).to_string()
    }
    fn empty() -> serde_json::Value {
        serde_json::json!({"entities":[],"triples":[],"pending":[],"complete":true})
    }
    fn extractor(url: String) -> QianfanGraphExtractor {
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = url;
        let mut e = QianfanGraphExtractor::new(cfg).unwrap();
        e.client = reqwest::Client::builder().no_proxy().build().unwrap();
        e
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

    #[test]
    fn extraction_defaults_share_ten_minutes() {
        assert_eq!(EXTRACTION_TIMEOUT_SECS, 600);
        let limits = GraphExtractLimits::default();
        assert_eq!(limits.source_timeout, Duration::from_secs(600));
        assert_eq!(limits.request_timeout, limits.source_timeout);
        assert_eq!(limits.concurrency, 1);
        assert_eq!(limits.max_requests, 72);
        assert_eq!(limits.max_splits, 3);
        assert_eq!(limits.max_depth, 2);
        assert_eq!(GRAPH_SOURCE_MAX_CHARS, 33_120);
        let minimum_advance = SOURCE_CHUNK_CHARS / 2 - SOURCE_CHUNK_OVERLAP;
        let maximum_initial_chunks = GRAPH_SOURCE_MAX_CHARS.div_ceil(minimum_advance);
        assert!(maximum_initial_chunks + 2 * limits.max_splits <= limits.max_requests);
    }

    #[tokio::test]
    async fn graph_request_override_does_not_keep_short_client_cutoff() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_millis(120));
            (200, envelope(empty(), "stop"))
        });
        let mut e = extractor(server.url.clone());
        e.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        e.limits.request_timeout = Duration::from_secs(2);
        e.limits.source_timeout = Duration::from_secs(2);
        e.extract(input("张三住在北京".into())).await.unwrap();
        assert_eq!(server.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn graph_retry_backoff_does_not_reset_source_deadline() {
        let server = Server::start(|_| (503, "private-upstream-body".into()));
        let mut e = extractor(server.url.clone());
        e.limits.source_timeout = Duration::from_millis(750);
        e.limits.request_timeout = Duration::from_secs(600);
        let start = Instant::now();
        let err = e
            .extract(input("张三住在北京".into()))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("total deadline"), "{err}");
        assert!(!err.contains("private-upstream-body"));
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(server.calls.load(Ordering::SeqCst) <= 2);
        tokio::task::yield_now().await;
        assert_eq!(e.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn chat_request_override_and_deadline_release_permits() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_millis(120));
            (200, envelope(empty(), "stop"))
        });
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let mut chat = QianfanChat::new(cfg).unwrap();
        chat.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        chat.complete_with_timeout(None, "fixture", Some(128), None, Duration::from_secs(2))
            .await
            .unwrap();
        let held = chat.permits.acquire_many(2).await.unwrap();
        let err = chat
            .complete_with_timeout(None, "fixture", None, None, Duration::from_millis(30))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("total deadline"));
        assert_eq!(server.calls.load(Ordering::SeqCst), 1);
        drop(held);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn chat_retry_backoff_uses_one_total_deadline() {
        let server = Server::start(|_| (503, "private-upstream-body".into()));
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let chat = QianfanChat::new(cfg).unwrap();
        let start = Instant::now();
        let err = chat
            .complete_with_timeout(None, "fixture", None, None, Duration::from_millis(750))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("total deadline"));
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(server.calls.load(Ordering::SeqCst) <= 2);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn chat_extraction_api_retains_ndjson_but_ordinary_api_rejects_length() {
        let ndjson = "{\"content\":\"完整第一行\"}\n{\"content\":\"完整第二行\"}\n{\"content\":";
        let server = Server::start(move |request| {
            let reason = request["messages"].as_array().unwrap().last().unwrap()["content"]
                .as_str()
                .unwrap();
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":reason,
                    "message":{"content":ndjson}}]})
                .to_string(),
            )
        });
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let mut chat = QianfanChat::new(cfg).unwrap();
        chat.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        for reason in ["length", "max_tokens"] {
            let error = chat
                .complete_with_timeout(None, reason, Some(128), None, Duration::from_secs(2))
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("chat response truncated (length)"));
            let completion = chat
                .extract_with_timeout(None, reason, Some(128), None, Duration::from_secs(2))
                .await
                .unwrap();
            assert_eq!(completion.text, ndjson);
            assert!(completion.truncated);
        }
        for reason in ["content_filter", "error"] {
            let error = chat
                .extract_with_timeout(None, reason, Some(128), None, Duration::from_secs(2))
                .await
                .unwrap_err()
                .to_string();
            assert!(!error.contains("完整第一行"));
        }
        let completion = chat
            .extract_with_timeout(None, "stop", Some(128), None, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(completion.text, ndjson);
        assert!(!completion.truncated);
        assert_eq!(server.calls.load(Ordering::SeqCst), 7);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn extraction_request_override_preserves_longer_explicit_deadline() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_millis(120));
            (200, envelope(empty(), "length"))
        });
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let mut chat = QianfanChat::new(cfg).unwrap();
        chat.client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        let completion = chat
            .extract_with_timeout(None, "fixture", Some(128), None, Duration::from_secs(2))
            .await
            .unwrap();
        assert!(completion.truncated);
        assert_eq!(completion.text, empty().to_string());
        assert_eq!(server.calls.load(Ordering::SeqCst), 1);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn extraction_deadline_cancels_inflight_calls_and_releases_both_permits() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_millis(600));
            (200, envelope(empty(), "length"))
        });
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let mut chat = QianfanChat::new(cfg).unwrap();
        chat.client = reqwest::Client::builder().no_proxy().build().unwrap();
        let calls: Vec<_> = (0..2)
            .map(|_| {
                let chat = chat.clone();
                tokio::spawn(async move {
                    chat.extract_with_timeout(
                        None,
                        "fixture",
                        Some(128),
                        None,
                        Duration::from_millis(300),
                    )
                    .await
                })
            })
            .collect();
        tokio::time::timeout(Duration::from_millis(250), async {
            while server.calls.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(chat.permits.available_permits(), 0);
        for call in calls {
            let error = call.await.unwrap().unwrap_err().to_string();
            assert!(error.contains("total deadline"), "{error}");
        }
        assert_eq!(chat.permits.available_permits(), 2);

        let held = chat.permits.acquire_many(2).await.unwrap();
        let error = chat
            .extract_with_timeout(None, "waiting", None, None, Duration::from_millis(30))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("total deadline"));
        assert_eq!(server.calls.load(Ordering::SeqCst), 2);
        drop(held);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn extraction_retry_backoff_uses_one_total_deadline_and_releases_permits() {
        let server = Server::start(|_| (503, "private-upstream-body".into()));
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        let mut chat = QianfanChat::new(cfg).unwrap();
        chat.client = reqwest::Client::builder().no_proxy().build().unwrap();
        let start = Instant::now();
        let error = chat
            .extract_with_timeout(None, "fixture", None, None, Duration::from_millis(750))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("total deadline"));
        assert!(!error.contains("private-upstream-body"));
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(server.calls.load(Ordering::SeqCst) <= 2);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    /// Manual/local smoke: actually cross all old 30/45/50-second cutoffs.
    /// No real provider, credentials or household data; both calls run in parallel.
    #[tokio::test]
    #[ignore = "65-second loopback smoke, run explicitly when changing timeout policy"]
    async fn slow_extraction_survives_old_thirty_and_fifty_second_cutoffs() {
        let server = Server::start(|_| {
            std::thread::sleep(Duration::from_secs(65));
            (200, envelope(empty(), "stop"))
        });
        let mut cfg = QianfanConfig::for_embed("fixture-only".into(), None, None, Some(8));
        cfg.chat_url = server.url.clone();
        // Production constructors/HTTP clients: do not replace their timeouts here.
        let graph = QianfanGraphExtractor::new(cfg.clone()).unwrap();
        let chat = QianfanChat::new(cfg).unwrap();
        let start = Instant::now();
        let (g, c) = tokio::join!(
            graph.extract(input("张三住在北京".into())),
            chat.complete_with_timeout(
                None,
                "fixture",
                Some(128),
                None,
                Duration::from_secs(EXTRACTION_TIMEOUT_SECS)
            )
        );
        g.unwrap();
        c.unwrap();
        assert!(start.elapsed() >= Duration::from_secs(65));
        assert!(start.elapsed() < Duration::from_secs(90));
        assert_eq!(server.calls.load(Ordering::SeqCst), 2);
        assert_eq!(graph.permits.available_permits(), 1);
        assert_eq!(chat.permits.available_permits(), 2);
    }

    #[test]
    fn adaptive_bisect_has_exactly_two_smaller_unicode_safe_children() {
        for size in [513, 900, 1500, 3000] {
            let text = "甲😀乙。".chars().cycle().take(size).collect::<String>();
            let parts = bisect_graph_chunk(&text).unwrap();
            assert!(parts
                .iter()
                .all(|p| p.chars().count() < size && text.contains(p.as_str())));
            assert!(parts.iter().map(|p| p.chars().count()).sum::<usize>() <= size + 80);
        }
        assert!(bisect_graph_chunk(&"甲".repeat(512)).is_none());
        assert!(!graph_error_allows_split(&MemoraError::Other(
            "kg request timeout".into()
        )));
        assert!(!graph_error_allows_split(&MemoraError::Other(
            "kg extract: no JSON object".into()
        )));
    }

    #[tokio::test]
    async fn deadline_includes_permit_wait_and_leaks_no_permits() {
        let mut e = extractor("http://127.0.0.1:1/chat".into());
        e.limits.source_timeout = Duration::from_millis(30);
        e.limits.request_timeout = Duration::from_secs(5);
        let held = e.permits.acquire_many(1).await.unwrap();
        let start = Instant::now();
        let error = e
            .extract(input("张三住在北京".into()))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("total deadline") && error.contains("requests=0"));
        assert!(start.elapsed() < Duration::from_secs(1));
        drop(held);
        tokio::task::yield_now().await;
        assert_eq!(e.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn total_deadline_bounds_source_even_before_first_request() {
        let server = Server::start(|_| (503, "not-safe-to-log".into()));
        let mut e = extractor(server.url.clone());
        e.limits.source_timeout = Duration::from_millis(100);
        let error = e
            .extract(input("张三住在北京".into()))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("total deadline") && !error.contains("not-safe-to-log"));
        // On a heavily loaded host the deadline can elapse before the first socket is scheduled.
        // Either outcome must respect the same end-to-end deadline; retries have separate tests.
        assert!(server.calls.load(Ordering::SeqCst) <= 1);
    }

    #[tokio::test]
    async fn request_budget_counts_every_transient_retry() {
        let server = Server::start(|_| (503, "upstream-secret".into()));
        let mut e = extractor(server.url.clone());
        e.limits.max_requests = 2;
        let error = e
            .extract(input("张三住在北京".into()))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("budget") && error.contains("no partial output committed"));
        assert!(!error.contains("upstream-secret"));
        assert_eq!(server.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn subdivision_has_global_limit_and_never_returns_partial_success() {
        let server = Server::start(|_| (200, envelope(empty(), "length")));
        let mut e = extractor(server.url.clone());
        e.limits.max_splits = 1;
        let error = e
            .extract(input("甲".repeat(SOURCE_CHUNK_CHARS)))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("no partial output committed"));
        assert!(server.calls.load(Ordering::SeqCst) <= 3);
    }

    #[tokio::test]
    async fn incomplete_contract_triggers_two_children_with_ordered_merge() {
        let server = Server::start(|request| {
            assert_eq!(request["max_tokens"], 4096);
            assert_eq!(request["enable_thinking"], false);
            assert_eq!(request["reasoning_split"], false);
            let prompt = request["messages"][0]["content"].as_str().unwrap();
            let data: serde_json::Value =
                serde_json::from_str(prompt.rsplit_once('\n').unwrap().1).unwrap();
            let text = data["source_text"].as_str().unwrap();
            if text.chars().count() > SOURCE_CHUNK_CHARS / 2 + 80 {
                return (
                    200,
                    envelope(
                        serde_json::json!({"entities":[],"triples":[],"pending":[],"complete":false}),
                        "stop",
                    ),
                );
            }
            let first = text.chars().next().unwrap().to_string();
            if first == "甲" {
                std::thread::sleep(Duration::from_millis(60));
            }
            (
                200,
                envelope(
                    serde_json::json!({"entities":[],"triples":[],"pending":[{"reason":"ambiguous","text":first}],"complete":true}),
                    "stop",
                ),
            )
        });
        let e = extractor(server.url.clone());
        let out = e
            .extract(input(format!(
                "{}{}",
                "甲".repeat(SOURCE_CHUNK_CHARS / 2 - 80),
                "乙".repeat(SOURCE_CHUNK_CHARS / 2 + 80)
            )))
            .await
            .unwrap();
        assert_eq!(server.calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            out.pending
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            vec!["甲", "乙"]
        );
    }

    #[tokio::test]
    async fn chunks_run_serially_and_one_failed_leaf_discards_all() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let a = active.clone();
        let p = peak.clone();
        let server = Server::start(move |_| {
            let n = a.fetch_add(1, Ordering::SeqCst) + 1;
            p.fetch_max(n, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(60));
            a.fetch_sub(1, Ordering::SeqCst);
            (200, envelope(empty(), "stop"))
        });
        let e = extractor(server.url.clone());
        e.extract(input("甲".repeat(9000))).await.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        let server = Server::start(|request| {
            let prompt = request["messages"][0]["content"].as_str().unwrap();
            let data: serde_json::Value =
                serde_json::from_str(prompt.rsplit_once('\n').unwrap().1).unwrap();
            if data["source_text"].as_str().unwrap().starts_with('乙') {
                (401, "secret".into())
            } else {
                (200, envelope(empty(), "stop"))
            }
        });
        let e = extractor(server.url.clone());
        let error = e
            .extract(input(format!("{}{}", "甲".repeat(3000), "乙".repeat(4000))))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("no partial output committed") && error.contains("401"));
    }

    async fn assert_bounded_serial_source_coverage(size: usize) {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let a = active.clone();
        let p = peak.clone();
        let r = received.clone();
        let server = Server::start(move |request| {
            let n = a.fetch_add(1, Ordering::SeqCst) + 1;
            p.fetch_max(n, Ordering::SeqCst);
            let prompt = request["messages"][0]["content"].as_str().unwrap();
            let data: serde_json::Value =
                serde_json::from_str(prompt.rsplit_once('\n').unwrap().1).unwrap();
            let text = data["source_text"].as_str().unwrap();
            assert!(text.chars().count() <= SOURCE_CHUNK_CHARS);
            r.lock().unwrap().push(text.to_string());
            std::thread::sleep(Duration::from_millis(5));
            a.fetch_sub(1, Ordering::SeqCst);
            (
                200,
                envelope(
                    serde_json::json!({"entities":[],"triples":[],"pending":[{"reason":"fixture coverage","text":text}],"complete":true}),
                    "stop",
                ),
            )
        });
        // Distinct non-ASCII characters make gaps/reordering visible; emoji also
        // prevents confusing the source character cap with its UTF-8 byte length.
        let source: String = (0..size)
            .map(|i| {
                if i % 97 == 96 {
                    '。'
                } else if i % 17 == 16 {
                    '😀'
                } else {
                    char::from_u32(0x4e00 + i as u32).unwrap()
                }
            })
            .collect();
        assert_eq!(source.chars().count(), size);
        let source_chars: Vec<_> = source.chars().collect();
        let e = extractor(server.url.clone());
        let out = e.extract(input(source)).await.unwrap();
        let chunks = received.lock().unwrap();
        assert_eq!(server.calls.load(Ordering::SeqCst), chunks.len());
        assert!(chunks.len() > 1 && chunks.len() <= e.limits.max_requests);
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(e.permits.available_permits(), 1);
        assert_eq!(
            out.pending
                .iter()
                .map(|item| &item.text)
                .collect::<Vec<_>>(),
            chunks.iter().collect::<Vec<_>>()
        );
        let mut covered_until = 0;
        for (index, chunk) in chunks.iter().enumerate() {
            let chars: Vec<_> = chunk.chars().collect();
            let start = if index == 0 {
                0
            } else {
                covered_until - SOURCE_CHUNK_OVERLAP
            };
            let end = start + chars.len();
            assert!(chars.len() <= SOURCE_CHUNK_CHARS && end > covered_until);
            assert_eq!(&source_chars[start..end], chars.as_slice());
            covered_until = end;
        }
        assert_eq!(
            covered_until, size,
            "every source character must be covered"
        );
    }

    #[tokio::test]
    async fn unicode_21200_character_document_has_small_serial_chunks_and_full_coverage() {
        assert_eq!(SOURCE_CHUNK_CHARS, 1500);
        assert_eq!(SOURCE_CHUNK_OVERLAP, 240);
        assert_bounded_serial_source_coverage(21_200).await;
    }

    #[tokio::test]
    async fn original_source_character_cap_remains_fully_supported() {
        assert_bounded_serial_source_coverage(GRAPH_SOURCE_MAX_CHARS).await;
    }

    #[tokio::test]
    async fn too_large_source_fails_before_any_request() {
        let server = Server::start(|_| panic!("must not call HTTP"));
        let e = extractor(server.url.clone());
        assert!(e
            .extract(input("甲".repeat(GRAPH_SOURCE_MAX_CHARS + 1)))
            .await
            .unwrap_err()
            .to_string()
            .contains("too large"));
        assert_eq!(server.calls.load(Ordering::SeqCst), 0);
    }
}
