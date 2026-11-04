use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use memora::{
    DreamOptions, ExportRequest, ForgetReason, GrantInput, GraphOverviewQuery, HouseholdMember,
    ImportMode, ListQuery, MemberRole, MemoryEngine, MemoryKind, MemoryService, MemoryState,
    MemoryTag, MemoryUpdate, PersonGraphQuery, QianfanChat, RecallQuery, ReconcileCheckpoint,
    ReconcileOutcome, RememberInput, RevokeGrantInput,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

use crate::reconcile::ReconcileRequest;

#[derive(Clone)]
pub struct AppState {
    pub engine: MemoryEngine,
    /// Android-aligned extract→ops curator; None when LLM_API_KEY missing.
    pub chat: Option<QianfanChat>,
    pub bind: String,
    pub db_path: String,
    pub embedder_fallback: String,
    pub tick_interval_secs: Option<u64>,
    pub dream_interval_secs: Option<u64>,
    pub reconcile_jobs: ReconcileJobs,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/memories", get(list_memories).post(remember))
        .route("/memories/{id}", get(get_memory).patch(revise))
        .route("/memories/{id}/explain", get(explain))
        .route("/memories/{id}/forget", post(forget))
        .route("/memories/{id}/restore", post(restore))
        .route("/recall", post(recall))
        .route("/reconcile", post(reconcile_turn))
        .route("/reconcile/jobs/{id}", get(reconcile_job))
        .route("/reconcile/resume/{id}", post(resume_reconcile))
        .route("/tick", post(tick))
        .route("/dream", post(dream))
        .route("/export", get(export_archive))
        .route("/import", post(import_archive))
        .route("/rebuild-index", post(rebuild_index))
        .route("/household/members", get(list_members).post(upsert_member))
        .route(
            "/household/members/{member_id}",
            axum::routing::delete(remove_member),
        )
        .route("/household/tags", get(list_tags).post(upsert_tag))
        .route("/graph/person", post(query_person_graph))
        .route(
            "/graph/overview",
            get(query_graph_overview).post(query_graph_overview_post),
        )
        .route("/graph/ingest", post(ingest_graph))
        .route("/graph/grants", get(list_grants).post(grant_access))
        .route("/graph/grants/revoke", post(revoke_access))
        .with_state(state)
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

struct ApiError(memora::MemoraError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            memora::MemoraError::NotFound(_) => StatusCode::NOT_FOUND,
            memora::MemoraError::InvalidState(_) => StatusCode::CONFLICT,
            memora::MemoraError::Rejected(_) => StatusCode::UNPROCESSABLE_ENTITY,
            memora::MemoraError::Json(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorBody {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

impl From<memora::MemoraError> for ApiError {
    fn from(value: memora::MemoraError) -> Self {
        Self(value)
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

#[derive(Serialize)]
struct StatusResponse {
    namespace: String,
    embedder: String,
    summarizer: String,
    classifier: String,
    reconcile_chat: bool,
    bind: String,
    db_path: String,
    tick_interval_secs: Option<u64>,
    dream_interval_secs: Option<u64>,
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        namespace: state.engine.namespace().to_string(),
        embedder: state
            .engine
            .embedder_model()
            .unwrap_or(&state.embedder_fallback)
            .to_string(),
        summarizer: state.engine.summarizer_id().to_string(),
        classifier: state.engine.classifier_id().to_string(),
        reconcile_chat: state.chat.is_some(),
        bind: state.bind.clone(),
        db_path: state.db_path.clone(),
        tick_interval_secs: state.tick_interval_secs,
        dream_interval_secs: state.dream_interval_secs,
    })
}

const ASYNC_RECONCILE_CHARS: usize = 1_200;
const MAX_RECONCILE_INPUT_CHARS: usize = 120_000;
const RECONCILE_JOB_CAPACITY: usize = 10;
const RECONCILE_CHECKPOINT_TTL: Duration = Duration::from_secs(30 * 60);
const MAX_JOB_FILES: usize = 256;

// One reservation travels from admission, through work, into a pending checkpoint.
// In particular a full pending store cannot admit work and silently lose its result.
struct PendingCheckpoint {
    created_at: Instant,
    checkpoint: ReconcileCheckpoint,
    slot: OwnedSemaphorePermit,
    job_id: Option<Uuid>,
    _expiry: CheckpointExpiry,
}

struct CheckpointExpiry(tokio::task::JoinHandle<()>);

impl Drop for CheckpointExpiry {
    fn drop(&mut self) {
        // A consumed/replaced checkpoint no longer needs its old timer. This
        // keeps timer tasks bounded along with checkpoints, even on fast resumes.
        self.0.abort();
    }
}

fn checkpoint_expiry(
    pending: &Arc<Mutex<HashMap<Uuid, PendingCheckpoint>>>,
    id: Uuid,
    created_at: Instant,
    ttl: Duration,
) -> CheckpointExpiry {
    let pending = Arc::downgrade(pending);
    // Capture no checkpoint, request or strong store reference while sleeping.
    CheckpointExpiry(tokio::spawn(async move {
        tokio::time::sleep_until(tokio::time::Instant::from_std(created_at + ttl)).await;
        if let Some(pending) = pending.upgrade() {
            let mut pending = pending.lock().unwrap_or_else(|error| error.into_inner());
            if pending.get(&id).is_some_and(|entry| {
                entry.created_at == created_at && entry.created_at.elapsed() >= ttl
            }) {
                // Exact generation only: also drops raw input and its reservation
                // if no client ever polls or submits another request.
                pending.remove(&id);
            }
        }
    }))
}

#[derive(Clone)]
pub struct ReconcileJobs {
    directory: Arc<PathBuf>,
    workers: Arc<Semaphore>,
    slots: Arc<Semaphore>,
    admission: Arc<tokio::sync::Mutex<()>>,
    // In-process only: never serialize checkpoints or expose a GET endpoint.
    pending: Arc<Mutex<HashMap<Uuid, PendingCheckpoint>>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JobRecord {
    job_id: Uuid,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    report: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl ReconcileJobs {
    /// Results live beside SQLite, never in it. No request payload is persisted.
    pub fn open(db_path: &FsPath) -> io::Result<Self> {
        let filename = db_path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "database filename required")
        })?;
        let mut dirname = filename.to_os_string();
        dirname.push(".reconcile-jobs");
        let directory = db_path.with_file_name(dirname);
        std::fs::create_dir_all(&directory)?;
        if std::fs::symlink_metadata(&directory)?
            .file_type()
            .is_symlink()
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "job directory cannot be a symlink",
            ));
        }
        restrict_job_permissions(&directory, true)?;
        let jobs = Self {
            directory: Arc::new(directory),
            workers: Arc::new(Semaphore::new(2)),
            slots: Arc::new(Semaphore::new(RECONCILE_JOB_CAPACITY)),
            admission: Arc::new(tokio::sync::Mutex::new(())),
            pending: Arc::new(Mutex::new(HashMap::new())),
        };
        for entry in std::fs::read_dir(jobs.directory.as_ref())? {
            let path = entry?.path();
            let Some(id) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| Uuid::parse_str(s).ok())
            else {
                continue;
            };
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "job file cannot be a symlink",
                ));
            }
            restrict_job_permissions(&path, false)?;
            let mut record = jobs.read(id)?;
            if matches!(record.status.as_str(), "queued" | "running") {
                record.status = "failed".into();
                record.error = Some("interrupted: server restarted; task was not replayed".into());
                record.report = None;
                jobs.write(&record)?;
            }
        }
        Ok(jobs)
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, HashMap<Uuid, PendingCheckpoint>> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Expiry is checked on every admission, resume and poll. Dropping a
        // checkpoint also releases its reserved slot, without persisting raw text.
        pending.retain(|_, entry| entry.created_at.elapsed() < RECONCILE_CHECKPOINT_TTL);
        pending
    }

    fn reserve(&self) -> Option<OwnedSemaphorePermit> {
        let _pending = self.pending();
        self.slots.clone().try_acquire_owned().ok()
    }

    fn take_checkpoint(&self, id: Uuid) -> Option<PendingCheckpoint> {
        self.pending().remove(&id)
    }

    fn finish_outcome(
        &self,
        outcome: ReconcileOutcome,
        slot: OwnedSemaphorePermit,
        job_id: Option<Uuid>,
    ) -> Value {
        match outcome {
            ReconcileOutcome::Complete(report) => serde_json::to_value(report)
                .unwrap_or_else(|_| json!({"error":"report serialization failed"})),
            ReconcileOutcome::Pending { report, checkpoint } => {
                let mut report = serde_json::to_value(report).unwrap_or_else(
                    |_| json!({"staged":true,"error":"report serialization failed"}),
                );
                let id = Uuid::now_v7();
                let created_at = Instant::now();
                let mut pending = self.pending();
                let expiry =
                    checkpoint_expiry(&self.pending, id, created_at, RECONCILE_CHECKPOINT_TTL);
                pending.insert(
                    id,
                    PendingCheckpoint {
                        created_at,
                        checkpoint,
                        slot,
                        job_id,
                        _expiry: expiry,
                    },
                );
                drop(pending);
                report["resume_id"] = json!(id);
                report["resume_expires_in_secs"] = json!(RECONCILE_CHECKPOINT_TTL.as_secs());
                report
            }
        }
    }

    fn hydrate_report(&self, record: &mut JobRecord) {
        let pending = self.pending();
        let Some(report) = record.report.as_mut() else {
            return;
        };
        if report.get("staged").and_then(Value::as_bool) != Some(true) {
            return;
        }
        if let Some(object) = report.as_object_mut() {
            object.remove("resume_id");
        }
        if let Some((id, entry)) = pending
            .iter()
            .find(|(_, entry)| entry.job_id == Some(record.job_id))
        {
            report["resume_id"] = json!(id);
            report["resume_expires_in_secs"] = json!(RECONCILE_CHECKPOINT_TTL
                .saturating_sub(entry.created_at.elapsed())
                .as_secs());
            if let Some(object) = report.as_object_mut() {
                object.remove("resume_error");
            }
        } else {
            report["resume_expires_in_secs"] = json!(0);
            report["resume_error"] = json!(
                "暂存已过期、已被继续请求领取或服务已重启；此暂存报告未写入记忆，旧令牌不可重放。"
            );
        }
    }

    fn path(&self, id: Uuid) -> PathBuf {
        self.directory.join(format!("{id}.json"))
    }

    fn read(&self, id: Uuid) -> io::Result<JobRecord> {
        let path = self.path(id);
        let meta = std::fs::symlink_metadata(&path)?;
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 16 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid job result file",
            ));
        }
        let record: JobRecord = serde_json::from_slice(&std::fs::read(path)?)?;
        if record.job_id != id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "job id mismatch",
            ));
        }
        Ok(record)
    }

    fn write(&self, record: &JobRecord) -> io::Result<()> {
        let bytes = serde_json::to_vec(record)?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "job result exceeds size limit",
            ));
        }
        let temp = self.directory.join(format!("{}.tmp", Uuid::now_v7()));
        let result = (|| {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            restrict_job_permissions(&temp, false)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, self.path(record.job_id))?;
            #[cfg(unix)]
            std::fs::File::open(self.directory.as_ref())?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }

    fn file_count(&self) -> io::Result<usize> {
        Ok(std::fs::read_dir(self.directory.as_ref())?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .count())
    }
}

#[cfg(unix)]
fn restrict_job_permissions(path: &FsPath, directory: bool) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(
        path,
        std::fs::Permissions::from_mode(if directory { 0o700 } else { 0o600 }),
    )
}

#[cfg(windows)]
fn restrict_job_permissions(path: &FsPath, directory: bool) -> io::Result<()> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "advapi32")]
    extern "system" {
        fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
            s: *const u16,
            revision: u32,
            descriptor: *mut *mut c_void,
            size: *mut u32,
        ) -> i32;
        fn SetFileSecurityW(path: *const u16, info: u32, descriptor: *mut c_void) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn LocalFree(memory: *mut c_void) -> *mut c_void;
    }
    // Protected DACL: only the file owner and SYSTEM. Inheritance applies to
    // children; each temporary file is also protected before any content write.
    let sddl = if directory {
        "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)"
    } else {
        "D:P(A;;FA;;;OW)(A;;FA;;;SY)"
    };
    let sddl: Vec<u16> = sddl.encode_utf16().chain(Some(0)).collect();
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe {
        let mut descriptor = std::ptr::null_mut();
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut descriptor,
            std::ptr::null_mut(),
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        let ok = SetFileSecurityW(path.as_ptr(), 0x80000004, descriptor);
        let error = if ok == 0 {
            Some(io::Error::last_os_error())
        } else {
            None
        };
        LocalFree(descriptor);
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn restrict_job_permissions(_: &FsPath, _: bool) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "private job storage unavailable",
    ))
}

#[derive(Deserialize)]
struct ReconcileBody {
    #[serde(flatten)]
    request: ReconcileRequest,
    #[serde(default, rename = "async", alias = "async_mode")]
    asynchronous: bool,
}

fn effective_input_chars(request: &ReconcileRequest) -> usize {
    let mut seen = HashSet::new();
    let mut total = 0usize;
    for source in &request.evidence {
        let text = source.text.trim();
        if !text.is_empty() && seen.insert(text) {
            total = total.saturating_add(text.chars().count());
        }
    }
    if seen.is_empty() {
        request.query.trim().chars().count()
    } else {
        total
    }
}

fn input_chars(request: &ReconcileRequest) -> usize {
    // Count all retained input, including repeated evidence and context, rather
    // than just effective extraction sources. No raw payload is queued above 120k.
    let mut total = request
        .query
        .chars()
        .count()
        .saturating_add(request.answer.chars().count())
        .saturating_add(request.transcript.chars().count());
    for value in [&request.speaker_id, &request.namespace, &request.session_id]
        .into_iter()
        .flatten()
    {
        total = total.saturating_add(value.chars().count());
    }
    for source in &request.evidence {
        for value in [&source.text, &source.id, &source.name, &source.kind] {
            total = total.saturating_add(value.chars().count());
        }
        total = total.saturating_add(source.locator.as_deref().unwrap_or("").chars().count());
    }
    total
}

fn persisted_report(mut report: Value) -> Value {
    // Raw model completions and warnings can echo full requests or provider errors.
    if let Some(object) = report.as_object_mut() {
        object.remove("extract_raw");
        object.remove("ops_raw");
        // Tokens are only recoverable in this process. Polling hydrates them from
        // memory, so a restarted server never advertises a fake continuation.
        object.remove("resume_id");
        object.remove("resume_expires_in_secs");
        if object.get("staged").and_then(Value::as_bool) == Some(true) {
            object.insert(
                "resume_error".into(),
                json!("暂存只在服务器内存保留30分钟；重启后失效，未写入记忆。"),
            );
        }
        let count = object
            .get("warnings")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        object.insert(
            "warnings".into(),
            if count == 0 {
                json!([])
            } else {
                json!([format!(
                    "{count} warning(s); sensitive diagnostic text omitted"
                )])
            },
        );
    }
    report
}

async fn reconcile_turn(
    State(state): State<AppState>,
    Json(body): Json<ReconcileBody>,
) -> ApiResult<Response> {
    let chat = state.chat.clone().ok_or_else(|| {
        ApiError(memora::MemoraError::InvalidState(
            "reconcile 需要 LLM_API_KEY".into(),
        ))
    })?;
    if input_chars(&body.request) > MAX_RECONCILE_INPUT_CHARS {
        return Ok((StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({"error":"输入（含上下文与附件）超过120000 Unicode字符；未启动抽取、未写入记忆"}))).into_response());
    }
    let jobs = state.reconcile_jobs.clone();
    let Some(slot) = jobs.reserve() else {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error":"抽取任务与暂存合计已达10个；请先继续已有暂存或等待30分钟过期"})),
        )
            .into_response());
    };
    if !body.asynchronous && effective_input_chars(&body.request) <= ASYNC_RECONCILE_CHARS {
        // Short compatibility requests are detached as well: dropping an HTTP
        // connection must not cancel an apply halfway through.
        let task =
            tokio::spawn(async move {
                let _worker =
                    jobs.workers.clone().acquire_owned().await.map_err(|_| {
                        memora::MemoraError::Other("reconcile workers closed".into())
                    })?;
                let outcome = memora::prepare_reconcile(&state.engine, &chat, body.request).await?;
                Ok::<_, memora::MemoraError>(jobs.finish_outcome(outcome, slot, None))
            });
        let report = task.await.map_err(|_| {
            ApiError(memora::MemoraError::Other(
                "reconcile interrupted; inspect memories before resubmitting".into(),
            ))
        })??;
        return Ok(([(header::CACHE_CONTROL, "no-store")], Json(report)).into_response());
    }
    start_reconcile_job(state, chat, ReconcileWork::Prepare(body.request), slot).await
}

async fn resume_reconcile(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let chat = state.chat.clone().ok_or_else(|| {
        ApiError(memora::MemoraError::InvalidState(
            "reconcile 需要 LLM_API_KEY".into(),
        ))
    })?;
    let checkpoint = Uuid::parse_str(&id)
        .ok()
        .and_then(|id| state.reconcile_jobs.take_checkpoint(id));
    let Some(checkpoint) = checkpoint else {
        return Ok((
            StatusCode::GONE,
            Json(json!({"error":
            "暂存不存在、已过期、服务已重启或已被另一继续请求领取；请勿重复提交或自动重试。"})),
        )
            .into_response());
    };
    // No await between atomic take and start_reconcile_job's detached supervisor.
    // Never put a consumed checkpoint back, including ambiguous apply failures.
    start_reconcile_job(
        state,
        chat,
        ReconcileWork::Resume(checkpoint.checkpoint),
        checkpoint.slot,
    )
    .await
}

enum ReconcileWork {
    Prepare(ReconcileRequest),
    Resume(ReconcileCheckpoint),
}

async fn start_reconcile_job(
    state: AppState,
    chat: QianfanChat,
    work: ReconcileWork,
    slot: OwnedSemaphorePermit,
) -> ApiResult<Response> {
    let jobs = state.reconcile_jobs.clone();
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel::<io::Result<Option<Uuid>>>();
    // Detach BEFORE the first admission await. An HTTP cancellation during the
    // blocking durable write must not leave a queued record without a worker.
    tokio::spawn(async move {
        // The supervisor owns capacity for admission, queued and running work.
        let admission = jobs.admission.lock().await;
        let storage = jobs.clone();
        let admission_result = tokio::task::spawn_blocking(move || -> io::Result<Option<Uuid>> {
            if storage.file_count()? >= MAX_JOB_FILES {
                return Ok(None);
            }
            let id = Uuid::now_v7();
            storage.write(&JobRecord {
                job_id: id,
                status: "queued".into(),
                report: None,
                error: None,
            })?;
            Ok(Some(id))
        })
        .await;
        drop(admission);
        let id = match admission_result {
            Ok(Ok(Some(id))) => id,
            Ok(result) => {
                let _ = accepted_tx.send(result);
                return;
            }
            Err(_) => {
                let _ = accepted_tx.send(Err(io::Error::new(
                    io::ErrorKind::Other,
                    "job storage unavailable",
                )));
                return;
            }
        };
        // Losing the receiver only means the caller disconnected; the durable
        // job must still run and persist its terminal state.
        let _ = accepted_tx.send(Ok(Some(id)));
        let Ok(_worker) = jobs.workers.clone().acquire_owned().await else {
            return;
        };
        let storage = jobs.clone();
        if !matches!(
            tokio::task::spawn_blocking(move || storage.write(&JobRecord {
                job_id: id,
                status: "running".into(),
                report: None,
                error: None
            }))
            .await,
            Ok(Ok(()))
        ) {
            tracing::error!(job_id = %id, "cannot start reconcile job: status storage unavailable");
            return;
        }
        let outcome = tokio::spawn(async move {
            match work {
                ReconcileWork::Prepare(request) => {
                    memora::prepare_reconcile(&state.engine, &chat, request).await
                }
                ReconcileWork::Resume(checkpoint) => {
                    memora::resume_reconcile(&state.engine, &chat, checkpoint).await
                }
            }
        })
        .await;
        let record = match outcome {
            Ok(Ok(outcome)) => JobRecord { job_id: id, status: "completed".into(),
                report: Some(persisted_report(jobs.finish_outcome(outcome, slot, Some(id)))), error: None },
            _ => JobRecord { job_id: id, status: "failed".into(), report: None,
                error: Some("reconcile failed; checkpoint consumed and cannot be replayed; if apply started memories may have changed, inspect before resubmitting".into()) },
        };
        if !matches!(
            tokio::task::spawn_blocking(move || jobs.write(&record)).await,
            Ok(Ok(()))
        ) {
            tracing::error!(job_id = %id, "cannot persist reconcile job terminal status");
        }
    });
    let admission_result = accepted_rx
        .await
        .map_err(|_| ApiError(memora::MemoraError::Other("job storage unavailable".into())))?
        .map_err(|_| ApiError(memora::MemoraError::Other("job storage unavailable".into())))?;
    let Some(id) = admission_result else {
        return Ok((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":"job result storage full; no work started; any claimed continuation was discarded without writing memories; archive completed results first"})),
        )
            .into_response());
    };
    Ok((
        StatusCode::ACCEPTED,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({
            "job_id": id, "status": "queued", "status_url": format!("/api/reconcile/jobs/{id}")
        })),
    )
        .into_response())
}

async fn reconcile_job(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Ok(id) = Uuid::parse_str(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    let jobs = state.reconcile_jobs;
    let storage = jobs.clone();
    match tokio::task::spawn_blocking(move || storage.read(id)).await {
        Ok(Ok(mut record)) => {
            jobs.hydrate_report(&mut record);
            ([(header::CACHE_CONTROL, "no-store")], Json(record)).into_response()
        }
        Ok(Err(error)) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":"job storage unavailable"})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct ListParams {
    namespace: Option<String>,
    state: Option<String>,
    kind: Option<String>,
    keyword: Option<String>,
    member_id: Option<String>,
    tag: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn list_memories(
    State(state): State<AppState>,
    Query(q): Query<ListParams>,
) -> ApiResult<Json<memora::MemoryPage>> {
    let state_filter = match q.state.as_deref() {
        None | Some("") | Some("all") => None,
        Some(s) => Some(MemoryState::parse(s).map_err(memora::MemoraError::from)?),
    };
    let kind_filter = match q.kind.as_deref() {
        None | Some("") | Some("all") => None,
        Some(s) => Some(MemoryKind::parse(s).map_err(memora::MemoraError::from)?),
    };
    let page = state
        .engine
        .list(ListQuery {
            namespace: q.namespace,
            state: state_filter,
            kind: kind_filter,
            keyword: q.keyword.filter(|s| !s.trim().is_empty()),
            member_id: q.member_id.filter(|s| !s.trim().is_empty()),
            tag: q.tag.filter(|s| !s.trim().is_empty()),
            limit: q.limit,
            offset: q.offset,
            ..ListQuery::default()
        })
        .await?;
    Ok(Json(page))
}

#[derive(Deserialize)]
struct RememberBody {
    content: String,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    kind: Option<MemoryKind>,
    #[serde(default)]
    importance: Option<f64>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    pin_to_stm: bool,
    #[serde(default)]
    stm_key: Option<String>,
    #[serde(default)]
    actor: Option<String>,
    #[serde(default)]
    skip_consolidation: bool,
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    domains: Vec<String>,
    /// Explicit household subjects (`[]` = public/shared). Aligned with JNI.
    #[serde(default)]
    subjects: Option<Vec<String>>,
    #[serde(default)]
    event_at: Option<String>,
    #[serde(default)]
    event_at_ms: Option<i64>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Serialize)]
struct IdResponse {
    id: Uuid,
}

fn subjects_metadata(subjects: &[String]) -> Value {
    let cleaned: Vec<Value> = subjects
        .iter()
        .map(|id| id.trim())
        .filter(|id| !id.is_empty())
        .map(|id| json!({ "member_id": id, "confidence": 1.0 }))
        .collect();
    let scope = if cleaned.is_empty() {
        "public"
    } else {
        "members"
    };
    // attribution=manual → engine.annotate 不再按正文重推归属
    json!({ "subjects": cleaned, "scope": scope, "attribution": "manual" })
}

fn overlay_metadata(base: Value, overlay: &Value) -> Value {
    let mut out = if base.is_object() { base } else { json!({}) };
    if let Some(obj) = overlay.as_object() {
        let map = out.as_object_mut().unwrap();
        for (k, v) in obj {
            if v.is_null() {
                map.remove(k);
            } else {
                map.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

async fn remember(
    State(state): State<AppState>,
    Json(body): Json<RememberBody>,
) -> ApiResult<Json<IdResponse>> {
    if body.content.trim().is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "content is required".into(),
        )));
    }
    let mut input = RememberInput::episodic(body.content);
    if let Some(kind) = body.kind {
        input.kind = kind;
    }
    input.summary = body.summary;
    input.importance = body.importance.unwrap_or(0.5).clamp(0.0, 1.0);
    input.confidence = body.confidence.unwrap_or(0.8).clamp(0.0, 1.0);
    input.session_id = body.session_id;
    input.namespace = body.namespace;
    input.entities = body.entities;
    input.pin_to_stm = body.pin_to_stm;
    input.stm_key = body.stm_key;
    input.actor = body.actor;
    input.skip_consolidation = body.skip_consolidation;
    input.member_id = body.member_id;
    input.domains = body.domains;
    // Align with JNI remember_input: subjects / event overlays live in metadata.
    // Tag is always inferred inside the engine.
    input.metadata = body.metadata.unwrap_or_else(|| json!({}));
    if let Some(subjects) = body.subjects {
        input.metadata = overlay_metadata(input.metadata, &subjects_metadata(&subjects));
    }
    if let Some(date) = body.event_at {
        input.metadata["event_at"] = json!(date);
    }
    if let Some(ms) = body.event_at_ms {
        input.metadata["event_at_ms"] = json!(ms);
    }
    let id = state.engine.remember(input).await?;
    Ok(Json(IdResponse { id }))
}

async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<memora::MemoryRecord>> {
    let rec = state
        .engine
        .get(id)
        .await?
        .ok_or_else(|| memora::MemoraError::NotFound(id.to_string()))?;
    Ok(Json(rec))
}

async fn revise(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(update): Json<MemoryUpdate>,
) -> ApiResult<Json<serde_json::Value>> {
    state.engine.revise(id, update).await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn explain(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(q): Query<ExplainParams>,
) -> ApiResult<Json<memora::MemoryExplanation>> {
    Ok(Json(
        state
            .engine
            .explain_visible(id, q.exclude_sensitive)
            .await?,
    ))
}

#[derive(Deserialize)]
struct ExplainParams {
    #[serde(default)]
    exclude_sensitive: bool,
}

#[derive(Deserialize)]
struct ForgetBody {
    #[serde(default)]
    reason: String,
}

async fn forget(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<ForgetBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let reason = if body.reason.trim().is_empty() {
        "user".into()
    } else {
        body.reason
    };
    state
        .engine
        .forget(id, ForgetReason::User { reason })
        .await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn restore(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<serde_json::Value>> {
    state.engine.restore(id).await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn recall(
    State(state): State<AppState>,
    Json(query): Json<RecallQuery>,
) -> ApiResult<Json<memora::RecallResult>> {
    if query.text.trim().is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "query text is required".into(),
        )));
    }
    Ok(Json(state.engine.recall(query).await?))
}

async fn tick(State(state): State<AppState>) -> ApiResult<Json<memora::TickReport>> {
    Ok(Json(state.engine.tick().await?))
}

async fn dream(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> ApiResult<Json<memora::DreamReport>> {
    let opts = if body.is_empty() {
        DreamOptions::default()
    } else {
        serde_json::from_slice(&body).map_err(memora::MemoraError::from)?
    };
    Ok(Json(state.engine.dream(opts).await?))
}

#[derive(Deserialize)]
struct ExportParams {
    namespace: Option<String>,
    #[serde(default)]
    include_forgotten: bool,
    #[serde(default)]
    include_vectors: bool,
    #[serde(default)]
    exclude_sensitive: bool,
}

async fn export_archive(
    State(state): State<AppState>,
    Query(q): Query<ExportParams>,
) -> ApiResult<Response> {
    let bytes = state
        .engine
        .export(ExportRequest {
            namespace: q.namespace,
            include_forgotten: q.include_forgotten,
            include_vectors: q.include_vectors,
            exclude_sensitive: q.exclude_sensitive,
        })
        .await?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/zip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"memora-export.zip\"",
            ),
        ],
        bytes,
    )
        .into_response())
}

#[derive(Deserialize)]
struct ImportQuery {
    mode: Option<String>,
}

async fn import_archive(
    State(state): State<AppState>,
    Query(q): Query<ImportQuery>,
    body: axum::body::Bytes,
) -> ApiResult<Json<memora::ImportReport>> {
    let mode = match q.mode.as_deref().unwrap_or("merge") {
        "merge" => ImportMode::Merge,
        "replace_namespace" => ImportMode::ReplaceNamespace,
        "dry_run" => ImportMode::DryRun,
        "reembed" => ImportMode::Reembed,
        other => {
            return Err(ApiError(memora::MemoraError::InvalidState(format!(
                "unknown import mode: {other}"
            ))))
        }
    };
    Ok(Json(state.engine.import(&body, mode).await?))
}

async fn rebuild_index(State(state): State<AppState>) -> ApiResult<Json<serde_json::Value>> {
    state.engine.rebuild_index().await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

#[derive(Deserialize)]
struct NamespaceQuery {
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Deserialize)]
struct MemberBody {
    member_id: String,
    display_name: String,
    role: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    age_band: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
}

async fn list_members(
    State(state): State<AppState>,
    Query(q): Query<NamespaceQuery>,
) -> ApiResult<Json<Vec<HouseholdMember>>> {
    Ok(Json(
        state.engine.list_members(q.namespace.as_deref()).await?,
    ))
}

async fn upsert_member(
    State(state): State<AppState>,
    Json(body): Json<MemberBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let role = MemberRole::parse(&body.role)?;
    state
        .engine
        .upsert_member(
            body.namespace.as_deref(),
            HouseholdMember {
                member_id: body.member_id,
                display_name: body.display_name,
                role,
                aliases: body.aliases,
                age_band: body.age_band,
            },
        )
        .await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn remove_member(
    State(state): State<AppState>,
    Path(member_id): Path<String>,
    Query(q): Query<NamespaceQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    state
        .engine
        .remove_member(q.namespace.as_deref(), &member_id)
        .await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

#[derive(Deserialize)]
struct TagBody {
    #[serde(default)]
    tag_id: Option<String>,
    display_name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    namespace: Option<String>,
}

async fn list_tags(
    State(state): State<AppState>,
    Query(q): Query<NamespaceQuery>,
) -> ApiResult<Json<Vec<MemoryTag>>> {
    Ok(Json(state.engine.list_tags(q.namespace.as_deref()).await?))
}

async fn upsert_tag(
    State(state): State<AppState>,
    Json(body): Json<TagBody>,
) -> ApiResult<Json<serde_json::Value>> {
    let display = body.display_name.trim();
    if display.is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "display_name is required".into(),
        )));
    }
    let tag_id = body
        .tag_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(display)
        .to_string();
    state
        .engine
        .upsert_tag(
            body.namespace.as_deref(),
            MemoryTag {
                tag_id,
                display_name: display.to_string(),
                aliases: body.aliases,
            },
        )
        .await?;
    Ok(Json(json!({ "ok": true })))
}

async fn query_person_graph(
    State(state): State<AppState>,
    Json(query): Json<PersonGraphQuery>,
) -> ApiResult<Json<memora::PersonGraph>> {
    if query.member_id.trim().is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "member_id is required".into(),
        )));
    }
    Ok(Json(state.engine.query_person_graph(query).await?))
}

#[derive(Deserialize)]
struct GraphOverviewParams {
    #[serde(default)]
    include_ended: bool,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    viewer_member_id: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

async fn query_graph_overview(
    State(state): State<AppState>,
    Query(q): Query<GraphOverviewParams>,
) -> ApiResult<Json<memora::GraphOverview>> {
    Ok(Json(
        state
            .engine
            .query_graph_overview(GraphOverviewQuery {
                namespace: q.namespace,
                domains: vec![],
                include_ended: q.include_ended,
                viewer_member_id: q.viewer_member_id,
                limit: q.limit,
            })
            .await?,
    ))
}

async fn query_graph_overview_post(
    State(state): State<AppState>,
    Json(query): Json<GraphOverviewQuery>,
) -> ApiResult<Json<memora::GraphOverview>> {
    Ok(Json(state.engine.query_graph_overview(query).await?))
}

#[derive(Deserialize)]
struct GraphIngestBody {
    text: String,
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
}

async fn ingest_graph(
    State(state): State<AppState>,
    Json(body): Json<GraphIngestBody>,
) -> ApiResult<Json<memora::IngestReport>> {
    if body.text.trim().is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "text is required".into(),
        )));
    }
    Ok(Json(
        state
            .engine
            .ingest_graph_text(body.text, body.member_id, body.namespace)
            .await?,
    ))
}

#[derive(Deserialize)]
struct ListGrantsParams {
    #[serde(default)]
    member_id: Option<String>,
    #[serde(default)]
    as_grantor: bool,
    #[serde(default)]
    namespace: Option<String>,
}

async fn list_grants(
    State(state): State<AppState>,
    Query(q): Query<ListGrantsParams>,
) -> ApiResult<Json<Vec<memora::KgGrant>>> {
    Ok(Json(
        state
            .engine
            .list_grants(q.member_id, q.as_grantor, q.namespace)
            .await?,
    ))
}

async fn grant_access(
    State(state): State<AppState>,
    Json(body): Json<GrantInput>,
) -> ApiResult<Json<memora::KgGrant>> {
    if body.grantor_member_id.trim().is_empty() || body.grantee_member_id.trim().is_empty() {
        return Err(ApiError(memora::MemoraError::InvalidState(
            "grantor_member_id and grantee_member_id are required".into(),
        )));
    }
    Ok(Json(state.engine.grant_access(body).await?))
}

async fn revoke_access(
    State(state): State<AppState>,
    Json(body): Json<RevokeGrantInput>,
) -> ApiResult<Json<serde_json::Value>> {
    let n = state.engine.revoke_access(body).await?;
    Ok(Json(json!({ "ok": true, "revoked": n })))
}

#[cfg(test)]
mod reconcile_job_tests {
    use super::*;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("memora-jobs-test-{}", Uuid::now_v7()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn request(value: Value) -> ReconcileBody {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn unicode_threshold_uses_effective_sources_not_context_or_duplicate_query() {
        let text = "记".repeat(ASYNC_RECONCILE_CHARS);
        let short = request(
            json!({"query":text,"answer":"答".repeat(50_000),"transcript":"史".repeat(50_000)}),
        );
        assert_eq!(effective_input_chars(&short.request), ASYNC_RECONCILE_CHARS);
        assert!(!short.asynchronous);
        let long =
            request(json!({"query":"指令", "evidence":[{"id":"a","text":"记".repeat(12_001)}]}));
        assert!(effective_input_chars(&long.request) > ASYNC_RECONCILE_CHARS);
        let duplicate = request(
            json!({"query":"同文", "evidence":[{"id":"a","text":"同文"},{"id":"b","text":" 同文 "}]}),
        );
        assert_eq!(effective_input_chars(&duplicate.request), 2);
        let empty = request(json!({"query":" 查询 ","evidence":[{"id":"a","text":"  "}]}));
        assert_eq!(effective_input_chars(&empty.request), 2);
        assert!(request(json!({"query":"短", "async":true})).asynchronous);
        assert!(request(json!({"query":"短", "async_mode":true})).asynchronous);
    }

    #[test]
    fn durable_recovery_never_replays_and_files_contain_only_state_and_result() {
        let directory = TestDirectory::new();
        let db = directory.0.join("memory.db");
        let jobs = ReconcileJobs::open(&db).unwrap();
        let queued = Uuid::now_v7();
        let running = Uuid::now_v7();
        let completed = Uuid::now_v7();
        for (id, status) in [
            (queued, "queued"),
            (running, "running"),
            (completed, "completed"),
        ] {
            jobs.write(&JobRecord {
                job_id: id,
                status: status.into(),
                report: None,
                error: None,
            })
            .unwrap();
        }
        let persisted: Value =
            serde_json::from_slice(&std::fs::read(jobs.path(queued)).unwrap()).unwrap();
        assert!(persisted.get("query").is_none());
        assert!(persisted.get("request").is_none());
        let restarted = ReconcileJobs::open(&db).unwrap();
        for id in [queued, running] {
            let record = restarted.read(id).unwrap();
            assert_eq!(record.status, "failed");
            assert!(record.error.unwrap().starts_with("interrupted"));
        }
        assert_eq!(restarted.read(completed).unwrap().status, "completed");
        assert_eq!(restarted.file_count().unwrap(), 3);
        assert!(restarted
            .path(queued)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(&queued.to_string()));
        assert!(!std::fs::read_dir(restarted.directory.as_ref())
            .unwrap()
            .any(
                |entry| entry.unwrap().path().extension().and_then(|s| s.to_str()) == Some("tmp")
            ));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(restarted.directory.as_ref())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(restarted.path(queued))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn worker_and_queue_semaphores_are_bounded() {
        let directory = TestDirectory::new();
        let jobs = ReconcileJobs::open(&directory.0.join("memory.db")).unwrap();
        let workers: Vec<_> = (0..2)
            .map(|_| jobs.workers.clone().try_acquire_owned().unwrap())
            .collect();
        assert!(jobs.workers.clone().try_acquire_owned().is_err());
        drop(workers);
        assert_eq!(jobs.workers.available_permits(), 2);
        let slots: Vec<_> = (0..RECONCILE_JOB_CAPACITY)
            .map(|_| jobs.slots.clone().try_acquire_owned().unwrap())
            .collect();
        assert!(jobs.slots.clone().try_acquire_owned().is_err());
        drop(slots);
        assert_eq!(jobs.slots.available_permits(), RECONCILE_JOB_CAPACITY);
    }

    #[tokio::test]
    async fn checkpoint_expiry_timer_runs_without_reads_or_strong_store_ownership() {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let timer = checkpoint_expiry(
            &pending,
            Uuid::now_v7(),
            Instant::now(),
            Duration::from_millis(1),
        );
        assert_eq!(Arc::strong_count(&pending), 1);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(timer.0.is_finished());
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn consumed_checkpoint_cancels_old_expiry_task() {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let timer = checkpoint_expiry(
            &pending,
            Uuid::now_v7(),
            Instant::now(),
            RECONCILE_CHECKPOINT_TTL,
        );
        let task = timer.0.abort_handle();
        drop(timer);
        tokio::task::yield_now().await;
        assert!(task.is_finished());
        assert_eq!(Arc::strong_count(&pending), 1);
    }

    #[test]
    fn input_limit_counts_unicode_context_and_duplicate_evidence() {
        let at_limit = request(json!({"query":"😀".repeat(MAX_RECONCILE_INPUT_CHARS)}));
        assert_eq!(input_chars(&at_limit.request), MAX_RECONCILE_INPUT_CHARS);
        let context = request(json!({"query":"😀", "answer":"答".repeat(60_000),
            "transcript":"史".repeat(60_000)}));
        assert_eq!(input_chars(&context.request), MAX_RECONCILE_INPUT_CHARS + 1);
        let duplicated = request(json!({"query":"指令", "evidence":[
            {"id":"a", "text":"证".repeat(60_000)}, {"id":"b", "text":"证".repeat(60_000)}]}));
        assert_eq!(effective_input_chars(&duplicated.request), 60_000);
        assert!(input_chars(&duplicated.request) > MAX_RECONCILE_INPUT_CHARS);
    }

    #[test]
    fn persisted_staged_report_has_no_raw_completions_or_live_tokens() {
        let report = persisted_report(json!({"staged":true, "resume_id":Uuid::now_v7(),
            "resume_expires_in_secs":1800, "extract_raw":"PRIVATE RAW", "ops_raw":"PRIVATE OPS",
            "warnings":["PRIVATE INPUT"], "extraction":{"remaining_ranges":[
                {"source_id":"source-a", "start":0, "end":4, "reason":"incomplete"}]}}));
        assert!(report.get("resume_id").is_none());
        assert!(report.get("resume_expires_in_secs").is_none());
        assert!(report.get("extract_raw").is_none());
        assert!(report.get("ops_raw").is_none());
        assert!(!report.to_string().contains("PRIVATE"));
        assert!(report["resume_error"]
            .as_str()
            .unwrap()
            .contains("重启后失效"));
        assert_eq!(report["extraction"]["remaining_ranges"][0]["end"], 4);
    }

    #[test]
    fn restart_hydration_never_advertises_stale_resume_tokens() {
        let directory = TestDirectory::new();
        let jobs = ReconcileJobs::open(&directory.0.join("memory.db")).unwrap();
        let mut record = JobRecord {
            job_id: Uuid::now_v7(),
            status: "completed".into(),
            report: Some(
                json!({"staged":true,"resume_id":Uuid::now_v7(),"resume_expires_in_secs":1800}),
            ),
            error: None,
        };
        jobs.hydrate_report(&mut record);
        let report = record.report.unwrap();
        assert!(report.get("resume_id").is_none());
        assert_eq!(report["resume_expires_in_secs"], 0);
        assert!(report["resume_error"].as_str().unwrap().contains("已重启"));
        assert!(jobs.take_checkpoint(Uuid::now_v7()).is_none());
        assert_eq!(jobs.slots.available_permits(), RECONCILE_JOB_CAPACITY);
    }

    #[test]
    fn job_identity_mismatch_is_rejected() {
        let directory = TestDirectory::new();
        let jobs = ReconcileJobs::open(&directory.0.join("memory.db")).unwrap();
        let id = Uuid::now_v7();
        jobs.write(&JobRecord {
            job_id: id,
            status: "completed".into(),
            report: Some(json!({"facts":[]})),
            error: None,
        })
        .unwrap();
        std::fs::rename(jobs.path(id), jobs.path(Uuid::now_v7())).unwrap();
        assert_eq!(jobs.read(id).unwrap_err().kind(), io::ErrorKind::NotFound);
    }
}
