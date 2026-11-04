const state = {
  offset: 0,
  limit: 20,
  selectedId: null,
  members: [],
  dialogue: [],
  sessionId: `web-${Date.now().toString(36)}`,
  reconcileChat: false,
  reconcileBusy: false,
  pendingReconcile: null,
  graphSim: null,
  lastGraph: null,
  graphFactFilter: null,
};

const KIND_LABEL = {
  episodic: "情景",
  semantic: "语义",
  pattern: "模式",
  summary: "摘要",
};

const ROLE_LABEL = {
  self: "本人",
  spouse: "配偶",
  child: "孩子",
  parent: "父母",
  other: "其他",
};

function $(id) {
  return document.getElementById(id);
}

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 2800);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: {
      ...(options.body && !(options.body instanceof ArrayBuffer) && !(options.body instanceof Blob)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (options.raw) {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      const error = new Error(err.error || res.statusText);
      error.status = res.status;
      throw error;
    }
    return res;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || res.statusText);
  }
  return data;
}

function fmtTime(ms) {
  if (!ms) return "—";
  return new Date(ms).toLocaleString("zh-CN", { hour12: false });
}

function shortId(id) {
  return (id || "").slice(0, 8);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function memberLabel(id) {
  const m = state.members.find((x) => x.member_id === id);
  return m ? m.display_name : id;
}

function card(memory, extra = "", displayText = null) {
  const el = document.createElement("article");
  el.className = `card kind-${memory.kind}`;
  const tag = memory.tag ? `<span class="badge">${escapeHtml(memory.tag)}</span>` : "";
  const subjects = (memory.subject_ids || [])
    .map((id) => `<span class="badge soft">${escapeHtml(memberLabel(id))}</span>`)
    .join("");
  el.innerHTML = `
    <header class="card-head">
      <div class="badges">
        <span class="badge kind-chip">${KIND_LABEL[memory.kind] || memory.kind}</span>
        <span class="badge state-chip state-${memory.state}">${memory.state}</span>
        ${tag}
        ${subjects}
        ${memory.embedding_pending ? `<span class="badge">嵌入中</span>` : ""}
        ${memory.consolidation_pending ? `<span class="badge">待巩固</span>` : ""}
      </div>
      <button type="button" class="id-chip" data-copy="${memory.id}" title="点击复制完整 ID"><code>${shortId(memory.id)}</code></button>
    </header>
    <p class="card-body">${escapeHtml(displayText ?? memory.content)}</p>
    <div class="meta card-meta">重要性 ${memory.importance.toFixed(2)} · 置信 ${memory.confidence.toFixed(2)} · ${fmtTime(memory.created_at_ms)}</div>
    <footer class="card-foot">
      <button class="btn ghost sm" data-act="explain">解释</button>
      ${memory.subject_ids?.[0]
        ? `<button class="btn ghost sm" data-act="graph">图谱</button>`
        : ""}
      ${memory.state === "forgotten" || memory.state === "archived"
        ? `<button class="btn ghost sm" data-act="restore">恢复</button>`
        : `<button class="btn danger sm" data-act="forget">遗忘</button>`}
      ${extra}
    </footer>
  `;
  const idChip = el.querySelector("[data-copy]");
  if (idChip) {
    idChip.onclick = () => {
      try {
        navigator.clipboard?.writeText(memory.id).then(
          () => toast("已复制完整 ID"),
          () => toast(memory.id)
        );
      } catch {
        toast(memory.id);
      }
    };
  }
  el.querySelector('[data-act="explain"]').onclick = () => openExplain(memory.id);
  const graphBtn = el.querySelector('[data-act="graph"]');
  if (graphBtn) {
    graphBtn.onclick = () => {
      $("graph-member").value = memory.subject_ids[0];
      showTab("graph");
      loadGraph().catch((err) => toast(err.message));
    };
  }
  const forgetBtn = el.querySelector('[data-act="forget"]');
  if (forgetBtn) {
    forgetBtn.onclick = async () => {
      const reason = prompt("遗忘原因", "web ui") || "user";
      await api(`/api/memories/${memory.id}/forget`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      });
      toast("已遗忘");
      loadLibrary();
    };
  }
  const restoreBtn = el.querySelector('[data-act="restore"]');
  if (restoreBtn) {
    restoreBtn.onclick = async () => {
      await api(`/api/memories/${memory.id}/restore`, { method: "POST", body: "{}" });
      toast("已恢复");
      loadLibrary();
    };
  }
  return el;
}

function fillMemberSelects() {
  const options =
    `<option value="">自动推断 / 全部</option>` +
    state.members
      .map(
        (m) =>
          `<option value="${escapeHtml(m.member_id)}">${escapeHtml(m.display_name)} (${escapeHtml(
            m.member_id
          )})</option>`
      )
      .join("");
  for (const id of ["member-id", "recall-member", "graph-member"]) {
    const el = $(id);
    if (!el) continue;
    const prev = el.value;
    if (id === "graph-member") {
      el.innerHTML =
        `<option value="">全部（overview）</option>` +
        state.members
          .map(
            (m) =>
              `<option value="${escapeHtml(m.member_id)}">${escapeHtml(m.display_name)} · ${
                ROLE_LABEL[m.role] || m.role
              }</option>`
          )
          .join("");
    } else {
      el.innerHTML = options;
    }
    if ([...el.options].some((o) => o.value === prev)) el.value = prev;
  }
  const list = $("member-list");
  if (list) {
    list.innerHTML = "";
    if (!state.members.length) {
      list.innerHTML = `<p class="muted">尚未登记成员。启动时会自动创建主驾；也可在此补充家庭成员。</p>`;
    } else {
      state.members.forEach((m) => {
        const el = document.createElement("article");
        el.className = "card compact";
        el.innerHTML = `<div class="meta">${escapeHtml(ROLE_LABEL[m.role] || m.role)} · <code>${escapeHtml(
          m.member_id
        )}</code></div><p>${escapeHtml(m.display_name)}${
          (m.aliases || []).length ? ` · ${escapeHtml(m.aliases.join("、"))}` : ""
        }</p>`;
        list.appendChild(el);
      });
    }
  }
}

async function loadMembers() {
  state.members = await api("/api/household/members");
  fillMemberSelects();
}

async function loadStatus() {
  const s = await api("/api/status");
  state.reconcileChat = !!s.reconcile_chat;
  const dreamLabel =
    s.dream_interval_secs != null && s.dream_interval_secs > 0
      ? `dream · ${formatInterval(s.dream_interval_secs)}`
      : "dream · 手动";
  const tickLabel =
    s.tick_interval_secs != null && s.tick_interval_secs > 0
      ? `tick · ${formatInterval(s.tick_interval_secs)}`
      : "tick · 手动";
  const reconcileLabel = state.reconcileChat ? "reconcile · 就绪" : "reconcile · 需 LLM_API_KEY";
  $("status-pills").innerHTML = `
    <span class="pill">ns · ${escapeHtml(s.namespace)}</span>
    <span class="pill">embed · ${escapeHtml(s.embedder)}</span>
    <span class="pill">sum · ${escapeHtml(s.summarizer)}</span>
    <span class="pill">cls · ${escapeHtml(s.classifier || "heuristic-v1")}</span>
    <span class="pill${state.reconcileChat ? "" : " pill-warn"}">${escapeHtml(reconcileLabel)}</span>
    <span class="pill">${escapeHtml(tickLabel)}</span>
    <span class="pill">${escapeHtml(dreamLabel)}</span>
    <span class="pill">${escapeHtml(s.db_path)}</span>
  `;
  const schedule = $("dream-schedule");
  if (schedule) {
    schedule.textContent =
      s.dream_interval_secs != null && s.dream_interval_secs > 0
        ? `后台定时 dream：每 ${formatInterval(s.dream_interval_secs)}（MEMORA_DREAM_INTERVAL）`
        : "后台定时 dream：已关闭（设置 MEMORA_DREAM_INTERVAL>0 开启）";
  }
  const btn = $("btn-reconcile");
  if (btn) btn.disabled = !state.reconcileChat || state.reconcileBusy;
}

function renderDialogue() {
  const log = $("dialogue-log");
  if (!log) return;
  if (!state.dialogue.length) {
    log.innerHTML = `<p class="muted dialogue-empty">对话为空。发送用户原话后走 Android 同款抽取链路。</p>`;
    return;
  }
  log.innerHTML = state.dialogue
    .map((turn) => {
      const role = turn.role === "user" ? "用户" : "助手";
      const cls = turn.role === "user" ? "user" : "assistant";
      return `<div class="dialogue-turn ${cls}"><span class="role">${role}</span><p>${escapeHtml(
        turn.text
      )}</p></div>`;
    })
    .join("");
  log.scrollTop = log.scrollHeight;
}

function buildTranscript() {
  return state.dialogue
    .map((t) => `${t.role === "user" ? "用户" : "助手"}：${t.text}`)
    .join("\n");
}

function showReconcileReport(report, onResume = null) {
  const el = $("reconcile-report");
  if (!el) return;
  el.hidden = false;
  if (report.skipped) {
    el.innerHTML = `<p class="muted">已跳过：${escapeHtml(report.skip_reason || "unknown")}</p>`;
    return;
  }
  const facts = (report.facts || [])
    .map(
      (f) =>
        `<li><code>${escapeHtml(f.tag || "")}</code> · ${escapeHtml(f.owner || "")} · ${escapeHtml(
          f.content || ""
        )}${f.event_at ? ` · ${escapeHtml(f.event_at)}` : ""}</li>`
    )
    .join("");
  const warns = (report.warnings || [])
    .map((w) => `<li>${escapeHtml(w)}</li>`)
    .join("");
  const extraction = report.extraction;
  const staged = report.staged === true;
  const remaining = (extraction?.remaining_ranges || []).map((range) =>
    `<li><code>${escapeHtml(range.source_id)}</code> [${escapeHtml(range.start)}, ${escapeHtml(range.end)}) · ${escapeHtml(range.reason || "待继续")}</li>`
  ).join("");
  const resumeAvailable = /^[0-9a-f-]{36}$/i.test(report.resume_id || "") && Number(report.resume_expires_in_secs) > 0;
  const stagedNote = staged ? `<div class="report-alert">结果暂存，未写入记忆。所有块完成后才统一写入；继续只处理未完成块。
    已完成 ${escapeHtml(extraction?.chunks_completed ?? 0)}/${escapeHtml(extraction?.chunks_total ?? 0)} 块。
    <p>完整原文默认不落盘；服务器仅内存暂存30分钟，过期或服务重启后失效。${resumeAvailable ? `剩余约 ${Math.ceil(Number(report.resume_expires_in_secs) / 60)} 分钟。` : escapeHtml(report.resume_error || "暂存不可继续，请确认没有运行中的任务后重新提交。")}</p>
    ${resumeAvailable ? '<button type="button" class="btn primary sm" data-resume-reconcile>继续未完成块</button>' : ""}
    <p class="resume-status" data-resume-status aria-live="polite"></p></div>` : "";
  const partialNote = !staged && extraction?.partial
    ? `<div class="report-alert">部分抽取：仅完成 ${extraction.chunks_completed ?? 0}/${extraction.chunks_total ?? 0} 块。已有内容可能已写入，请勿直接重复提交全部输入。失败来源：${escapeHtml(
        (extraction.failed_source_ids || []).join("、")
      )}。无效记录：${extraction.invalid_records ?? 0}；输出截断来源：${escapeHtml(
        (extraction.truncated_source_ids || []).join("、") || "无"
      )}</div>`
    : "";
  el.innerHTML = `
    ${stagedNote}${partialNote}
    <div class="reconcile-stats">
      <span>抽取 ${report.memories_extracted ?? 0}</span>
      <span>保留 ${report.memories_kept ?? 0}</span>
      <span>登记 ${report.registers ?? 0}</span>
      <span>撤回 ${report.retractions ?? 0}</span>
      <span>${staged ? "未写入（暂存）" : `写入 ${(report.written_ids || []).length}`}</span>
      <span>修订 ${(report.revised_ids || []).length}</span>
      <span>遗忘 ${report.forgotten ?? 0}</span>
      <span>恢复块 ${escapeHtml(extraction?.recovered_chunks ?? 0)}</span>
      <span>规划调用 ${escapeHtml(extraction?.planning_calls ?? 0)}</span>
      <span>抽取调用 ${escapeHtml(extraction?.extraction_calls ?? 0)}</span>
      <span>截断尝试 ${escapeHtml(extraction?.truncated_attempts ?? 0)}</span>
      <span>有效记录 ${escapeHtml(extraction?.valid_records ?? 0)}</span>
      <span>无效尝试记录 ${escapeHtml(extraction?.invalid_attempt_records ?? 0)}</span>
      ${report.fallback_used ? "<span>fallback</span>" : ""}
    </div>
    ${remaining ? `<h3 class="subtle">未完成范围（原来源 Unicode 字符，0起始 [start, end)）</h3><ul class="fact-list remaining-ranges">${remaining}</ul>` : ""}
    ${facts ? `<h3 class="subtle">${staged ? "暂存事实（尚未写入）" : "事实"}</h3><ul class="fact-list">${facts}</ul>` : ""}
    ${warns ? `<h3 class="subtle">警告</h3><ul class="warn-list">${warns}</ul>` : ""}
    <details class="dream-raw">
      <summary>extract / ops 原文</summary>
      <pre class="log">${escapeHtml(report.extract_raw || "")}\n\n—— ops ——\n${escapeHtml(
        report.ops_raw || ""
      )}</pre>
    </details>
  `;
  const resume = el.querySelector?.("[data-resume-reconcile]");
  if (resume) {
    resume.disabled = !onResume;
    resume.onclick = onResume;
  }
}

async function reconcileReport(response, onStatus, request = api, sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))) {
  if (response.status !== 202) return response.json();
  const accepted = await response.json();
  const statusUrl = accepted.status_url;
  if (!/^\/api\/reconcile\/jobs\/[0-9a-f-]{36}$/i.test(statusUrl || "")) {
    throw new Error("异步任务返回无效查询地址");
  }
  // Only individual polls time out. A multi-chunk task has no overall 60s deadline.
  for (;;) {
    onStatus(`任务 ${accepted.job_id}：等待查询；关闭页面不会取消任务`);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    let job;
    try {
      job = await request(statusUrl, { signal: controller.signal });
    } catch (error) {
      onStatus(`任务 ${accepted.job_id}：暂时无法查询，将继续重试；请勿重复提交`);
      await sleep(3000);
      continue;
    } finally {
      clearTimeout(timeout);
    }
    if (job.status === "completed" && job.report) return job.report;
    if (job.status === "failed") throw new Error(`任务 ${accepted.job_id} 失败：${job.error || "unknown"}`);
    if (!["queued", "running"].includes(job.status)) throw new Error(`任务 ${accepted.job_id} 状态无效`);
    onStatus(`任务 ${accepted.job_id}：${job.status === "queued" ? "排队中" : "抽取中"}；关闭页面不会取消任务`);
    await sleep(1500);
  }
}

async function onReconcile(ev) {
  ev.preventDefault();
  if (state.reconcileBusy) return;
  if (!state.reconcileChat) {
    toast("未配置 LLM_API_KEY，无法走 Android 抽取链路");
    return;
  }
  if (state.pendingReconcile?.token && Date.now() < state.pendingReconcile.expiresAt) {
    toast("已有未写入的暂存结果，请点击继续未完成块；请勿重复提交全文");
    return;
  }
  const query = $("query").value.trim();
  if (!query) {
    toast("请输入用户原话");
    return;
  }
  const includeAnswer = $("include-answer")?.checked;
  const answer = includeAnswer ? ($("answer").value.trim() || "") : "";
  const transcriptBefore = buildTranscript();
  const btn = $("btn-reconcile");
  const snapshot = { query, answer, token: null, completed: false };
  state.pendingReconcile = null;
  state.reconcileBusy = true;
  btn.disabled = true;
  btn.textContent = "抽取中…";
  try {
    // Do not send speaker_id — Memora infers ownership from content.
    const body = {
      query,
      answer,
      transcript: transcriptBefore,
      session_id: state.sessionId,
      message_at_ms: Date.now(),
      async: true,
    };
    const response = await api("/api/reconcile", {
      method: "POST",
      body: JSON.stringify(body),
      raw: true,
    });
    const report = await reconcileReport(response, (message) => {
      btn.textContent = "后台任务中…";
      const el = $("reconcile-report");
      if (el) { el.hidden = false; el.textContent = message; }
    });
    await finishReconcileReport(report, snapshot);
  } catch (error) {
    const el = $("reconcile-report");
    if (el) {
      el.hidden = false;
      el.textContent = `请求失败或连接中断：${error.message}。后台任务可能仍在运行，不能据此认定未写入；请勿自动重复提交全文。`;
    }
    toast(error.message);
  } finally {
    state.reconcileBusy = false;
    btn.disabled = !state.reconcileChat;
    btn.textContent = "抽取并写入";
  }
}

async function finishReconcileReport(report, snapshot) {
  snapshot.report = report;
  if (report.staged === true) {
    snapshot.token = /^[0-9a-f-]{36}$/i.test(report.resume_id || "") ? report.resume_id : null;
    snapshot.expiresAt = Date.now() + Math.max(0, Number(report.resume_expires_in_secs) || 0) * 1000;
    state.pendingReconcile = snapshot;
    showReconcileReport(report, () => onResumeReconcile());
    toast(snapshot.token && Date.now() < snapshot.expiresAt
      ? "结果暂存，未写入记忆；可继续未完成块"
      : "结果未写入记忆，但暂存已失效或不可继续");
    return;
  }
  state.pendingReconcile = null;
  if (!snapshot.completed) {
    snapshot.completed = true;
    state.dialogue.push({ role: "user", text: snapshot.query });
    if (snapshot.answer) state.dialogue.push({ role: "assistant", text: snapshot.answer });
    renderDialogue();
  }
  showReconcileReport(report);
  // Preserve anything the user typed while the background job was running.
  if ($("query").value.trim() === snapshot.query) $("query").value = "";
  if ($("answer")?.value.trim() === snapshot.answer) $("answer").value = "";
  const wrote = (report.written_ids || []).length + (report.revised_ids || []).length + (report.forgotten || 0);
  toast(report.skipped ? `已跳过：${report.skip_reason || ""}`
    : `${report.extraction?.partial ? "部分完成（请核查失败来源）" : "完成"} · 保留 ${report.memories_kept ?? 0} · 变更 ${wrote}`);
  state.offset = 0;
  // Updating data must not navigate away from the report or force a graph filter.
  await loadMembers().catch(() => toast("抽取已完成，但成员列表刷新失败"));
  await loadLibrary().catch(() => toast("抽取已完成，但记忆库刷新失败"));
  await loadGraph().catch(() => {});
}

async function onResumeReconcile() {
  if (state.reconcileBusy) return;
  const snapshot = state.pendingReconcile;
  if (!snapshot?.token) return;
  if (Date.now() >= snapshot.expiresAt) {
    snapshot.token = null;
    showReconcileReport({ ...snapshot.report, resume_id: null, resume_expires_in_secs: 0,
      resume_error: "暂存已过期，未写入记忆；请重新提交原文。" });
    return;
  }
  const token = snapshot.token;
  // Claim once locally too. A lost HTTP response must never cause an automatic
  // POST retry; the server may already have consumed this generation's token.
  snapshot.token = null;
  state.reconcileBusy = true;
  const submit = $("btn-reconcile");
  submit.disabled = true;
  const panel = $("reconcile-report");
  const button = panel.querySelector("[data-resume-reconcile]");
  if (button) { button.disabled = true; button.textContent = "继续中…"; }
  const onStatus = (message) => {
    const status = panel.querySelector("[data-resume-status]");
    if (status) status.textContent = message;
  };
  onStatus("正在领取暂存并继续未完成块…");
  try {
    const response = await api(`/api/reconcile/resume/${token}`, { method: "POST", raw: true });
    const report = await reconcileReport(response, onStatus);
    await finishReconcileReport(report, snapshot);
  } catch (error) {
    if ([404, 409, 410].includes(error.status)) {
      onStatus(`此报告是未写入的暂存快照。${error.message} 若已由另一请求继续，请查看该任务结果。`);
    } else {
      onStatus(`继续请求失败或连接中断：${error.message}。此前结果为未写入的暂存；本次后台任务可能仍在运行或已经写入，请核查任务结果，勿重发全文。`);
    }
    toast(error.message);
  } finally {
    state.reconcileBusy = false;
    submit.disabled = !state.reconcileChat;
  }
}

function formatInterval(secs) {
  const n = Number(secs) || 0;
  if (n >= 3600 && n % 3600 === 0) return `${n / 3600}h`;
  if (n >= 60 && n % 60 === 0) return `${n / 60}m`;
  return `${n}s`;
}

async function loadLibrary() {
  const stateFilter = $("filter-state").value;
  const kindFilter = $("filter-kind").value;
  const q = new URLSearchParams({
    limit: String(state.limit),
    offset: String(state.offset),
    state: stateFilter,
    kind: kindFilter,
  });
  const page = await api(`/api/memories?${q}`);
  const root = $("library");
  root.innerHTML = "";
  if (!page.items.length) {
    root.innerHTML = `<p class="muted">还没有记忆。在上方写一条开始。</p>`;
  } else {
    page.items.forEach((m) => root.appendChild(card(m)));
  }
  $("lib-count").textContent = `${page.total} 条 · 第 ${Math.floor(state.offset / state.limit) + 1} 页`;
  $("prev-page").disabled = state.offset <= 0;
  $("next-page").disabled = state.offset + state.limit >= page.total;
}

const PAGE_META = {
  chat: { title: "对话测试", sub: "与 Android 同路径 · extract → ops → 写入" },
  library: { title: "记忆库", sub: "浏览、筛选与维护已保存的记忆" },
  recall: { title: "召回测试", sub: "检索质量与 Context Pack 预览" },
  graph: { title: "知识图谱", sub: "家庭实体与关系 · 点击节点或边查看详情" },
  explain: { title: "记忆解释", sub: "版本链与审计追踪" },
  dream: { title: "Dream", sub: "巩固、去重与聚类摘要" },
  ops: { title: "维护", sub: "名册、索引与数据导入导出" },
};

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    if (on) t.setAttribute("aria-current", "page");
    else t.removeAttribute("aria-current");
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  const meta = PAGE_META[name];
  if (meta) {
    const title = $("page-title");
    const sub = $("page-sub");
    if (title) title.textContent = meta.title;
    if (sub) sub.textContent = meta.sub;
  }
}

async function openExplain(id) {
  state.selectedId = id;
  showTab("explain");
  $("explain-empty").hidden = true;
  const view = $("explain-view");
  view.hidden = false;
  view.innerHTML = `<p class="muted">加载 ${shortId(id)}…</p>`;
  const data = await api(`/api/memories/${id}/explain`);
  const scores = data.last_scores
    ? `<div class="meta">RRF ${data.last_scores.rrf.toFixed(4)} · decay ${data.last_scores.decay.toFixed(3)} · final ${data.last_scores.final_score.toFixed(4)}</div>`
    : "";
  view.innerHTML = `
    <article class="card kind-${data.memory.kind}">
      <header>
        <div class="badges">
          <span class="badge">${KIND_LABEL[data.memory.kind] || data.memory.kind}</span>
          <span class="badge">${data.memory.state}</span>
          ${data.memory.consolidation_pending ? `<span class="badge">待巩固</span>` : ""}
        </div>
        <code>${data.memory.id}</code>
      </header>
      <p>${escapeHtml(data.memory.content)}</p>
      ${scores}
    </article>
    <h3 class="subtle">版本链</h3>
    <div class="versions">
      ${data.versions
        .map(
          (v) =>
            `<div class="card compact"><div class="meta">${v.operation} · ${v.actor} · ${fmtTime(v.created_at_ms)}</div><div>${escapeHtml(v.reason || "")}</div></div>`
        )
        .join("") || `<p class="muted">无版本</p>`}
    </div>
    <h3 class="subtle">审计</h3>
    <div class="versions">
      ${data.recent_audit
        .map(
          (a) =>
            `<div class="card compact"><div class="meta">#${a.seq} ${a.event_type} · ${fmtTime(a.created_at_ms)}</div><code>${escapeHtml(a.hash.slice(0, 16))}…</code></div>`
        )
        .join("") || `<p class="muted">无审计</p>`}
    </div>
  `;
}

async function onRemember(ev) {
  ev.preventDefault();
  const content = $("content").value.trim();
  if (!content) {
    toast("请填写要直接记住的内容");
    return;
  }
  const entities = $("entities")
    .value.split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);
  const body = {
    content,
    kind: $("kind").value,
    session_id: $("session").value.trim() || null,
    entities,
    importance: Number($("importance").value),
    confidence: Number($("confidence").value),
    pin_to_stm: $("pin-stm").checked,
  };
  const mid = $("member-id").value.trim();
  if (mid) {
    body.member_id = mid;
    body.subjects = [mid];
  }
  const eventAt = $("event-at").value.trim();
  if (eventAt) body.event_at = eventAt;
  const res = await api("/api/memories", { method: "POST", body: JSON.stringify(body) });
  toast(`已记住 ${shortId(res.id)}`);
  $("content").value = "";
  state.offset = 0;
  await loadLibrary();
  await loadGraph({ showTab: true }).catch(() => {});
}

async function onRecall(ev) {
  ev.preventDefault();
  const topK = Number($("recall-k").value) || 10;
  const minScore = Number($("recall-score").value);
  const threshold = Number.isFinite(minScore) ? minScore : 0;
  const memberId = $("recall-member").value.trim() || null;
  const query = {
    text: $("recall-text").value.trim(),
    top_k: topK,
    kinds: [],
    member_id: memberId,
  };
  const result = await api("/api/recall", { method: "POST", body: JSON.stringify(query) });
  const filtered = result.hits.filter((h) => h.scores.final_score >= threshold);
  const graphHits = result.graph_hits || [];
  showTab("recall");
  $("recall-meta").textContent =
    `query「${query.text}」 · TopK ${topK} · score≥${threshold.toFixed(2)} · ` +
    `命中 ${filtered.length}/${result.hits.length} · 图谱 ${graphHits.length} · STM ${result.stm_hits.length} · ` +
    `tokens ${result.context.estimated_tokens}/${result.context.budget}`;
  const hits = $("recall-hits");
  hits.innerHTML = "";
  filtered.forEach((h) => {
    const extra = `<span class="meta">score ${h.scores.final_score.toFixed(3)} · rrf ${h.scores.rrf.toFixed(3)}${h.from_stm ? " · STM" : ""}</span>`;
    hits.appendChild(card(h.memory, extra, h.display_text || null));
  });
  if (!filtered.length) {
    hits.innerHTML = `<p class="muted">没有达到阈值的召回结果</p>`;
  }
  const gRoot = $("recall-graph-hits");
  gRoot.innerHTML = "";
  if (!graphHits.length) {
    gRoot.innerHTML = `<p class="muted">无图谱命中</p>`;
  } else {
    graphHits.forEach((g) => {
      const el = document.createElement("article");
      el.className = "card graph-fact";
      const rel = g.fact?.relation || "";
      const domains = (g.fact?.domains || []).join(", ");
      el.innerHTML = `
        <header>
          <div class="badges">
            <span class="badge">graph</span>
            <span class="badge">${escapeHtml(rel)}</span>
            ${domains ? `<span class="badge soft">${escapeHtml(domains)}</span>` : ""}
          </div>
          <span class="meta">score ${(g.score ?? 0).toFixed(2)}</span>
        </header>
        <p>${escapeHtml(g.display_text || `${g.subject_name} → ${g.object_label}`)}</p>
      `;
      gRoot.appendChild(el);
    });
  }
  const pack = $("context-pack");
  pack.innerHTML = "";
  const keepIds = new Set(filtered.map((h) => h.memory.id));
  const packItems = result.context.items.filter(
    (item) => item.role === "stm_fact" || item.role === "graph_fact" || keepIds.has(item.memory_id)
  );
  packItems.forEach((item) => {
    const el = document.createElement("article");
    el.className = "card compact";
    el.innerHTML = `<div class="meta">${item.role} · ${item.tokens} tok</div><p>${escapeHtml(item.text)}</p>`;
    pack.appendChild(el);
  });
  if (!packItems.length) {
    pack.innerHTML = `<p class="muted">无 Context Pack</p>`;
  }
}

function factObjectLabel(object) {
  if (!object) return "—";
  if (object.entity_id) return object.entity_id;
  if (object.value != null) {
    return typeof object.value === "string" ? object.value : JSON.stringify(object.value);
  }
  return "—";
}

function stopGraphSim() {
  if (state.graphSim) {
    cancelAnimationFrame(state.graphSim);
    state.graphSim = null;
  }
}

function buildGraphModel(entities, facts, hubId) {
  const nodes = new Map();
  const ensure = (id, label, kind) => {
    if (!nodes.has(id)) {
      nodes.set(id, { id, label: label || id, kind: kind || "other" });
    } else if (label && nodes.get(id).label === id) {
      nodes.get(id).label = label;
    }
    return nodes.get(id);
  };
  (entities || []).forEach((e) => {
    const kind = e.entity_type === "person" || e.id === hubId ? "person" : "other";
    ensure(e.id, e.canonical_name, kind);
  });
  const links = [];
  (facts || []).forEach((f, idx) => {
    const subj = ensure(f.subject_id, f.subject_id, "other");
    let objId;
    let objLabel;
    let objKind = "other";
    if (f.object?.entity_id) {
      objId = f.object.entity_id;
      objLabel = f.object.entity_id;
    } else {
      objLabel = factObjectLabel(f.object);
      objId = `val:${f.id || idx}:${objLabel}`;
      objKind = "value";
    }
    const obj = ensure(objId, objLabel, objKind);
    links.push({
      id: f.id || `edge-${idx}`,
      source: subj.id,
      target: obj.id,
      relation: f.relation || "",
      fact: f,
    });
  });
  return { nodes: [...nodes.values()], links };
}

function renderForceGraph(entities, facts, hubId) {
  stopGraphSim();
  const svg = $("graph-svg");
  if (!svg) return;
  const wrap = svg.parentElement;
  const width = Math.max(wrap?.clientWidth || 640, 320);
  const height = 480;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const model = buildGraphModel(entities, facts, hubId);
  if (!model.nodes.length) {
    state.graphScene = null;
    state.graphView = { scale: 1, tx: 0, ty: 0 };
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#b7aa98" font-size="13">暂无图谱节点 · 用对话或记住写入后会出现</text>`;
    return;
  }

  const ns = "http://www.w3.org/2000/svg";
  const defs = document.createElementNS(ns, "defs");
  const marker = document.createElementNS(ns, "marker");
  marker.setAttribute("id", "graph-direction-arrow");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "10");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "9");
  marker.setAttribute("markerHeight", "9");
  marker.setAttribute("markerUnits", "userSpaceOnUse");
  marker.setAttribute("orient", "auto");
  const arrow = document.createElementNS(ns, "path");
  arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 Z");
  arrow.setAttribute("fill", "#d4a574");
  marker.appendChild(arrow);
  defs.appendChild(marker);
  svg.appendChild(defs);
  // Everything drawable lives in one viewport group so zoom/pan is a single
  // transform. The view resets whenever the graph re-renders.
  const viewport = document.createElementNS(ns, "g");
  svg.appendChild(viewport);
  const gLinks = document.createElementNS(ns, "g");
  const gLabels = document.createElementNS(ns, "g");
  const gNodes = document.createElementNS(ns, "g");
  viewport.appendChild(gLinks);
  viewport.appendChild(gLabels);
  viewport.appendChild(gNodes);
  state.graphView = { scale: 1, tx: 0, ty: 0 };
  const applyView = () => {
    const v = state.graphView;
    viewport.setAttribute("transform", `translate(${v.tx} ${v.ty}) scale(${v.scale})`);
  };
  applyView();

  const nodeById = new Map();
  model.nodes.forEach((n, i) => {
    const angle = (i / model.nodes.length) * Math.PI * 2;
    n.x = width / 2 + Math.cos(angle) * Math.min(width, height) * 0.28;
    n.y = height / 2 + Math.sin(angle) * Math.min(width, height) * 0.28;
    n.vx = 0;
    n.vy = 0;
    if (hubId && n.id === hubId) {
      n.x = width / 2;
      n.y = height / 2;
    }
    nodeById.set(n.id, n);
  });

  // Allocate lanes for the unordered pair, not just the source/target order:
  // reciprocal family facts and multiple predicates must all remain readable.
  const pairLinks = new Map();
  model.links.forEach((link) => {
    const pair = [link.source, link.target].sort();
    const key = JSON.stringify(pair);
    if (!pairLinks.has(key)) pairLinks.set(key, []);
    pairLinks.get(key).push(link);
  });
  pairLinks.forEach((links) => links.forEach((link, index) => {
    link.curveOffset = (index - (links.length - 1) / 2) * 64;
    link.loopIndex = index;
  }));
  const nodeRadius = (node) => node.kind === "person" ? 16 : node.kind === "value" ? 10 : 12;
  const linkEls = model.links.map((l) => {
    const line = document.createElementNS(ns, "path");
    line.classList.add("graph-link");
    line.setAttribute("marker-end", "url(#graph-direction-arrow)");
    line.dataset.edge = l.id;
    if (l.fact?.source_memory_id) {
      line.style.cursor = "pointer";
      line.onclick = () => openExplain(l.fact.source_memory_id);
    }
    gLinks.appendChild(line);
    const label = document.createElementNS(ns, "text");
    label.classList.add("graph-link-label");
    label.textContent = l.relation;
    gLabels.appendChild(label);
    return { line, label, link: l };
  });

  const nodeEls = [];
  function highlightGraphNode(nodeId) {
    // Selecting a node scopes the right-hand fact list to that entity.
    state.graphFactFilter = nodeId;
    if (typeof renderFactList === "function" && state.lastGraph) {
      renderFactList(state.lastGraph.entities, state.lastGraph.facts, state.lastGraph.person);
    }
    nodeEls.forEach(({ g, node }) => g.classList.toggle("hot", node.id === nodeId));
    linkEls.forEach(({ line, link }) => {
      const hot = link.source === nodeId || link.target === nodeId;
      line.classList.toggle("hot", hot);
    });
    document.querySelectorAll("#graph-facts .graph-fact").forEach((el) => {
      const sid = el.dataset.subject;
      const oid = el.dataset.object;
      el.classList.toggle("hot", sid === nodeId || oid === nodeId);
    });
  }

  model.nodes.forEach((n) => {
    const g = document.createElementNS(ns, "g");
    g.classList.add(
      "graph-node",
      n.kind === "person" ? "person" : n.kind === "value" ? "value" : "other"
    );
    g.dataset.node = n.id;
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("r", nodeRadius(n));
    const t = document.createElementNS(ns, "text");
    t.setAttribute("dy", "28");
    t.textContent = (n.label || "").slice(0, 12);
    g.appendChild(c);
    g.appendChild(t);
    g.onclick = () => highlightGraphNode(n.id);
    gNodes.appendChild(g);
    nodeEls.push({ g, c, t, node: n });
  });

  let ticks = 0;
  let tickTarget = 180;
  const applyGeometry = () => {
    linkEls.forEach(({ line, label, link }) => {
      const a = nodeById.get(link.source);
      const b = nodeById.get(link.target);
      if (!a || !b) return;
      if (link.source === link.target) {
        // A self relation needs a visible loop, rather than a zero-length line.
        const rim = (nodeRadius(a) + 3) / Math.SQRT2;
        const loop = 44 + link.loopIndex * 48;
        const side = a.y >= height / 2 ? -1 : 1;
        const sy = a.y + side * rim;
        const cy = a.y + side * loop;
        line.setAttribute("d", `M ${a.x + rim} ${sy} C ${a.x + loop} ${cy} ${a.x - loop} ${cy} ${a.x - rim} ${sy}`);
        label.setAttribute("x", a.x);
        label.setAttribute("y", sy * 0.25 + cy * 0.75 + side * 6);
        return;
      }
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.hypot(dx, dy);
      const ux = distance > 0 ? dx / distance : 1;
      const uy = distance > 0 ? dy / distance : 0;
      // Use a canonical normal so reverse edges occupy different physical lanes.
      const direction = link.source < link.target ? 1 : -1;
      const cx = (a.x + b.x) / 2 - uy * direction * link.curveOffset;
      const cy = (a.y + b.y) / 2 + ux * direction * link.curveOffset;
      const startLength = Math.hypot(cx - a.x, cy - a.y) || 1;
      const endLength = Math.hypot(cx - b.x, cy - b.y) || 1;
      const startRadius = nodeRadius(a) + 3;
      const endRadius = nodeRadius(b) + 3;
      const sx = a.x + (cx - a.x) / startLength * startRadius;
      const sy = a.y + (cy - a.y) / startLength * startRadius;
      const ex = b.x + (cx - b.x) / endLength * endRadius;
      const ey = b.y + (cy - b.y) / endLength * endRadius;
      line.setAttribute("d", `M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`);
      label.setAttribute("x", sx * 0.25 + cx * 0.5 + ex * 0.25);
      label.setAttribute("y", sy * 0.25 + cy * 0.5 + ey * 0.25 - 4);
    });
    nodeEls.forEach(({ g, node }) => {
      g.setAttribute("transform", `translate(${node.x},${node.y})`);
    });
  };
  const step = () => {
    ticks += 1;
    const nodes = model.nodes;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dist = Math.hypot(dx, dy) || 1;
        const force = 900 / (dist * dist);
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        a.vx += dx;
        a.vy += dy;
        b.vx -= dx;
        b.vy -= dy;
      }
    }
    model.links.forEach((l) => {
      const a = nodeById.get(l.source);
      const b = nodeById.get(l.target);
      if (!a || !b) return;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ideal = 110;
      const force = (dist - ideal) * 0.04;
      dx = (dx / dist) * force;
      dy = (dy / dist) * force;
      a.vx += dx;
      a.vy += dy;
      b.vx -= dx;
      b.vy -= dy;
    });
    nodes.forEach((n) => {
      n.vx += (width / 2 - n.x) * 0.004;
      n.vy += (height / 2 - n.y) * 0.004;
      n.vx *= 0.85;
      n.vy *= 0.85;
      // A node being dragged (or placed by the user) holds its position while
      // the remaining nodes settle around it.
      if (n.pinned) {
        n.vx = 0;
        n.vy = 0;
        return;
      }
      n.x = Math.min(width - 24, Math.max(24, n.x + n.vx));
      n.y = Math.min(height - 36, Math.max(24, n.y + n.vy));
    });
    applyGeometry();
    if (ticks < tickTarget) {
      state.graphSim = requestAnimationFrame(step);
    } else {
      state.graphSim = null;
    }
  };
  // Node drags reheat a shortened settle instead of the full 180-tick layout.
  state.graphKick = () => {
    ticks = 0;
    tickTarget = 90;
    stopGraphSim();
    state.graphSim = requestAnimationFrame(step);
  };
  state.graphScene = { model, viewport, width, height, apply: applyGeometry, applyView };
  state.graphSim = requestAnimationFrame(step);
}

// A node-focused list shows only facts where the node is subject or object.
function graphFactMatchesNode(fact, nodeId) {
  if (!nodeId) return true;
  return fact.subject_id === nodeId || fact.object?.entity_id === nodeId;
}

function renderFactList(entities, facts, person) {
  const factRoot = $("graph-facts");
  const entityName = Object.fromEntries((entities || []).map((e) => [e.id, e.canonical_name]));
  if (person) entityName[person.id] = person.canonical_name;
  const filterId = state.graphFactFilter || null;
  const visible = facts.filter((f) => graphFactMatchesNode(f, filterId));
  factRoot.innerHTML = filterId
    ? `<div class="fact-filter">
        <h3 class="subtle">关系事实 · ${escapeHtml(entityName[filterId] || filterId)}</h3>
        <button type="button" class="btn ghost sm" id="graph-fact-clear">显示全部（${facts.length}）</button>
      </div>`
    : `<h3 class="subtle">关系事实</h3>`;
  const clearBtn = factRoot.querySelector("#graph-fact-clear");
  if (clearBtn) {
    clearBtn.onclick = () => {
      state.graphFactFilter = null;
      document.querySelectorAll("#graph-svg .hot, #graph-facts .hot").forEach((el) => {
        el.classList.remove("hot");
      });
      if (state.lastGraph) {
        renderFactList(state.lastGraph.entities, state.lastGraph.facts, state.lastGraph.person);
      }
    };
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = filterId
      ? "该节点暂无关系事实。"
      : "暂无图谱事实。用「对话测试」或「记住」写入后会出现。";
    factRoot.appendChild(empty);
    return;
  }

  visible.forEach((f) => {
    const el = document.createElement("article");
    el.className = "card graph-fact";
    const subj = entityName[f.subject_id] || f.subject_id;
    const objId = f.object?.entity_id || null;
    const obj = objId != null ? entityName[objId] || objId : factObjectLabel(f.object);
    el.dataset.subject = f.subject_id;
    el.dataset.object = objId || "";
    const polarity = f.polarity === "negative" ? "不" : "";
    const status =
      f.status === "paused"
        ? "（已暂停）"
        : f.status === "ended"
          ? "（已结束）"
          : f.status === "planned"
            ? "（计划中）"
            : "";
    const cond = f.condition_text ? `（条件：${f.condition_text}）` : "";
    el.innerHTML = `
      <header>
        <div class="badges">
          <span class="badge">${escapeHtml(f.relation)}</span>
          <span class="badge soft">${escapeHtml(f.status || "active")}</span>
          ${(f.domains || []).map((d) => `<span class="badge soft">${escapeHtml(d)}</span>`).join("")}
        </div>
        <span class="meta">置信 ${(f.confidence ?? 0).toFixed(2)}</span>
      </header>
      <p>${escapeHtml(`${subj} ${polarity}${f.relation} ${obj}${status}${cond}`)}</p>
      ${f.evidence_text ? `<div class="meta">证据：${escapeHtml(f.evidence_text)}</div>` : ""}
      ${
        f.source_memory_id
          ? `<footer><button class="btn ghost" data-mid="${f.source_memory_id}">查看来源记忆</button></footer>`
          : ""
      }
    `;
    const btn = el.querySelector("button[data-mid]");
    if (btn) btn.onclick = () => openExplain(btn.dataset.mid);
    factRoot.appendChild(el);
  });
}

// Zoom / pan / node-drag listeners live outside renderForceGraph: the renderer
// must stay testable against a minimal DOM, and these bind exactly once.
function bindGraphInteractions() {
  const svg = $("graph-svg");
  if (!svg) return;
  const MIN_SCALE = 0.4;
  const MAX_SCALE = 4;
  let drag = null;

  const view = () => state.graphView || (state.graphView = { scale: 1, tx: 0, ty: 0 });
  const applyView = (scene) => {
    const v = view();
    scene.viewport.setAttribute("transform", `translate(${v.tx} ${v.ty}) scale(${v.scale})`);
  };
  const toViewBox = (event, rect, scene) => [
    ((event.clientX - rect.left) / Math.max(rect.width, 1)) * scene.width,
    ((event.clientY - rect.top) / Math.max(rect.height, 1)) * scene.height,
  ];
  const endDrag = () => {
    drag = null;
    svg.classList.remove("panning");
  };

  svg.addEventListener(
    "wheel",
    (e) => {
      const scene = state.graphScene;
      if (!scene) return;
      e.preventDefault();
      const v = view();
      const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      if (next === v.scale) return;
      const [px, py] = toViewBox(e, svg.getBoundingClientRect(), scene);
      // Keep the point under the cursor stationary while the scale changes.
      v.tx = px - ((px - v.tx) / v.scale) * next;
      v.ty = py - ((py - v.ty) / v.scale) * next;
      v.scale = next;
      applyView(scene);
    },
    { passive: false }
  );

  svg.addEventListener("pointerdown", (e) => {
    const scene = state.graphScene;
    if (!scene || e.button !== 0) return;
    const rect = svg.getBoundingClientRect();
    const nodeEl = e.target?.closest?.(".graph-node");
    if (nodeEl) {
      const node = scene.model.nodes.find((n) => n.id === nodeEl.dataset.node);
      if (!node) return;
      const v = view();
      const [px, py] = toViewBox(e, rect, scene);
      drag = {
        kind: "node",
        node,
        dx: (px - v.tx) / v.scale - node.x,
        dy: (py - v.ty) / v.scale - node.y,
        rect,
      };
      node.pinned = true;
      stopGraphSim();
    } else {
      const [px, py] = toViewBox(e, rect, scene);
      drag = { kind: "pan", px, py, tx: view().tx, ty: view().ty, rect, moved: false };
    }
    svg.classList.add("panning");
    e.preventDefault();
  });

  window.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const scene = state.graphScene;
    if (!scene) {
      endDrag();
      return;
    }
    const [px, py] = toViewBox(e, drag.rect, scene);
    if (drag.kind === "pan") {
      if (Math.abs(px - drag.px) + Math.abs(py - drag.py) > 4) drag.moved = true;
      const v = view();
      v.tx = drag.tx + (px - drag.px);
      v.ty = drag.ty + (py - drag.py);
      applyView(scene);
      return;
    }
    const v = view();
    drag.node.x = Math.min(scene.width - 24, Math.max(24, (px - v.tx) / v.scale - drag.dx));
    drag.node.y = Math.min(scene.height - 36, Math.max(24, (py - v.ty) / v.scale - drag.dy));
    scene.apply();
  });

  window.addEventListener("pointerup", () => {
    if (!drag) return;
    const { kind, moved } = drag;
    if (kind === "node" && state.graphKick) state.graphKick();
    endDrag();
    if (kind === "pan" && moved) {
      // A pan that started on a link must not count as a click on release.
      const swallow = (e) => {
        e.stopPropagation();
        e.preventDefault();
        svg.removeEventListener("click", swallow, true);
      };
      svg.addEventListener("click", swallow, true);
      setTimeout(() => svg.removeEventListener("click", swallow, true), 0);
    }
  });
  window.addEventListener("pointercancel", endDrag);

  svg.addEventListener("dblclick", () => {
    const scene = state.graphScene;
    if (!scene) return;
    state.graphView = { scale: 1, tx: 0, ty: 0 };
    applyView(scene);
  });
}

async function loadGraph(opts = {}) {
  const memberId = $("graph-member")?.value?.trim() || "";
  const includeEnded = $("graph-include-ended")?.checked || false;
  let entities = [];
  let facts = [];
  let person = null;
  let truncated = false;

  if (!memberId) {
    const q = new URLSearchParams({ include_ended: String(includeEnded) });
    const graph = await api(`/api/graph/overview?${q}`);
    entities = graph.entities || [];
    facts = graph.facts || [];
    truncated = !!graph.truncated;
    $("graph-meta").textContent =
      `全部 · 实体 ${entities.length} · 事实 ${facts.length}${truncated ? " · 已截断" : ""}`;
  } else {
    const graph = await api("/api/graph/person", {
      method: "POST",
      body: JSON.stringify({ member_id: memberId, include_ended: includeEnded }),
    });
    person = graph.person || null;
    entities = graph.entities || [];
    facts = graph.facts || [];
    if (person && !entities.some((e) => e.id === person.id)) {
      entities = [person, ...entities];
    }
    $("graph-meta").textContent =
      `${person?.canonical_name || memberId} · 实体 ${entities.length} · 事实 ${facts.length}`;
  }

  state.lastGraph = { entities, facts, person, memberId };
  state.graphFactFilter = null;
  const hubId = person?.id || (memberId ? `person:${memberId}` : null);
  renderForceGraph(entities, facts, hubId);
  renderFactList(entities, facts, person);
  if (opts.showTab) showTab("graph");
}

async function loadPersonGraph(opts) {
  return loadGraph(opts);
}

async function onGraphIngest(ev) {
  ev.preventDefault();
  const text = $("graph-ingest-text").value.trim();
  if (!text) return;
  const memberId = $("graph-member").value.trim() || null;
  const report = await api("/api/graph/ingest", {
    method: "POST",
    body: JSON.stringify({ text, member_id: memberId }),
  });
  toast(
    `图谱写入 · 实体 ${report.entities_upserted} · 事实 ${report.facts_written} · 确认 ${report.facts_confirmed}`
  );
  $("graph-ingest-text").value = "";
  await loadGraph();
}

async function onUpsertMember(ev) {
  ev.preventDefault();
  const aliases = $("new-member-aliases")
    .value.split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);
  await api("/api/household/members", {
    method: "POST",
    body: JSON.stringify({
      member_id: $("new-member-id").value.trim(),
      display_name: $("new-member-name").value.trim(),
      role: $("new-member-role").value,
      aliases,
    }),
  });
  toast("成员已登记");
  $("member-form").reset();
  $("new-member-role").value = "child";
  await loadMembers();
}

async function onDream() {
  const buttons = ["btn-dream", "btn-dream-top", "btn-dream-ops"]
    .map((id) => $(id))
    .filter(Boolean);
  buttons.forEach((b) => {
    b.disabled = true;
  });
  showTab("dream");
  $("dream-empty").textContent = "Dream 运行中…";
  $("dream-empty").hidden = false;
  $("dream-result").hidden = true;
  try {
    const report = await api("/api/dream", {
      method: "POST",
        body: JSON.stringify({
          lookback_ms: 7 * 24 * 3600 * 1000,
          candidate_limit: 50,
          archive_sources: true,
        }),
    });
    await renderDreamResult(report);
    toast(
      `dream 完成 · 巩固 ${report.consolidated} · 合并 ${report.merged ?? 0} · 清除 ${report.purged ?? 0} · 聚类 ${report.clusters}`
    );
    await loadLibrary();
  } finally {
    buttons.forEach((b) => {
      b.disabled = false;
    });
  }
}

async function renderDreamResult(report) {
  $("dream-empty").hidden = true;
  $("dream-result").hidden = false;
  $("dream-stats").innerHTML = `
    <div class="stat"><b>${report.consolidated ?? 0}</b><span>巩固</span></div>
    <div class="stat"><b>${report.merged ?? 0}</b><span>合并去重</span></div>
    <div class="stat"><b>${report.purged ?? 0}</b><span>物理清除</span></div>
    <div class="stat"><b>${report.dedup_groups ?? 0}</b><span>去重组</span></div>
    <div class="stat"><b>${report.archived_low_value ?? 0}</b><span>低价值归档</span></div>
    <div class="stat"><b>${report.clusters ?? 0}</b><span>聚类</span></div>
    <div class="stat"><b>${report.patterns ?? 0}</b><span>模式</span></div>
    <div class="stat"><b>${(report.output_ids || []).length}</b><span>产出</span></div>
    <div class="stat wide"><b>${escapeHtml(report.classifier_id || "-")}</b><span>分类器</span></div>
    <div class="stat wide"><b>${escapeHtml(report.summarizer_id || "-")}</b><span>摘要器</span></div>
  `;
  $("dream-log").textContent = JSON.stringify(report, null, 2);

  const outputs = $("dream-outputs");
  outputs.innerHTML = "";
  const ids = report.output_ids || [];
  if (!ids.length) {
    outputs.innerHTML = `<p class="muted">本次无新产出记忆（可能仅做了巩固分类）。</p>`;
    return;
  }
  const memories = await Promise.all(
    ids.slice(0, 30).map(async (id) => {
      try {
        return await api(`/api/memories/${id}`);
      } catch {
        return null;
      }
    })
  );
  memories.filter(Boolean).forEach((m) => outputs.appendChild(card(m)));
  if (ids.length > 30) {
    const more = document.createElement("p");
    more.className = "muted";
    more.textContent = `另有 ${ids.length - 30} 条产出未展开，见原始 JSON。`;
    outputs.appendChild(more);
  }
}

function bind() {
  $("importance").oninput = () => {
    $("importance-val").textContent = Number($("importance").value).toFixed(2);
  };
  $("confidence").oninput = () => {
    $("confidence-val").textContent = Number($("confidence").value).toFixed(2);
  };
  $("reconcile-form").onsubmit = (e) => onReconcile(e).catch((err) => toast(err.message));
  $("remember-form").onsubmit = (e) => onRemember(e).catch((err) => toast(err.message));
  $("recall-form").onsubmit = (e) => onRecall(e).catch((err) => toast(err.message));
  $("graph-ingest-form").onsubmit = (e) => onGraphIngest(e).catch((err) => toast(err.message));
  $("member-form").onsubmit = (e) => onUpsertMember(e).catch((err) => toast(err.message));
  $("btn-load-graph").onclick = () => loadGraph().catch((err) => toast(err.message));
  $("graph-include-ended").onchange = () => loadGraph().catch((err) => toast(err.message));
  $("graph-member").onchange = () => loadGraph().catch((err) => toast(err.message));
  $("include-answer").onchange = () => {
    $("answer-wrap").hidden = !$("include-answer").checked;
  };
  $("btn-clear-dialogue").onclick = () => {
    if (state.reconcileBusy || (state.pendingReconcile?.token && Date.now() < state.pendingReconcile.expiresAt)) {
      toast("请先完成当前抽取或继续暂存，清空对话不会取消后台任务");
      return;
    }
    state.pendingReconcile = null;
    state.dialogue = [];
    state.sessionId = `web-${Date.now().toString(36)}`;
    renderDialogue();
    const report = $("reconcile-report");
    if (report) {
      report.hidden = true;
      report.innerHTML = "";
    }
  };
  $("query").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      $("reconcile-form").requestSubmit();
    }
  });
  $("content").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      $("remember-form").requestSubmit();
    }
  });
  const themeBtn = $("theme-toggle");
  if (themeBtn) {
    themeBtn.onclick = () => {
      const root = document.documentElement;
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("memora-theme", next);
      } catch {}
    };
  }
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target;
    if (
      target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable)
    ) {
      return;
    }
    showTab("recall");
    $("recall-text")?.focus();
    e.preventDefault();
  });
  renderDialogue();
  bindGraphInteractions();
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      showTab(tab.dataset.tab);
      if (tab.dataset.tab === "graph") {
        loadGraph().catch((err) => toast(err.message));
      }
    };
  });
  $("refresh-lib").onclick = () => loadLibrary().catch((err) => toast(err.message));
  $("filter-state").onchange = () => {
    state.offset = 0;
    loadLibrary().catch((err) => toast(err.message));
  };
  $("filter-kind").onchange = () => {
    state.offset = 0;
    loadLibrary().catch((err) => toast(err.message));
  };
  $("prev-page").onclick = () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadLibrary().catch((err) => toast(err.message));
  };
  $("next-page").onclick = () => {
    state.offset += state.limit;
    loadLibrary().catch((err) => toast(err.message));
  };
  $("btn-tick").onclick = async () => {
    try {
      const r = await api("/api/tick", { method: "POST", body: "{}" });
      $("tick-log").textContent = JSON.stringify(r, null, 2);
      toast("tick 完成");
      await loadLibrary();
    } catch (err) {
      toast(err.message);
    }
  };
  const runDream = () => onDream().catch((err) => toast(err.message));
  $("btn-dream").onclick = runDream;
  $("btn-dream-top").onclick = runDream;
  $("btn-dream-ops").onclick = runDream;
  $("btn-rebuild").onclick = async () => {
    try {
      await api("/api/rebuild-index", { method: "POST", body: "{}" });
      toast("索引已重建");
    } catch (err) {
      toast(err.message);
    }
  };
  $("btn-export").onclick = async () => {
    try {
      const res = await api("/api/export?include_vectors=true", { raw: true });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "memora-export.zip";
      a.click();
      URL.revokeObjectURL(url);
      $("io-log").textContent = `exported ${blob.size} bytes`;
    } catch (err) {
      toast(err.message);
    }
  };
  $("import-file").onchange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const buf = await file.arrayBuffer();
      const mode = $("import-mode").value;
      const report = await api(`/api/import?mode=${mode}`, {
        method: "POST",
        body: buf,
        headers: { "Content-Type": "application/zip" },
      });
      $("io-log").textContent = JSON.stringify(report, null, 2);
      toast("导入完成");
      await loadLibrary();
    } catch (err) {
      toast(err.message);
    } finally {
      e.target.value = "";
    }
  };
}

bind();
Promise.all([loadStatus(), loadMembers(), loadLibrary()])
  .then(() => loadGraph().catch(() => {}))
  .catch((err) => toast(err.message));
