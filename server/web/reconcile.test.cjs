const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const start = source.indexOf('async function reconcileReport(');
const end = source.indexOf('\nasync function onReconcile(', start);
assert.ok(start >= 0 && end > start);
const context = { setTimeout, clearTimeout, AbortController, Error, api: () => { throw Error('unexpected poll'); } };
vm.createContext(context);
vm.runInContext(source.slice(start, end) + '\nthis.poll = reconcileReport;', context);
const accepted = () => ({ status: 202, json: async () => ({ job_id: '01900000-0000-7000-8000-000000000001', status_url: '/api/reconcile/jobs/01900000-0000-7000-8000-000000000001' }) });
const renderStart = source.indexOf('function showReconcileReport(');
const renderEnd = source.indexOf('\nasync function reconcileReport(', renderStart);
assert.ok(renderStart >= 0 && renderEnd > renderStart);
const panel = { hidden: true, innerHTML: '' };
context.$ = () => panel;
context.escapeHtml = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
vm.runInContext(source.slice(renderStart, renderEnd) + '\nthis.render = showReconcileReport;', context);

test('short synchronous report remains compatible', async () => {
  const report = { memories_kept: 1 };
  const actual = await context.poll({ status: 200, json: async () => report }, () => {}, () => { throw Error('unexpected poll'); });
  assert.equal(actual, report);
});
test('many pending polls have no total timeout and GET failures retry only GET', async () => {
  let calls = 0;
  const report = { memories_kept: 4 };
  const actual = await context.poll(accepted(), () => {}, async (url, options) => {
    assert.match(url, /^\/api\/reconcile\/jobs\//);
    assert.ok(options.signal);
    calls += 1;
    if (calls === 1) throw Error('transient network failure');
    return calls <= 65 ? { status: 'running' } : { status: 'completed', report };
  }, async () => {});
  assert.equal(calls, 66);
  assert.equal(actual, report);
});
test('failed job is terminal rather than resubmitted', async () => {
  await assert.rejects(context.poll(accepted(), () => {}, async () => ({ status: 'failed', error: 'interrupted' }), async () => {}), /interrupted/);
});
test('rejects attacker-controlled status URLs', async () => {
  await assert.rejects(context.poll({ status: 202, json: async () => ({ status_url: 'https://example.invalid/' }) }, () => {}), /无效查询地址/);
});
test('NDJSON partial report shows invalid and truncated sources without HTML injection', () => {
  context.render({ memories_kept: 1, written_ids: ['memory1'], extraction: {
    partial: true, chunks_completed: 0, chunks_total: 1, invalid_records: 2,
    failed_source_ids: ['document:<script>'], truncated_source_ids: ['document:<script>']
  }, facts: [{ owner: 'member1', tag: '饮食', content: '喜欢<茶>' }] });
  assert.equal(panel.hidden, false);
  assert.match(panel.innerHTML, /无效记录：2/);
  assert.match(panel.innerHTML, /输出截断来源：document:&lt;script&gt;/);
  assert.match(panel.innerHTML, /已有内容可能已写入/);
  assert.match(panel.innerHTML, /喜欢&lt;茶&gt;/);
  assert.ok(!panel.innerHTML.includes('<script>'));
});
test('legacy partial report without new counters remains renderable', () => {
  context.render({ extraction: { partial: true, chunks_completed: 1, chunks_total: 2 } });
  assert.match(panel.innerHTML, /无效记录：0/);
  assert.match(panel.innerHTML, /输出截断来源：无/);
});

const token1 = '01900000-0000-7000-8000-000000000001';
const token2 = '01900000-0000-7000-8000-000000000002';
const stagedReport = (token = token1) => ({
  staged: true, memories_extracted: 3, memories_kept: 0, written_ids: [],
  resume_id: token, resume_expires_in_secs: 1800,
  extraction: { partial: true, chunks_completed: 1, chunks_total: 2, recovered_chunks: 2,
    planning_calls: 1, extraction_calls: 4, truncated_attempts: 2, invalid_attempt_records: 3,
    valid_records: 3, remaining_ranges: [{ source_id: '文档:<script>', start: 12, end: 35, reason: '<img onerror=alert(1)>' }] }
});

test('staged report clearly says unwritten and escapes original Unicode ranges', () => {
  context.render(stagedReport());
  assert.match(panel.innerHTML, /结果暂存，未写入记忆/);
  assert.match(panel.innerHTML, /继续只处理未完成块/);
  assert.match(panel.innerHTML, /恢复块 2/);
  assert.match(panel.innerHTML, /规划调用 1/);
  assert.match(panel.innerHTML, /截断尝试 2/);
  assert.match(panel.innerHTML, /无效尝试记录 3/);
  assert.match(panel.innerHTML, /Unicode 字符，0起始/);
  assert.match(panel.innerHTML, /文档:&lt;script&gt;.*\[12, 35\)/);
  assert.match(panel.innerHTML, /&lt;img onerror=alert\(1\)&gt;/);
  assert.match(panel.innerHTML, /继续未完成块/);
  assert.match(panel.innerHTML, /30分钟.*重启后失效/);
  assert.doesNotMatch(panel.innerHTML, /已有内容可能已写入|<script>|<img/);
});

test('expired or restarted staged report has no continuation button or stale token', () => {
  context.render({ ...stagedReport(), resume_id: undefined, resume_expires_in_secs: 0, resume_error: '服务已重启，暂存失效' });
  assert.match(panel.innerHTML, /未写入记忆/);
  assert.match(panel.innerHTML, /服务已重启，暂存失效/);
  assert.doesNotMatch(panel.innerHTML, /data-resume-reconcile/);
  context.render({ ...stagedReport('https://evil.invalid/'), resume_expires_in_secs: 1800 });
  assert.doesNotMatch(panel.innerHTML, /data-resume-reconcile|https:\/\/evil/);
});

test('completed asynchronous staged job returns the continuation rather than polling forever', async () => {
  const report = stagedReport();
  const result = await context.poll(accepted(), () => {}, async () => ({ status: 'completed', report }), async () => {});
  assert.equal(result, report);
});

function uiHarness(request) {
  const resumeButton = { disabled: false, textContent: '' };
  const resumeStatus = { textContent: '' };
  const elements = {
    query: { value: '原始长文' }, answer: { value: '助手上下文' },
    'include-answer': { checked: true }, 'btn-reconcile': { disabled: false, textContent: '' },
    'reconcile-report': { hidden: true, innerHTML: '', textContent: '',
      querySelector: (selector) => selector === '[data-resume-reconcile]' ? resumeButton : resumeStatus }
  };
  const calls = [], refreshes = [], toasts = [];
  const sandbox = {
    state: { reconcileChat: true, reconcileBusy: false, pendingReconcile: null, dialogue: [], sessionId: 'test-session' },
    $: (id) => elements[id], api: async (path, options) => { calls.push({ path, options }); return request(path, options); },
    buildTranscript: () => '', renderDialogue: () => {},
    escapeHtml: context.escapeHtml, toast: (message) => toasts.push(message),
    loadMembers: async () => refreshes.push('members'), loadLibrary: async () => refreshes.push('library'),
    loadGraph: async () => refreshes.push('graph'), showTab: () => { throw Error('must not navigate'); },
    setTimeout, clearTimeout, AbortController, Date, Error
  };
  vm.createContext(sandbox);
  const finish = source.indexOf('\nfunction formatInterval(', renderStart);
  vm.runInContext(source.slice(renderStart, finish), sandbox);
  return { sandbox, elements, calls, refreshes, toasts, resumeButton, resumeStatus };
}
const response = (report) => ({ status: 200, json: async () => report });
const event = { preventDefault() {} };

test('initial UI requests async and preserves input/dialogue while staged', async () => {
  const ui = uiHarness(() => response(stagedReport()));
  await ui.sandbox.onReconcile(event);
  assert.equal(ui.calls.length, 1);
  assert.equal(ui.calls[0].path, '/api/reconcile');
  assert.equal(JSON.parse(ui.calls[0].options.body).async, true);
  assert.equal(ui.elements.query.value, '原始长文');
  assert.equal(ui.elements.answer.value, '助手上下文');
  assert.equal(ui.sandbox.state.dialogue.length, 0);
  assert.equal(ui.sandbox.state.pendingReconcile.token, token1);
  assert.equal(ui.resumeButton.disabled, false);
  assert.deepEqual(ui.refreshes, []);
  await ui.sandbox.onReconcile(event);
  assert.equal(ui.calls.length, 1, 'do not replace a live checkpoint with a repeated full input');
});

test('continuations send only rotating tokens and append the captured dialogue once after completion', async () => {
  let resumes = 0;
  const complete = { staged: false, memories_kept: 3, written_ids: ['a', 'b', 'c'] };
  const ui = uiHarness((path) => path === '/api/reconcile' ? response(stagedReport())
    : response(++resumes === 1 ? stagedReport(token2) : complete));
  await ui.sandbox.onReconcile(event);
  const captured = ui.sandbox.state.pendingReconcile;
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.sandbox.state.pendingReconcile.token, token2);
  assert.equal(ui.sandbox.state.dialogue.length, 0);
  await ui.sandbox.onResumeReconcile();
  assert.deepEqual(ui.calls.map((call) => call.path), ['/api/reconcile', `/api/reconcile/resume/${token1}`, `/api/reconcile/resume/${token2}`]);
  for (const call of ui.calls.slice(1)) {
    assert.equal(call.options.method, 'POST');
    assert.equal(call.options.body, undefined, 'must not replay the old input payload');
  }
  assert.equal(ui.sandbox.state.dialogue.length, 2);
  assert.equal(ui.sandbox.state.dialogue[0].text, '原始长文');
  assert.equal(ui.sandbox.state.dialogue[1].text, '助手上下文');
  assert.equal(ui.sandbox.state.pendingReconcile, null);
  assert.equal(ui.elements.query.value, '');
  assert.deepEqual(ui.refreshes, ['members', 'library', 'graph']);
  await ui.sandbox.finishReconcileReport(complete, captured);
  assert.equal(ui.sandbox.state.dialogue.length, 2, 'same completion callback cannot duplicate conversation');
});

test('double resume click claims token once while async job is running', async () => {
  let release;
  const wait = new Promise((resolve) => { release = resolve; });
  const ui = uiHarness(async (path) => path === '/api/reconcile' ? response(stagedReport())
    : path.includes('/resume/') ? accepted() : wait);
  await ui.sandbox.onReconcile(event);
  const first = ui.sandbox.onResumeReconcile();
  await Promise.resolve();
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.sandbox.state.reconcileBusy, true);
  assert.equal(ui.resumeButton.disabled, true);
  assert.match(ui.resumeButton.textContent, /继续中/);
  assert.equal(ui.calls.filter((call) => call.path.includes('/resume/')).length, 1);
  release({ status: 'completed', report: { staged: false, written_ids: ['one'] } });
  await first;
  assert.equal(ui.sandbox.state.dialogue.length, 2);
  assert.equal(ui.sandbox.state.reconcileBusy, false);
});

test('completion preserves a new draft typed during a pending operation', async () => {
  const ui = uiHarness((path) => response(path === '/api/reconcile' ? stagedReport() : { staged: false, written_ids: ['one'] }));
  await ui.sandbox.onReconcile(event);
  ui.elements.query.value = '下一条新输入';
  ui.elements.answer.value = '新的上下文';
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.elements.query.value, '下一条新输入');
  assert.equal(ui.elements.answer.value, '新的上下文');
  assert.equal(ui.sandbox.state.dialogue[0].text, '原始长文');
});

test('resume transport failure is never POST-retried and does not claim no writes for running job', async () => {
  const ui = uiHarness((path) => {
    if (path === '/api/reconcile') return response(stagedReport());
    throw Error('connection reset');
  });
  await ui.sandbox.onReconcile(event);
  await ui.sandbox.onResumeReconcile();
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.calls.length, 2);
  assert.equal(ui.sandbox.state.pendingReconcile.token, null);
  assert.equal(ui.sandbox.state.dialogue.length, 0);
  assert.match(ui.resumeStatus.textContent, /此前结果为未写入的暂存/);
  assert.match(ui.resumeStatus.textContent, /本次后台任务可能仍在运行或已经写入/);
  assert.equal(ui.resumeButton.disabled, true);
});

test('gone continuation explains restart/expiry and never replays a consumed generation', async () => {
  const ui = uiHarness((path) => {
    if (path === '/api/reconcile') return response(stagedReport());
    const error = Error('暂存已过期或服务已重启'); error.status = 410; throw error;
  });
  await ui.sandbox.onReconcile(event);
  await ui.sandbox.onResumeReconcile();
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.calls.length, 2);
  assert.match(ui.resumeStatus.textContent, /未写入的暂存快照/);
  assert.match(ui.resumeStatus.textContent, /暂存已过期或服务已重启/);
  assert.match(ui.resumeStatus.textContent, /另一请求继续/);
});

test('client-side expired checkpoint does not dispatch a request', async () => {
  const ui = uiHarness(() => response(stagedReport()));
  await ui.sandbox.onReconcile(event);
  ui.sandbox.state.pendingReconcile.expiresAt = 0;
  await ui.sandbox.onResumeReconcile();
  assert.equal(ui.calls.length, 1);
  assert.equal(ui.sandbox.state.pendingReconcile.token, null);
  assert.match(ui.elements['reconcile-report'].innerHTML, /暂存已过期，未写入记忆/);
});

test('successful completion remains successful when a data refresh fails', async () => {
  const ui = uiHarness(() => response({ staged: false, written_ids: ['one'] }));
  ui.sandbox.loadMembers = async () => { throw Error('refresh offline'); };
  await ui.sandbox.onReconcile(event);
  assert.equal(ui.sandbox.state.dialogue.length, 2);
  assert.deepEqual(ui.refreshes, ['library', 'graph']);
  assert.ok(ui.toasts.includes('抽取已完成，但成员列表刷新失败'));
  assert.doesNotMatch(ui.elements['reconcile-report'].textContent, /请求失败或连接中断/);
});
