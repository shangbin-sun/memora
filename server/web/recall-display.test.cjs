const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const start = source.indexOf('function card(');
const end = source.indexOf('\nasync function ', start);
assert.ok(start >= 0 && end > start);
const context = {
  document: { createElement: () => ({ innerHTML: '', querySelector: () => ({}) }) },
  KIND_LABEL: { episodic: '情景' },
  escapeHtml: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
  memberLabel: id => id, shortId: id => id.slice(0, 8), fmtTime: () => '创建时间',
};
vm.createContext(context);
vm.runInContext(source.slice(start, end) + '\nthis.renderCard = card;', context);
const memory = { id: 'memory001', kind: 'episodic', state: 'active', content: '原始<正文>',
  subject_ids: [], importance: 0.8, confidence: 1.0, created_at_ms: 1 };
test('recall card renders server snippet and escapes member/body HTML', () => {
  const rendered = context.renderCard(memory, '', '[主驾<script> · 记录于2026-10-26 11:20] 我有点头疼');
  assert.match(rendered.innerHTML, /\[主驾&lt;script&gt; · 记录于2026-10-26 11:20\] 我有点头疼/);
  assert.ok(!rendered.innerHTML.includes('<script>'));
  assert.ok(!rendered.innerHTML.includes('原始&lt;正文&gt;'));
  assert.equal(memory.content, '原始<正文>');
  assert.ok(source.includes('card(h.memory, extra, h.display_text || null)'));
});
test('library cards and legacy recall keep raw content', () => {
  assert.match(context.renderCard(memory).innerHTML, /原始&lt;正文&gt;/);
  assert.match(context.renderCard(memory, '', null).innerHTML, /原始&lt;正文&gt;/);
});
