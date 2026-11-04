const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Execute the actual SVG renderer, without booting unrelated network/UI code.
// This checks DOM attributes and geometry, not browser paint or text metrics.
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const start = source.indexOf('function factObjectLabel(');
const end = source.indexOf('\nfunction renderFactList(', start);
assert.ok(start >= 0 && end > start, 'actual graph renderer boundaries exist');

function fixture() {
  const htmlWrites = [];
  class Element {
    constructor(tagName) {
      this.tagName = tagName;
      this.children = [];
      this.attributes = {};
      this.dataset = {};
      this.style = {};
      this.textContent = '';
      const classes = new Set();
      this.classList = {
        add: (...names) => names.forEach(name => classes.add(name)),
        contains: name => classes.has(name),
        toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
      };
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name] ?? null; }
    appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
    set innerHTML(value) { htmlWrites.push(value); this.children = []; }
  }
  const svg = new Element('svg');
  svg.parentElement = { clientWidth: 640 };
  const frames = new Map();
  let nextFrame = 1;
  const cancelled = [];
  const explained = [];
  const context = {
    state: { graphSim: null },
    $: id => id === 'graph-svg' ? svg : null,
    document: {
      createElementNS: (_ns, tag) => new Element(tag),
      querySelectorAll: () => [],
    },
    requestAnimationFrame: callback => { const id = nextFrame++; frames.set(id, callback); return id; },
    cancelAnimationFrame: id => { cancelled.push(id); frames.delete(id); },
    openExplain: id => explained.push(id),
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end) + '\nthis.render = renderForceGraph;', context);
  function descendants(root = svg) {
    return root.children.flatMap(child => [child, ...descendants(child)]);
  }
  return {
    svg, context, htmlWrites, cancelled, explained,
    render: (entities, facts, hub = null) => context.render(entities, facts, hub),
    frame() {
      const next = frames.entries().next().value;
      assert.ok(next, 'renderer schedules an animation frame');
      frames.delete(next[0]);
      next[1]();
    },
    nodes: () => descendants().filter(el => el.classList.contains('graph-node')),
    paths: () => descendants().filter(el => el.classList.contains('graph-link')),
    labels: () => descendants().filter(el => el.classList.contains('graph-link-label')),
    all: descendants,
  };
}
function person(id, label = id) { return { id, canonical_name: label, entity_type: 'person' }; }
function fact(id, subject, relation, object, extra = {}) {
  return { id, subject_id: subject, relation, object: { entity_id: object }, ...extra };
}
function numbers(text) { return (text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) || []).map(Number); }
function position(node) { return numbers(node.getAttribute('transform')); }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function labelPosition(label) { return [label.getAttribute('x'), label.getAttribute('y')].join(','); }
function assertDirected(f, edge, sourceId, targetId) {
  assert.equal(edge.tagName, 'path');
  assert.equal(edge.getAttribute('marker-end'), 'url(#graph-direction-arrow)');
  assert.equal(edge.getAttribute('marker-start'), null);
  assert.match(edge.getAttribute('d'), /^M .* Q /);
  const coords = numbers(edge.getAttribute('d'));
  assert.equal(coords.length, 6);
  const a = position(f.nodes().find(node => node.dataset.node === sourceId));
  const b = position(f.nodes().find(node => node.dataset.node === targetId));
  const from = coords.slice(0, 2);
  const to = coords.slice(-2);
  assert.ok(distance(from, a) < distance(from, b), 'path begins at its subject');
  assert.ok(distance(to, b) < distance(to, a), 'arrow points to its object');
  assert.ok(Math.abs(distance(from, a) - 19) < 1e-6, 'path starts beyond person circle');
  assert.ok(Math.abs(distance(to, b) - 19) < 1e-6, 'arrow tip stays beyond target circle');
}

test('reciprocal father/son and wife/husband edges point to their own targets on distinct curves', () => {
  const f = fixture();
  const facts = [
    fact('dad', 'child', '爸爸', 'father'),
    fact('son', 'father', '儿子', 'child'),
    fact('wife', 'father', '老婆', 'mother'),
    fact('husband', 'mother', '老公', 'father'),
  ];
  f.render([person('child', '林子航'), person('father', '林志远'), person('mother', '宋娜娜')], facts);
  f.frame();
  const marker = f.all().find(el => el.tagName === 'marker');
  assert.equal(marker.getAttribute('id'), 'graph-direction-arrow');
  assert.equal(marker.getAttribute('orient'), 'auto');
  assert.equal(marker.getAttribute('refX'), '10');
  assert.equal(marker.children[0].getAttribute('d'), 'M 0 0 L 10 5 L 0 10 Z');
  facts.forEach((value, index) => assertDirected(f, f.paths()[index], value.subject_id, value.object.entity_id));
  assert.deepEqual(f.labels().map(label => label.textContent), ['爸爸', '儿子', '老婆', '老公']);
  for (const [left, right] of [[0, 1], [2, 3]]) {
    assert.notEqual(f.paths()[left].getAttribute('d'), f.paths()[right].getAttribute('d'));
    assert.notDeepEqual(numbers(f.paths()[left].getAttribute('d')).slice(2, 4),
      numbers(f.paths()[right].getAttribute('d')).slice(2, 4), 'reciprocal control points differ');
    assert.notEqual(labelPosition(f.labels()[left]), labelPosition(f.labels()[right]));
  }
});

test('multiple predicates in one direction plus reverse predicates use separate lanes and label positions', () => {
  const f = fixture();
  const facts = [
    fact('one', 'a', '爸爸', 'b'), fact('two', 'a', '监护人', 'b'),
    fact('three', 'a', '紧急联系人', 'b'), fact('four', 'b', '儿子', 'a'),
  ];
  f.render([person('a'), person('b')], facts);
  f.frame();
  assert.equal(new Set(f.paths().map(path => path.getAttribute('d'))).size, 4);
  assert.equal(new Set(f.paths().map(path => numbers(path.getAttribute('d')).slice(2, 4).join(','))).size, 4);
  assert.equal(new Set(f.labels().map(labelPosition)).size, 4);
  facts.forEach((value, index) => assertDirected(f, f.paths()[index], value.subject_id, value.object.entity_id));
});

test('self relations retain finite directed loops with independently positioned labels', () => {
  const f = fixture();
  f.render([person('self')], [fact('loop1', 'self', '自我描述', 'self'), fact('loop2', 'self', '自我评价', 'self')]);
  f.frame();
  for (const path of f.paths()) {
    assert.match(path.getAttribute('d'), /^M .* C /);
    assert.equal(path.getAttribute('marker-end'), 'url(#graph-direction-arrow)');
    const coords = numbers(path.getAttribute('d'));
    assert.equal(coords.length, 8);
    assert.ok(coords.every(Number.isFinite));
    assert.notDeepEqual(coords.slice(0, 2), coords.slice(-2), 'loop has distinct rim endpoints');
  }
  assert.notEqual(f.paths()[0].getAttribute('d'), f.paths()[1].getAttribute('d'));
  assert.notEqual(labelPosition(f.labels()[0]), labelPosition(f.labels()[1]));
});

test('untrusted relation and entity labels stay in textContent, facts are unchanged, and source links still work', () => {
  const f = fixture();
  const relation = '<script>alert(1)</script>';
  const entities = [person('a', '<img src=x>'), person('b')];
  const facts = [fact('edge" onload="bad', 'a', relation, 'b', { source_memory_id: 'memory-id' })];
  const before = JSON.stringify({ entities, facts });
  f.render(entities, facts);
  f.frame();
  assert.equal(f.labels()[0].textContent, relation);
  assert.equal(f.labels()[0].children.length, 0);
  assert.equal(f.nodes()[0].children[1].textContent, '<img src=x>');
  assert.deepEqual(f.htmlWrites, [''], 'only static SVG clearing uses innerHTML');
  assert.ok(f.all().every(el => !['img', 'script'].includes(el.tagName)));
  f.paths()[0].onclick();
  assert.deepEqual(f.explained, ['memory-id']);
  f.nodes()[0].onclick();
  assert.ok(f.paths()[0].classList.contains('hot'), 'path retains incident highlight behavior');
  assert.equal(JSON.stringify({ entities, facts }), before, 'rendering must not modify graph data');
});

test('rerender cancels previous simulation, recreates one marker, and handles literal value nodes', () => {
  const f = fixture();
  const values = [{ id: 'nickname', subject_id: 'a', relation: '小名', object: { value: '航航' } }];
  f.render([person('a', '林子航')], values);
  const previous = f.context.state.graphSim;
  f.render([person('a', '林子航')], values);
  assert.deepEqual(f.cancelled, [previous]);
  assert.equal(f.all().filter(el => el.tagName === 'marker').length, 1);
  f.frame();
  const valueNode = f.nodes().find(node => node.classList.contains('value'));
  assert.ok(valueNode);
  assert.equal(valueNode.children[1].textContent, '航航');
  const endpoint = numbers(f.paths()[0].getAttribute('d')).slice(-2);
  assert.ok(Math.abs(distance(endpoint, position(valueNode)) - 13) < 1e-6);
  assert.equal(f.labels()[0].textContent, '小名');
});
