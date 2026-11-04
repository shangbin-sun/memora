const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// Execute the real bindGraphInteractions with a minimal DOM so the zoom
// anchor, pan conversion, and node-drag math stay locked down.
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const start = source.indexOf('function bindGraphInteractions(');
const end = source.indexOf('\nasync function loadGraph(', start);
assert.ok(start >= 0 && end > start, 'graph interaction binder boundaries exist');

function makeEventTarget() {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(name, fn) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(fn);
    },
    removeEventListener(name, fn) {
      const list = listeners.get(name);
      if (list) list.splice(list.indexOf(fn) >>> 0, 1);
    },
    fire(name, event) {
      (listeners.get(name) || []).slice().forEach((fn) => fn(event));
    },
  };
}

function fixture() {
  const svg = Object.assign(makeEventTarget(), {
    classSet: new Set(),
    classList: {
      add: (n) => svg.classSet.add(n),
      remove: (n) => svg.classSet.delete(n),
      contains: (n) => svg.classSet.has(n),
    },
    rect: { left: 0, top: 0, width: 320, height: 240 },
    getBoundingClientRect() { return svg.rect; },
    attributes: {},
    setAttribute(name, value) { svg.attributes[name] = String(value); },
    getAttribute(name) { return svg.attributes[name] ?? null; },
  });
  const viewport = Object.assign(makeEventTarget(), { attributes: {}, setAttribute(n, v) { viewport.attributes[n] = String(v); } });
  const windowTarget = makeEventTarget();
  const stopped = [];
  const state = { graphSim: null };
  const context = {
    state,
    $: (id) => (id === 'graph-svg' ? svg : null),
    stopGraphSim: () => stopped.push('stop'),
    setTimeout,
    window: windowTarget,
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end) + '\nthis.bind = bindGraphInteractions;', context);
  context.bind();
  return {
    svg, viewport, state, stopped, windowTarget,
    wheel: (clientX, clientY, deltaY) => svg.fire('wheel', { clientX, clientY, deltaY, preventDefault() {} }),
    down: (event) => svg.fire('pointerdown', event),
    move: (event) => windowTarget.fire('pointermove', event),
    up: () => windowTarget.fire('pointerup', {}),
    cancel: () => windowTarget.fire('pointercancel', {}),
    transform: () => viewport.attributes.transform,
  };
}

function svgEvent(target, extra = {}) {
  return { target, button: 0, preventDefault() {}, ...extra };
}

test('wheel zooms toward the cursor within clamp bounds', () => {
  const f = fixture();
  f.state.graphScene = { model: { nodes: [] }, viewport: f.viewport, width: 640, height: 480, apply() {}, applyView() {} };
  f.wheel(160, 120, -100);
  const view = f.state.graphView;
  assert.equal(view.scale, 1.12);
  // viewBox units are 2x client units here; the point under the cursor stays fixed.
  assert.ok(Math.abs((320 - view.tx) / view.scale - 320) < 1e-9);
  assert.ok(Math.abs((240 - view.ty) / view.scale - 240) < 1e-9);
  assert.equal(f.transform(), `translate(${view.tx} ${view.ty}) scale(1.12)`);
  for (let i = 0; i < 40; i++) f.wheel(160, 120, -100);
  assert.ok(f.state.graphView.scale <= 4 + 1e-9, 'zoom clamps at max scale');
  const zoomedIn = f.state.graphView.scale;
  f.wheel(160, 120, 100);
  assert.ok(f.state.graphView.scale < zoomedIn, 'wheel down zooms out');
  for (let i = 0; i < 80; i++) f.wheel(160, 120, 100);
  assert.ok(f.state.graphView.scale >= 0.4 - 1e-9, 'zoom clamps at min scale');
});

test('background drag pans the viewport and a moved drag suppresses the click', () => {
  const f = fixture();
  f.state.graphScene = { model: { nodes: [] }, viewport: f.viewport, width: 640, height: 480, apply() {}, applyView() {} };
  f.down(svgEvent({}, { clientX: 160, clientY: 120 }));
  assert.ok(f.svg.classSet.has('panning'));
  f.move({ clientX: 260, clientY: 170 });
  const view = f.state.graphView;
  assert.equal(view.tx, 200);
  assert.equal(view.ty, 100);
  f.up();
  assert.ok(!f.svg.classSet.has('panning'));
  assert.ok(f.svg.listeners.has('click'), 'capture-phase click swallow is armed');
  await_setTimeout_then_assert();
  function await_setTimeout_then_assert() {
    setTimeout(() => {
      assert.ok(!f.svg.listeners.get('click')?.length, 'click swallow disarms itself');
    }, 0);
  }
  f.cancel();
  assert.equal(f.state.graphView.tx, 200);
});

test('node drag pins the node, follows the pointer in graph coordinates, and kicks settle', () => {
  const f = fixture();
  const node = { id: 'a', x: 100, y: 200, kind: 'person' };
  let applied = 0;
  f.state.graphScene = {
    model: { nodes: [node] },
    viewport: f.viewport,
    width: 640,
    height: 480,
    apply: () => { applied += 1; },
    applyView() {},
  };
  const kicks = [];
  f.state.graphKick = () => kicks.push('kick');
  const nodeEl = { dataset: { node: 'a' }, closest: () => nodeEl };
  // client (160,120) -> viewBox (320,240); grab offset (20,-160).
  f.down(svgEvent(nodeEl, { clientX: 160, clientY: 120 }));
  assert.deepEqual(f.stopped, ['stop'], 'drag freezes the running layout');
  assert.equal(node.pinned, true);
  f.move({ clientX: 170, clientY: 130 });
  assert.equal(node.x, 120);
  assert.equal(node.y, 220);
  assert.ok(applied >= 1, 'geometry reapplied while dragging');
  // Dragging beyond the canvas edge clamps to the same margins as the simulation.
  f.move({ clientX: -500, clientY: 900 });
  assert.equal(node.x, 24);
  assert.equal(node.y, 480 - 36);
  f.up();
  assert.deepEqual(kicks, ['kick'], 'release reheats a short settle');
  assert.ok(!f.svg.classSet.has('panning'));
});

test('fact list filter keeps only facts touching the selected node', () => {
  const filterStart = source.indexOf('function graphFactMatchesNode(');
  const filterEnd = source.indexOf('\nfunction renderFactList(', filterStart);
  assert.ok(filterStart >= 0 && filterEnd > filterStart);
  const context = {};
  vm.createContext(context);
  vm.runInContext(source.slice(filterStart, filterEnd) + '\nthis.match = graphFactMatchesNode;', context);
  const fact = (id, subject, object) => ({ id, subject_id: subject, object: { entity_id: object } });
  assert.equal(context.match(fact('f1', 'a', 'b'), 'a'), true, 'subject match');
  assert.equal(context.match(fact('f1', 'a', 'b'), 'b'), true, 'object match');
  assert.equal(context.match(fact('f1', 'a', 'b'), 'c'), false, 'unrelated node filtered out');
  assert.equal(context.match(fact('f1', 'a', 'b'), null), true, 'no filter keeps everything');
  assert.equal(context.match({ id: 'v', subject_id: 'a', object: { value: '航航' } }, 'a'), true);
  assert.equal(context.match({ id: 'v', subject_id: 'a', object: { value: '航航' } }, '航航'), false, 'value objects never match');
});
