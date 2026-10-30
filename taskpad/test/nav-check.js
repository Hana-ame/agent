#!/usr/bin/env node
'use strict';

/**
 * 面包屑导航验证（无头 Edge + CDP）。
 *
 *   node test/nav-check.js [--url http://127.0.0.1:8335]
 *
 * 流程：造一个临时模块 → 给它建「内部流程图」→ 进入子图 →
 *   1. 面包屑应该是「主图 › 子图」两段，当前段不可点
 *   2. 点第一段能回到主图，面包屑收缩成一段
 *   3. 返回按钮在子图里可用、在主图里禁用
 * 跑完把临时模块和子图删掉。
 */

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
let BASE = arg('--url', '');

const { launch, sleep } = require('./cdp');
const { startServer } = require('./server');

async function api(method, path, body) {
  const r = await fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined
  });
  const text = await r.text();
  return { ok: r.ok, status: r.status, body: text ? JSON.parse(text) : null };
}

let failed = 0;
const ok = (n) => console.log('  ok   ' + n);
const bad = (n, m) => { failed++; console.log('  FAIL ' + n + ' :: ' + m); };

async function main() {
  let srv = null;
  if (!BASE) { srv = await startServer(); BASE = srv.base; }
  const cdp = await launch(BASE);
  if (cdp.edgeMissing) {
    console.log('跳过：本机没找到 Edge');
    if (srv) await srv.stop();
    return;
  }

  let taskId = null;
  let graphId = null;

  try {
    // 造现场
    const state0 = (await api('GET', '/api/state')).body;
    const rootId = state0.graphs[0].id;
    const t = (await api('POST', '/api/tasks', { name: '面包屑测试节点', command: 'echo nav', graphId: rootId })).body.task;
    taskId = t.id;
    const g = (await api('POST', '/api/graphs', { name: '面包屑测试节点 · 内部流程', parentNodeId: taskId })).body.graph;
    graphId = g.id;

    // 刷新前端状态，然后进入子图
    await cdp.evalJs('refresh().then(() => { enterGraph(' + JSON.stringify(graphId) + '); return 1; })');
    await sleep(600);

    const crumbs = await cdp.evalJs(`(() => {
      const items = [...document.querySelectorAll('#graph-path .gp-item')];
      return JSON.stringify({
        count: items.length,
        texts: items.map(i => i.querySelector('.gp-txt').textContent),
        clickable: items.map(i => i.hasAttribute('data-goto-graph')),
        currentIndex: items.findIndex(i => i.classList.contains('current')),
        seps: document.querySelectorAll('#graph-path .gp-sep').length,
        backDisabled: document.getElementById('btn-graph-back').disabled,
        overflowX: getComputedStyle(document.getElementById('graph-path')).overflowX,
        pathWidth: document.getElementById('graph-path').getBoundingClientRect().width,
        firstWidth: items[0] ? items[0].getBoundingClientRect().width : 0
      });
    })()`);
    const c = JSON.parse(crumbs);

    if (c.count === 2) ok('子图里面包屑两段：' + c.texts.join(' › '));
    else bad('面包屑段数', '期望 2，实际 ' + c.count + '（' + c.texts.join(' › ') + '）');

    if (c.currentIndex === 1 && c.clickable[1] === false) ok('当前段不可点（无 data-goto-graph）');
    else bad('当前段', 'currentIndex=' + c.currentIndex + ' clickable=' + JSON.stringify(c.clickable));

    if (c.clickable[0] === true) ok('上级段可点');
    else bad('上级段', '缺少 data-goto-graph');

    if (c.backDisabled === false) ok('子图里「返回」按钮可用');
    else bad('返回按钮', '在子图里被禁用');

    if (c.pathWidth > 40 && c.firstWidth > 10) ok('面包屑有实际宽度 ' + Math.round(c.pathWidth) + 'px（旧版被挤成竖排）');
    else bad('面包屑宽度', 'pathWidth=' + c.pathWidth + ' firstItem=' + c.firstWidth);

    const shot = arg('--shot', '');
    if (shot) {
      await cdp.screenshot(shot, 1440, 900);
      console.log('  shot ' + shot);
    }

    // 点第一段回主图
    await cdp.evalJs("document.querySelector('#graph-path .gp-item[data-goto-graph]').click(); 1");
    await sleep(600);
    const after = JSON.parse(await cdp.evalJs(`(() => {
      const items = [...document.querySelectorAll('#graph-path .gp-item')];
      return JSON.stringify({
        count: items.length,
        text: items[0] ? items[0].querySelector('.gp-txt').textContent : '',
        active: state.activeGraphId,
        backDisabled: document.getElementById('btn-graph-back').disabled
      });
    })()`));

    if (after.active === rootId && after.count === 1) ok('点上级段回到「' + after.text + '」，面包屑收成一段');
    else bad('点击回退', 'active=' + after.active + ' count=' + after.count + ' root=' + rootId);

    if (after.backDisabled === true) ok('主图里「返回」按钮禁用');
    else bad('返回按钮', '在主图里仍可点');
  } finally {
    // 清场：先删子图（连带节点），再删测试模块
    try { if (graphId) await api('DELETE', '/api/graphs/' + graphId); } catch (e) { /* ignore */ }
    try { if (taskId) await api('DELETE', '/api/tasks/' + taskId); } catch (e) { /* ignore */ }
    const err = await cdp.close();
    if (failed && err) console.log('\n[edge stderr]\n' + err.split('\n').slice(0, 5).join('\n'));
    if (srv) { try { await srv.stop(); } catch (e) { /* ignore */ } }
  }

  console.log(failed ? '\n失败 ' + failed + ' 项' : '\n全部通过');
  process.exitCode = failed ? 1 : 0;
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
