'use strict';

const { uid } = require('./ids');

/**
 * 端口 / 测试的归一化。纯函数，不依赖 store，供 store 迁移与领域层共用。
 * 放在 core 是为了避免「store 需要 domain 的工具」这种反向依赖。
 */

function normalizeTests(list) {
  const out = [];
  for (const t of list || []) {
    const command = String((t && t.command) || '').trim();
    if (!command) continue;
    out.push({
      id: String((t && t.id) || uid()).slice(0, 12),
      name: String((t && t.name) || '').trim() || 'test',
      command
    });
  }
  return out;
}

function normalizePorts(list, isOutput) {
  const out = [];
  for (const p of list || []) {
    if (!p) continue;
    const name = String(p.name || '').trim() || 'port';
    out.push({
      id: String(p.id || uid()).slice(0, 12),
      name,
      spec: String(p.spec || ''),
      tests: normalizeTests(p.tests),
      capture: isOutput ? (p.capture === 'file' ? 'file' : 'stdout') : undefined,
      glob: isOutput && p.capture === 'file' ? String(p.glob || '*') : undefined
    });
  }
  if (out.length) return out;
  return isOutput
    ? [{ id: 'out', name: 'out', spec: '', tests: [], capture: 'stdout' }]
    : [{ id: 'in', name: 'dep', spec: '', tests: [] }];
}

/** 极简 glob：只支持 * 通配 */
function simpleGlobMatch(name, pattern) {
  if (pattern === '*' || pattern === '' || pattern == null) return true;
  if (pattern === name) return true;
  const rx = '^' + String(pattern).split('*').map((x) => x.replace(/[.+^${}()|[\]\\]/g, '\\$&')).join('.*') + '$';
  return new RegExp(rx).test(name);
}

module.exports = { normalizeTests, normalizePorts, simpleGlobMatch };
