'use strict';

const crypto = require('crypto');

/** 8 字节随机 id（16 hex 字符），与旧 store.json 里的 id 形态一致 */
function uid() {
  return crypto.randomBytes(8).toString('hex');
}

function nowIso() {
  return new Date().toISOString();
}

function pad2(n) {
  return String(n).padStart(2, '0');
}

/** 把任意名字压成可安全用于环境变量后缀的 key */
function envKey(name, fallback) {
  const s = String(name == null ? '' : name).replace(/[^\w\u4e00-\u9fa5]+/g, '_').replace(/^_+|_+$/g, '');
  return s || fallback || 'port';
}

module.exports = { uid, nowIso, pad2, envKey };
