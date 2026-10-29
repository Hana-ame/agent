'use strict';

/**
 * 定时表达式 → 下次触发时刻。纯时间计算，无状态、无 I/O，可单测。
 * 支持 interval（每 N 分钟）、daily（HH:MM）、cron（5 段）。
 */

function parseCron(expr) {
  const parts = String(expr || '').trim().split(/\s+/);
  if (parts.length !== 5) return null;

  function field(spec, min, max) {
    const set = new Set();
    for (const token of spec.split(',')) {
      if (token === '*') { for (let i = min; i <= max; i++) set.add(i); continue; }
      const m = token.match(/^(\d+|\*)\/(\d+)$/);
      if (m) { const step = parseInt(m[2], 10); if (step > 0) for (let i = min; i <= max; i += step) set.add(i); continue; }
      const a = parseInt(token.split('-')[0], 10);
      const b = token.includes('-') ? parseInt(token.split('-')[1], 10) : a;
      if (Number.isNaN(a) || Number.isNaN(b)) continue;
      for (let i = a; i <= b && i <= max; i++) set.add(i);
    }
    return set;
  }

  const minutes = field(parts[0], 0, 59);
  const hours = field(parts[1], 0, 23);
  const dom = field(parts[2], 1, 31);
  const months = field(parts[3], 1, 12);
  const dow = field(parts[4], 0, 7);
  if (dow.has(7)) dow.add(0);
  return { minutes, hours, dom, months, dow };
}

function cronMatches(c, d) {
  if (!c.months.has(d.getMonth() + 1)) return false;
  const domMatch = c.dom.has(d.getDate());
  const dowMatch = c.dow.has(d.getDay());
  const domFull = c.dom.size >= 31;
  const dowFull = c.dow.size >= 7;
  if (domFull && dowFull) return true;
  if (!domFull && !dowFull) return domMatch && dowMatch;
  // cron 语义：DOM 与 DOW 只有一个受限时，用受限的那个
  return domFull ? dowMatch : domMatch;
}

function nextDailyAt(timeStr, after) {
  const m = String(timeStr || '').match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return null;
  const h = parseInt(m[1], 10);
  const mi = parseInt(m[2], 10);
  if (h > 23 || mi > 59) return null;
  const d = new Date(after);
  d.setHours(h, mi, 0, 0);
  if (d.getTime() <= after) return d.getTime() + 24 * 3600 * 1000;
  return d.getTime();
}

function nextCronAt(expr, after) {
  const c = parseCron(expr);
  if (!c) return null;
  let t = new Date(Math.floor(after / 60000) * 60000 + 60000);
  for (let steps = 0; steps < 400000; steps++) {
    if (cronMatches(c, t)) return t.getTime();
    t = new Date(t.getTime() + 60000);
  }
  return null;
}

function labelOf(s) {
  if (!s) return '';
  if (s.type === 'interval') return '每 ' + s.minutes + ' 分钟';
  if (s.type === 'daily') return '每日 ' + s.time;
  if (s.type === 'cron') return 'cron ' + s.cron;
  return '';
}

function computeNextAt(s, now) {
  if (!s.enabled) return null;
  if (s.type === 'interval') return now + (parseInt(s.minutes, 10) || 5) * 60000;
  if (s.type === 'daily') return nextDailyAt(s.time, now);
  if (s.type === 'cron') return nextCronAt(s.cron, now);
  return null;
}

function advanceSchedule(s, now) {
  if (s.type === 'interval') s.nextAt = now + (parseInt(s.minutes, 10) || 5) * 60000;
  else if (s.type === 'daily') s.nextAt = nextDailyAt(s.time, now + 1000);
  else if (s.type === 'cron') s.nextAt = nextCronAt(s.cron, now + 1000);
  else s.nextAt = null;
  return s.nextAt;
}

module.exports = { parseCron, cronMatches, nextDailyAt, nextCronAt, labelOf, computeNextAt, advanceSchedule };
