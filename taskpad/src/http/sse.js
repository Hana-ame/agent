'use strict';

const { onSafe } = require('../core/events');

/**
 * SSE 通道。
 *
 * 它是唯一「知道 HTTP 连接存在」的订阅者：领域层只发事件，这里负责转成
 * text/event-stream 写出去。因此领域层永远不会有 res.write 这种代码。
 *
 * 两类消息：
 *   - 增量：run:start / run:update / run:output / run:end / queue
 *   - 全量：state（50ms 合并窗口，避免批量导入/级联删除时把事件循环打满）
 */
function createSseHub({ bus, cfg, buildState, queueSnapshot, log }) {
  const logger = log || console;
  const clients = new Set();
  let stateTimer = null;

  function frame(payload) {
    return 'data: ' + JSON.stringify(payload) + '\n\n';
  }

  function broadcast(payload) {
    if (!clients.size) return;
    const data = frame(payload);
    for (const client of clients) {
      try { client.write(data); } catch (e) { clients.delete(client); }
    }
  }

  function attach(req, res) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no'
    });
    res.write('retry: 3000\n\n');
    res.write(frame({ type: 'state', data: buildState() }));
    clients.add(res);

    const hb = setInterval(() => {
      try { res.write(': ping\n\n'); } catch (e) { /* 由 close 清理 */ }
    }, cfg.limits.ssePingMs);
    if (hb.unref) hb.unref();

    const close = () => { clearInterval(hb); clients.delete(res); };
    req.on('close', close);
    req.on('error', close);
  }

  function scheduleState() {
    if (stateTimer) return;
    stateTimer = setTimeout(() => {
      stateTimer = null;
      try {
        broadcast({ type: 'state', data: buildState() });
      } catch (e) {
        logger.error('[sse] state build failed:', e && e.message);
      }
    }, cfg.limits.stateDebounceMs);
    if (stateTimer.unref) stateTimer.unref();
  }

  onSafe(bus, 'state:dirty', scheduleState);
  onSafe(bus, 'queue:dirty', () => broadcast({ type: 'queue', data: queueSnapshot() }));
  for (const topic of ['run:start', 'run:update', 'run:end']) {
    onSafe(bus, topic, (payload) => broadcast({ type: topic, run: payload.run }));
  }
  onSafe(bus, 'run:output', (payload) => broadcast({
    type: 'run:output', runId: payload.runId, stream: payload.stream, text: payload.text
  }));

  return {
    attach,
    broadcast,
    scheduleState,
    clientCount: () => clients.size,
    closeAll: () => { for (const c of clients) { try { c.end(); } catch (e) { /* ignore */ } } clients.clear(); }
  };
}

module.exports = { createSseHub };
