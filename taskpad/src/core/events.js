'use strict';

const { EventEmitter } = require('events');

/**
 * 进程内事件总线。
 *
 * 约定（防止循环依赖的关键）：
 *   - 领域层只管 emit，不知道谁在听；
 *   - SSE / trace / 图队列 都是订阅者；
 *   - 事件名固定为 `<域>:<动作>`，禁止订阅方反向调用领域层造成环。
 *
 * 事件清单：
 *   run:start   { run }
 *   run:update  { run }
 *   run:output  { runId, stream, text }
 *   run:end     { run }
 *   run:stale   { run }         心跳超时被回收
 *   queue:dirty {}
 *   state:dirty {}              任一实体变更，SSE 层合并后推全量
 */
function createBus() {
  const bus = new EventEmitter();
  bus.setMaxListeners(0); // 订阅者数量随前端/定时器增长，不设上限
  return bus;
}

/** emit 绝不让监听器异常打断调用方（领域层不应因为某个前端写失败而回滚） */
function emitSafe(bus, topic, payload) {
  try {
    bus.emit(topic, payload);
  } catch (e) {
    console.error('[bus] listener failed on ' + topic + ':', e && e.message);
  }
}

function onSafe(bus, topic, fn) {
  bus.on(topic, (...args) => {
    try {
      fn(...args);
    } catch (e) {
      console.error('[bus] handler failed on ' + topic + ':', e && e.message);
    }
  });
}

module.exports = { createBus, emitSafe, onSafe };
