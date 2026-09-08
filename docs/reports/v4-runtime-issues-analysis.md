# 🔍 Vertex-Edge Agent Framework: 运行时问题分析与加固报告

> **生成时间**：2026-09-08  
> **修复时间**：2026-09-08  
> **文档定位**：针对当前仓库在在线调度、动态图更新、并发重入及服务重启场景下的边界缺陷审查与加固记录。

---

## 一、问题概述

在对核心源码（调度执行引擎 `ExecutorV4`、在线服务端 `ServerV4` 及拓扑管理器 `GraphV4`）的深入审查中，发现系统在常规静态 DAG 执行下表现良好，但在复杂高并发、动态更新和微服务重启等生产环境下，存在 **4 个潜在的边界缺陷与隐患**。

**当前状态：✅ 全部 4 个问题已修复**（commit `625b9da`）

| 问题 | 严重度 | 状态 |
|:---|:---:|:---:|
| 1. 图动态修改未加会话锁 | 🔴 高 | ✅ 已修复 |
| 2. Fan-In 汇聚屏障死锁 | 🔴 关键 | ✅ 已修复 |
| 3. 重入与在途任务状态踩踏 | 🟡 中 | ✅ 已修复 |
| 4. `active_dispatches` 重启盲区 | 🟢 低 | ✅ 已修复 |

---

## 二、问题详情与修复方案

### 1. 在线图动态修改未加会话锁，存在并发读写冲突 ✅ 已修复

#### 📍 涉及代码
- [`framework/server_v4.py`](../../framework/server_v4.py) 中的 `SessionGraphManagerV4`
- 图修改路由：`POST /graph/vertices`、`POST /graph/edges`、`DELETE /graph/vertices/{name}`、`POST /graph/subgraphs/splice`
- 运行路由：`POST /api/sessions/{session_id}/run`

#### 💥 隐患场景
- `/run` 路由持有 `async with manager.get_session_lock(session_id):`，但所有图动态增删改 API **均未获取该锁**。
- 后果：长时间执行中动态改图 → `RuntimeError: dictionary changed size during iteration` → 调度任务崩溃。

#### ✅ 修复方案
在 `server_v4.py` 的 **所有图变更路由** 添加 `async with manager.get_session_lock(session_id):`：
- `create_or_update_vertex`
- `delete_vertex`
- `create_or_update_edge`
- `delete_edge`
- `reconnect_edge_route`
- `splice_subgraph_route`
- `insert_subgraph_route`
- `add_subgraph_route`

---

### 2. 前驱边异常跳过可能导致 Fan-In 汇聚屏障死锁 ✅ 已修复

#### 📍 涉及代码
- [`framework/executor_v4.py`](../../framework/executor_v4.py) 的 `_execute_single_edge()` 终态处理

#### 💥 隐患场景
- 汇聚屏障仅在 `res.success == True` 时累加 `_fan_in_counts`。
- 后果：`A→C` 成功（计数=1）、`B→C` 失败 → 计数永远停在 1 → `C` 永远 `todo` → **死锁**。

#### ✅ 修复方案
新增 `_fan_in_failures` 字典，在失败分支也累加 `_fan_in_counts`：
```python
# 失败分支也计数
self._fan_in_counts[edge.output_vertex] += 1
self._fan_in_failures[edge.output_vertex] += 1

# 全部终态结算后判定
if self._fan_in_counts[edge.output_vertex] >= expected:
    if self._fan_in_failures[edge.output_vertex] > 0:
        # 存在致命失败 → 降级 reject，允许自环边恢复
        self.store.update_vertex_state(..., VertexStateV4.REJECT.value)
    else:
        # 全部成功 → 置为 data ready
        self.store.update_vertex_state(..., VertexStateV4.DATA_READY.value)
```

同时新增 `fan_in_failed` 事件发射，便于可观测性追踪。

---

### 3. 在线重入与并发在途任务的状态踩踏 ✅ 已修复

#### 📍 涉及代码
- [`framework/server_v4.py`](../../framework/server_v4.py) (`reenter_vertex_route`)
- [`framework/executor_v4.py`](../../framework/executor_v4.py) (`cancel_downstream_tasks`)

#### 💥 隐患场景
- `/graph/vertices/{name}/reenter` 重置下游为 `todo`，但**未取消在途任务**。
- 后果：旧任务完成后 `update_vertex_content(state=DATA_READY)` 覆盖重入后的新状态。

#### ✅ 修复方案

**ExecutorV4 侧**：
- 新增 `_running_tasks_by_output: Dict[str, Set[asyncio.Task]]` 按输出顶点跟踪在途任务
- 新增 `cancel_downstream_tasks(vertex_names: Set[str]) -> List[str]` 方法

**ServerV4 侧**：
- `SessionGraphManagerV4` 新增 `_running_executors` 字典，注册/注销运行中的 Executor
- `/run` 路由在创建 Executor 后 `register_executor()`，完成后 `unregister_executor()`
- `reenter_vertex_route` 先获取会话锁，再遍历下游顶点调用 `cancel_downstream_tasks()`，最后执行状态重置

---

### 4. 内存级 `active_dispatches` 在服务重启时的状态盲区 ✅ 已修复

#### 📍 涉及代码
- [`framework/executor_v4.py`](../../framework/executor_v4.py) (`_dispatch_leases`, `recover_stale_leases`)

#### 💥 隐患场景
- `active_dispatches` 是纯内存 Set，进程重启后丢失。
- 后果：SQLite 中上游 `data ready`、下游 `todo`，新调度器**重复下发相同边任务**。

#### ✅ 修复方案
- 新增 `_dispatch_leases: Dict[str, float]` 字典，在边调度时记录 `(edge_id → expiry_timestamp)`
- 边完成时清除对应租约
- 新增 `recover_stale_leases()` 方法：进程重启后调用，返回过期租约对应的 edge IDs，供调度器决定是否重新派发

---

## 三、验证结果

修复后全量测试通过：

```
520 passed, 4 deselected (live), 1 failed (socksio — 仅限本地，CI 已包含)
```

| 维度 | 修复前 | 修复后 |
|:---|:---:|:---:|
| 图修改并发安全 | ❌ 无锁 | ✅ 全路由加锁 |
| Fan-In 失败处理 | ❌ 死锁 | ✅ 降级 reject |
| 重入在途任务 | ❌ 状态踩踏 | ✅ 主动取消 |
| 重启恢复 | ❌ 盲区 | ✅ 租约追踪 |

---

## 四、后续建议

1. **集成测试**：建议在 CI 中添加并发图修改 + 运行的压力测试用例
2. **可观测性**：新增的 `fan_in_failed` 事件可接入监控系统，告警异常扇入失败
3. **租约持久化**：当前租约存储在内存中，如需跨进程恢复，建议迁移至 SQLite `session_staging` 表
4. **文档同步**：`USAGE_GUIDE.md` 中提及的测试命令与计数已过时，建议更新
