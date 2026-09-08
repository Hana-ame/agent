# 🔍 Vertex-Edge Agent Framework v4.0: 运行时热破（Hot-Breakage）问题分析与防范规范

> **生成时间**：2026-09-08  
> **文档定位**：针对系统在在线调度、动态热修改（Hot Mutation）、并发热重入（Hot Re-entry）及热重启（Hot Reload）场景下的隐患审查与加固建议。

---

## 一、什么是“热破”（Hot Breakage）？

在 Vertex-Edge Agent Framework 的 DAG 执行引擎中，“热破”指**系统处于在线运行状态、并发执行状态或热动态更新时，由于拓扑结构变更、异常跳过、重入踩踏或状态丢失，导致调度器出现死锁、字典并发修改异常、状态污染或重复执行的边界缺陷**。

---

## 二、当前发现的 4 大核心热破问题

### 1. 在线热修改未加锁引发并发读写冲突 (Hot Mutation Race Condition)

#### 📍 涉及代码
- [`framework/server_v4.py`](framework/server_v4.py) 中的 `SessionGraphManagerV4`
- 图修改路由：`POST /api/sessions/{session_id}/graph/vertices`、`POST /graph/edges`、`DELETE /graph/vertices/{name}`、`POST /graph/subgraphs/splice`
- 运行路由：`POST /api/sessions/{session_id}/run`

#### 💥 隐患场景
- 当客户端调用 `/run` 运行工作流时，服务端获取了 `async with manager.get_session_lock(session_id):`。
- 但是，所有的在线图动态增删改 API **均未获取该 `session_lock`**。
- **热破后果**：如果在长时间执行（如真实 SenseNova 大模型推理、长耗时批处理）的会话中，外部调用 API 或在 Web 仪表盘动态增删节点/连线：
  - `ExecutorV4` 调度主循环正在遍历 `self.graph.edges.values()`；
  - 在线修改直接在后台对字典进行结构性修改；
  - 立即引发 Python 的 **`RuntimeError: dictionary changed size during iteration`**，导致调度任务意外崩溃。

---

### 2. 前驱边异常跳过导致 Fan-In 汇聚屏障永久死锁 (Fan-In Settlement Deadlock)

#### 📍 涉及代码
- [`framework/executor_v4.py:298-307`](framework/executor_v4.py#L298-L307)
  ```python
  if res.success:
      if not edge.is_reflexive and not isinstance(edge, ReflexiveEdgeV4):
          self._fan_in_counts[edge.output_vertex] += 1
          expected = len([e for e in self.graph.get_incoming_edges(edge.output_vertex)...])
          if self._fan_in_counts[edge.output_vertex] >= expected:
              self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.DATA_READY.value)
              self._fan_in_counts[edge.output_vertex] = 0
          else:
              self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.TODO.value)
  ```

#### 💥 隐患场景
- 目前汇聚屏障仅在 `res.success == True` 的分支累加 `_fan_in_counts`。
- **热破后果**：
  - 假设顶点 `C` 有两个前驱入边 `A -> C` 和 `B -> C`（`expected = 2`）。
  - 若 `A -> C` 成功（计数置为 1，状态维持在 `todo`）；
  - 若前驱 `B` 节点抛出致命异常，导致 `B -> C` 边执行失败（`res.success == False`）或被握手逻辑跳过（`res.skipped == True`）；
  - `_fan_in_counts["C"]` 将永远停留在 `1`，永远无法达到 `2`；
  - 目标节点 `C` 永远停留在 `todo`，后续调度器检测不到可调度的边，直接判定为 **Deadlock（死锁）**，任务挂起直至超时。

---

### 3. 热重入与并发在途任务的状态踩踏 (Hot Re-entry State Stamping)

#### 📍 涉及代码
- [`framework/server_v4.py:370-388`](framework/server_v4.py#L370-L388) (`reenter_vertex`)
- [`framework/graph_v4.py:605-671`](framework/graph_v4.py#L605-L671) (`reset_affected_vertices`)

#### 💥 隐患场景
- `/graph/vertices/{name}/reenter` 用于在不重启 Session 的前提下重新触发某节点的重算。重入时系统会将下游所有受影响的顶点重置为 `TODO`。
- **热破后果**：
  - 如果调用重入的瞬间，下游某些节点对应的边正在后台协程中运行（In-flight 任务）；
  - 重入刚把下游顶点重置为 `TODO`；
  - 紧接着在途任务执行完成，调用 `update_vertex_content(state=DATA_READY)`；
  - **旧任务的输出直接覆盖了重入后的新状态**，造成数据竞态与拓扑执行序列混乱。

---

### 4. 内存级 `active_dispatches` 在热重启时的状态盲区 (Hot Reload Amnesia)

#### 📍 涉及代码
- [`framework/executor_v4.py:130`](framework/executor_v4.py#L130)
- [`v4-architecture-spec.md:22`](v4-architecture-spec.md#L22)
  > *"No `running` state in database: In-flight task tracking is deduplicated in memory via `active_dispatches: Set[Tuple[str, str, str]]`."*

#### 💥 隐患场景
- 为了减少对 SQLite 的高频写入，规范设计为仅在内存 Set `active_dispatches` 中记录执行中的任务。
- **热破后果**：
  - 在长时间任务运行期间，若服务进程发生热重启（Uvicorn Reload、Worker 重建、容器漂移）；
  - 内存中的 `active_dispatches` 丢失；
  - 此时 SQLite 中上游依然为 `data ready`，下游依然为 `todo`；
  - 新重启的调度器无法感知旧任务是否还在执行，会**无条件重复下发相同的边任务**，造成外部 API 双重调用或重复扣费。

---

## 三、代码级加固方案建议

1. **图修改互斥锁 (Session Graph Lock)**：
   在 `server_v4.py` 的所有图动态变更端点添加会话锁保护，避免在运行中发生字典并发修改：
   ```python
   async with manager.get_session_lock(session_id):
       manager.add_or_update_vertex(...)
   ```

2. **终态结算屏障 (Settled Barrier)**：
   将 `_fan_in_counts` 升级为记录“终态数量（成功 + 失败 + 跳过）”。当所有入边全部终态结算后：
   - 若全部成功：目标顶点置为 `data ready`；
   - 若存在未恢复的致命失败：目标顶点安全置为 `reject`，防止死锁挂死。

3. **热重入主动取消在途任务 (Cancel In-Flight on Reentry)**：
   在执行 `reenter_vertex` 时，关联 `ExecutorV4._running_tasks`，主动对处于下游依赖链上的协程调用 `task.cancel()`，避免旧结果污染。

4. **轻量租约标记 (Task Lease in Staging)**：
   边调度开始时在 `session_staging` 中记录租约 Token 与有效截止时间，即使进程重启也能识别在途任务，实现状态恢复与防重放。
