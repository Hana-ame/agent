# Complex 示例 — 多源 Fan-out / Fan-in + 外部子类

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「多输入并发 + 汇聚依赖 + 外部子类挂接」。

---

## 问题

框架需要一个「中等复杂度」示例覆盖三个能力，旧文档只在最大示例里零散体现：
1. **多源并发**：两个输入节点同时提供初始数据；
2. **Fan-out + Fan-in**：一个节点数据分到多条边并行，汇聚节点等齐所有入边；
3. **外部子类挂接**：数据变换逻辑放在外部 `.py` 脚本，而非内联 JSON/顶层 hook。

## 方案

**Topology**：

```
input_a ─e1─▶ transform ─e3─▶ merge ─e5─▶ output
input_a ─e4─────────┬───────────▶ merge
input_b ─e2─────────▶ transform
```

- `input_a`/`input_b`：双源。
- `transform`：挂 `script: ../scripts/uppercase_handler.py`（`UpperVertex` 子类），
  `on_receive` 转大写。
- `merge`：汇聚节点（依赖 `e3` + `e4` 都到齐才 READY —— Settlement Barrier）。
- `e3`：挂 `script: ../scripts/prefix_handler.py`（`PrefixEdge` 子类），
  `pre_process` 加 `[PRE]` 前缀、`post_process` 加 `[POST]` 后缀。

## 修改

- `examples/complex/config.json`：4 顶点 + 5 边拓扑；`script` 引用 `../scripts/*.py`。
- `examples/scripts/uppercase_handler.py`：`UpperVertex(Vertex)`（`on_receive`/`on_ready`）。
- `examples/scripts/prefix_handler.py`：`PrefixEdge(Edge)`（`pre_process`/`post_process`）。
- 没有顶层 hook 函数；所有自定义逻辑都在子类方法里。

## 测试

**测试方案**：双源并发、transform 大写、merge 等齐双入边、prefix 前后缀生效。
**测试方法**：
```bash
python examples/run.py examples/complex/config.json
```
**测试结果**：
- `input_a` 与 `input_b` 并发进入，`transform` 收到两条数据并转大写；
- `e4`（`input_a→merge`）与 `e3`（`transform→merge`）都到达后 `merge` 才 READY
  （锁无关并发同步，经 `EdgeSignal` barrier 实现）；
- `e5` 输出带 `[PRE]`…`[POST]`（prefix 子类生效），`output` 汇聚后 DONE。

> 该示例用于回答「外部子类是脚本挂接的推荐形式」；`custom_classes` 是它的精简版。