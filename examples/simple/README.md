# Simple 示例 — 最小 3 节点串行管线

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「框架最小可运行形态长什么样」。

---

## 问题

新用户需要一个最简拓扑来理解 `Vertex → Edge → Vertex` 的数据流动，而不是先面对 fan-out/
条件路由/子图等高级概念。旧文档一上来就讲大图，认知负担大。

## 方案

建 3 节点串行图 `input → processor → output`，每条边一个 LLM 步骤：
- `input`：注入初始数据（`initial_data`），无入边 → 自动 `READY`。
- `processor` / `output`：普通中间/汇聚节点。
- `e1` / `e2`：标准 `Edge`，把数据过一遍 LLM（框架默认 MockAgent 前缀 `[hy3-free]`）。

## 修改

- `examples/simple/config.json`：3 顶点 + 2 边拓扑。
- 无自定义脚本——纯 config 驱动。

## 测试

**测试方案**：验证无入边节点自动 READY、数据逐边流动、汇聚节点结束。
**测试方法**：
```bash
python examples/run.py examples/simple/config.json
```
**测试结果**：`input` READY → `e1` MockAgent 前缀 `[hy3-free]` → `processor` →
`e2` 再前缀 → `output` 汇聚 → 全图 `DONE`。无自定义脚本即可跑通。

## 数据流（测试轨迹）

```
input(RDY) --e1--> processor(RDY) --e2--> output(RDY) --settle--> DONE
```

> 该示例同时用于 README「快速开始」的最小入口。