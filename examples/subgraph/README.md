# Subgraph — 嵌套子图（multi-agent 团队）

> 按「问题/方案/修改/测试」记录：解决「把一整张独立图封装成父图的一个节点」。v3.0 特性。

## 问题

多 agent 团队协作时希望「整支团队」作为一个可复用、可隔离的节点嵌进父流程，带输入/输出边界映射，而不是把团队内部的顶点全部摊平到父图。

## 方案

`SubgraphVertex`：一个顶点封装一张独立图（`settings.graph_config` 指向子图 config 或内嵌 dict）。边界用 `input_map`/`output_map` 翻译；事件冒泡为 `subgraph_*`；checkpoint 命名空间隔离。

## 修改

- `framework/subgraph.py`：`SubgraphVertex`（已核实存在）。
- `examples/subgraph/demo.py`（已核实存在）：父图 import 一个 `research_team.json` 子图，边界路由。

## 测试

**测试方案**：子图输入/输出正确翻译、事件冒泡、边界数据可达。**测试方法**：
```bash
python examples/subgraph/demo.py
```
**测试结果**：父图调用子图团队 → `input_map` 注入 → 内部多 agent 协作 → `output_map` 汇出 → 父图继续；`tests/test_subgraph.py` 覆盖嵌套、事件冒泡、checkpoint 命名空间。