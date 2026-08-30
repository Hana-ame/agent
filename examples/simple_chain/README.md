# Simple Chain — 免 JSON 的程序化拓扑

> 按「问题 / 方案 / 修改 / 测试」记录：解决「不想手写 JSON 时怎么建最小 A→B→C 图」。

## 问题

手写 `config.json` 建串行图冗余（metadata + vertices + edges + settings）。对 3 节点串行这种
最常用形态，应有一行式 API。

## 方案

`LinearChain.build(prompts: List[str]) -> Graph`：`prompts` 长度 = 边数，自动生成
A→B→C… 拓扑与逐边 prompt。

## 修改

- `framework/builders/chain.py`：`LinearChain.build(prompts)`（已核实存在）。
- `examples/simple_chain/demo.py`：`LinearChain.build(["Step1", "Step2"])` → `Executor(graph)`。

## 测试

**测试方案**：prompts 自动生成 N+1 节点、N 边。**测试方法**：
```bash
python examples/simple_chain/demo.py
```
**测试结果**：`A→B→C` 图生成并跑通（框架 `tests/test_improvements.py` 覆盖 `LinearChain`）。