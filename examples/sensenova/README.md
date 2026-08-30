# SenseNova — 免代理直连真实端点

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「免费 SenseNova 端点直连（无需代理）」——
> 与 `real_llm`（经传输代理）相对。

---

## 问题

`real_llm` 通过 `https_proxy` 出网；但有些端点（如 SenseNova → https://token.sensenova.cn）
**直连可达**，不需要任何代理。旧做法一刀切走 proxy 环境变量，反而绕路。

## 方案

`script: sensenova_edge.py:SensenovaEdge` 加载 Edge，Edge 在 `__init__` 自持 `HttpLLMAgent`，
`base_url`/`model` 在 `settings` 声明，API key 从 `SENSENOVA_API_KEY` 环境变量读取。
不注入 agent、无默认回退、无 proxy 依赖。

## 修改

- `examples/sensenova/sensenova_edge.py`：`SensenovaEdge` 在 `__init__` 自持 agent，
  `base_url=https://token.sensenova.cn`、`model=sensenova-6.8-flash-lite`。
- `examples/sensenova/config.json`：`script: sensenova_edge.py:SensenovaEdge` + settings。

## 测试

**测试方案**：无 `HTTPS_PROXY` 也能直连出结果。
**测试方法**：
```bash
export SENSENOVA_API_KEY=sk-...   # 或读取 ~/.config/opencode/opencode.json
python examples/run.py examples/sensenova/config.json
```
**测试结果**：`user_input -- e_sensenova (SensenovaEdge) --> sensenova_output`，直连返回。
无需任何 proxy 环境变量。