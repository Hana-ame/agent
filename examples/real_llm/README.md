# 接驳真实大模型接口示例 (Real LLM Endpoint)

本示例展示了如何通过覆写 `Edge` 的子类，完全架空框架自带的测试用 `MockPIAgent`，从而直接向真实的外部大模型服务商发起请求。

不同于普通的 `pre_process`（修改 Prompt）或 `post_process`（修改返回结果），`RealLLMEdge` 彻底重构了内部的工作流。它通过内置的 `urllib` 库配合 `asyncio.to_thread` 向兼容 OpenAI 的端点（例如 `https://opencode.ai/zen/v1/chat/completions`）发起真实的 HTTP POST 异步网络请求，请求所使用的模型由 `config.json` 动态指定。

## 运行原理 (How it works)

1. 在 `config.json` 中，边 `e_real_llm` 被配置为使用外部扩展 `"script": "llm_edge.py"`。
2. 框架加载 `llm_edge.py` 并在构建图时，自动用 `RealLLMEdge` 子类替换默认的 `Edge` 类。
3. 当调度器 (Executor) 激活这条边时，不仅不会使用自带的 PI Agent，反而会执行我们在子类中重写的定制化调用逻辑。
4. 这个调用逻辑包括：从上游 Vertex 读取数据、拼接 JSON 请求体、发起非阻塞式 HTTP 请求、解析并提取大模型的回复、并通过 `handle_edge_signal` 完整写入到下游目标 Vertex。

## 运行示例 (Execution)

使用统一运行脚本，指向本目录的 `config.json`：

```bash
python examples/run.py examples/real_llm/config.json
```
