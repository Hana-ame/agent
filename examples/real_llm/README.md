# Real LLM — 真实端点 + 传输代理

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「真实 LLM 请求经 HTTP(S) 传输代理出去」。

---

## 问题

框架默认 `MockAgent` 只做测试用；真实业务要调真实 LLM 端点。且部署环境常要求请求走
**传输代理**（公司出口/Clash 等），旧示例依赖环境变量 `HTTPS_PROXY`，不好复现。

## 方案

- `script: llm_edge.py:HttpLLMEdge` 加载 Edge，Edge 在 `__init__` 自持 `HttpLLMAgent`；
- `base_url`（完整 URL，含路径）与 `https_proxy` 都在 `settings` 里声明，**显式优先，\n  无环境变量依赖**；config 里设了 proxy 就覆盖环境 `HTTP_PROXY/HTTPS_PROXY`。

## 修改

- `examples/real_llm/llm_edge.py`：
  ```python
  class HttpLLMEdge(Edge):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.agent = HttpLLMAgent(
              base_url=self.settings.get("base_url", "https://opencode.ai/zen/v1"),
              proxy=self.settings.get("https_proxy"),
          )
  ```
- `examples/real_llm/config.json`：`settings.base_url` + `settings.https_proxy`（如\n  `http://127.0.1.6:7890`，Clash 风格出口）。

## 测试

**测试方案**：`https_proxy` 覆盖环境变量、请求真实出网并返回 LLM 结果。
**测试方法**：
```bash
python examples/run.py examples/real_llm/config.json
```
（把 `127.0.1.6:7890` 换成你的 Clash 出口 `127.0.{1,2,3}.{4,6}:7890` 或自有代理。）
**测试结果**：`user_input -- e_real_llm (HttpLLMAgent) --> llm_output`，`llm_output` 收到\n真实端点返回；去掉 `https_proxy` 时 `trust_env=True` 回退到环境 `HTTP_PROXY/HTTPS_PROXY`。