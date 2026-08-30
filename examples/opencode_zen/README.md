# OpenCode Zen — 委派本地 `opencode` CLI

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「用本地 opencode CLI 当推理后端」。

---

## 问题

框架的 `HttpLLMAgent` 直接调 HTTP 端点；但有些场景要委派给**本地 CLI 子进程**（如
opencode 的 agent 循环、联网检索、缓存）。旧做法没有现成的 runner 可挂。

## 方案

`OpenCodeAgentRunner`：把推理委托给 `opencode run` 子进程。边由 `script: zen_edge.py:OpenCodeEdge`
加载，该 Edge 在 `__init__` 自持 `OpenCodeAgentRunner`；runner 不注入 agent、无默认回退。

## 修改

- `examples/opencode_zen/zen_edge.py`：
  ```python
  class OpenCodeEdge(Edge):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.agent = OpenCodeAgentRunner()   # 自持，非注入
  ```
- `examples/opencode_zen/config.json`：`script: zen_edge.py:OpenCodeEdge`，
  `settings.prompt/model` 声明在 config。
- `examples/opencode_zen/run.py`：只加载图并 `Executor(graph)` 运行（不传 agent）。

## 测试

**测试方案**：opencode CLI 子进程被调用、返回结果流到下游。
**测试方法**：
```bash
python examples/opencode_zen/run.py
```
（要求 `opencode` CLI 在 PATH，配好 `HTTPS_PROXY` 或 CLI 自带代理。）
**测试结果**：`prompt_in -- e_zen --> zen_out` 管线跑通，`zen_out` 收到 CLI 结果。

### 同类示例
- `examples/real_pi/`：委派本地 `pi` CLI（`PiAgentRunner`）——同一模式的 pi 版。
- `examples/opencode_zen/proxy_demo.py`：HTTP agent 的传输代理演示（框架特性，与 CLI agent 无关）。