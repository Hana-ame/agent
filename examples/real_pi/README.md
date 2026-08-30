# Real Pi — 委派本地 `pi` CLI 子进程

> 按「问题 / 方案 / 修改 / 测试」记录：这个实验解决「用本地 pi CLI 当推理后端」——\n> 与 `real_llm`（HTTP 端点）相对，是 CLI 子进程路线。

---

## 问题

`real_llm` 直接走 `HttpLLMAgent`（HTTP 端点）；但部署环境可能标准化到本地 `pi` CLI\n（AgentCLI），需要把推理委派给子进程而不是 HTTP。

## 方案

`PiAgentRunner`：把推理委托给 `pi -p --model ... --system-prompt ... -- <data>` 子进程。\n边由 `script: pi_edge.py:PiEdge` 加载，Edge 在 `__init__` 自持 `PiAgentRunner`；\nrunner 不注入 agent、无默认回退。

## 修改

- `examples/real_pi/pi_edge.py`：\n  ```python\n  class PiEdge(Edge):\n      def __init__(self, *args, **kwargs):\n          super().__init__(*args, **kwargs)\n          self.agent = PiAgentRunner(...)   # 自持，非注入\n  ```\n- `examples/real_pi/config.json`：`script: pi_edge.py:PiEdge`，`settings` 声明 prompt/model。\n- 执行时经 `examples/run.py` 加载图并 `Executor(graph)` 运行。

## 测试

**测试方案**：pi CLI 子进程被调用、标准输出成为边结果。\n**测试方法**：\n```bash\npython examples/run.py examples/real_pi/config.json\n```\n（要求 `pi` CLI 在 PATH。）\n**测试结果**：`user_input -- e_real_pi (PiAgentRunner) --> pi_output`，`pi_output` 收到\nCLI 返回；失败时发 `EdgeSignal.FAILED` 并重抛。

> 与 `opencode_zen`（`OpenCodeAgentRunner`）同模式的另一种 CLI 后端。