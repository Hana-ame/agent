# Edge Agent 选择问题

## 背景

framework 当前对 LLM agent 的处理方式:

- `Executor.__init__` 接受一个全局 `agents` 参数,默认 `MockAgent()`。`Executor` 持有 `self.agents`,在执行每条 edge 时把它传给 `edge.execute()`。
- `Edge` 类**不持有 agent**。`Edge.execute(self, source_vertex, dest_vertex, agents, **kwargs)` 只有一行委托:`return await self._pipeline.run(..., agents=agents, **kwargs)`。
- 因此所有 edge 共用同一个全局 agent 实例。

## 问题

agent 类型的选择(MockAgent / HttpLLMAgent / PiAgentRunner)只能靠调用方字符串匹配 config 路径:

```python
# examples/run.py
from framework import MockAgent, HttpLLMAgent
from framework.agents import PiAgentRunner
if "real_llm" in config_path:
    agent = HttpLLMAgent()
elif "real_pi" in config_path:
    agent = PiAgentRunner()
else:
    agent = MockAgent()
```

## 为什么是问题

1. **不可移植**:agent 配置没有跟着 config 走。同一份 config 换个运行入口(不经过 `run.py`,或 `run.py` 路径写法变了),agent 类型就变了。配置不自描述。

2. **不可 per-edge**:Executor 只持有一个全局 agent,所有 edge 共用。无法让"edge A 用 mock、edge B 用 http、edge C 用 pi"。

3. **违背 config 自描述原则**:`config.json` 的 edge 块里有 `prompt`、`model`、`settings` 给 agent 用,却**没有"用哪个 agent 类型"的字段**。

4. **设计与实现错位**:Edge 的"agent 配置"属性组把 `prompt`/`model` 当一等公民,却把 agent 选择权留给了调用方。Edge 框架有完备的 hook / 子类 / script 扩展机制,唯独缺"per-edge 声明 agent 类型"的能力。

## 根因

Edge 设计上把 prompt/model 当一等公民,把 agent 选择权留给了调用方。Edge 缺少:

- 一个 `agent` 配置字段(声明该 edge 用哪种 agent)
- 一个 agent 类型字符串 → 类的映射(registry)

## 涉及代码

- `framework/executor.py`:`self.agents = agents or MockAgent()`(全局);`edge.execute(src, dst, self.agents, ...)`(注入)
- `framework/edge.py`:`Edge.__init__` 无 `agent` 参数;`Edge.execute` 仅透传 `agents`
- `framework/pipeline.py`:`result = await agents.process(data=..., prompt=..., model=..., settings=...)`
- `examples/run.py`:路径字符串匹配选择 agent

## 期望方向(待定)

让 agent 选择下沉到 Edge / config 层面:

- 给 `Edge.__init__` 加 `agent` 参数,Edge 持有 `self._agent`
- 在 `framework/agents.py` 加 agent registry + `make_agent(name)` 工厂
- `graph.py` 加载 edge 时从 `ec.get("agent")` 读取
- `Executor` 优先用 `edge._agent`,否则 fallback 全局 `self.agents`
- `run.py` 不再路径匹配
