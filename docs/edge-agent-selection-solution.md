# Edge Agent 选择与配置重构方案设计

## 1. 问题背景与现状

在现有框架中，LLM 计算的执行由 `Edge` 的第 4 阶段（Compute）负责，但关于 Agent 实例的选择和生命周期管理存在以下设计缺陷：

1. **全局单例与缺乏细粒度控制（No Per-Edge Agent）**：`Executor` 在初始化时接收一个全局 `agents` 参数，每条 Edge 在执行时强制使用此全局 Agent，无法在同一张图中为不同 Edge 指定不同的 Agent 实现（如 Edge 1 使用 `HttpLLMAgent`，Edge 2 使用 `PiAgentRunner`，Edge 3 使用 `MockAgent`）。
2. **配置不自描述（Non-self-describing Config）**：`config.json` 的 Edge 节点中定义了 `prompt`、`model`、`settings`，却缺失了 `agent` 类型字段。
3. **入口硬编码与路径匹配**：`examples/run.py` 依赖判断配置文件路径字符串（如 `if "real_llm" in config_path`）来选择 Agent，严重破坏了可移植性与模块封装性。

---

## 2. 方案总体设计

本方案的目标是将 **Agent 的解析与绑定下沉到 Edge 级别**，并建立规范的 **Agent 注册表（Agent Registry & Factory）**，同时保持对既有 API 的完全向后兼容。

### 核心架构改造

```
                  ┌───────────────────────────────┐
                  │      AgentRegistry (Factory)  │
                  │   "mock"     -> MockAgent     │
                  │   "http"     -> HttpLLMAgent  │
                  │   "pi_agent" -> PiAgentRunner │
                  └───────────────┬───────────────┘
                                  │ creates / resolves
                                  ▼
┌──────────────┐          ┌───────────────┐
│ config.json  │ ───────▶ │   Edge /      │ (holds optional self.agent)
│ "agent": ... │          │  EdgePipeline │
└──────────────┘          └───────┬───────┘
                                  │
                                  ▼
                      Stage 4: Compute Phase
                      agent = self.agent or fallback_agents
                      await agent.process(...)
```

---

## 3. 详细改造步骤与模块设计

### 3.1 `framework/agents.py`: 建立 AgentRegistry 与工厂方法

1. 维护全局注册表 `_AGENT_REGISTRY: Dict[str, Type[BaseAgent]]`。
2. 提供注册装饰器 `@register_agent(name)` / `register_agent_class(name, cls)`。
3. 默认注册内建 Agent：
   - `"mock"` / `"default"` -> `MockAgent`
   - `"http"` / `"httpllm"` / `"real_llm"` -> `HttpLLMAgent`
   - `"pi"` / `"pi_agent"` / `"real_pi"` -> `PiAgentRunner`
4. 提供工厂解析方法 `get_agent(agent_spec: Union[str, BaseAgent, Dict, None]) -> Optional[BaseAgent]`：
   - 若传入 `BaseAgent` 实例，直接返回；
   - 若传入 `str`，根据名称从注册表查找并实例化；
   - 若传入 `dict`（如 `{"type": "http", "base_url": "..."}`），实例化并注入参数。

### 3.2 `framework/edge.py` & `framework/pipeline.py`: Edge 支持持有 Agent

1. **`Edge.__init__`** 增加 `agent: Optional[Union[str, BaseAgent, Dict]] = None` 参数。
2. **`EdgePipeline`** 增加 `self.agent = get_agent(agent)`。
3. **`EdgePipeline.run(..., agents=None, **kwargs)`** 的 Compute 阶段解析执行 Agent：
   ```python
   # 优先使用当前 Edge 配置的 Agent，否则回退到 Executor 传入的全局 agents，最后回退到 MockAgent()
   active_agent = self.agent or agents or MockAgent()
   result = await active_agent.process(
       data=data,
       prompt=current_prompt,
       model=self.model,
       settings=self.settings,
   )
   ```

### 3.3 `framework/graph.py`: JSON 声明式解析

在解析 `edges` 配置项时，提取 `ec.get("agent")` 并传递给 `Edge` 构造函数：

```python
# config.json
{
  "edges": [
    {
      "id": "e_extract",
      "source": "v1",
      "destination": "v2",
      "agent": "http",        // 支持字符串注册名，或包含参数的字典配置
      "prompt": "Extract JSON",
      "model": "gemini-1.5-flash"
    }
  ]
}
```

### 3.4 `framework/executor.py` & `examples/run.py`: 解耦与清理

1. **`Executor`**：`agents` 参数变为可选（默认为 `None`），若 Edge 自身声明了 Agent，则直接使用 Edge 独立的 Agent 实例。
2. **`examples/run.py`**：移除基于路径关键词匹配 Agent 的临时逻辑，改为：
   - 直接通过配置自描述运行 `Executor(graph)`；
   - 若图内所有 Edge 均未指定 Agent 且调用方也未传入，默认使用 `MockAgent()`。

---

## 4. 兼容性与迁移保证

1. **向后兼容**：调用方显式使用 `Executor(graph, agents=custom_agent)` 的旧代码行为保持完全一致（当 Edge 未配置 `agent` 时，自动 Fallback 至该全局 `agents`）。
2. **渐进式升级**：既有的 `config.json` 如果没有 `"agent"` 字段，不会报错，平滑走默认回退流程。
