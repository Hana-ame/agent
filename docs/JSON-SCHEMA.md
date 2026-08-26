# Vertex-Edge Agent 框架配置指南（JSON Schema）

本文档详细解释 `config.json` 的每个字段，以及如何自定义 Vertex / Edge。
配套可直接运行的模板见仓库根目录的 [`config.template.json`](../config.template.json)。

---

## 1. 顶层结构

```jsonc
{
  "metadata": { ... },       // 元信息（仅描述，不参与执行）
  "vertices": [ ... ],       // 顶点列表
  "edges":    [ ... ]        // 边列表
}
```

- `vertices` / `edges` 均可为空，但引用必须完整（见「校验」）。
- 加载方式：`Graph.from_json("config.json")`，脚本相对路径会自动以 JSON 文件所在目录为基准解析。

---

## 2. `metadata` —— 元信息（可选）

```jsonc
{
  "name": "Pipeline Name",
  "description": "What this pipeline does"
}
```

仅用于说明，不参与执行逻辑。

---

## 3. `vertices[]` —— 顶点

每个顶点是一个"数据节点 + 状态机"：

```jsonc
{
  "id": "processor",              // 必填：唯一 ID，被边的 source/destination 引用
  "settings": { "min_length": 3 },// 可选：任意自定义配置，透传给脚本钩子
  "script": "path/to/vertex.py",  // 可选：外部脚本（on_receive / on_ready 钩子）
  "initial_data": [               // 可选：预置数据（源顶点用它喂初始内容）
    {
      "data_id": "text",          // 数据键
      "tags": ["raw"],            // 标签（与 data_id 一起定位数据）
      "value": "any string or json"  // 数据本体
    }
  ]
}
```

### 3.1 顶点类型约定（`settings.type`，约定俗成）
| type | 含义 |
|---|---|
| `"source"` | 无入边，靠 `initial_data` 提供数据 |
| `"process"` / `"transform"` | 中间处理节点 |
| `"merge"` | 汇聚多个入边的结果 |
| `"sink"` | 无出边，存放最终结果 |

> 这些只是 `settings` 里的约定，框架本身不强制。

### 3.2 `initial_data` 注意
- **不经过 `set()` / `on_receive`**，直接在构造时写入数据存储。
- 若初始数据也需要清洗/校验，请让数据先经过一条边的写入，或改用在 `set()` 中处理。

---

## 4. `edges[]` —— 边

每条边做一次「读取 → AI 处理 → 写入」：

```jsonc
{
  "id": "e1",                          // 必填：边唯一 ID
  "source": "input_a",                 // 必填：源顶点 ID（从此处 get 数据）
  "destination": "transform",          // 必填：目标顶点 ID（把结果 set 进去）
  "data_id": "text",                   // 可选：数据键，默认 "default"
  "tags": ["raw"],                     // 可选：标签列表，默认 []
  "prompt": "Summarize this:",         // 可选：发给 LLM 的指令
  "model": "DeepSeek-V4-Flash-0731-Event", // 可选：模型名，默认 "default"
  "settings": { "prefix": "[PRE]" },   // 可选：任意配置，透传给 agent 与脚本
  "script": "path/to/edge.py"          // 可选：外部脚本（pre_process / post_process）
}
```

### 4.1 数据定位：`(data_id, tags)` 就是"坐标"
- 顶点内部是一个字典：`(data_id, tuple(sorted(tags))) -> value`
- 边读数据：`source.get(data_id, tags)`；写数据：`dest.set(result, data_id, tags)`
- **tags 顺序无关**：`["b","a"]` 与 `["a","b"]` 是同一个键。
- 同一顶点可存多份数据（不同 `data_id`/`tags` 组合）。

### 4.2 一条边的完整执行流程
```
① data = source.get(data_id, tags)                 ← 读源
② [可选] pre_process(data, settings)                ← 脚本预处理
③ result = pi_agent.process(data, prompt, model, settings)  ← AI 处理
④ [可选] post_process(result, settings)             ← 脚本后处理
⑤ dest.set(result, data_id, tags)                   ← 写目标（目标计数+1，可能触发 READY）
```

---

## 5. 数据流与执行原理

```
 ① Executor 启动: 所有无入边顶点(源) → READY
 ② 扫描循环: 找 READY 顶点 → PROCESSING → 并发触发其所有出边(信号量限流)
 ③ 每条边按 4.2 执行 → 目标 set() 计数+1
 ④ 目标计数 >= required_input_count(入边数) → 目标 READY
 ⑤ 回到 ②，直到全部 DONE/ERROR；死锁/超时兜底
```

状态机：

```
 IDLE ─(所有入边数据到齐)→ READY ─(Executor拾取)→ PROCESSING ─(出边全成功)→ DONE
                              │                        └─(任一出边失败)→ ERROR
```

---

## 6. 顶点脚本钩子（`vertex["script"]`）

```python
def on_receive(data, data_id, tags, settings):
    """数据到达时调用。
    - 返回: 转换后的数据
    - 抛异常: 拒绝该数据 → 框架抛出 DataRejectedError，整条边失败
    """

def on_ready(all_data, settings):
    """所有输入到齐、出边触发前调用。
    返回 {(data_id, (tags,...)): value}，会被合并进该顶点的数据存储。
    常用于把多份输入合并成一份输出。
    """
    return {("result", ("merged",)): " | ".join(str(v) for v in all_data.values())}
```

---

## 7. 边脚本钩子（`edge["script"]`）

```python
def pre_process(data, settings):
    """AI 处理之前。返回加工后的数据。"""

def post_process(data, settings):
    """AI 处理之后。返回加工后的结果。"""
```

---

## 8. 完整示例（JSONC，带注释）

```jsonc
{
  "metadata": {
    "name": "My Pipeline",
    "description": "source -> process -> sink"
  },
  "vertices": [
    {
      "id": "source",
      "settings": { "type": "source" },
      "initial_data": [
        { "data_id": "text", "tags": ["raw"], "value": "Hello" }
      ]
    },
    {
      "id": "processor",
      "settings": { "min_length": 3 },
      "script": "examples/scripts/uppercase_handler.py"
    },
    {
      "id": "sink",
      "settings": { "type": "sink" }
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "source",
      "destination": "processor",
      "data_id": "text",
      "tags": ["raw"],
      "prompt": "Summarize this text:",
      "model": "DeepSeek-V4-Flash-0731-Event",
      "script": "examples/scripts/prefix_handler.py",
      "settings": { "prefix": "[IN]", "suffix": "[/IN]" }
    },
    {
      "id": "e2",
      "source": "processor",
      "destination": "sink",
      "data_id": "text",
      "tags": ["raw"],
      "prompt": "Restate the text:",
      "model": "DeepSeek-V4-Flash-0731-Event"
    }
  ]
}
```

> JSON 标准不支持注释，运行前请把 `//` 说明去掉。
> 可参考 `examples/simple/config.json`（合法 JSON）与 `examples/complex/config.json`。

---

## 9. 如何自定义 Vertex / Edge

### Vertex：外部脚本（推荐日常用）
JSON 里 `"script": "my_vertex.py"`，实现 `on_receive` / `on_ready`（见第 6 节）。
完整示例：`examples/scripts/uppercase_handler.py`、`examples/scripts/validator.py`。

### Vertex：继承 `Vertex` 类（需要额外状态/复杂逻辑）
```python
from framework.vertex import Vertex

class MyVertex(Vertex):
    def __init__(self, vid, **kw):
        super().__init__(vid, **kw)
        self.extra_state = []

    async def set(self, data, data_id="default", tags=None):
        data = transform(data)
        return await super().set(data, data_id, tags)  # 必调

    async def prepare_outputs(self):
        await self._store({"total": 3}, "stats", ["summary"])  # 直写，别用 self.set()
        await super().prepare_outputs()
```
完整可运行示例：`examples/custom_vertex/`。

### Edge：外部脚本（推荐日常用）
JSON 里 `"script": "my_edge.py"`，实现 `pre_process` / `post_process`（见第 7 节）。
完整示例：`examples/scripts/prefix_handler.py`。

### Edge：继承 `Edge` 类（完全自定义）
```python
from framework.edge import Edge

class MyEdge(Edge):
    async def execute(self, src, dst, agent):
        raw = await src.get(self.data_id, self.tags)
        r1 = await agent.process(raw, "step1", self.model, self.settings)
        r2 = await agent.process(r1, "step2", self.model, self.settings)
        await dst.set(r2, self.data_id, self.tags)
        self.completed, self.result = True, r2
        return r2
```

> 注意：自定义 Vertex/Edge 子类**不能**通过 `Graph.from_json` 构造（它只建内置类），
> 需手动组装图并登记边的关联关系（见 `examples/custom_vertex/run.py` 的 `add_edge`）。

---

## 10. 常见问题

| 问题 | 原因 / 解决 |
|---|---|
| 顶点永远不 READY | 入边数据没到齐：检查 `required_input_count` 与各边是否成功 `set` |
| 源顶点没数据 | `initial_data` 直接写入、不经过 `set()`；确认 `data_id`/`tags` 与边一致 |
| 数据取不到（返回 None） | 边的 `data_id`/`tags` 与源顶点写入时不一致 |
| 脚本不生效 | 检查 `script` 路径（相对 JSON 目录）、钩子函数名拼写 |
| `Graph validation failed` | 引用了不存在的顶点 ID，或图存在环（必须 DAG） |
| 死锁 / 超时 | 某顶点入边未到齐且无新 READY；检查是否有边指向它却没被触发 |
| 想用真实模型 | `PICLIPIAgent`(pi CLI) 或 `OpenCodeAgent`(opencode CLI)，见 `scripts/run_graph.py` |

---

## 9.5 Agent 后端（多实现）

框架的 `PIAgent` 是抽象接口，下面是它的各种后端实现，可自由切换：

| Agent | 后端 | 用途 |
|---|---|---|
| `MockPIAgent` | 无（纯模拟） | 调试拓扑/脚本，确定性、秒回 |
| `OpenCodeAgent` | opencode CLI（`opencode run`） | **默认后端**，模型默认 `opencode-zen/hy3-free`(免费) |
| `PICLIPIAgent` | pi CLI（`pi --print`） | 真实 LLM（pi 配置的 provider/model） |
| `ExternalPIAgent` | 第三方 `pi_agent` Python 包 | 预留接口 |

> 注意：构造器显式指定的 `model` 优先级高于 config 边级 `model`，
> 因此切后端时只需在构造器里传入对应后端的模型名。

使用 `scripts/run_graph.py` 可一行切换后端（**默认后端 = opencode-zen/hy3-free**）：

```bash
python3 scripts/run_graph.py my_config.json                       # 默认: opencode + opencode-zen/hy3-free
python3 scripts/run_graph.py my_config.json --agent mock          # Mock
python3 scripts/run_graph.py my_config.json --agent pi --model DeepSeek-V4-Flash-0731-Event   # pi CLI
python3 scripts/run_graph.py my_config.json --agent opencode --model opencode/hy3-free
```

---

## 参考

- 简单示例：`examples/simple/config.json`
- 复杂示例：`examples/complex/config.json`
- 自定义顶点：`examples/custom_vertex/`
- 可运行模板：`../config.template.json`
