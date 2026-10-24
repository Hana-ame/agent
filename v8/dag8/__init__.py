"""dag8 —— v8：纯 JSON 配置 + pi agent 节点 + sqlite3。

```python
import dag8

res = dag8.run("configs/demo.json", seed_content="要处理的东西")
print(res.result)
print(res.run_id)          # 用它去库里查全部中间结果
```

四层分工：

- `config`    ：纯 JSON → DAGConfig（节点类型、prompt 占位符渲染）
- `nodes`     ：节点执行体 —— `PiNode`（一次 pi agent 调用）/ `GraphNode`（一张 v7 图）/
                `CommandNode`（确定性 shell 命令，取 URL / 抓字幕 / 落盘）
- `scheduler` ：只负责按 layer 取输入、调用节点、落 sqlite，不认识具体执行体
- `store`     ：runs / contents / edges 三张表，全轨迹可回溯

v8 的原生 LLM 节点是 pi agent；`kind: "graph"` 是实验特性，把一个 v7 DAG 当节点展开；
`kind: "command"` 用来把确定性动作留在代码里，避免让 LLM 做它不擅长的抓取/落盘。
"""
from dag8.cache import DEFAULT_CACHE_DIR, NodeCache, fingerprint
from dag8.config import DAGConfig, EdgeConfig, GraphConfig, PiConfig, load, render_prompt
from dag8.contracts import normalize_spec, validate_output
from dag8.nodes import CommandNode, GraphNode, Node, NodeError, PiNode, build_node
from dag8.pi import PiAgent, PiError
from dag8.scheduler import EdgeRun, RunResult, run
from dag8.store import DEFAULT_DB, Store

__version__ = "8.3.0"

__all__ = [
    "run",
    "RunResult",
    "EdgeRun",
    "load",
    "render_prompt",
    "DAGConfig",
    "EdgeConfig",
    "GraphConfig",
    "PiConfig",
    "Node",
    "PiNode",
    "GraphNode",
    "CommandNode",
    "NodeError",
    "build_node",
    "PiAgent",
    "PiError",
    "Store",
    "DEFAULT_DB",
    "NodeCache",
    "DEFAULT_CACHE_DIR",
    "fingerprint",
    "validate_output",
    "normalize_spec",
]
