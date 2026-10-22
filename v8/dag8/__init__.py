"""dag8 —— v8：JSON 配置驱动、edge 由 pi agent 执行、内容落 sqlite3。

```python
import dag8

res = dag8.run("configs/demo.json", seed_content="要处理的东西")
print(res.result)
print(res.run_id)          # 用它去库里查全部中间结果
```

三层分工：

- `config`：纯 JSON → DAGConfig（含 prompt 占位符渲染）
- `pi`    ：一条 prompt → subprocess 调 `pi -p`（支持本机 / SSH 到 WSL）
- `store` ：runs / contents / edges 三张表，全轨迹可回溯

`engine.run()` 把它们串起来：按 layer 跑边，缺输入就跳过，出错就记 error 并停下。
"""
from dag8.config import DAGConfig, EdgeConfig, PiConfig, load, render_prompt
from dag8.engine import EdgeRun, RunResult, run
from dag8.pi import PiAgent, PiError
from dag8.store import DEFAULT_DB, Store

__version__ = "8.0.0"

__all__ = [
    "run",
    "RunResult",
    "EdgeRun",
    "load",
    "render_prompt",
    "DAGConfig",
    "EdgeConfig",
    "PiConfig",
    "PiAgent",
    "PiError",
    "Store",
    "DEFAULT_DB",
]
