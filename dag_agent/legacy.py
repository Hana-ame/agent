"""v6 遗留子系统门面（legacy）。

## 定位

2026-10-22 起仓库只有**一条主线**：`dag_agent`（v7）。原 `vea_v6` 不再是并列主线，
降级为「自举子系统」——它的核心价值是**逐字节自复现的固定点**（SPEC → 源码，
生成的框架能重建自己），而不是一套通用 DAG API。

v6 里对 v7 有普适价值的两项能力已经**真正搬进 v7**（不是门面，是重新实现）：

| 能力 | v7 落点 |
|---|---|
| SSE 流式 + reasoning 兜底 | `dag_agent.llm.OpenAICompatibleClient(stream=True)` |
| 重试白名单（403/1010/10054/429/5xx/超时） | `dag_agent.llm.OpenAICompatibleClient(max_retries=N)` |

## 用法

```python
from dag_agent.legacy import v6
v6.verify_fixed_point()      # 校验固定点
v6.bootstrap(...)            # 自举
```

惰性导入：只有真的取属性时才 import `vea_v6`，`import dag_agent` 不会被 1386 行的
bootstrap 模块拖慢。

## 铁律：不要物理移动 `vea_v6/parts/`

那些文件的**路径是固定点锚点**：
- `vea_v6/parts/bootstrap.py` 里的 `FRAME_FILES` 写死 7 个相对路径；
- 主源 payload 是这 7 个文件的 gzip+base64 内嵌，`fixpoint_v3.py --check` 逐字节比对。

一旦挪目录，`tools/fixpoint_v3.py --check`、`gen3_probe.py`、`oneshot_chain.py`
全部失效，163835 字节的自复现链断掉。要搬家必须同步改 SPEC 并重新固化，
不能顺手挪。
"""

from __future__ import annotations

from typing import Any

__all__ = ["v6"]


def __getattr__(name: str) -> Any:
    if name != "v6":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return importlib.import_module("vea_v6")
