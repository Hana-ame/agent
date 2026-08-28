"""Pipeline module — DEPRECATED. Pipeline 编排逻辑已搬入 Edge。

改造背景：Pipeline 是无状态、每次 execute 都 new、所有字段从 Edge 拷贝的"伪对象"。
现 5 阶段编排逻辑（guard/pre_process/compute/retry+timeout/post_process/schema/memory/
telemetry）全部在 Edge 内部（Edge.execute / Edge._run_compute / Edge.compute 等）。

此文件仅保留向后兼容：`from framework.pipeline import Pipeline` 仍可用，
但 Pipeline 已等价于 Edge（Pipeline = Edge 别名）。新代码请直接用 Edge。
"""

from .edge import Edge

# 向后兼容别名：旧代码 `from framework.pipeline import Pipeline` 不崩。
# ⚠️ Pipeline 已不是独立类——它就是 Edge。不应再用 Pipeline 做任何事。
Pipeline = Edge

# AbortPipeline 异常定义在 utils/errors.py，这里 re-export 保持旧 import 路径可用。
from .utils.errors import AbortPipeline, GuardAbortError, HookError, ComputeError

__all__ = ["Pipeline", "AbortPipeline", "GuardAbortError", "HookError", "ComputeError"]
