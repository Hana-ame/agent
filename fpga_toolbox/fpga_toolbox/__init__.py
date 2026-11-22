"""FPGA 验证工具箱（融合版 v0.5.0）。

导出采用惰性方式：`import fpga_toolbox` 不会立刻加载 registry 和全部工具模块，
这样 `python -m fpga_toolbox.<tool>` 不会因为包已提前导入该模块而触发
runpy 的 "found in sys.modules" 警告。
"""

from typing import Any

__version__ = "0.5.0"
__all__ = ["TOOL_REGISTRY", "call_tool", "list_tool_schemas", "__version__"]

_LAZY = {"TOOL_REGISTRY", "call_tool", "list_tool_schemas"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
