"""预设检查 DAG：对一个对话，挂 N 条“预设”，每条预设是一段完整 system prompt + 一个 LLM，
并行检查 session 状态是否满足该预设，最后聚合出命中清单。

用法：
    python -m v8.presets --presets presets/examples/presets \
        --conv presets/examples/conv_example.json --state mainline
"""
from __future__ import annotations

from .builder import DEFAULT_PI, build_dag  # noqa: F401
from .presets import Preset, load_presets  # noqa: F401
from .state import render_state  # noqa: F401

__all__ = ["Preset", "load_presets", "render_state", "build_dag", "DEFAULT_PI"]
