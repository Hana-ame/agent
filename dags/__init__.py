"""可扩展 DAG 库。

用法::

    python3 -m dags list
    python3 -m dags run <name>

新增 DAG：在 `dag_defs/` 下新建模块，用 `@register(...)` 装饰一个返回 `DAG` 的 build 函数，
`import dags` 时会自动发现，无需改库代码。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from dags.registry import (  # noqa: E402
    DAG_SPECS,
    DagSpec,
    list_dags,
    load_definitions,
    print_timings,
    register,
    run_dag,
    timed,
)

# 自动发现任务 DAG 定义（目录可配置，库本身不含任何具体任务）
DEFINITIONS_DIR = ROOT / "dag_defs"
load_definitions(DEFINITIONS_DIR)

__all__ = [
    "DAG_SPECS",
    "DEFINITIONS_DIR",
    "load_definitions",
    "DagSpec",
    "list_dags",
    "register",
    "run_dag",
    "timed",
    "print_timings",
]
