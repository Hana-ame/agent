"""可扩展 DAG 库。

用法::

    python3 -m dags list
    python3 -m dags run stage1st-ai-24h
    python3 -m dags run cf-subdomain-daily --output /tmp/cf.md

新增 DAG：在 `dags/` 下新建模块，用 `@register(...)` 装饰一个返回 `DAG` 的 build 函数，
并在本文件末尾 import 该模块即可自动注册。
"""
from dags.registry import (
    DAG_SPECS,
    DagSpec,
    list_dags,
    print_timings,
    register,
    run_dag,
    timed,
)

# 导入即注册
from dags import cf_usage  # noqa: F401,E402
from dags import stage1st_ai  # noqa: F401,E402

__all__ = [
    "DAG_SPECS",
    "DagSpec",
    "list_dags",
    "register",
    "run_dag",
    "timed",
    "print_timings",
]
