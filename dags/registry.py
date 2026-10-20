"""DAG 注册表：把每个任务写成一个可注册、可单次运行的 DAG。

新增一个 DAG 只需三步：
1. 新建 `dags/xxx.py`
2. `@register("name", "描述", seed_tag=..., seed_content=..., final_tag=..., default_output=...)`
3. 写一个返回 `DAG` 的 build 函数

之后 `python3 -m dags list` 能看到它，`python3 -m dags run name` 能单次跑出结果。
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# 单次运行内的每条 edge 耗时（键 = 处理器函数名）
_timings: Dict[str, float] = {}


def timed(fn):
    """包一层计时。functools.wraps 保留签名，供引擎 inspect.signature 正常识别入参。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _timings[fn.__name__] = time.perf_counter() - t0

    return wrapper


@dataclass
class DagSpec:
    name: str
    description: str
    build: Callable[[], object]
    seed_tag: str
    seed_content: str
    final_tag: str
    default_output: str


DAG_SPECS: Dict[str, DagSpec] = {}


def register(
    name: str,
    description: str,
    *,
    seed_tag: str,
    seed_content: str,
    final_tag: str,
    default_output: str,
):
    def deco(build: Callable[[], object]):
        DAG_SPECS[name] = DagSpec(
            name=name,
            description=description,
            build=build,
            seed_tag=seed_tag,
            seed_content=seed_content,
            final_tag=final_tag,
            default_output=default_output,
        )
        return build

    return deco


def list_dags() -> List[DagSpec]:
    return [DAG_SPECS[k] for k in sorted(DAG_SPECS)]


def run_dag(
    name: str,
    *,
    output: Optional[str] = None,
    max_steps: int = 30,
) -> Dict[str, object]:
    """一次调用跑完整条 DAG：build → 注入种子 → run(until_tag) → 写文件。

    返回 {name, dag, report, output, elapsed, steps, timings}。
    """
    spec = DAG_SPECS.get(name)
    if spec is None:
        raise KeyError(f"未知 DAG: {name!r}；可用: {sorted(DAG_SPECS)}")

    _timings.clear()
    t0 = time.perf_counter()
    dag = spec.build()
    dag.add_content(contents=spec.seed_content, tags=[spec.seed_tag], gen=0)
    steps = dag.run(until_tag=spec.final_tag, max_steps=max_steps)
    elapsed = time.perf_counter() - t0

    report = dag.get_result(spec.final_tag) or ""
    out_path = Path(output) if output else ROOT / spec.default_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    return {
        "name": name,
        "dag": dag,
        "report": report,
        "output": str(out_path),
        "elapsed": elapsed,
        "steps": steps,
        "timings": dict(_timings),
    }


def print_timings(timings: Dict[str, float], total: float) -> None:
    if not timings:
        return
    print("\n=== 各 edge 耗时 ===")
    print(f"{'处理器':<22}{'耗时(s)':>10}{'占比':>10}")
    print("-" * 44)
    for fn_name in sorted(timings, key=lambda k: -timings[k]):
        sec = timings[fn_name]
        pct = (sec / total * 100.0) if total else 0.0
        print(f"{fn_name:<22}{sec:>10.3f}{pct:>9.1f}%")
    print("-" * 44)
    print(f"{'合计(边)':<22}{sum(timings.values()):>10.3f}")
    print(f"{'总计(run)':<22}{total:>10.3f}")
