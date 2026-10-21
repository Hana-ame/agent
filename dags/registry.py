"""DAG 注册表：把每个任务写成一个可注册、可单次运行的 DAG。

新增一个 DAG 只需三步：
1. 新建 `dags/xxx.py`
2. `@register("name", "描述", seed_tag=..., seed_content=..., final_tag=..., default_output=...)`
3. 写一个返回 `DAG` 的 build 函数

之后 `python3 -m dags list` 能看到它，`python3 -m dags run name` 能单次跑出结果。
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

# stem -> 已加载的定义模块。定义模块自己需要兄弟模块（如稳定性脚本要拿
# stage1st_ai 的函数）时用 get_definition_module()，不要靠猜 sys.modules 的键名。
LOADED_MODULES: Dict[str, Any] = {}


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
        old = DAG_SPECS.get(name)
        if old is not None and old.build is not build:
            # 静默覆盖会让人以为新定义生效、实际跑的是别的模块 —— 只警告不阻断，
            # 保留热重载能力。
            warnings.warn(
                f"重复注册 DAG {name!r}：{old.build.__module__} 被 {build.__module__} 覆盖",
                RuntimeWarning,
                stacklevel=2,
            )
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


def get_definition_module(stem: str) -> Any:
    """取已加载的定义模块；不存在时抛 KeyError 并列出可用的。"""
    try:
        return LOADED_MODULES[stem]
    except KeyError:
        raise KeyError(f"定义模块未加载: {stem!r}；已加载: {sorted(LOADED_MODULES)}") from None


def load_definitions(directory: Union[str, Path], *, force: bool = False) -> List[str]:
    """自动发现并加载一个目录下的所有任务 DAG 模块（忽略 `_` 开头）。

    两种加载方式，优先走包导入：
    - 目录有 `__init__.py`（是个包）→ `importlib.import_module("dag_defs.xxx")`。
      这样 `from dag_defs.xxx import y` 与自动发现拿到的是**同一个模块对象**，
      不会出现两份实例各注册一次（早先用文件加载，两种写法混用时必重复注册）。
    - 否则（散装脚本目录）→ 按文件路径加载，模块名带目录指纹，避免多目录同名串味。
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    # 只有「父目录已在 sys.path 上」的包才走包导入。临时目录（同名且不在 sys.path）
    # 若也走包导入，会撞 sys.modules 缓存拿到别的目录的模块。
    # `sys.path` 里的空串表示当前工作目录，不能过滤掉（`python -c` / `-m` 都靠它）
    search_roots = {str(Path(p or ".").resolve()) for p in sys.path}
    is_package = (d / "__init__.py").exists() and str(d.parent.resolve()) in search_roots
    pkg_root = d.parent
    digest = hashlib.sha1(str(d.resolve()).encode("utf-8")).hexdigest()[:8]
    loaded: List[str] = []

    for path in sorted(d.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stem = path.stem
        module: Any = None

        if is_package:
            qualname = f"{d.name}.{stem}"
            if qualname in sys.modules and not force:
                module = sys.modules[qualname]
            else:
                root_str = str(pkg_root)
                if root_str not in sys.path:
                    sys.path.insert(0, root_str)
                try:
                    module = importlib.import_module(qualname)
                    if force:
                        module = importlib.reload(module)
                except Exception as exc:  # noqa: BLE001 - 包导入失败回落到文件加载
                    warnings.warn(
                        f"包导入 {qualname} 失败（{type(exc).__name__}: {exc}），回落到文件加载",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    module = None

        if module is None:
            mod_name = f"dag_defs_{digest}_{stem}"
            if mod_name in sys.modules and not force:
                module = sys.modules[mod_name]
            else:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)

        LOADED_MODULES[stem] = module
        loaded.append(stem)
    return loaded


def list_dags() -> List[DagSpec]:
    return [DAG_SPECS[k] for k in sorted(DAG_SPECS)]


def run_dag(
    name: str,
    *,
    output: Optional[str] = None,
    max_steps: int = 30,
    dry_run: bool = False,
    seed: Optional[str] = None,
) -> Dict[str, object]:
    """一次调用跑完整条 DAG：build → 注入种子 → run(until_tag) → 写文件。

    dry_run=True 时只跑到取结果，不落盘（测试与 CI 用，避免污染工作树）。
    seed 覆盖注册时写死的 seed_content（评测类 DAG 靠它换目标，如 dag-eval）。

    返回 {name, dag, report, output, written, elapsed, steps, timings}。
    """
    spec = DAG_SPECS.get(name)
    if spec is None:
        raise KeyError(f"未知 DAG: {name!r}；可用: {sorted(DAG_SPECS)}")

    _timings.clear()
    t0 = time.perf_counter()
    dag = spec.build()
    dag.add_content(
        contents=spec.seed_content if seed is None else seed,
        tags=[spec.seed_tag],
        gen=0,
    )
    steps = dag.run(until_tag=spec.final_tag, max_steps=max_steps)
    elapsed = time.perf_counter() - t0

    report = dag.get_result(spec.final_tag) or ""
    out_path = Path(output) if output else ROOT / spec.default_output
    written = False
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        written = True

    return {
        "name": name,
        "dag": dag,
        "report": report,
        "output": str(out_path),
        "written": written,
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
