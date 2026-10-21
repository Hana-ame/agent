"""所有已注册 DAG 的契约测试。

一个 DAG 注册进来就要满足这些硬约束，否则 `python -m dags run` 要到运行时才炸：
- 元数据齐全（name / description / seed_tag / final_tag / default_output）
- build() 能构造出可用的 DAG 对象，且不依赖真实端点
- 种子能注入、终态标签从种子**可达**（否则 run(until_tag) 会空转到 max_steps 才返回空）
- 默认 max_steps 够走完全图
- 默认产物落在 out/ 下，不往仓库根散落

新增 DAG 后无需改这个文件，参数化会自动带上。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dags import list_dags

SPECS = list_dags()
IDS = [s.name for s in SPECS]


def _reachable_tags(dag, start: str) -> set[str]:
    """按 input_tags ⊆ 已见集合 反复展开，算标签可达闭包。"""
    seen = {start}
    changed = True
    while changed:
        changed = False
        for edge in dag.edges:
            if set(edge.input_tags) <= seen:
                new = set(edge.output_tags) - seen
                if new:
                    seen |= new
                    changed = True
    return seen


def test_registry_is_not_empty():
    assert SPECS, "dags 自动发现没加载到任何定义（检查 dag_defs/ 与 DEFINITIONS_DIR）"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_metadata_complete(spec):
    assert spec.name.strip(), "name 不能为空"
    assert spec.description.strip(), "description 不能为空（list 命令要显示）"
    assert spec.seed_tag.strip(), "seed_tag 不能为空"
    assert spec.final_tag.strip(), "final_tag 不能为空"
    assert spec.seed_tag != spec.final_tag, "seed_tag 与 final_tag 不能相同"
    assert isinstance(spec.seed_content, str) and spec.seed_content.strip(), "seed_content 不能为空"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_build_produces_usable_dag(spec):
    dag = spec.build()
    for attr in ("add_content", "run", "get_result", "has_tag", "edges"):
        assert hasattr(dag, attr), f"{spec.name}: build() 返回对象缺少 {attr}"
    assert dag.edges, f"{spec.name}: 没有任何边"

    dag.add_content(contents=spec.seed_content, tags=[spec.seed_tag], gen=0)
    assert dag.has_tag(spec.seed_tag), f"{spec.name}: 种子注入后 has_tag({spec.seed_tag}) 仍为 False"

    reachable = _reachable_tags(dag, spec.seed_tag)
    assert spec.final_tag in reachable, (
        f"{spec.name}: 从 {spec.seed_tag} 不可达 {spec.final_tag} —— "
        f"run(until_tag={spec.final_tag}) 会空转到 max_steps 并返回空报告。"
        f"可达标签：{sorted(reachable)}"
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_default_max_steps_is_enough(spec):
    dag = spec.build()
    # run_dag 默认 max_steps=30；图里的边数超过它就一定跑不完
    assert len(dag.edges) + 1 <= 30, f"{spec.name}: {len(dag.edges)} 条边，默认 max_steps=30 不够"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_default_output_under_out_dir(spec):
    p = Path(spec.default_output)
    assert not p.is_absolute(), f"{spec.name}: default_output 应是仓库内相对路径，实际 {p}"
    assert "out" in p.parts, f"{spec.name}: 产物应落在 out/ 下，实际 {spec.default_output}"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_build_is_repeatable(spec):
    a, b = spec.build(), spec.build()
    assert a is not b, f"{spec.name}: build() 每次都应返回新实例（共享状态会串味）"
    assert len(a.edges) == len(b.edges)
