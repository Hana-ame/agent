"""浸泡 DAG（dag-soak）的测试：收敛、不收敛、时间预算，全离线。"""
from __future__ import annotations

import json
import time

import pytest

from dag_agent import DAG, MockLLMClient
from dags.registry import DAG_SPECS, DagSpec


def _build_soak():
    return DAG_SPECS["dag-soak"].build()


def _spec(name: str, processor) -> DagSpec:
    def build():
        d = DAG(name=name, models={}, llm_client=MockLLMClient())
        d.add_edge(input_tags=["seed"], processor=processor, output_tags=["final"])
        return d

    return DagSpec(
        name=name,
        description="浸泡用假目标",
        build=build,
        seed_tag="seed",
        seed_content="x",
        final_tag="final",
        default_output="dag_defs/out/fake.md",
    )


def _run(request: dict, targets: dict[str, DagSpec] | None = None) -> str:
    for name, spec in (targets or {}).items():
        DAG_SPECS[name] = spec
    dag = _build_soak()
    dag.add_content(contents=json.dumps(request, ensure_ascii=False), tags=["soak_request"], gen=0)
    dag.run(until_tag="soak_report", max_steps=10)
    report = dag.get_result("soak_report")
    assert report, "dag-soak 没产出报告"
    return report


STABLE = "## 摘要\n稳定输出。\n\n## 主题索引\n- [a](u)\n- [b](u)\n- [c](u)\n"


def test_stable_target_converges(clean_specs):
    report = _run(
        {"target": "selftest-stable", "hours": 0.001, "window": 3, "min_runs": 3, "max_runs": 50}
    )
    assert "结论：**已收敛**" in report
    assert "连续窗口达标，提前停止" in report
    assert "末窗平均 Jaccard：**1.0" in report


def test_noisy_target_does_not_converge(clean_specs):
    report = _run(
        {"target": "selftest-noisy", "hours": 0.001, "window": 3, "stability": 0.99,
         "min_runs": 3, "max_runs": 8}
    )
    assert "结论：**未收敛**" in report
    assert "达到 max_runs 上限" in report


def test_budget_stops_early(clean_specs):
    """目标慢 + 预算极小：跑够 min_runs 后应因预算停下。"""

    def slow(x):
        time.sleep(0.05)
        return STABLE

    report = _run(
        # 0.000003h ≈ 0.011s 预算；每轮 0.05s，跑到第 2 轮就已超时
        {"target": "tmp-slow", "hours": 0.000003, "window": 2, "min_runs": 2, "max_runs": 100},
        {"tmp-slow": _spec("tmp-slow", slow)},
    )
    assert "结论：**未收敛**" in report
    assert "时间预算耗尽" in report


def test_unknown_target_raises(clean_specs):
    dag = _build_soak()
    dag.add_content(contents=json.dumps({"target": "nope"}), tags=["soak_request"], gen=0)
    with pytest.raises(KeyError, match="未知 DAG"):
        dag.run(until_tag="soak_report", max_steps=5)


def test_report_contains_all_sections(clean_specs):
    report = _run({"target": "selftest-stable", "hours": 0.001, "window": 2, "min_runs": 2})
    for section in ("## 速度", "## 稳定性轨迹", "## 检查通过率", "## 代表输出"):
        assert section in report
    assert "```markdown" in report


def test_empty_report_never_converges(clean_specs):
    """目标产出空串：ends_clean/non_empty 不过，绝不能判收敛。"""
    report = _run(
        {"target": "tmp-empty", "hours": 0.001, "window": 2, "min_runs": 2, "max_runs": 5},
        {"tmp-empty": _spec("tmp-empty", lambda x: "")},
    )
    assert "结论：**未收敛**" in report
    assert "| non_empty | 0% |" in report
