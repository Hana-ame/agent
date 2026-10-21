"""评测 DAG（dag-eval）的测试：用假目标验证三项指标与判定逻辑，离线。"""
from __future__ import annotations

import json

import pytest

from dag_agent import DAG, MockLLMClient
from dags.registry import DAG_SPECS, DagSpec


def _build_eval():
    """从注册表取，别直接 import 定义模块（避免与自动发现产生两份实例）。"""
    return DAG_SPECS["dag-eval"].build()


def _spec(name: str, processor) -> DagSpec:
    def build():
        d = DAG(name=name, models={}, llm_client=MockLLMClient())
        d.add_edge(input_tags=["seed"], processor=processor, output_tags=["final"])
        return d

    return DagSpec(
        name=name,
        description="评测用假目标",
        build=build,
        seed_tag="seed",
        seed_content="x",
        final_tag="final",
        default_output="dag_defs/out/fake.md",
    )


STABLE_TEXT = "## 标题\n第一行内容。\n第二行。\n第三行。\n第四行。\n第五行。"


def _run(request: dict, targets: dict[str, DagSpec]) -> str:
    for name, spec in targets.items():
        DAG_SPECS[name] = spec
    dag = _build_eval()
    dag.add_content(contents=json.dumps(request, ensure_ascii=False), tags=["eval_request"], gen=0)
    dag.run(until_tag="eval_report", max_steps=10)
    report = dag.get_result("eval_report")
    assert report, "dag-eval 没产出报告"
    return report


def test_registered():
    assert "dag-eval" in DAG_SPECS
    spec = DAG_SPECS["dag-eval"]
    assert spec.seed_tag == "eval_request"
    assert spec.final_tag == "eval_report"
    assert "out" in spec.default_output


def test_stable_target_passes(clean_specs):
    report = _run({"target": "tmp-stable", "runs": 3}, {"tmp-stable": _spec("tmp-stable", lambda x: STABLE_TEXT)})

    assert "判定：**PASS**" in report
    assert "文本 3-gram Jaccard 均值：**1.000**" in report
    assert "| chars | 1.000 |" in report
    assert "| non_empty | 100% |" in report
    assert "运行次数：**3**" in report


def test_jittery_target_warns(clean_specs):
    counter = {"n": 0}

    def jitter(x):
        counter["n"] += 1
        return "## 标题\n" + "\n".join(f"第{i}行不同内容。" for i in range(counter["n"] * 6))

    report = _run({"target": "tmp-jitter", "runs": 2}, {"tmp-jitter": _spec("tmp-jitter", jitter)})
    assert "判定：**WARN**" in report
    assert "结构字段不稳定" in report or "长度变异" in report or "文本相似度" in report


def test_crashing_target_fails(clean_specs):
    def boom(x):
        raise RuntimeError("模拟失败")

    report = _run({"target": "tmp-boom", "runs": 2}, {"tmp-boom": _spec("tmp-boom", boom)})
    assert "判定：**FAIL**" in report
    assert "运行期异常 2 次" in report
    assert "RuntimeError: 模拟失败" in report
    assert "| non_empty | 0% |" in report


def test_truncated_output_fails_structure_checks(clean_specs):
    truncated = "## 标题\n说到一半就断"
    report = _run({"target": "tmp-trunc", "runs": 2}, {"tmp-trunc": _spec("tmp-trunc", lambda x: truncated)})
    # ends_clean 是硬条件（防截断）→ 直接 FAIL；min_lines 只是 WARN
    assert "判定：**FAIL**" in report
    assert "正确率检查未全过：ends_clean" in report
    assert "正确率检查未全过：min_lines" in report


def test_unknown_target_raises(clean_specs):
    dag = _build_eval()
    dag.add_content(contents=json.dumps({"target": "no-such-dag", "runs": 1}), tags=["eval_request"], gen=0)
    with pytest.raises(KeyError, match="未知 DAG"):
        dag.run(until_tag="eval_report", max_steps=5)


def test_default_request_targets_stage1st():
    req = json.loads(DAG_SPECS["dag-eval"].seed_content)
    assert req["target"] == "stage1st-ai-24h"
    assert req["runs"] >= 1


# ---------------------------------------------------------------------------
# 目标 DAG 自带的语义验收器
# ---------------------------------------------------------------------------
GOOD_S1 = "## 摘要\n今天有 AI 讨论。\n\n## 主题索引\n- [a](u)\n- [b](u)\n- [c](u)\n"


def test_stage1st_verify_report_accepts_complete_report():
    from dag_defs.stage1st_ai import verify_report

    ok, why = verify_report(GOOD_S1)
    assert ok, why


@pytest.mark.parametrize(
    "bad, reason",
    [
        ("没有摘要", "缺 `## 摘要` 段"),
        ("## 摘要\n正文。\n", "缺 `## 主题索引` 段"),
        ("## 摘要\n（LLM 摘要不可用，以下为抽取式摘要）\n\n## 主题索引\n- [a](u)\n", "兜底"),
        ("## 摘要\n说到一半\n\n## 主题索引\n- [a](u)\n- [b](u)\n- [c](u)\n", "截断"),
        ("## 摘要\n完整。\n\n## 主题索引\n- [a](u)\n", "只有 1 条"),
    ],
)
def test_stage1st_verify_report_rejects(bad, reason):
    from dag_defs.stage1st_ai import verify_report

    ok, why = verify_report(bad)
    assert not ok
    assert reason in why


def test_custom_verifier_discovery():
    from dag_defs.dag_eval import _custom_verifier

    assert _custom_verifier("stage1st-ai-24h") is not None
    assert _custom_verifier("no-such-dag") is None
