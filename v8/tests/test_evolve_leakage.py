"""泄漏惩罚：候选出现仅答案拥有的片段 → 分数 ×0.1。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from evolve.leakage import audit, audit_optional  # noqa: E402
from evolve.tasks import check_output  # noqa: E402


SOURCE = "源材料只说了甲乙丙三件事，没有别的判断。"
ANSWER = "答案独有的短语XYZ必须出现在这里。\n\n| 原文识别词 | 实际对应词 | 语境释义 |\n|---|---|---|\n| 独有表头词 | 正确词 | 解释 |\n"


def test_audit_flags_prompt_leak():
    res = audit(SOURCE, ANSWER, {"prompt": "请输出：答案独有的短语XYZ", "output": ""})
    assert res["leak"] is True
    assert res["multiplier"] == 0.1
    assert any("prompt" == m["where"] for m in res["matches"])


def test_audit_clean_candidate():
    res = audit(SOURCE, ANSWER, {"prompt": "请总结甲乙丙", "output": "甲乙丙"})
    assert res["leak"] is False
    assert res["multiplier"] == 1.0


def test_check_output_applies_penalty():
    leak = {"enabled": True, "leak": True, "multiplier": 0.1,
            "reason": "test", "matches": []}
    res = check_output("甲" * 20, {"min_chars": 10}, leakage=leak)
    assert res["score_raw"] == 100.0
    assert res["score"] == 10.0


def test_optional_requires_source_and_answer():
    assert audit_optional(None, "answer.md", {"prompt": ""})["enabled"] is False
    assert audit_optional(SOURCE, None, {"prompt": ""})["enabled"] is False


def test_runner_records_leakage(tmp_path):
    from evolve.genome import Genome
    from evolve.runner import run_candidate
    from evolve.tasks import Task

    class LeakAgent:
        def run(self, prompt: str, **_kw) -> str:
            return "答案独有的短语XYZ " + "甲" * 40

    cfg = tmp_path / "dag.json"
    cfg.write_text(json.dumps({
        "name": "leak", "seed_tag": "topic", "seed_content": "x",
        "result_tag": "final",
        "pi": {"provider": "laptop", "model": "sensenova-6.8-flash-lite"},
        "edges": [{"id": "w", "layer": 0, "input_tags": ["topic"],
                   "output_tags": ["final"], "prompt": "写：{contents}"}],
    }, ensure_ascii=False), encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(SOURCE, encoding="utf-8")
    answer = tmp_path / "answer.txt"
    answer.write_text(ANSWER, encoding="utf-8")
    task = Task(id="t", title="t", config=str(cfg), seed="x",
                checks={"min_chars": 10})
    rec = run_candidate(
        Genome.from_config(cfg), task, tmp_path / "run",
        agent=LeakAgent(), use_cache=False,
        source_path=str(source), answer_path=str(answer),
    )
    assert rec["leakage"]["leak"] is True
    assert rec["auto"]["score"] == 10.0
