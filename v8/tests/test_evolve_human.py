"""evolve 人工评价闭环的离线测试：不碰 pi / 网络。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from evolve.genome import Genome  # noqa: E402
from evolve.review import build_review, rate_candidate, render_table, select_candidate  # noqa: E402
from evolve.runner import run_candidate  # noqa: E402
from evolve.tasks import Task, check_output  # noqa: E402
from evolve.from_runs import build_from_runs, load_runs  # noqa: E402


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    def run(self, prompt: str, **_kw) -> str:
        self.calls.append(prompt)
        return (
            "# 标题\n\n"
            "## 一、 一句话核心概括（Executive Summary）\n\n"
            "这是一个标签驱动 DAG 的离线测试输出，覆盖 DAG 与调度。\n\n"
            "## 二、 完整内容摘要（Comprehensive Abstract）\n\n"
            "1. **标签**：用标签表达依赖。\n\n"
            "## 三、 结构化深度阅读笔记（Structured Reading Notes）\n\n"
            "### 1. 案例对标：单一指标考核的系统性失灵\n\n"
            "| 对比维度 | A | B | C |\n|---|---|---|---|\n| 核心 | x | y | z |\n\n"
            "### 2. 核心管理学推演：SOP 是如何杀死组织的？\n\n"
            "1. **第一阶段**：x\n2. **第二阶段**：x\n3. **第三阶段**：x\n"
            "4. **第四阶段**：x\n5. **第五阶段**：x\n\n"
            "### 3. 好组织运作的“双支柱模型”\n\n"
            "* **支柱一**：x\n* **支柱二**：y\n\n"
            "### 4. 原文口语/谐音词汇对照勘误表（Transcript Glossary）\n\n"
            "| 原文识别词 | 实际对应词 / 典故 | 语境释义 |\n|---|---|---|\n"
            "| a | b | c |\n\n"
            "## 四、 树状思维导图（Text & Mermaid Mind Map）\n\n"
            "### 1. 结构化层级导图（ASCII Tree）\n\n```text\nDAG\n└─ tag\n```\n\n"
            "### 2. Mermaid 思维导图代码\n\n```mermaid\nmindmap\n  root((DAG))\n    tag\n```\n"
        )


def _config(path: Path) -> Path:
    data = {
        "name": "human_eval_test",
        "seed_tag": "topic",
        "seed_content": "标签驱动 DAG",
        "result_tag": "final",
        "pi": {"provider": "laptop", "model": "sensenova-6.8-flash-lite",
               "thinking": "off"},
        "edges": [
            {"id": "draft", "layer": 0, "input_tags": ["topic"],
             "output_tags": ["draft"], "prompt": "写初稿：{contents}"},
            {"id": "write", "layer": 1, "input_tags": ["draft"],
             "output_tags": ["final"], "prompt": "定稿：{contents}"},
        ],
    }
    p = path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _task(cfg: Path) -> Task:
    return Task(
        id="t", title="测试任务", config=str(cfg), seed="标签驱动 DAG",
        criteria=["覆盖", "准确"],
        checks={"min_chars": 20, "required_terms": ["DAG"]},
    )


def test_genome_materialize_and_mutate(tmp_path):
    cfg = _config(tmp_path)
    base = Genome.from_config(cfg)
    child = base.mutate(__import__("random").Random(1))
    assert child.id != base.id
    assert child.lineage["parent"] == base.id
    out = child.materialize(tmp_path / "cand")
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "human_eval_test"


def test_critique_mutation_wires_tag(tmp_path):
    cfg = _config(tmp_path)
    base = Genome.from_config(cfg)
    child = base.mutate(__import__("random").Random(3), allowed=["critique_node"])
    edges = child.config["edges"]
    crit = [e for e in edges if e["id"] == "self_critique"]
    assert crit, child.lineage
    final = [e for e in edges if "final" in e["output_tags"]][0]
    assert "critique" in final["input_tags"]
    assert "{tag:critique}" in final["prompt"]


def test_run_candidate_and_review_select(tmp_path):
    cfg = _config(tmp_path)
    task = _task(cfg)
    agent = FakeAgent()
    run_dir = tmp_path / "run"
    base = Genome.from_config(cfg, name="base")
    r1 = run_candidate(base, task, run_dir, agent=agent, use_cache=False)
    assert r1["status"] == "ok", r1
    assert "DAG" in Path(r1["output_path"]).read_text(encoding="utf-8")
    assert r1["auto"]["score"] == 100.0

    child = base.mutate(__import__("random").Random(2))
    r2 = run_candidate(child, task, run_dir, agent=agent, use_cache=False)
    assert r2["status"] == "ok", r2

    review_md = build_review(run_dir, task, [r1, r2])
    assert review_md.is_file()
    assert "评分表" in review_md.read_text(encoding="utf-8")
    assert "人工分" in render_table(run_dir)

    rate_candidate(run_dir, base.id, 5, "更完整")
    rate_candidate(run_dir, child.id, 3, "略短")
    sel = select_candidate(run_dir, by="human")
    assert sel["winner"] == base.id
    assert sel["ranking"][0]["human"] == 5

    sel_auto = select_candidate(run_dir, by="auto")
    assert sel_auto["winner"] in {base.id, child.id}


def test_select_human_requires_scores(tmp_path):
    cfg = _config(tmp_path)
    task = _task(cfg)
    run_dir = tmp_path / "run"
    rec = run_candidate(Genome.from_config(cfg), task, run_dir,
                        agent=FakeAgent(), use_cache=False)
    build_review(run_dir, task, [rec])
    with pytest.raises(ValueError, match="还没人工评分"):
        select_candidate(run_dir, by="human")


def test_from_runs_builds_packet(tmp_path):
    cfg = _config(tmp_path)
    task = _task(cfg)
    run_dir = tmp_path / "seedrun"
    rec = run_candidate(Genome.from_config(cfg), task, run_dir,
                        agent=FakeAgent(), use_cache=False)
    db = Path(rec["output_path"]).parent / "run.sqlite3"
    assert db.is_file()
    runs = load_runs(db, last=3)
    assert runs and runs[0]["output"]

    out = tmp_path / "review"
    path = build_from_runs(db, out, last=3, title="历史")
    assert path.is_file()
    assert "评分表" in path.read_text(encoding="utf-8")
    assert (out / "ratings.json").is_file()


def test_check_output_reports_failures():
    res = check_output("短文本", {"min_chars": 100, "required_terms": ["DAG"]})
    assert res["score"] < 100
    assert any(not c["ok"] for c in res["checks"])
