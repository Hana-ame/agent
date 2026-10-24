"""库式 Dag.step()：脚本驱动、卡住时 set 外部结果。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8 import Dag  # noqa: E402


class _Agent:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def run(self, prompt, **kw):
        self.calls += 1
        if self.fail and self.calls == 1:
            raise RuntimeError("模型失败")
        return "ok"


def _cfg(tmp_path, edges, result="out", seed="S"):
    data = {"name": "session_demo", "seed_tag": "task", "seed_content": seed,
            "result_tag": result,
            "pi": {"provider": "laptop", "model": "deepseek-flash",
                   "timeout": 10, "retries": 1},
            "edges": edges}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_step_runs_one_edge_at_a_time(tmp_path):
    cfg = _cfg(tmp_path, [
        {"id": "e1", "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "P {contents}"},
        {"id": "e2", "input_tags": ["a"], "output_tags": ["out"],
         "prompt": "Q {contents}"},
    ])
    d = Dag(cfg, db_path=tmp_path / "s.sqlite3", agent=_Agent())
    r1 = d.step()
    assert r1.status == "ran" and r1.ran == ["e1"]
    assert d.contents["a"] == ["ok"]
    r2 = d.step()
    assert r2.status == "done"
    assert d.result == "ok"
    d.close()


def test_blocked_then_set_external(tmp_path):
    cfg = _cfg(tmp_path, [
        {"id": "e", "input_tags": ["needed"], "output_tags": ["out"],
         "prompt": "P {contents}"},
    ])
    d = Dag(cfg, db_path=tmp_path / "s.sqlite3", agent=_Agent())
    r = d.step()
    assert r.status == "blocked"
    assert r.blocked[0]["missing"] == ["needed"]
    d.set("needed", "外部结果")
    r2 = d.step()
    assert r2.status == "done"
    assert d.result == "ok"
    d.close()


def test_error_then_set_output_continues(tmp_path):
    cfg = _cfg(tmp_path, [
        {"id": "e1", "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "P {contents}"},
        {"id": "e2", "input_tags": ["a"], "output_tags": ["out"],
         "prompt": "Q {contents}"},
    ])
    d = Dag(cfg, db_path=tmp_path / "s.sqlite3", agent=_Agent(fail=True))
    r = d.step()
    assert r.status == "error"
    d.set("a", "人工补的结果")
    r2 = d.step()
    assert r2.status == "done"
    d.close()
