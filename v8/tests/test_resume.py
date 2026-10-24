"""断点续跑（--resume）的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402

CONFIG = {
    "name": "resume_demo", "seed_tag": "task", "seed_content": "S",
    "result_tag": "out",
    "pi": {"provider": "laptop", "model": "deepseek-flash",
           "timeout": 10, "retries": 1},
    "edges": [
        {"id": "e1", "layer": 0, "input_tags": ["task"], "output_tags": ["mid"],
         "prompt": "A {contents}"},
        {"id": "e2", "layer": 1, "input_tags": ["mid"], "output_tags": ["out"],
         "prompt": "B {contents}"},
    ],
}


class _FailSecond:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, **kw):
        self.calls += 1
        if self.calls == 1:
            return "first"
        raise dag8.PiError("boom", retryable=False)


class _Agent:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, **kw):
        self.calls += 1
        return "settled"


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(CONFIG), encoding="utf-8")
    return p


def test_resume_only_pending(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"

    first = _FailSecond()
    res = dag8.run(cfg, db_path=db, agent=first)
    assert res.status == "error"
    assert first.calls == 2

    again = _Agent()
    res2 = dag8.run(cfg, db_path=db, agent=again, resume=res.run_id)
    assert res2.run_id == res.run_id
    assert res2.status == "ok", res2.summary()
    assert res2.result == "settled"
    assert again.calls == 1, "只应续跑失败的那条边"


def test_resume_completed_does_nothing(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"
    agent = _Agent()
    res = dag8.run(cfg, db_path=db, agent=agent)
    assert res.status == "ok"
    assert agent.calls == 2

    again = _Agent()
    res2 = dag8.run(cfg, db_path=db, agent=again, resume=res.run_id)
    assert res2.status == "ok"
    assert res2.result == "settled"
    assert again.calls == 0


def test_resume_unknown_run(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"
    with pytest.raises(ValueError, match="不存在"):
        dag8.run(cfg, db_path=db, agent=_Agent(), resume="nope")


def test_cli_resume_completed(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"
    res = dag8.run(cfg, db_path=db, agent=_Agent())
    assert res.status == "ok"
    assert dag8_main(["--db", str(db), "run", str(cfg), "--resume", res.run_id]) == 0


def test_resume_last_finds_failed_run(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"
    first = _FailSecond()
    res = dag8.run(cfg, db_path=db, agent=first)
    assert res.status == "error"

    again = _Agent()
    res2 = dag8.run(cfg, db_path=db, agent=again, resume_last=True)
    assert res2.run_id == res.run_id
    assert res2.status == "ok"
    assert again.calls == 1


def test_resume_last_starts_fresh_when_none(tmp_path):
    cfg, db = _write(tmp_path), tmp_path / "r.sqlite3"
    agent = _Agent()
    res = dag8.run(cfg, db_path=db, agent=agent, resume_last=True)
    assert res.status == "ok"
    assert agent.calls == 2
