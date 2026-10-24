"""approval 人工闸门的测试（含 resume 续跑）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402


class _Agent:
    def run(self, prompt, **kw):
        return "ok"


def _cfg(tmp_path: Path, marker: str = "OK.flag") -> Path:
    data = {
        "name": "approval_demo", "seed_tag": "task", "seed_content": "S",
        "result_tag": "out",
        "pi": {"provider": "laptop", "model": "deepseek-flash",
               "timeout": 10, "retries": 1},
        "edges": [
            {"id": "e1", "input_tags": ["task"], "output_tags": ["a"],
             "prompt": "P {contents}"},
            {"id": "gate", "kind": "approval", "input_tags": ["a"],
             "output_tags": ["out"], "marker": marker},
        ],
    }
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_approval_blocks_without_marker(tmp_path):
    res = dag8.run(_cfg(tmp_path), db_path=tmp_path / "r.sqlite3", agent=_Agent())
    assert res.status == "error"
    gate = [e for e in res.edges if e.edge_id == "gate"][0]
    assert "人工批准" in (gate.error or "")


def test_approval_passes_after_marker_and_resume(tmp_path):
    cfg, db = _cfg(tmp_path), tmp_path / "r.sqlite3"
    res = dag8.run(cfg, db_path=db, agent=_Agent())
    assert res.status == "error"

    (tmp_path / "OK.flag").write_text("ok", encoding="utf-8")
    res2 = dag8.run(cfg, db_path=db, agent=_Agent(), resume_last=True)
    assert res2.run_id == res.run_id
    assert res2.status == "ok"
    assert res2.result == "ok"


def test_approval_explicit_reject(tmp_path):
    (tmp_path / "OK.flag").write_text('{"approved": false, "reason": "不行"}',
                                      encoding="utf-8")
    res = dag8.run(_cfg(tmp_path), db_path=tmp_path / "r.sqlite3", agent=_Agent())
    assert res.status == "error"
    assert "人工拒绝" in (res.edges[-1].error or "")


def test_approval_requires_marker(tmp_path):
    data = {"name": "t", "seed_tag": "a", "seed_content": "x",
            "result_tag": "out",
            "edges": [{"id": "g", "kind": "approval", "input_tags": ["a"],
                       "output_tags": ["out"]}]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="marker"):
        cfg_mod.load(p)
