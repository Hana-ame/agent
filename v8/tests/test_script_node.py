"""script 节点：默认边就是脚本，JSON 输入、stdout 输出。"""
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


def _write(tmp_path: Path, script: str = "run.py") -> Path:
    data = {"name": "script_demo", "seed_tag": "task", "seed_content": "abc",
            "result_tag": "out",
            "edges": [{"id": "e", "input_tags": ["task"],
                       "output_tags": ["out"], "script": script}]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_script_is_default_kind_and_reads_json_env(tmp_path):
    (tmp_path / "run.py").write_text(
        "import os, json\n"
        "data = json.loads(os.environ['DAG8_INPUTS'])\n"
        "print(data['task'].upper())\n", encoding="utf-8")
    cfg = _write(tmp_path)
    assert cfg_mod.load(cfg).edges[0].kind == "script"
    res = dag8.run(cfg, db_path=tmp_path / "r.sqlite3")
    assert res.status == "ok", res.summary()
    assert res.result == "ABC"


def test_script_nonzero_exit_is_error(tmp_path):
    (tmp_path / "run.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    res = dag8.run(_write(tmp_path), db_path=tmp_path / "r.sqlite3")
    assert res.status == "error"
    assert "退出码 3" in (res.edges[0].error or "")


def test_script_missing_file(tmp_path):
    res = dag8.run(_write(tmp_path, "nope.py"), db_path=tmp_path / "r.sqlite3")
    assert res.status == "error"
    assert "不存在" in (res.edges[0].error or "")


def test_script_requires_script_or_command(tmp_path):
    data = {"name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
            "edges": [{"id": "e", "kind": "script", "input_tags": ["a"],
                       "output_tags": ["out"]}]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="script 或 command"):
        cfg_mod.load(p)
