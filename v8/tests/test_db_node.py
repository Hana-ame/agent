"""db 节点的本地 SQLite 集成测试。"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "data.sqlite3"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO items VALUES (?, ?)",
                     [(1, "a"), (2, "b"), (3, "c")])
    conn.commit()
    conn.close()
    return p


def _cfg(tmp_path: Path, **edge) -> Path:
    data = {
        "name": "db_demo", "seed_tag": "task", "seed_content": "0",
        "result_tag": "out",
        "edges": [{"id": "q", "kind": "db", "input_tags": ["task"],
                   "output_tags": ["out"],
                   "db_path": "data.sqlite3",
                   "query": "SELECT name FROM items WHERE id > ?",
                   "params": ["{tag:task}"], **edge}],
    }
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_db_query(tmp_path):
    _db(tmp_path)
    res = dag8.run(_cfg(tmp_path), db_path=tmp_path / "r.sqlite3")
    assert res.status == "ok", res.summary()
    data = json.loads(res.result)
    assert data["columns"] == ["name"]
    assert data["rows"] == [["a"], ["b"], ["c"]]


def test_db_max_rows_truncates(tmp_path):
    _db(tmp_path)
    res = dag8.run(_cfg(tmp_path, max_rows=2), db_path=tmp_path / "r.sqlite3")
    data = json.loads(res.result)
    assert data["rows"] == [["a"], ["b"]]
    assert data["truncated"] is True


def test_db_read_only_blocks_writes(tmp_path):
    _db(tmp_path)
    res = dag8.run(_cfg(tmp_path, query="DELETE FROM items", params=[]),
                   db_path=tmp_path / "r.sqlite3")
    assert res.status == "error"
    assert "readonly" in (res.edges[0].error or "").lower()


def test_db_config_requires_query(tmp_path):
    data = {"name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
            "edges": [{"id": "q", "kind": "db", "input_tags": ["a"],
                       "output_tags": ["out"], "db_path": "x.sqlite3"}]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="query"):
        cfg_mod.load(p)
