"""show / runs 的 JSON 输出测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed(tmp_path: Path):
    db = tmp_path / "s.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        c = st.add_content(r, ["out"], "x")
        st.log_edge(r, "e", 0, ["seed"], ["out"], model="m",
                    status="ok", elapsed=1.0, content_id=c)
        st.finish_run(r, "ok", result_tag="out", result_content_id=c)
    return db, r


def test_show_json(tmp_path, capsys):
    db, r = _seed(tmp_path)
    assert dag8_main(["--db", str(db), "show", r, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == r
    assert data["edges"][0]["edge"] == "e"
    assert data["contents"][0]["chars"] == 1


def test_runs_json(tmp_path, capsys):
    db, r = _seed(tmp_path)
    assert dag8_main(["--db", str(db), "runs", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert data[0]["run_id"] == r


def test_show_unknown_run_exit_1(tmp_path, capsys):
    db = tmp_path / "empty.sqlite3"
    with Store(db):
        pass
    assert dag8_main(["--db", str(db), "show", "nope"]) == 1
    assert "不存在" in capsys.readouterr().out
