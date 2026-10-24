"""运行对比（diff）的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.diff import diff_runs, format_report  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed(tmp_path: Path):
    db = tmp_path / "d.sqlite3"
    with Store(db) as st:
        a = st.new_run("A", "cfg", "seed", "S")
        st.log_edge(a, "e1", 0, ["seed"], ["mid"], model="m1",
                    status="ok", elapsed=5.0)
        st.log_edge(a, "e2", 1, ["mid"], ["out"], model="m1",
                    status="error", elapsed=10.0, error="boom")
        st.log_edge(a, "only_a", 2, ["out"], ["x"], model="m1",
                    status="ok", elapsed=1.0)
        st.finish_run(a, "error", error="boom")

        b = st.new_run("B", "cfg", "seed", "S")
        st.log_edge(b, "e1", 0, ["seed"], ["mid"], model="m2",
                    status="ok", elapsed=4.0)
        cid = st.add_content(b, ["out"], "result!")
        st.log_edge(b, "e2", 1, ["mid"], ["out"], model="m2",
                    status="ok", elapsed=6.0, content_id=cid)
        st.finish_run(b, "ok", result_tag="out", result_content_id=cid)
    return db, a, b


def test_diff_rows(tmp_path):
    db, a, b = _seed(tmp_path)
    with Store(db) as st:
        data = diff_runs(st, a, b)

    rows = {r["edge"]: r for r in data["edges"]}
    assert rows["e1"]["delta"] == -1.0
    assert rows["e1"]["model_a"] == "m1" and rows["e1"]["model_b"] == "m2"
    assert rows["e2"]["status_a"] == "error" and rows["e2"]["status_b"] == "ok"
    assert rows["e2"]["delta"] == -4.0
    assert rows["only_a"]["status_b"] == "-"
    assert data["a"]["status"] == "error"
    assert data["b"]["result_chars"] == 7


def test_format_report(tmp_path):
    db, a, b = _seed(tmp_path)
    with Store(db) as st:
        report = format_report(diff_runs(st, a, b))
    assert "edge" in report
    assert "only_a" in report
    assert "+" in report or "-" in report


def test_diff_unknown_run(tmp_path):
    db, a, _ = _seed(tmp_path)
    with Store(db) as st:
        try:
            diff_runs(st, a, "nope")
        except KeyError:
            pass
        else:
            raise AssertionError("应当抛 KeyError")


def test_cli_diff_json(tmp_path, capsys):
    db, a, b = _seed(tmp_path)
    assert dag8_main(["--db", str(db), "diff", a, b, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["b"]["status"] == "ok"
    assert any(r["edge"] == "only_a" for r in data["edges"])
