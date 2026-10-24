"""explain 的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.explain import explain, format_report  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed_error(tmp_path: Path) -> tuple[Path, str]:
    db = tmp_path / "e.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        c = st.add_content(r, ["mid"], "x")
        st.log_edge(r, "e1", 0, ["seed"], ["mid"], model="m",
                    status="ok", elapsed=5.0, content_id=c)
        st.log_edge(r, "e2", 1, ["mid"], ["out"], model="m",
                    status="error", elapsed=10.0, error="429 Too Many Requests")
        st.finish_run(r, "error", error="429")
    return db, r


def _seed_skipped(tmp_path: Path) -> tuple[Path, str]:
    db = tmp_path / "s.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        st.log_edge(r, "e1", 0, ["seed"], ["mid"], model="m",
                    status="ok", elapsed=1.0)
        st.log_edge(r, "e2", 1, ["mid", "missing"], ["out"], model="m",
                    status="skipped", error="缺输入 tag: ['missing']")
        st.finish_run(r, "incomplete", result_tag="out")
    return db, r


def test_explain_error_429(tmp_path):
    db, rid = _seed_error(tmp_path)
    with Store(db) as st:
        data = explain(st, rid)
    assert data["run"]["status"] == "error"
    assert data["result_chars"] is None
    joined = " ".join(data["hints"])
    assert "--rpm" in joined
    assert "--resume" in joined


def test_explain_skipped(tmp_path):
    db, rid = _seed_skipped(tmp_path)
    with Store(db) as st:
        data = explain(st, rid)
    joined = " ".join(data["hints"])
    assert "optional_inputs" in joined
    assert any(e["status"] == "skipped" for e in data["edges"])


def test_format_report(tmp_path):
    db, rid = _seed_error(tmp_path)
    with Store(db) as st:
        report = format_report(explain(st, rid))
    assert "建议" in report
    assert "429" in report


def test_cli_explain_json(tmp_path, capsys):
    db, rid = _seed_error(tmp_path)
    assert dag8_main(["--db", str(db), "explain", rid, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run"]["run_id"] == rid
    assert data["edges"]


def test_explain_partial(tmp_path):
    db = tmp_path / "p.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        c = st.add_content(r, ["mid"], "x")
        st.log_edge(r, "e1", 0, ["seed"], ["mid"], status="ok", elapsed=1.0,
                    content_id=c)
        st.log_edge(r, "e2", 1, ["mid"], ["out"], status="error", elapsed=2.0,
                    error="boom")
        st.finish_run(r, "partial", error="boom")
    with Store(db) as st:
        data = explain(st, r)
    joined = " ".join(data["hints"])
    assert "部分失败" in joined
    assert "--resume-last" in joined


def test_explain_over_budget(tmp_path):
    db = tmp_path / "b.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        st.log_edge(r, "e", 0, ["seed"], ["out"], status="skipped",
                    error="超出成本预算 1.0")
        st.finish_run(r, "over_budget", error="超出成本预算 1.0")
    with Store(db) as st:
        data = explain(st, r)
    assert any("成本" in h for h in data["hints"])
