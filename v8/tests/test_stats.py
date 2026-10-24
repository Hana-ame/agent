"""运行统计的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.stats import collect, format_report  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed(tmp_path: Path) -> Path:
    db = tmp_path / "s.sqlite3"
    with Store(db) as st:
        r1 = st.new_run("d1", "cfg", "seed", "S")
        c1 = st.add_content(r1, ["out"], "x")
        st.log_edge(r1, "e1", 0, ["seed"], ["out"],
                    model="laptop/deepseek-flash", status="ok",
                    elapsed=5.0, content_id=c1)
        st.finish_run(r1, "ok", result_tag="out", result_content_id=c1)

        r2 = st.new_run("d2", "cfg", "seed", "S")
        st.log_edge(r2, "e2", 0, ["seed"], ["out"],
                    model="laptop/deepseek-flash", status="error",
                    elapsed=10.0, error="429 Too Many Requests")
        st.finish_run(r2, "error", error="429")

        r3 = st.new_run("d3", "cfg", "seed", "S")
        c3 = st.add_content(r3, ["out"], "y")
        st.log_edge(r3, "e3", 0, ["seed"], ["out"],
                    model="laptop/glm", status="ok", elapsed=20.0,
                    content_id=c3, cached=True)
        st.finish_run(r3, "ok", result_tag="out", result_content_id=c3)
    return db


def test_collect(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        data = collect(st)
    assert data["runs"]["n"] == 3
    assert data["runs"]["by_status"]["ok"] == 2
    assert data["runs"]["by_status"]["error"] == 1

    ds = [m for m in data["models"] if m["model"] == "laptop/deepseek-flash"][0]
    assert ds["n"] == 2
    assert ds["ok_rate"] == 0.5
    assert ds["p50"] == 5.0

    glm = [m for m in data["models"] if m["model"] == "laptop/glm"][0]
    assert glm["cached"] == 1
    assert any("429" in e["message"] for e in data["errors"])


def test_format_report(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        report = format_report(collect(st))
    assert "模型" in report
    assert "错误 Top" in report
    assert "429" in report


def test_empty_db(tmp_path):
    db = tmp_path / "empty.sqlite3"
    with Store(db) as st:
        data = collect(st)
    assert data["runs"]["n"] == 0
    assert format_report(data)


def test_cli_stats_json(tmp_path, capsys):
    db = _seed(tmp_path)
    assert dag8_main(["--db", str(db), "stats", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"]["n"] == 3
    assert data["models"]


def test_cached_calls_do_not_skew_latency(tmp_path):
    db = tmp_path / "c.sqlite3"
    with Store(db) as st:
        r = st.new_run("d", "cfg", "seed", "S")
        c = st.add_content(r, ["out"], "x")
        st.log_edge(r, "fast", 0, ["seed"], ["out"], model="laptop/deepseek-flash",
                    status="ok", elapsed=0.0, cached=True, content_id=c)
        st.log_edge(r, "real", 0, ["seed"], ["out"], model="laptop/deepseek-flash",
                    status="ok", elapsed=10.0, content_id=c)
        st.finish_run(r, "ok")
    with Store(db) as st:
        data = collect(st)
    m = [x for x in data["models"] if x["model"] == "laptop/deepseek-flash"][0]
    assert m["p50"] == 10.0
    assert m["cached"] == 1
    assert m["n_uncached"] == 1
