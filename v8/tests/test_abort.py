"""abort（中断标记）的离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.explain import explain  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed_running(tmp_path: Path):
    db = tmp_path / "a.sqlite3"
    with Store(db) as st:
        r = st.new_run("demo", "cfg", "seed", "S")
        c = st.add_content(r, ["mid"], "x")
        st.log_edge(r, "e1", 0, ["seed"], ["mid"], status="ok",
                    elapsed=1.0, content_id=c)
        # 故意不 finish，模拟 Ctrl-C/崩溃
    return db, r


def test_abort_cli(tmp_path):
    db, r = _seed_running(tmp_path)
    assert dag8_main(["--db", str(db), "abort", r]) == 0
    with Store(db) as st:
        assert st.run_info(r)["status"] == "interrupted"
        assert st.abort_run(r) is False


def test_abort_unknown(tmp_path):
    db = tmp_path / "a.sqlite3"
    with Store(db):
        pass
    assert dag8_main(["--db", str(db), "abort", "nope"]) == 1


def test_explain_interrupted_hint(tmp_path):
    db, r = _seed_running(tmp_path)
    with Store(db) as st:
        st.abort_run(r)
    with Store(db) as st:
        data = explain(st, r)
    joined = " ".join(data["hints"])
    assert "中断" in joined
    assert "--resume-last" in joined
