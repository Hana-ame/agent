"""prune（旧运行清理）的离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.store import Store  # noqa: E402


def _seed(tmp_path: Path, n: int = 3) -> Path:
    db = tmp_path / "p.sqlite3"
    with Store(db) as st:
        for i in range(n):
            r = st.new_run(f"d{i}", "cfg", "seed", "S")
            c = st.add_content(r, ["out"], "x")
            st.log_edge(r, "e", 0, ["seed"], ["out"], status="ok", content_id=c)
            st.finish_run(r, "ok", result_tag="out", result_content_id=c)
    return db


def _count(db: Path, table: str) -> int:
    with Store(db) as st:
        return st.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_prune_dry_run(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        assert st.prune(1, dry_run=True) == 2
    assert _count(db, "runs") == 3
    assert _count(db, "contents") == 3


def test_prune_apply(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        assert st.prune(1, dry_run=False) == 2
    assert _count(db, "runs") == 1
    assert _count(db, "contents") == 1
    assert _count(db, "edges") == 1


def test_cli_prune(tmp_path):
    db = _seed(tmp_path)
    assert dag8_main(["--db", str(db), "prune", "--keep", "1"]) == 0
    assert _count(db, "runs") == 3

    assert dag8_main(["--db", str(db), "prune", "--keep", "1", "--apply"]) == 0
    assert _count(db, "runs") == 1
