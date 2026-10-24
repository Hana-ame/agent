"""provider 自动路由的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.router import choose_model  # noqa: E402
from dag8.store import Store  # noqa: E402

CATALOG = [
    {"provider": "laptop", "model": "fast", "price_in": 0, "price_out": 0},
    {"provider": "laptop", "model": "slow", "price_in": 100, "price_out": 100},
]


def _add(st, model: str, status: str, elapsed: float, i: int) -> None:
    r = st.new_run(f"d{i}", "cfg", "seed", "S")
    st.log_edge(r, "e", 0, ["seed"], ["out"], model=model,
                status=status, elapsed=elapsed)
    st.finish_run(r, status)


def test_choose_prefers_success_and_speed(tmp_path):
    db = tmp_path / "r.sqlite3"
    with Store(db) as st:
        for i in range(3):
            _add(st, "laptop/fast", "ok", 5.0, i)
        _add(st, "laptop/slow", "ok", 40.0, 10)
        _add(st, "laptop/slow", "error", 40.0, 11)
        best = choose_model(st, CATALOG)
    assert best is not None
    assert best["key"] == "laptop/fast"


def test_price_weight_can_flip(tmp_path):
    db = tmp_path / "r.sqlite3"
    with Store(db) as st:
        _add(st, "laptop/fast", "ok", 1.0, 0)
        _add(st, "laptop/slow", "ok", 1.0, 1)
        cheap = choose_model(st, CATALOG, price_weight=1.0)
        free = choose_model(st, CATALOG)
    assert cheap["key"] == "laptop/fast"   # fast 价格 0，权重后仍优
    assert free["key"] in ("laptop/fast", "laptop/slow")


def test_min_calls_filters(tmp_path):
    db = tmp_path / "r.sqlite3"
    with Store(db) as st:
        _add(st, "laptop/fast", "ok", 1.0, 0)
        best = choose_model(st, CATALOG, min_calls=3)
    assert best is None


def test_cli_route_json(tmp_path, capsys):
    db = tmp_path / "r.sqlite3"
    with Store(db) as st:
        for i in range(2):
            _add(st, "laptop/deepseek-flash", "ok", 5.0, i)
        _add(st, "laptop/glm-5.2", "error", 40.0, 9)
    assert dag8_main(["--db", str(db), "route", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["key"] == "laptop/deepseek-flash"
