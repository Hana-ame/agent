"""模型记分卡的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.models import format_report, load_catalog, scorecard  # noqa: E402
from dag8.store import Store  # noqa: E402


def _prices(tmp_path: Path) -> Path:
    p = tmp_path / "prices.json"
    p.write_text(json.dumps({
        "laptop": {"models": {
            "deepseek-flash": {"price_in": 1, "price_out": 2},
            "glm": {"price_in": 3, "price_out": 4},
        }}
    }), encoding="utf-8")
    return p


def _seed(tmp_path: Path) -> Path:
    db = tmp_path / "m.sqlite3"
    with Store(db) as st:
        r = st.new_run("d", "cfg", "seed", "S")
        c = st.add_content(r, ["out"], "x")
        st.log_edge(r, "e", 0, ["seed"], ["out"], model="laptop/deepseek-flash",
                    status="ok", elapsed=5.0, content_id=c)
        st.finish_run(r, "ok")
        r2 = st.new_run("d", "cfg", "seed", "S")
        st.log_edge(r2, "e", 0, ["seed"], ["out"], model="other/x",
                    status="ok", elapsed=7.0)
        st.finish_run(r2, "ok")
    return db


def test_scorecard_joins_prices(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        rows = {r["key"]: r for r in scorecard(st, load_catalog(_prices(tmp_path)))}

    ds = rows["laptop/deepseek-flash"]
    assert ds["price_in"] == 1 and ds["price_out"] == 2
    assert ds["n"] == 1 and ds["ok_rate"] == 1.0 and ds["p50"] == 5.0

    glm = rows["laptop/glm"]
    assert glm["n"] == 0 and glm["ok_rate"] is None
    assert glm["price_in"] == 3


def test_unknown_model_has_no_price(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        rows = {r["key"]: r for r in scorecard(st, load_catalog(_prices(tmp_path)))}
    other = rows["other/x"]
    assert other["price_in"] is None
    assert other["n"] == 1


def test_format_report(tmp_path):
    db = _seed(tmp_path)
    with Store(db) as st:
        report = format_report(scorecard(st, load_catalog(_prices(tmp_path))))
    assert "in/1M" in report
    assert "laptop/deepseek-flash" in report


def test_cli_models_json(tmp_path, capsys):
    db = _seed(tmp_path)
    prices = _prices(tmp_path)
    assert dag8_main(["--db", str(db), "models", "--providers", str(prices),
                      "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    keys = {r["key"] for r in data}
    assert "laptop/deepseek-flash" in keys
    assert "laptop/glm" in keys
