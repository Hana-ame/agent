"""成本估算的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.cost import format_report, load_prices, run_costs  # noqa: E402
from dag8.store import Store  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402


def _seed(db: Path, model: str = "laptop/deepseek-flash",
          content: str = "x" * 100, prompt_chars: int = 100) -> str:
    with Store(db) as st:
        rid = st.new_run("demo", "cfg", "seed", "S")
        cid = st.add_content(rid, ["out"], content, gen=1, source_edge="e")
        st.log_edge(rid, "e", 0, ["seed"], ["out"], model=model,
                    prompt_chars=prompt_chars, status="ok", content_id=cid)
        st.finish_run(rid, "ok", result_tag="out", result_content_id=cid)
        return rid


def test_run_costs(tmp_path):
    db = tmp_path / "t.sqlite3"
    rid = _seed(db)
    with Store(db) as st:
        rows = run_costs(st, {("laptop", "deepseek-flash"): (1.0, 2.0)})
    r = rows[0]
    assert r["run_id"] == rid
    assert r["tokens_in"] == 50
    assert r["tokens_out"] == 50
    assert abs(r["cost"] - (50 * 1.0 + 50 * 2.0) / 1e6) < 1e-9


def test_unknown_model_is_flagged(tmp_path):
    db = tmp_path / "t.sqlite3"
    _seed(db, model="graph:v7.json")
    with Store(db) as st:
        rows = run_costs(st, {("laptop", "deepseek-flash"): (1.0, 2.0)})
    assert rows[0]["unknown_models"] == ["graph:v7.json"]
    assert rows[0]["cost"] == 0.0


def test_load_prices(tmp_path):
    p = tmp_path / "prices.json"
    p.write_text(json.dumps({
        "laptop": {"models": {"m": {"price_in": 1, "price_out": 2}}}
    }), encoding="utf-8")
    assert load_prices(p)[("laptop", "m")] == (1.0, 2.0)


def test_format_report(tmp_path):
    db = tmp_path / "t.sqlite3"
    rid = _seed(db)
    with Store(db) as st:
        rows = run_costs(st, {("laptop", "deepseek-flash"): (1.0, 2.0)})
    report = format_report(rows)
    assert rid in report
    assert "合计" in report


def test_cli_cost(tmp_path, capsys):
    db = tmp_path / "t.sqlite3"
    rid = _seed(db)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({
        "laptop": {"models": {"deepseek-flash": {"price_in": 1, "price_out": 2}}}
    }), encoding="utf-8")

    assert dag8_main(["--db", str(db), "cost", "--providers", str(prices)]) == 0
    out = capsys.readouterr().out
    assert rid in out
    assert "合计" in out


def test_cli_cost_json(tmp_path, capsys):
    db = tmp_path / "t.sqlite3"
    rid = _seed(db)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({
        "laptop": {"models": {"deepseek-flash": {"price_in": 1, "price_out": 2}}}
    }), encoding="utf-8")

    assert dag8_main(["--db", str(db), "cost", "--providers", str(prices),
                      "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["run_id"] == rid
