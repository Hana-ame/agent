"""汇报检查 DAG 的离线测试。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402

DIR = V8 / "configs" / "report_check"
SCRIPT = DIR / "check_report.py"
CONFIG = DIR / "dag.json"


def _metrics(text: str) -> dict:
    out = subprocess.run([sys.executable, str(SCRIPT)], input=text,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_metrics_flags_bad_report():
    text = (DIR / "examples" / "bad.md").read_text(encoding="utf-8")
    data = _metrics(text)
    assert data["empty"] is False
    assert data["sections"]["conclusion"] is False
    assert data["sections"]["next_steps"] is False
    assert "DAG" in data["jargon"]
    assert "大概" in data["vague"]


def test_metrics_accepts_structured_report():
    text = (DIR / "examples" / "structured.md").read_text(encoding="utf-8")
    data = _metrics(text)
    assert data["sections"]["conclusion"] is True
    assert data["sections"]["evidence"] is True
    assert data["sections"]["next_steps"] is True
    assert data["first_screen_conclusion"] is True
    assert data["numbers"] >= 3


def test_report_check_lint_and_dry_run(tmp_path):
    assert dag8_main(["lint", str(CONFIG)]) == 0
    res = dag8.run(CONFIG, seed_content="# 汇报\n\n## 结论\n\n这是一段示例。\n",
                   db_path=tmp_path / "r.sqlite3", dry_run=True)
    assert res.status == "ok", res.summary()
    assert len(res.edges) == 5
    assert res.result
