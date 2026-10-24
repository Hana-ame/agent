"""离线 demo 的测试。"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.demo import run_demo  # noqa: E402


def test_run_demo_offline():
    res, tmp = run_demo()
    try:
        assert res.status == "ok"
        assert res.result == "A|B"
        assert len(res.edges) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_demo(capsys):
    assert dag8_main(["demo", "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "A|B" in out
