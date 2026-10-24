"""CLI 面审计：所有子命令的 --help 都应正常退出。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.__main__ import main as dag8_main  # noqa: E402

COMMANDS = ["run", "show", "runs", "explain", "diff", "models", "stats",
            "convert", "cost", "lint", "doctor", "new", "scaffold",
            "demo", "abort", "prune", "cache"]


def test_all_subcommands_have_help():
    for cmd in COMMANDS:
        with pytest.raises(SystemExit) as exc:
            dag8_main([cmd, "--help"])
        assert exc.value.code == 0, cmd
