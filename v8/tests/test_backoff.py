"""抖动退避的离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.backoff import jittered  # noqa: E402


def test_jitter_bounds():
    assert jittered(2.0, 0, rand=lambda: 0.0) == 1.0
    assert jittered(2.0, 0, rand=lambda: 1.0) == 3.0
    assert jittered(2.0, 1, rand=lambda: 0.5) == 4.0


def test_jitter_cap():
    assert jittered(100.0, 5, cap=60.0, rand=lambda: 1.0) == 90.0


def test_jitter_zero_base():
    assert jittered(0.0, 3, rand=lambda: 0.5) == 0.0
