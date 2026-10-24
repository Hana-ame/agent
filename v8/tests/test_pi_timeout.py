"""pi 超时总预算 / 快速错误重试的离线测试（用假 pi 可执行文件）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
import dag8.pi as pi_mod  # noqa: E402


def _fake_pi(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "pi"
    p.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    p.chmod(0o755)
    return p


def _cfg(timeout: float, retries: int) -> dag8.PiConfig:
    return dag8.PiConfig(provider="laptop", model="deepseek-flash",
                         timeout=timeout, retries=retries, ssh_host=None)


def test_pi_timeout_is_total_budget(tmp_path, monkeypatch):
    fake = _fake_pi(tmp_path, "sleep 5")
    monkeypatch.setattr(pi_mod, "find_pi", lambda: str(fake))

    t0 = time.perf_counter()
    with pytest.raises(dag8.PiError, match="总预算"):
        dag8.PiAgent(_cfg(timeout=0.4, retries=3)).run("hi")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"总预算没生效：{elapsed:.2f}s"


def test_pi_retries_quick_429(tmp_path, monkeypatch):
    counter = tmp_path / "n"
    fake = _fake_pi(tmp_path, f'''
n=$(cat "{counter}" 2>/dev/null || echo 0)
n=$((n+1))
echo "$n" > "{counter}"
if [ "$n" -eq 1 ]; then
  echo "429 rate_limit" >&2
  exit 1
fi
cat >/dev/null
echo ok
''')
    monkeypatch.setattr(pi_mod, "find_pi", lambda: str(fake))

    out = dag8.PiAgent(_cfg(timeout=10, retries=2)).run("hi")
    assert out == "ok"
    assert counter.read_text().strip() == "2"
