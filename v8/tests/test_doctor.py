"""doctor 健康检查的离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8 import doctor  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402


class _Ok:
    def run(self, prompt, **kw):
        return "就绪"


class _Down:
    def run(self, prompt, **kw):
        raise doctor.PiError("pi 超时（5s）", retryable=True)


def test_check_pi_missing(monkeypatch):
    monkeypatch.setattr(doctor, "find_pi", lambda: None)
    c = doctor._check_pi()
    assert not c.ok
    assert "ssh_host" in c.hint


def test_check_gateway_ok():
    c = doctor._check_gateway("laptop", "m", 5, agent=_Ok())
    assert c.ok
    assert "就绪" in c.detail


def test_check_gateway_down_hint():
    c = doctor._check_gateway("laptop", "m", 5, agent=_Down())
    assert not c.ok
    assert "--rpm" in c.hint


def test_format_report():
    report = doctor.format_report([doctor.Check("x", False, "bad", "fix it")])
    assert "✗ x" in report
    assert "建议：fix it" in report


def test_check_all_skip_gateway(monkeypatch):
    monkeypatch.setattr(doctor, "find_pi", lambda: "/bin/pi")
    checks = doctor.check_all(skip_gateway=True)
    assert len(checks) == 2
    assert all(c.ok for c in checks)


def test_cli_doctor_skip(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "find_pi", lambda: "/bin/pi")
    assert dag8_main(["doctor", "--skip-gateway"]) == 0
    out = capsys.readouterr().out
    assert "pi 二进制" in out
    assert "provider 注册表" in out


def test_cli_doctor_failure_exit(monkeypatch):
    monkeypatch.setattr(
        doctor, "check_all",
        lambda *a, **k: [doctor.Check("x", False, "bad", "")],
    )
    assert dag8_main(["doctor", "--skip-gateway"]) == 1


class _Resp:
    def __init__(self, code: int):
        self.status_code = code


def test_http_check_530(monkeypatch):
    monkeypatch.setattr(doctor.requests, "get", lambda *a, **k: _Resp(530))
    c = doctor._check_http("laptop", 5)
    assert not c.ok
    assert "530" in c.detail
    assert "源站" in c.hint


def test_http_check_ok(monkeypatch):
    monkeypatch.setattr(doctor.requests, "get", lambda *a, **k: _Resp(401))
    c = doctor._check_http("laptop", 5)
    assert c.ok, "401/403 说明端点可达，只是鉴权"


def test_http_check_network_error(monkeypatch):
    def boom(*a, **k):
        raise doctor.requests.RequestException("dns fail")
    monkeypatch.setattr(doctor.requests, "get", boom)
    c = doctor._check_http("laptop", 5)
    assert not c.ok
    assert "dns fail" in c.detail
