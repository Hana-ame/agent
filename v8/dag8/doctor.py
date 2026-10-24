"""健康检查：pi 二进制、provider 注册表、网关真实调用。

定时轮次/换机器时先跑它，别把「网关挂了」误判成「框架坏了」：

    python -m dag8 doctor
    python -m dag8 doctor --timeout 5 --skip-gateway

返回码：全绿 0，否则 1。
"""
from __future__ import annotations

import json
import time

import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from dag8.pi import PiAgent, PiConfig, PiError, find_pi

V8 = Path(__file__).resolve().parent.parent
PROVIDERS = V8 / "configs" / "providers.json"

__all__ = ["Check", "check_all", "format_report"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _check_pi() -> Check:
    exe = find_pi()
    if exe:
        return Check("pi 二进制", True, exe)
    return Check("pi 二进制", False, "PATH 与 ~/.nvm/versions/node/*/bin 都没有",
                 "安装 pi，或在配置里设置 pi.ssh_host")


def _check_providers() -> Check:
    try:
        data = json.loads(PROVIDERS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check("provider 注册表", False, f"{PROVIDERS}：{exc}",
                     "确认 configs/providers.json 存在且是合法 JSON")
    models = sum(len((p or {}).get("models") or {})
                 for k, p in data.items() if not str(k).startswith("_"))
    providers = [k for k in data if not str(k).startswith("_")]
    return Check("provider 注册表", bool(models),
                 f"{len(providers)} providers / {models} models：{', '.join(providers)}",
                 "检查 configs/providers.json")


def _provider_http(provider: str) -> tuple:
    try:
        data = json.loads(PROVIDERS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    cfg = data.get(provider) or {}
    base = cfg.get("base_url") or cfg.get("api") or ""
    return str(base), str(cfg.get("key") or "")


def _check_http(provider: str, timeout: float) -> Check:
    """原始 HTTP 探测：区分「网络/源站挂了」和「pi 坏了」。"""
    base, key = _provider_http(provider)
    name = f"网关可达性 {provider}"
    if not base:
        return Check(name, False, f"providers.json 里 {provider} 没有 api",
                     "检查 configs/providers.json")
    url = base.rstrip("/") + "/models"
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {key}"},
                            timeout=min(timeout, 5.0))
    except Exception as exc:  # noqa: BLE001
        return Check(name, False, f"{type(exc).__name__}: {exc}",
                     "检查网络/代理；pi 走同一出口，先修通路")
    code = resp.status_code
    if code >= 500:
        return Check(name, False, f"HTTP {code}",
                     "源站/代理故障（530 常见于 Cloudflare origin 不可达），不是配置问题")
    return Check(name, True, f"HTTP {code}（端点可达）")


def _check_gateway(provider: str, model: str, timeout: float,
                   agent: Optional[Any] = None) -> Check:
    cfg = PiConfig(provider=provider, model=model, timeout=timeout, retries=1)
    t0 = time.perf_counter()
    try:
        out = (agent or PiAgent(cfg)).run("只回复两个字：就绪", timeout=timeout)
        elapsed = time.perf_counter() - t0
        return Check(f"网关 {provider}/{model}", True,
                     f"{elapsed:.1f}s，返回 {out.strip()[:20]!r}")
    except PiError as exc:
        elapsed = time.perf_counter() - t0
        msg = str(exc)
        hint = ""
        if "429" in msg or "rate" in msg.lower() or "超时" in msg:
            hint = ("限流/超时：调度加 --rpm，配置 pi.fallbacks；"
                    "graph 子图会放大调用数，优先降并发")
        return Check(f"网关 {provider}/{model}", False,
                     f"{elapsed:.1f}s：{msg[:160]}", hint)


def check_all(provider: str = "laptop", model: str = "deepseek-flash",
              timeout: float = 8.0, *, skip_gateway: bool = False,
              agent: Optional[Any] = None) -> List[Check]:
    checks = [_check_pi(), _check_providers()]
    if not skip_gateway:
        checks.append(_check_http(provider, timeout))
        checks.append(_check_gateway(provider, model, timeout, agent))
    return checks


def format_report(checks: List[Check]) -> str:
    lines = []
    for c in checks:
        mark = "✓" if c.ok else "✗"
        lines.append(f"{mark} {c.name}: {c.detail}")
        if not c.ok and c.hint:
            lines.append(f"  建议：{c.hint}")
    return "\n".join(lines)
