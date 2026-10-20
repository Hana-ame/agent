"""普通 Python 模块：CF 数据拉取与验证，作为 dag 的函数边（processor.fn）被 step() 调用。

真实环境：直接 subprocess 调仓库里的 tools/cf_usage.py + tools/cf_verify.sh。
本工作区没有这两脚本，故内置示例数据兜底（与用户笔记中 2026-10-13~19 的数字一致）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_CF_USAGE = Path(__file__).resolve().parent.parent / "tools" / "cf_usage.py"
_CF_VERIFY = Path(__file__).resolve().parent.parent / "tools" / "cf_verify.sh"

# 用户笔记中的示例数据（7 天，3 域）
SAMPLE_USAGE = {
    "window": "2026-10-13 ~ 2026-10-19",
    "total_requests": 42_881_383,
    "total_traffic_gb": 5_526.89,
    "by_domain": [
        {"domain": "moonchan.xyz", "requests": 22_722_235, "traffic_gb": 5_347.99},
        {"domain": "810114.xyz", "requests": 13_009_968, "traffic_gb": 100.09},
        {"domain": "4545810.xyz", "requests": 7_149_180, "traffic_gb": 78.81},
    ],
    "daily_trend": [
        {"date": "2026-10-18", "requests": 3_600_000, "traffic_gb": 997.72},
        {"date": "2026-10-19", "requests": 3_710_000, "traffic_gb": 453.78},
    ],
}


def fetch_cf_usage(days: int = 7) -> str:
    """拉取 CF 数据，返回 JSON 字符串（函数边产出）。

    - 若 tools/cf_usage.py 存在：subprocess 调 `--days {days}` 拿真实数据。
    - 否则：返回内置示例数据，并打上 source=sample 标记。
    """
    if _CF_USAGE.exists():
        out = subprocess.run(
            [sys.executable, str(_CF_USAGE), "--days", str(days)],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode == 0 and out.stdout.strip():
            try:
                data = json.loads(out.stdout)
                data["source"] = "live:tools/cf_usage.py"
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                pass  # 脚本输出非 JSON，回落示例
    data = dict(SAMPLE_USAGE)
    data["source"] = "sample"
    data["note"] = "工作区无 tools/cf_usage.py，返回内置示例数据（真实环境会自动调脚本）"
    return json.dumps(data, ensure_ascii=False)


def verify_checks(data: str) -> str:
    """对 CF 数据跑三道闸（quick 版：唯一 + 自洽），返回 JSON 字符串。

    - 若 tools/cf_verify.sh 存在：subprocess 调 `--quick` 拿真实校验输出。
    - 否则：对传入 JSON 做确定性校验：域请求/流量加总 = 合计；占比 <= 100%。
    """
    if _CF_VERIFY.exists():
        out = subprocess.run(
            ["bash", str(_CF_VERIFY), "--quick"],
            capture_output=True, text=True, timeout=120,
        )
        return json.dumps({
            "check": "cf_verify.sh --quick",
            "exit_code": out.returncode,
            "output": out.stdout.strip(),
            "pass": out.returncode == 0,
        }, ensure_ascii=False)

    usage = json.loads(data)
    checks = {}
    # ① 唯一：域列表无重复
    domains = [d["domain"] for d in usage.get("by_domain", [])]
    checks["unique_domains"] = len(domains) == len(set(domains))
    # ② 自洽：分域求和 == 合计
    sum_req = sum(d["requests"] for d in usage.get("by_domain", []))
    sum_gb = sum(d["traffic_gb"] for d in usage.get("by_domain", []))
    checks["sum_requests_eq_total"] = sum_req == usage.get("total_requests")
    checks["sum_traffic_eq_total"] = abs(sum_gb - usage.get("total_traffic_gb")) < 0.01
    # 占比 <= 100%
    checks["shares_le_100pct"] = all(
        d["requests"] <= usage.get("total_requests", 0)
        for d in usage.get("by_domain", [])
    )
    passed = all(checks.values())
    return json.dumps({
        "check": "local-quick (unique + self-consistent)",
        "checks": checks,
        "pass": passed,
    }, ensure_ascii=False)
