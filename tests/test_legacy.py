"""v6 → v7 收编关系的守门测试。

v6 已降级为自举子系统（legacy）：允许它存在、允许它慢，但不允许
① 拖慢 `import dag_agent`  ② 门面断链  ③ 固定点被悄悄改坏。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_import_dag_agent_does_not_pull_in_v6():
    """惰性门面：普通 import 不该把 1386 行的 bootstrap 模块拉起来。"""
    code = (
        "import sys, dag_agent\n"
        "print('vea_v6' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_legacy_facade_exposes_v6_public_api():
    from dag_agent.legacy import v6

    for name in ("DAG", "Edge", "ContentItem", "bootstrap", "verify_fixed_point", "SmartMockLLM"):
        assert hasattr(v6, name), f"legacy 门面缺 {name}"


def test_v6_fixed_point_still_holds():
    """搬家/改路径会让这条断掉：它是判断「v6 还能不能自复现」的唯一硬指标。"""
    from dag_agent.legacy import v6

    ok, why = v6.verify_fixed_point()
    assert ok, why


def test_v7_and_v6_dag_types_are_distinct():
    """两套 DAG 不是同一个类型，别混用（v6 的只服务自举链）。"""
    import dag_agent
    from dag_agent.legacy import v6

    assert dag_agent.DAG is not v6.DAG
