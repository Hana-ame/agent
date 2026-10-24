"""真实调用 pi 的测试。

默认就跑（用户要求测试必须真的碰 pi）。pi 不可用时直接 fail，不会假绿：
`_probe.py` 只做 `command -v pi`，不调模型。本机有 pi 时自动走本地，
没有本地 pi 才走 `DAG8_SSH_HOST`（可用 `DAG8_LOCAL_PI=1/0` 强制）。

三种粒度：

1. 单次直调 —— 验证 PiAgent → ssh → pi → 模型这条链路（约 5~20 秒）
2. 边级执行 —— 验证调度器把 prompt 交给节点、结果写进 sqlite（约 10 秒）
3. 整图三边 —— 验证 layer 调度与多输入拼接（约 45 秒，标 slow）

跳过方式：
    cd v8 && python -m pytest tests/ -q -m "not slow"   # 跳过整图那条
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import PiConfig  # noqa: E402
from dag8.store import Store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe import pi_probe  # noqa: E402

AVAILABLE, WHY = pi_probe()
pytestmark = [pytest.mark.live]


@pytest.fixture(autouse=True)
def _require_pi():
    """pi 不可用就直接失败。

    不用 skip：skip 会把「环境没准备好」伪装成「没问题」——CI 上悄悄跳过
    跟全部通过长得一样，这是假绿。宁可红。
    """
    if not AVAILABLE:
        pytest.fail(f"pi 不可用：{WHY}", pytrace=False)


def _pi_cfg(**kw) -> PiConfig:
    """跟 configs/demo.json 一致：pi 在 WSL，Windows 侧走 ssh。"""
    host = os.environ.get("DAG8_SSH_HOST", "lumin@172.29.89.192")
    key = os.environ.get("DAG8_SSH_KEY", "~/.ssh/id_rsa")
    # 本机有 pi 就优先本地；没有才走 SSH；显式环境变量可覆盖
    local = AVAILABLE and WHY.startswith("local:")
    force = os.environ.get("DAG8_LOCAL_PI")
    use_local = force == "1" or (force != "0" and local)
    base = dict(provider="laptop", model="deepseek-v4-flash", tools=False,
                timeout=120.0, retries=3,
                ssh_host=None if use_local else host,
                ssh_key=key)
    base.update(kw)
    return PiConfig(**base)


# ---------------------------------------------------------------------------
# 1. 单次直调：链路是否活着
# ---------------------------------------------------------------------------
def test_pi_direct_call_returns_text():
    agent = dag8.PiAgent(_pi_cfg())
    t0 = time.perf_counter()
    out = agent.run("只回复两个字：收到")
    elapsed = time.perf_counter() - t0
    assert out.strip(), "pi 返回空内容"
    assert "收到" in out, f"预期「收到」，实际：{out[:200]!r}"
    assert elapsed < 120, f"单次调用耗时 {elapsed:.1f}s，超出预期"


def test_pi_reports_error_for_unknown_provider():
    """provider 写错时必须抛 PiError，而不是静默返回空串。"""
    agent = dag8.PiAgent(_pi_cfg(provider="no-such-provider", retries=1))
    with pytest.raises(dag8.PiError):
        agent.run("hi")


# ---------------------------------------------------------------------------
# 2. 边级：scheduler + store 打通
# ---------------------------------------------------------------------------
def test_single_edge_writes_to_sqlite(tmp_path):
    cfg_path = tmp_path / "one.json"
    # 拷 demo.json，但去掉输出契约：这个用例喂的是 2 个字的种子，
    # 不该被 min_chars=80 拦住（契约本身由 test_contracts / test_dag8 覆盖）。
    import json as _json
    data = _json.loads((V8 / "configs" / "demo.json").read_text(encoding="utf-8"))
    for edge in data.get("edges", []):
        edge.pop("expect", None)
        edge.pop("repair", None)
    cfg_path.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    res = dag8.run(
        cfg_path,
        seed_content="只回复两个字：收到",
        db_path=tmp_path / "live.sqlite3",
        agent=dag8.PiAgent(_pi_cfg(timeout=120)),
    )

    # demo.json 有三条边；seed 改成极短任务后 draft 会产出，后面两条也会走完
    assert res.status == "ok", res.summary()
    assert res.result and len(res.result) > 10
    assert all(e.status == "ok" for e in res.edges), res.summary()

    with Store(tmp_path / "live.sqlite3") as st:
        contents = st.all_contents(res.run_id)
        assert len(contents) == 1 + len(res.edges)
        assert all(e["elapsed"] and e["elapsed"] > 0 for e in st.edge_records(res.run_id))
        assert st.run_info(res.run_id)["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. 整图三边（慢）
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_pipeline_live(tmp_path):
    res = dag8.run(
        V8 / "configs" / "demo.json",
        db_path=tmp_path / "live_full.sqlite3",
        agent=dag8.PiAgent(_pi_cfg(timeout=180)),
    )
    assert res.status == "ok", res.summary()
    assert len(res.edges) == 3
    assert [e.status for e in res.edges] == ["ok", "ok", "ok"]
    assert res.result and len(res.result) > 50

    # 多输入那条边（draft + critique）确实拿到了两份输入
    final_edge = res.edges[-1]
    assert final_edge.prompt_chars > 100
