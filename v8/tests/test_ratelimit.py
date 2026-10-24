"""RPM 限速器与调度器节流的离线测试。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8.ratelimit import RateLimiter  # noqa: E402


def test_limiter_noop_when_disabled():
    lim = RateLimiter(None)
    t0 = time.perf_counter()
    for _ in range(5):
        lim.acquire()
    assert time.perf_counter() - t0 < 0.05


def test_limiter_waits_for_window():
    lim = RateLimiter(rpm=2, window=0.1)
    t0 = time.perf_counter()
    for _ in range(3):
        lim.acquire()
    assert time.perf_counter() - t0 >= 0.09


class _Agent:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, **_kw):
        self.calls += 1
        return f"out{self.calls}"


def test_scheduler_rpm_spaces_api_calls(tmp_path):
    data = {
        "name": "rl",
        "seed_tag": "task",
        "seed_content": "Q",
        "result_tag": "a",
        "edges": [
            {"id": "a", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
             "prompt": "A {contents}"},
            {"id": "b", "layer": 0, "input_tags": ["task"], "output_tags": ["b"],
             "prompt": "B {contents}"},
        ],
    }
    cfg = tmp_path / "dag.json"
    cfg.write_text(json.dumps(data), encoding="utf-8")

    agent = _Agent()
    t0 = time.perf_counter()
    res = dag8.run(cfg, db_path=tmp_path / "rl.sqlite3", agent=agent,
                   max_workers=2, rpm=1, rate_window=0.2)
    elapsed = time.perf_counter() - t0

    assert res.status == "ok", res.summary()
    assert agent.calls == 2
    assert elapsed >= 0.18, f"第二次调用应被窗口挡住：{elapsed:.2f}s"


class _Inner:
    def __init__(self):
        self.completes = 0

    def complete(self, *a, **k):
        self.completes += 1
        return "ok"

    def embed(self, *a, **k):
        return [0.0] * 32


class _Lim:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        self.calls += 1


def test_limited_client_counts_complete_only():
    from dag8.nodes import _LimitedClient

    inner, lim = _Inner(), _Lim()
    client = _LimitedClient(inner, lim)
    client.complete("x")
    client.embed("y")
    client.complete("z")
    assert lim.calls == 2
    assert inner.completes == 2


def test_graph_node_limits_each_v7_call(tmp_path):
    import json

    v7 = {
        "name": "sub",
        "models": {"mock": {"api": "http://unused.invalid/v1", "key": "not-needed", "model": "x"}},
        "edges": [
            {"input_tags": ["a"], "output_tags": ["b"], "layer": 0,
             "processor": {"model": "mock", "prompt": "1 {contents}"}},
            {"input_tags": ["b"], "output_tags": ["final"], "layer": 1,
             "processor": {"model": "mock", "prompt": "2 {contents}"}},
        ],
    }
    (tmp_path / "v7.json").write_text(json.dumps(v7), encoding="utf-8")
    cfg = {
        "name": "t", "seed_tag": "a", "seed_content": "S", "result_tag": "final",
        "edges": [{"id": "g", "kind": "graph", "layer": 0,
                   "input_tags": ["a"], "output_tags": ["final"],
                   "graph": {"path": "v7.json", "seed_tag": "a", "result_tag": "final"}}],
    }
    cfg_path = tmp_path / "dag.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    client = _Inner()
    t0 = time.perf_counter()
    res = dag8.run(cfg_path, db_path=tmp_path / "g.sqlite3",
                   graph_llm_client=client, rpm=1, rate_window=0.15)
    elapsed = time.perf_counter() - t0

    assert res.status == "ok", res.summary()
    assert client.completes == 2
    assert elapsed >= 0.12, f"两次 v7 调用应被错开：{elapsed:.2f}s"


def test_shared_limiter_reuses_instance():
    from dag8.ratelimit import shared_limiter

    a = shared_limiter(2, 60)
    assert a is shared_limiter(2, 60)
    assert shared_limiter(3, 60) is not a
