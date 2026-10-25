"""dag8 的离线测试：配置渲染、sqlite 读写、调度器，全部不碰 pi / 网络。

真实 pi 调用只在 `test_live_pi.py` 里做（默认跳过，需要 WSL + pi）。
graph 节点用 v7 的 MockLLMClient 离线验证。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
for p in (str(V8), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
from dag.llm import MockLLMClient as V7MockLLM  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402
from dag8.store import Store  # noqa: E402


# ---------------------------------------------------------------------------
# 配置与占位符
# ---------------------------------------------------------------------------
def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


BASE = {
    "name": "t",
    "seed_tag": "task",
    "seed_content": "hello",
    "result_tag": "out",
    "edges": [
        {
            "id": "e1",
            "layer": 0,
            "input_tags": ["task"],
            "output_tags": ["out"],
            "prompt": "处理：{contents}",
        }
    ],
}


def test_load_config(tmp_path):
    cfg = cfg_mod.load(_write(tmp_path, BASE))
    assert cfg.name == "t"
    assert cfg.seed_tag == "task"
    assert len(cfg.edges) == 1
    assert cfg.edges[0].prompt == "处理：{contents}"
    assert cfg.pi.provider == "laptop"


def test_load_pi_thinking(tmp_path):
    data = dict(BASE, pi={"provider": "laptop", "model": "m", "thinking": "off"})
    cfg = cfg_mod.load(_write(tmp_path, data))
    assert cfg.pi.thinking == "off"


def test_load_rejects_missing_fields(tmp_path):
    bad = dict(BASE, edges=[{"id": "x", "input_tags": ["a"]}])
    with pytest.raises(ValueError, match="缺字段"):
        cfg_mod.load(_write(tmp_path, bad))


def test_render_contents_and_tags():
    inputs = {"draft": ["第一稿"], "critique": ["太啰嗦"]}
    assert cfg_mod.render_prompt("草稿 {tag:draft} / 批评 {tag:critique}", inputs) == "草稿 第一稿 / 批评 太啰嗦"
    assert cfg_mod.render_prompt("全部：{contents}", inputs) == "全部：第一稿\n\n---\n\n太啰嗦"
    assert cfg_mod.render_prompt("最新：{content}", inputs) == "最新：太啰嗦"


def test_render_keeps_unknown_placeholders():
    """prompt 里给模型看的 JSON 示例不能被吃掉。"""
    text = '只输出 {"ai": [1,3]} 这样的格式，数据：{contents}'
    out = cfg_mod.render_prompt(text, {"x": ["DATA"]})
    assert '{"ai": [1,3]}' in out
    assert "DATA" in out


def test_render_multi_tag_shorthand():
    inputs = {"a": ["A"], "b": ["B"]}
    assert cfg_mod.render_prompt("{a,b}", inputs) == "A\n\n---\n\nB"


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------
def test_store_roundtrip(tmp_path):
    with Store(tmp_path / "t.sqlite3") as st:
        rid = st.new_run("d", "cfg.json", "seed", "S")
        cid = st.add_content(rid, ["a", "b"], "内容", gen=1, source_edge="e1")
        st.log_edge(rid, "e1", 0, ["seed"], ["a", "b"], model="m", status="ok",
                    elapsed=0.1, content_id=cid)
        st.finish_run(rid, "ok", result_tag="a", result_content_id=cid)

        got = st.contents_for_tag(rid, "b")
        assert len(got) == 1 and got[0]["content"] == "内容"
        assert st.result(rid, "a") == "内容"
        assert st.run_info(rid)["status"] == "ok"
        assert st.edge_records(rid)[0]["edge_id"] == "e1"


def test_store_isolates_runs(tmp_path):
    with Store(tmp_path / "t.sqlite3") as st:
        r1 = st.new_run("d", "c", "seed", "one")
        r2 = st.new_run("d", "c", "seed", "two")
        st.add_content(r1, ["x"], "A")
        st.add_content(r2, ["x"], "B")
        assert [i["content"] for i in st.contents_for_tag(r1, "x")] == ["A"]
        assert [i["content"] for i in st.contents_for_tag(r2, "x")] == ["B"]


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
def test_scheduler_dry_run(tmp_path):
    cfg_path = _write(tmp_path, BASE)
    res = dag8.run(cfg_path, db_path=tmp_path / "e.sqlite3", dry_run=True)
    assert res.status == "ok"
    assert res.result == "处理：hello"
    assert res.edges[0].status == "ok"


def test_scheduler_missing_input_is_skipped(tmp_path):
    data = dict(
        BASE,
        edges=[
            {"id": "e1", "layer": 0, "input_tags": ["nope"], "output_tags": ["mid"], "prompt": "x"},
            {"id": "e2", "layer": 1, "input_tags": ["mid"], "output_tags": ["out"], "prompt": "y"},
        ],
    )
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "e.sqlite3", dry_run=True)
    assert res.status == "incomplete"
    assert res.edges[0].status == "skipped"
    assert res.edges[1].status == "skipped"


def test_scheduler_chains_two_layers(tmp_path):
    data = dict(
        BASE,
        edges=[
            {"id": "e1", "layer": 0, "input_tags": ["task"], "output_tags": ["mid"], "prompt": "A>{contents}"},
            {"id": "e2", "layer": 1, "input_tags": ["mid"], "output_tags": ["out"], "prompt": "B>{contents}"},
        ],
    )
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "e.sqlite3", dry_run=True)
    assert res.result == "B>A>hello"
    assert [e.status for e in res.edges] == ["ok", "ok"]


class _Boom:
    def run(self, prompt, **_kw):
        raise dag8.PiError("429 rate_limit", retryable=False)


def test_scheduler_records_pi_error(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", agent=_Boom())
    assert res.status == "error"
    assert res.edges[0].status == "error"
    assert "429" in (res.edges[0].error or "")
    with Store(tmp_path / "e.sqlite3") as st:
        assert st.run_info(res.run_id)["status"] == "error"


class _Upper:
    def run(self, prompt, **_kw):
        return prompt.upper()


def test_scheduler_uses_injected_agent(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", agent=_Upper())
    assert res.result == "处理：HELLO"


def test_run_summary_is_readable(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", dry_run=True)
    s = res.summary()
    assert "✓ [0] e1" in s
    assert "result: out" in s


# ---------------------------------------------------------------------------
# graph 节点：一个 v8 节点 = 一张 v7 图（实验特性）
# ---------------------------------------------------------------------------
def _v7_graph(tmp_path: Path) -> Path:
    data = {
        "name": "v7_sub",
        "models": {"mock": {"api": "http://unused.invalid/v1", "key": "not-needed", "model": "x"}},
        "edges": [
            {"input_tags": ["topic"], "output_tags": ["final"], "layer": 0,
             "processor": {"model": "mock", "prompt": "压缩：{contents}"}}
        ],
    }
    p = tmp_path / "v7.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _graph_node_cfg(tmp_path: Path, **graph_overrides) -> Path:
    graph = {"path": "v7.json", "seed_tag": "topic", "result_tag": "final"}
    graph.update(graph_overrides)
    data = {
        "name": "nested",
        "seed_tag": "topic",
        "seed_content": "标签驱动的 DAG",
        "result_tag": "answer",
        "edges": [{
            "id": "g", "kind": "graph", "layer": 0,
            "input_tags": ["topic"], "output_tags": ["answer"], "graph": graph,
        }],
    }
    return _write(tmp_path, data)


def test_load_graph_node_config(tmp_path):
    cfg = cfg_mod.load(_graph_node_cfg(tmp_path))
    e = cfg.edges[0]
    assert e.kind == "graph"
    assert e.graph.path == "v7.json"
    assert e.graph.seed_tag == "topic"
    assert e.graph.result_tag == "final"
    assert e.graph.max_steps == 20


def test_graph_kind_inferred_from_graph_field(tmp_path):
    data = {
        "name": "nested", "seed_tag": "topic", "seed_content": "x", "result_tag": "out",
        "edges": [{"id": "g", "input_tags": ["topic"], "output_tags": ["out"],
                   "graph": {"path": "g.json", "seed_tag": "topic", "result_tag": "final"}}],
    }
    cfg = cfg_mod.load(_write(tmp_path, data))
    assert cfg.edges[0].kind == "graph"


@pytest.mark.parametrize("bad_edge, match", [
    ({"kind": "fn", "prompt": "x"}, "kind"),
    ({"kind": "graph", "graph": {"path": "x.json", "seed_tag": "a"}}, "result_tag"),
    ({"kind": "pi"}, "prompt"),
])
def test_invalid_node_configs_rejected(tmp_path, bad_edge, match):
    edge = {"id": "x", "input_tags": ["a"], "output_tags": ["b"], **bad_edge}
    data = dict(BASE, edges=[edge])
    with pytest.raises(ValueError, match=match):
        cfg_mod.load(_write(tmp_path, data))


def test_graph_node_runs_v7_dag(tmp_path):
    _v7_graph(tmp_path)
    res = dag8.run(_graph_node_cfg(tmp_path), db_path=tmp_path / "g.sqlite3",
                   graph_llm_client=V7MockLLM())
    assert res.status == "ok", res.summary()
    assert res.result == "[mock] 压缩：标签驱动的 DAG"
    with Store(tmp_path / "g.sqlite3") as st:
        rec = st.edge_records(res.run_id)[0]
        assert rec["model"] == "graph:v7.json"
        assert rec["status"] == "ok"


def test_graph_node_dry_run_previews_seed(tmp_path):
    _v7_graph(tmp_path)
    res = dag8.run(_graph_node_cfg(tmp_path), db_path=tmp_path / "g.sqlite3", dry_run=True)
    assert res.result == "标签驱动的 DAG"


def test_graph_node_missing_v7_file_is_error(tmp_path):
    res = dag8.run(_graph_node_cfg(tmp_path, path="nope.json"),
                   db_path=tmp_path / "g.sqlite3", graph_llm_client=V7MockLLM())
    assert res.status == "error"
    assert "v7 图不存在" in (res.edges[0].error or "")
    with Store(tmp_path / "g.sqlite3") as st:
        assert st.run_info(res.run_id)["status"] == "error"


def test_graph_node_dry_run_also_validates_path(tmp_path):
    res = dag8.run(_graph_node_cfg(tmp_path, path="nope.json"),
                   db_path=tmp_path / "g.sqlite3", dry_run=True)
    assert res.status == "error"
    assert "v7 图不存在" in (res.edges[0].error or "")


def test_graph_node_without_models_is_error(tmp_path):
    """v7 的 LLM 边没有模型配置时，要提前报清楚，而不是落到默认 openai 端点。"""
    v7 = {
        "name": "no_models",
        "edges": [
            {"input_tags": ["topic"], "output_tags": ["final"], "layer": 0,
             "processor": {"model": "m", "prompt": "x {contents}"}}
        ],
    }
    (tmp_path / "v7.json").write_text(json.dumps(v7, ensure_ascii=False), encoding="utf-8")
    res = dag8.run(_graph_node_cfg(tmp_path), db_path=tmp_path / "g.sqlite3",
                   graph_llm_client=V7MockLLM())
    assert res.status == "error"
    assert "模型配置" in (res.edges[0].error or "")


def test_default_node_kind_is_pi(tmp_path):
    cfg = cfg_mod.load(_write(tmp_path, BASE))
    assert cfg.edges[0].kind == "pi"


# ---------------------------------------------------------------------------
# command 节点：确定性 shell，不碰 pi
# ---------------------------------------------------------------------------
def _command_cfg(tmp_path: Path, command: str, *, stdin: bool = False) -> Path:
    data = {
        "name": "cmd",
        "seed_tag": "url",
        "seed_content": "https://example.com/BV1234567890/",
        "result_tag": "out",
        "edges": [{
            "id": "fetch", "kind": "command", "layer": 0,
            "input_tags": ["url"], "output_tags": ["out"],
            "command": command, "stdin": stdin,
        }],
    }
    return _write(tmp_path, data)


def test_command_kind_inferred_from_command_field(tmp_path):
    cfg = cfg_mod.load(_command_cfg(tmp_path, "printf ok"))
    assert cfg.edges[0].kind == "command"
    assert cfg.edges[0].command == "printf ok"
    assert cfg.edges[0].stdin is False


def test_command_node_runs_with_stdin(tmp_path):
    res = dag8.run(
        _command_cfg(tmp_path, "python3 -c 'import sys; print(sys.stdin.read().strip().upper())'",
                     stdin=True),
        db_path=tmp_path / "c.sqlite3",
    )
    assert res.status == "ok", res.summary()
    assert res.result == "HTTPS://EXAMPLE.COM/BV1234567890/"
    with Store(tmp_path / "c.sqlite3") as st:
        assert st.edge_records(res.run_id)[0]["model"] == "command"


def test_command_node_nonzero_exit_is_error(tmp_path):
    res = dag8.run(
        _command_cfg(tmp_path, "printf boom >&2; exit 3"),
        db_path=tmp_path / "c.sqlite3",
    )
    assert res.status == "error"
    assert "退出码 3" in (res.edges[0].error or "")


def test_command_node_missing_command_rejected(tmp_path):
    data = dict(BASE, edges=[{"id": "x", "kind": "command",
                              "input_tags": ["task"], "output_tags": ["out"]}])
    with pytest.raises(ValueError, match="command"):
        cfg_mod.load(_write(tmp_path, data))


def test_command_node_dry_run_does_not_execute(tmp_path):
    marker = tmp_path / "ran.txt"
    res = dag8.run(
        _command_cfg(tmp_path, f"touch {marker}"),
        db_path=tmp_path / "c.sqlite3", dry_run=True,
    )
    assert res.status == "ok", res.summary()
    assert not marker.exists()


# ---------------------------------------------------------------------------
# 调度形式：波次并行
# ---------------------------------------------------------------------------
class _SlowAgent:
    def __init__(self, delay: float = 0.3):
        self.delay = delay
        self.calls = []

    def run(self, prompt, **_kw):
        import time as _t
        _t.sleep(self.delay)
        self.calls.append(prompt)
        return f"ok:{prompt}"


def _fanout_cfg(tmp_path: Path) -> Path:
    data = {
        "name": "fanout",
        "seed_tag": "task",
        "seed_content": "Q",
        "result_tag": "a",
        "edges": [
            {"id": "a", "layer": 0, "input_tags": ["task"], "output_tags": ["a"], "prompt": "A {contents}"},
            {"id": "b", "layer": 0, "input_tags": ["task"], "output_tags": ["b"], "prompt": "B {contents}"},
        ],
    }
    return _write(tmp_path, data)


def test_wave_parallel_is_faster_than_serial(tmp_path):
    import time as _t

    t0 = _t.perf_counter()
    res = dag8.run(_fanout_cfg(tmp_path), db_path=tmp_path / "s.sqlite3",
                   agent=_SlowAgent(0.3), max_workers=1)
    serial = _t.perf_counter() - t0
    assert res.status == "ok"

    t0 = _t.perf_counter()
    res = dag8.run(_fanout_cfg(tmp_path), db_path=tmp_path / "p.sqlite3",
                   agent=_SlowAgent(0.3), max_workers=3)
    parallel = _t.perf_counter() - t0
    assert res.status == "ok"
    assert parallel < serial * 0.85, f"parallel={parallel:.2f}s serial={serial:.2f}s"


def test_parallel_workers_respect_dependencies(tmp_path):
    data = dict(
        BASE,
        edges=[
            {"id": "e1", "layer": 0, "input_tags": ["task"], "output_tags": ["mid"], "prompt": "A>{contents}"},
            {"id": "e2", "layer": 1, "input_tags": ["mid"], "output_tags": ["out"], "prompt": "B>{contents}"},
        ],
    )
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "e.sqlite3",
                   agent=_Upper(), max_workers=3)
    assert res.result == "B>A>HELLO"
    assert [e.edge_id for e in res.edges] == ["e1", "e2"]


# ---------------------------------------------------------------------------
# 节点缓存：同 节点+输入+模型 命中后不再调用
# ---------------------------------------------------------------------------
class _CountingAgent:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, **_kw):
        self.calls += 1
        return f"out{self.calls}:{prompt}"


class _CountingLLM:
    """v7 的替身客户端：complete 计数，embed 返回确定性向量。"""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **_kw):
        self.calls += 1
        return f"v7out{self.calls}:{prompt}"

    def embed(self, text, **_kw):
        from dag.llm import deterministic_semantic_embedding
        return deterministic_semantic_embedding(text)


def test_cache_hit_skips_agent(tmp_path):
    agent = _CountingAgent()
    cfg_path = _write(tmp_path, BASE)
    cache_dir = tmp_path / "cache"

    r1 = dag8.run(cfg_path, db_path=tmp_path / "a.sqlite3", agent=agent,
                  use_cache=True, cache_dir=cache_dir)
    r2 = dag8.run(cfg_path, db_path=tmp_path / "b.sqlite3", agent=agent,
                  use_cache=True, cache_dir=cache_dir)

    assert agent.calls == 1, "第二次运行应命中缓存，不能再调 agent"
    assert r2.result == r1.result
    assert r2.edges[0].cached is True
    assert r2.edges[0].elapsed == 0.0
    with Store(tmp_path / "b.sqlite3") as st:
        assert st.edge_records(r2.run_id)[0]["cached"] == 1


def test_cache_off_by_default(tmp_path):
    agent = _CountingAgent()
    cfg_path = _write(tmp_path, BASE)
    dag8.run(cfg_path, db_path=tmp_path / "a.sqlite3", agent=agent)
    dag8.run(cfg_path, db_path=tmp_path / "b.sqlite3", agent=agent)
    assert agent.calls == 2


def test_cache_key_includes_model(tmp_path):
    agent = _CountingAgent()
    cache_dir = tmp_path / "cache"

    d1 = dict(BASE, edges=[dict(BASE["edges"][0], model="glm-5.2")])
    d2 = dict(BASE, edges=[dict(BASE["edges"][0], model="kimi-k3")])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    c1 = _write(tmp_path / "a", d1)
    c2 = _write(tmp_path / "b", d2)

    dag8.run(c1, db_path=tmp_path / "a.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    dag8.run(c2, db_path=tmp_path / "b.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    assert agent.calls == 2, "换模型必须换缓存键"


def test_cache_refresh_reexecutes(tmp_path):
    agent = _CountingAgent()
    cfg_path = _write(tmp_path, BASE)
    cache_dir = tmp_path / "cache"

    dag8.run(cfg_path, db_path=tmp_path / "a.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    dag8.run(cfg_path, db_path=tmp_path / "b.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir, refresh_cache=True)
    assert agent.calls == 2


def test_graph_node_cache_hits(tmp_path):
    _v7_graph(tmp_path)
    client = _CountingLLM()
    cfg_path = _graph_node_cfg(tmp_path)
    cache_dir = tmp_path / "cache"

    r1 = dag8.run(cfg_path, db_path=tmp_path / "g1.sqlite3",
                  graph_llm_client=client, use_cache=True, cache_dir=cache_dir)
    r2 = dag8.run(cfg_path, db_path=tmp_path / "g2.sqlite3",
                  graph_llm_client=client, use_cache=True, cache_dir=cache_dir)

    assert client.calls == 1, "graph 节点命中缓存后不应再跑 v7"
    assert r2.result == r1.result
    assert r2.edges[0].cached is True


# ---------------------------------------------------------------------------
# 输出契约：校验 + 失败回灌重试
# ---------------------------------------------------------------------------
def _expect_cfg(tmp_path: Path, expect, repair: int = 0) -> Path:
    edge = dict(BASE["edges"][0], expect=expect, repair=repair)
    return _write(tmp_path, dict(BASE, edges=[edge]))


class _InvalidThenValid:
    def __init__(self):
        self.prompts = []

    def run(self, prompt, **_kw):
        self.prompts.append(prompt)
        return '{"ok": true}' if len(self.prompts) > 1 else "这不是 JSON"


class _AlwaysInvalid:
    def run(self, prompt, **_kw):
        return "这不是 JSON"


def test_load_expect_and_repair(tmp_path):
    cfg = cfg_mod.load(_expect_cfg(tmp_path, {"type": "json", "required_keys": ["ok"]}, repair=1))
    assert cfg.edges[0].expect == {"type": "json", "required_keys": ["ok"]}
    assert cfg.edges[0].repair == 1


def test_invalid_expect_rejected(tmp_path):
    with pytest.raises(ValueError, match="expect"):
        cfg_mod.load(_expect_cfg(tmp_path, {"type": "yaml"}))


def test_contract_repair_succeeds(tmp_path):
    agent = _InvalidThenValid()
    res = dag8.run(_expect_cfg(tmp_path, {"type": "json", "required_keys": ["ok"]}, repair=1),
                   db_path=tmp_path / "c.sqlite3", agent=agent)
    assert res.status == "ok", res.summary()
    assert json.loads(res.result) == {"ok": True}
    assert len(agent.prompts) == 2
    assert "格式校验" in agent.prompts[1]


def test_contract_failure_without_repair(tmp_path):
    res = dag8.run(_expect_cfg(tmp_path, {"type": "json"}),
                   db_path=tmp_path / "c.sqlite3", agent=_AlwaysInvalid())
    assert res.status == "error"
    assert "输出契约" in (res.edges[0].error or "")


def test_contract_failure_after_repair(tmp_path):
    res = dag8.run(_expect_cfg(tmp_path, {"type": "json"}, repair=1),
                   db_path=tmp_path / "c.sqlite3", agent=_AlwaysInvalid())
    assert res.status == "error"
    assert "输出契约" in (res.edges[0].error or "")


def test_contract_skipped_in_dry_run(tmp_path):
    res = dag8.run(_expect_cfg(tmp_path, {"type": "json"}),
                   db_path=tmp_path / "c.sqlite3", dry_run=True)
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# 可选输入：缺失也不阻塞（治「等不该等的 task」）
# ---------------------------------------------------------------------------
def _optional_cfg(tmp_path: Path, *, produce_b: bool, optional: bool = True) -> Path:
    edges = [
        {"id": "a", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "A {contents}"},
    ]
    if produce_b:
        edges.append({"id": "b", "layer": 0, "input_tags": ["task"], "output_tags": ["b"],
                      "prompt": "B {contents}"})
    edge = {"id": "opt", "layer": 1, "input_tags": ["a", "b"], "output_tags": ["out"],
            "prompt": "A={tag:a} | B={tag:b}"}
    if optional:
        edge["optional_inputs"] = ["b"]
    edges.append(edge)
    return _write(tmp_path, dict(BASE, edges=edges))


def test_optional_input_missing_still_runs(tmp_path):
    res = dag8.run(_optional_cfg(tmp_path, produce_b=False),
                   db_path=tmp_path / "o.sqlite3", agent=_Upper())
    assert res.status == "ok", res.summary()
    assert res.result == "A=A HELLO | B="


def test_optional_input_present_is_used(tmp_path):
    res = dag8.run(_optional_cfg(tmp_path, produce_b=True),
                   db_path=tmp_path / "o.sqlite3", agent=_Upper())
    assert res.status == "ok", res.summary()
    assert res.result == "A=A HELLO | B=B HELLO"


def test_required_input_missing_is_still_skipped(tmp_path):
    res = dag8.run(_optional_cfg(tmp_path, produce_b=False, optional=False),
                   db_path=tmp_path / "o.sqlite3", agent=_Upper())
    assert res.status == "incomplete"
    assert res.edges[-1].status == "skipped"


def test_optional_inputs_must_be_subset_of_input_tags(tmp_path):
    data = dict(BASE, edges=[{
        "id": "x", "input_tags": ["task"], "output_tags": ["out"],
        "prompt": "x {contents}", "optional_inputs": ["nope"],
    }])
    with pytest.raises(ValueError, match="optional_inputs"):
        cfg_mod.load(_write(tmp_path, data))


# ---------------------------------------------------------------------------
# graph 节点：429 重试（重建子图）
# ---------------------------------------------------------------------------
class _FlakyLLM:
    def __init__(self):
        self.complete_calls = 0

    def complete(self, prompt, **_kw):
        self.complete_calls += 1
        if self.complete_calls == 1:
            raise RuntimeError("429 Too Many Requests")
        return "重试成功"

    def embed(self, text, **_kw):
        from dag.llm import deterministic_semantic_embedding
        return deterministic_semantic_embedding(text)


def test_graph_node_retries_429(tmp_path, monkeypatch):
    import dag8.nodes as nodes_mod
    monkeypatch.setattr(nodes_mod, "_RETRY_SLEEPS", (0.0, 0.0))
    _v7_graph(tmp_path)
    client = _FlakyLLM()
    res = dag8.run(_graph_node_cfg(tmp_path), db_path=tmp_path / "g.sqlite3",
                   graph_llm_client=client)
    assert res.status == "ok", res.summary()
    assert res.result == "重试成功"
    assert client.complete_calls == 2


def test_edge_cache_false_disables_cache(tmp_path):
    """易变节点（如扫描当前文件）显式禁用缓存。"""
    agent = _CountingAgent()
    data = dict(BASE, edges=[dict(BASE["edges"][0], cache=False)])
    cfg_path = _write(tmp_path, data)
    cache_dir = tmp_path / "cache"
    dag8.run(cfg_path, db_path=tmp_path / "a.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    dag8.run(cfg_path, db_path=tmp_path / "b.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    assert agent.calls == 2


# ---------------------------------------------------------------------------
# graph 节点总超时
# ---------------------------------------------------------------------------
class _SlowLLM:
    def __init__(self, delay: float = 0.1):
        import time as _t
        self.delay = delay
        self.calls = 0
        self._t = _t

    def complete(self, *a, **k):
        self.calls += 1
        self._t.sleep(self.delay)
        return "x"

    def embed(self, text, **k):
        from dag.llm import deterministic_semantic_embedding
        return deterministic_semantic_embedding(text)


def test_graph_timeout_parsed(tmp_path):
    cfg = cfg_mod.load(_graph_node_cfg(tmp_path, timeout=12.5))
    assert cfg.edges[0].graph.timeout == 12.5


def test_graph_node_total_timeout(tmp_path, monkeypatch):
    import dag8.nodes as nodes_mod
    monkeypatch.setattr(nodes_mod, "_RETRY_SLEEPS", (0.0, 0.0))
    _v7_graph(tmp_path)
    client = _SlowLLM(0.1)
    res = dag8.run(_graph_node_cfg(tmp_path, timeout=0.05),
                   db_path=tmp_path / "g.sqlite3", graph_llm_client=client)
    assert res.status == "error"
    assert "总预算" in (res.edges[0].error or "")
    assert client.calls == 1


# ---------------------------------------------------------------------------
# 模型回退：主模型 429 → 备用模型；回退结果不写缓存
# ---------------------------------------------------------------------------
class _FallbackAgent:
    def __init__(self):
        self.calls = []

    def run(self, prompt, **kw):
        self.calls.append((kw.get("provider"), kw.get("model")))
        if kw.get("model") == "bad-model":
            raise dag8.PiError("429 rate_limit", retryable=True)
        return "fallback-ok"


def _fallback_cfg(tmp_path: Path) -> Path:
    data = dict(BASE)
    data["pi"] = {"provider": "laptop", "model": "bad-model",
                  "fallbacks": ["glm-5.2"]}
    return _write(tmp_path, data)


def test_pi_fallbacks_switch_model(tmp_path):
    agent = _FallbackAgent()
    res = dag8.run(_fallback_cfg(tmp_path), db_path=tmp_path / "f.sqlite3", agent=agent)
    assert res.status == "ok", res.summary()
    assert res.result == "fallback-ok"
    assert agent.calls == [("laptop", "bad-model"), ("laptop", "glm-5.2")]
    with Store(tmp_path / "f.sqlite3") as st:
        assert st.edge_records(res.run_id)[0]["model"] == "laptop/glm-5.2"


def test_fallback_result_not_cached(tmp_path):
    agent = _FallbackAgent()
    cache_dir = tmp_path / "cache"
    dag8.run(_fallback_cfg(tmp_path), db_path=tmp_path / "a.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    dag8.run(_fallback_cfg(tmp_path), db_path=tmp_path / "b.sqlite3", agent=agent,
             use_cache=True, cache_dir=cache_dir)
    assert len(agent.calls) == 4, "回退结果不能写进主模型的缓存键"


def test_pi_fallbacks_parsed(tmp_path):
    cfg = cfg_mod.load(_fallback_cfg(tmp_path))
    assert cfg.pi.fallbacks == ["glm-5.2"]


# ---------------------------------------------------------------------------
# 运行级总预算
# ---------------------------------------------------------------------------
def test_run_budget_exceeded_skips_remaining(tmp_path):
    data = dict(
        BASE,
        edges=[
            {"id": "e1", "layer": 0, "input_tags": ["task"], "output_tags": ["mid"], "prompt": "A {contents}"},
            {"id": "e2", "layer": 1, "input_tags": ["mid"], "output_tags": ["out"], "prompt": "B {contents}"},
        ],
    )
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "b.sqlite3",
                   agent=_Upper(), max_seconds=0)
    assert res.status == "timeout"
    assert [e.status for e in res.edges] == ["skipped", "skipped"]
    with Store(tmp_path / "b.sqlite3") as st:
        assert st.run_info(res.run_id)["status"] == "timeout"


def test_run_budget_ok_when_enough(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "b.sqlite3",
                   agent=_Upper(), max_seconds=10)
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# engine="v8" 的递归子图（用 command 节点离线验证）
# ---------------------------------------------------------------------------
def _nested_v8_config(tmp_path: Path) -> Path:
    nested = {
        "name": "nested", "seed_tag": "start", "seed_content": "",
        "result_tag": "final",
        "edges": [
            {"id": "a", "layer": 0, "kind": "command", "input_tags": ["start"],
             "output_tags": ["a"], "command": "printf A"},
            {"id": "b", "layer": 0, "kind": "command", "input_tags": ["start"],
             "output_tags": ["b"], "command": "printf B"},
            {"id": "m", "layer": 1, "kind": "command", "input_tags": ["a", "b"],
             "output_tags": ["final"],
             "command": "printf '%s|%s' \"{tag:a}\" \"{tag:b}\""},
        ],
    }
    (tmp_path / "nested.json").write_text(json.dumps(nested), encoding="utf-8")
    parent = dict(BASE)
    parent["edges"] = [{
        "id": "g", "kind": "graph", "layer": 0,
        "input_tags": ["task"], "output_tags": ["out"],
        "graph": {"path": "nested.json", "engine": "v8", "timeout": 20, "workers": 2},
    }]
    return _write(tmp_path, parent)


def test_v8_graph_node_nested(tmp_path):
    res = dag8.run(_nested_v8_config(tmp_path), db_path=tmp_path / "n.sqlite3")
    assert res.status == "ok", res.summary()
    assert res.result == "A|B"


def test_v8_graph_engine_parsed(tmp_path):
    cfg = cfg_mod.load(_nested_v8_config(tmp_path))
    g = cfg.edges[0].graph
    assert g.engine == "v8"
    assert g.workers == 2


def test_v8_graph_engine_invalid(tmp_path):
    data = dict(BASE)
    data["edges"] = [{
        "id": "g", "kind": "graph", "input_tags": ["task"], "output_tags": ["out"],
        "graph": {"path": "x.json", "engine": "v9"},
    }]
    with pytest.raises(ValueError):
        cfg_mod.load(_write(tmp_path, data))


class _PickAgent:
    def run(self, prompt, **kw):
        if "FAIL" in prompt:
            raise dag8.PiError("boom", retryable=False)
        return "ok"


def _keep_cfg(tmp_path):
    data = dict(BASE, edges=[
        {"id": "bad", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "FAIL {contents}"},
        {"id": "good", "layer": 0, "input_tags": ["task"], "output_tags": ["b"],
         "prompt": "B {contents}"},
        {"id": "later", "layer": 1, "input_tags": ["b"], "output_tags": ["out"],
         "prompt": "C {contents}"},
    ])
    return _write(tmp_path, data)


def test_keep_going_false_stops_after_error(tmp_path):
    res = dag8.run(_keep_cfg(tmp_path), db_path=tmp_path / "k.sqlite3",
                   agent=_PickAgent())
    assert res.status == "error"
    assert "later" not in [e.edge_id for e in res.edges]


def test_keep_going_true_runs_independent_branch(tmp_path):
    res = dag8.run(_keep_cfg(tmp_path), db_path=tmp_path / "k.sqlite3",
                   agent=_PickAgent(), keep_going=True)
    assert res.status == "partial"
    assert res.result == "ok"
    assert [e.edge_id for e in res.edges].count("later") == 1


class _BoomAgent:
    def run(self, prompt, **kw):
        raise RuntimeError("kaboom")


def test_unexpected_exception_recorded(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "boom.sqlite3",
                   agent=_BoomAgent())
    assert res.status == "error"
    assert "kaboom" in (res.edges[0].error or "")
    assert "节点内部错误" in (res.edges[0].error or "")


# ---------------------------------------------------------------------------
# 自适应并发（AIMD）
# ---------------------------------------------------------------------------
class _RateAgent:
    def run(self, prompt, **kw):
        if "FAIL" in prompt:
            raise dag8.PiError("429 Too Many Requests", retryable=False)
        return "ok"


def test_adaptive_controller_aimd():
    from dag8.scheduler import AdaptiveConcurrency

    c = AdaptiveConcurrency(4)
    assert c.current == 4
    assert c.observe(["429 rate limit"], 0) == 2
    assert c.observe(["429"], 0) == 1
    assert c.observe([], 3) == 2
    assert c.observe([], 3) == 3
    assert c.observe([], 3) == 4

    c2 = __import__("dag8.scheduler", fromlist=["AdaptiveConcurrency"]).AdaptiveConcurrency(4)
    assert c2.observe(["pi 超时（60s）"], 0) == 2


def test_adaptive_scheduler_reduces_next_wave(tmp_path):
    data = dict(BASE, edges=[
        {"id": "a", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "OK {contents}"},
        {"id": "b", "layer": 0, "input_tags": ["task"], "output_tags": ["b"],
         "prompt": "FAIL {contents}"},
        {"id": "c", "layer": 0, "input_tags": ["task"], "output_tags": ["c"],
         "prompt": "FAIL {contents}"},
        {"id": "d", "layer": 0, "input_tags": ["task"], "output_tags": ["d"],
         "prompt": "OK {contents}"},
        {"id": "e", "layer": 1, "input_tags": ["a"], "output_tags": ["out"],
         "prompt": "OK {contents}"},
    ])
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "ad.sqlite3",
                   agent=_RateAgent(), max_workers=4, keep_going=True,
                   adaptive=True)
    assert res.workers_used == [4, 2]
    assert res.status == "partial"
    assert res.result == "ok"


def test_pi_fallback_provider_prefix(tmp_path):
    data = dict(BASE)
    data["pi"] = {"provider": "laptop", "model": "bad-model",
                  "fallbacks": ["sensenova/deepseek-v4-flash"]}
    agent = _FallbackAgent()
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "fp.sqlite3",
                   agent=agent)
    assert res.status == "ok"
    assert agent.calls == [("laptop", "bad-model"),
                           ("sensenova", "deepseek-v4-flash")]


# ---------------------------------------------------------------------------
# 条件边 when
# ---------------------------------------------------------------------------
class _WhenAgent:
    def __init__(self, first: str):
        self.first = first

    def run(self, prompt, **kw):
        if "A1" in prompt:
            return self.first
        return "ok"


def _when_cfg(tmp_path):
    data = dict(BASE, edges=[
        {"id": "a1", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "A1 {contents}"},
        {"id": "e2", "layer": 1, "input_tags": ["a"], "output_tags": ["x"],
         "prompt": "A2 {contents}", "when": {"tag": "a", "pattern": "YES"}},
        {"id": "e3", "layer": 1, "input_tags": ["a"], "output_tags": ["out"],
         "prompt": "A3 {contents}"},
    ])
    return _write(tmp_path, data)


def test_when_skips_edge(tmp_path):
    res = dag8.run(_when_cfg(tmp_path), db_path=tmp_path / "w.sqlite3",
                   agent=_WhenAgent("NO"))
    assert res.status == "ok", res.summary()
    e2 = [e for e in res.edges if e.edge_id == "e2"][0]
    assert e2.status == "skipped"
    assert "when" in (e2.error or "")


def test_when_runs_when_matches(tmp_path):
    res = dag8.run(_when_cfg(tmp_path), db_path=tmp_path / "w.sqlite3",
                   agent=_WhenAgent("YES"))
    assert res.status == "ok"
    e2 = [e for e in res.edges if e.edge_id == "e2"][0]
    assert e2.status == "ok"


def test_when_tag_must_be_input(tmp_path):
    data = dict(BASE, edges=[
        {"id": "e", "input_tags": ["task"], "output_tags": ["out"],
         "prompt": "A {contents}", "when": {"tag": "missing", "present": True}},
    ])
    with pytest.raises(ValueError, match="when.tag"):
        cfg_mod.load(_write(tmp_path, data))


# ---------------------------------------------------------------------------
# 成本预算
# ---------------------------------------------------------------------------
def test_max_cost_zero_stops(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "c.sqlite3",
                   agent=_Upper(), max_cost=0)
    assert res.status == "over_budget"
    assert res.edges[0].status == "skipped"
    assert "成本" in (res.edges[0].error or "")
    assert res.cost_estimate >= 0


def test_max_cost_high_runs(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "c.sqlite3",
                   agent=_Upper(), max_cost=1e6)
    assert res.status == "ok"


def test_auto_model_uses_stats(tmp_path):
    db = tmp_path / "auto.sqlite3"
    with Store(db) as st:
        for i in range(3):
            r = st.new_run(f"d{i}", "cfg", "seed", "S")
            st.log_edge(r, "e", 0, ["seed"], ["out"],
                        model="laptop/deepseek-flash", status="ok", elapsed=5.0)
            st.finish_run(r, "ok")
    data = dict(BASE)
    data["pi"] = {"provider": "laptop", "model": "glm-5.2",
                  "timeout": 10, "retries": 1}
    res = dag8.run(_write(tmp_path, data), db_path=db,
                   agent=_Upper(), auto_model=True)
    assert res.status == "ok"
    with Store(db) as st:
        assert st.edge_records(res.run_id)[0]["model"] == "laptop/deepseek-flash"


# ---------------------------------------------------------------------------
# judge 质量闸门
# ---------------------------------------------------------------------------
class _JudgeAgent:
    def __init__(self, score):
        self.score = score

    def run(self, prompt, **kw):
        if "评审" in prompt:
            return json.dumps({"score": self.score, "reason": "test"})
        return "content"


def _judge_cfg(tmp_path):
    data = dict(BASE)
    data["edges"] = [{"id": "e", "input_tags": ["task"], "output_tags": ["out"],
                      "prompt": "P {contents}", "judge": {"min_score": 8}}]
    return _write(tmp_path, data)


def test_judge_pass(tmp_path):
    res = dag8.run(_judge_cfg(tmp_path), db_path=tmp_path / "j.sqlite3",
                   agent=_JudgeAgent(9))
    assert res.status == "ok"
    assert res.result == "content"


def test_judge_fail(tmp_path):
    res = dag8.run(_judge_cfg(tmp_path), db_path=tmp_path / "j.sqlite3",
                   agent=_JudgeAgent(5))
    assert res.status == "error"
    assert "judge 分数" in (res.edges[0].error or "")


class _BadJudgeAgent:
    def run(self, prompt, **kw):
        return "not json" if "评审" in prompt else "content"


def test_judge_bad_json(tmp_path):
    res = dag8.run(_judge_cfg(tmp_path), db_path=tmp_path / "j.sqlite3",
                   agent=_BadJudgeAgent())
    assert res.status == "error"
    assert "judge 输出" in (res.edges[0].error or "")


def test_auto_model_ignores_tiny_samples(tmp_path):
    db = tmp_path / "auto2.sqlite3"
    with Store(db) as st:
        # n=1 的“快”模型不应压过 n=3 的稳定模型
        r = st.new_run("one", "cfg", "seed", "S")
        st.log_edge(r, "e", 0, ["seed"], ["out"],
                    model="laptop/glm-5.2", status="ok", elapsed=1.0)
        st.finish_run(r, "ok")
        for i in range(3):
            r = st.new_run(f"d{i}", "cfg", "seed", "S")
            st.log_edge(r, "e", 0, ["seed"], ["out"],
                        model="laptop/deepseek-flash", status="ok", elapsed=5.0)
            st.finish_run(r, "ok")
    data = dict(BASE)
    data["pi"] = {"provider": "laptop", "model": "glm-5.2",
                  "timeout": 10, "retries": 1}
    res = dag8.run(_write(tmp_path, data), db_path=db,
                   agent=_Upper(), auto_model=True)
    assert res.status == "ok"
    with Store(db) as st:
        assert st.edge_records(res.run_id)[0]["model"] == "laptop/deepseek-flash"


def test_json_contract_strips_fences(tmp_path):
    class _Fenced:
        def run(self, prompt, **kw):
            return '```json\n{"score": 5, "ok": true}\n```'

    data = dict(BASE)
    data["edges"] = [{"id": "e", "input_tags": ["task"], "output_tags": ["out"],
                      "prompt": "P {contents}",
                      "expect": {"type": "json", "required_keys": ["score"]}}]
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "j.sqlite3",
                   agent=_Fenced())
    assert res.status == "ok"
    assert "```" not in res.result
    assert json.loads(res.result)["score"] == 5


def test_join_any_with_pruned_branch(tmp_path):
    data = dict(BASE, seed_content="x", edges=[
        {"id": "e1", "layer": 0, "input_tags": ["task"], "output_tags": ["a"],
         "prompt": "A {contents}"},
        {"id": "e2", "layer": 0, "input_tags": ["task"], "output_tags": ["b"],
         "prompt": "B {contents}", "when": {"tag": "task", "pattern": "YES"}},
        {"id": "e3", "layer": 1, "input_tags": ["a", "b"], "join": "any",
         "output_tags": ["out"], "prompt": "C {contents}"},
    ])
    res = dag8.run(_write(tmp_path, data), db_path=tmp_path / "j.sqlite3",
                   agent=_Upper())
    assert res.status == "ok", res.summary()
    assert [e.status for e in res.edges if e.edge_id == "e2"] == ["skipped"]
    assert [e.status for e in res.edges if e.edge_id == "e3"] == ["ok"]
