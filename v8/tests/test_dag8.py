"""dag8 的离线测试：配置渲染、sqlite 读写、引擎调度，全部不碰 pi。

真实 pi 调用只在 `test_live_pi.py` 里做（默认跳过，需要 WSL + pi）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V8))

import dag8  # noqa: E402
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
def test_engine_dry_run(tmp_path):
    cfg_path = _write(tmp_path, BASE)
    res = dag8.run(cfg_path, db_path=tmp_path / "e.sqlite3", dry_run=True)
    assert res.status == "ok"
    assert res.result == "处理：hello"
    assert res.edges[0].status == "ok"


def test_engine_missing_input_is_skipped(tmp_path):
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


def test_engine_chains_two_layers(tmp_path):
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


def test_engine_records_pi_error(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", agent=_Boom())
    assert res.status == "error"
    assert res.edges[0].status == "error"
    assert "429" in (res.edges[0].error or "")
    with Store(tmp_path / "e.sqlite3") as st:
        assert st.run_info(res.run_id)["status"] == "error"


class _Upper:
    def run(self, prompt, **_kw):
        return prompt.upper()


def test_engine_uses_injected_agent(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", agent=_Upper())
    assert res.result == "处理：HELLO"


def test_run_summary_is_readable(tmp_path):
    res = dag8.run(_write(tmp_path, BASE), db_path=tmp_path / "e.sqlite3", dry_run=True)
    s = res.summary()
    assert "✓ [0] e1" in s
    assert "result: out" in s
