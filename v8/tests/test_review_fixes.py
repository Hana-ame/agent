"""针对架构评审里已确认的底层问题：动态 PATH、排序稳定性、tag 查询。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8 import config as cfg_mod  # noqa: E402
from dag8.pi import PiAgent, PiConfig, _remote_path_prefix  # noqa: E402
from dag8.store import Store  # noqa: E402


def test_remote_path_is_dynamic():
    prefix = _remote_path_prefix()
    assert "v22.23.1" not in prefix
    assert "ls -d" in prefix
    cfg = PiConfig(provider="laptop", model="deepseek-flash",
                   ssh_host="example", ssh_key="~/.ssh/id_rsa")
    remote = PiAgent(cfg)._command("laptop", "deepseek-flash", False)[-1]
    assert "v22.23.1" not in remote
    assert "ls -d" in remote


def test_sorted_edges_is_stable_and_layer_ordered(tmp_path):
    data = {
        "name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
        "edges": [
            {"id": "late", "layer": 2, "input_tags": ["a"], "output_tags": ["out"],
             "prompt": "p {contents}"},
            {"id": "first", "layer": 0, "input_tags": ["a"], "output_tags": ["b"],
             "prompt": "p {contents}"},
            {"id": "second", "layer": 0, "input_tags": ["a"], "output_tags": ["c"],
             "prompt": "p {contents}"},
        ],
    }
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert [e.id for e in cfg_mod.load(p).sorted_edges()] == ["first", "second", "late"]


def test_contents_for_tag_uses_tag_filter(tmp_path):
    with Store(tmp_path / "s.sqlite3") as st:
        run = st.new_run("d", "cfg", "seed", "S")
        st.add_content(run, ["a", "b"], "hit")
        st.add_content(run, ["a"], "miss")
        hits = st.contents_for_tag(run, "b")
        assert [c["content"] for c in hits] == ["hit"]
