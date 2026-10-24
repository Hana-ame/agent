"""v7 → v8 转换的离线测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.convert import convert_v7, convert_v7_to_graphs  # noqa: E402
from dag8.lint import lint_config  # noqa: E402
from dag8.scaffold import write_config as scaffold_write  # noqa: E402

V7 = {
    "name": "v7demo",
    "models": {"laptop": {"api": "http://unused.invalid/v1",
                          "key": "k", "model": "deepseek-flash"}},
    "contents": [{"tags": ["topic"], "contents": "hello", "gen": 0}],
    "edges": [
        {"id": "a", "layer": 0, "input_tags": ["topic"], "output_tags": ["mid"],
         "processor": {"model": "laptop", "prompt": "A {contents}"}},
        {"id": "b", "layer": 1, "input_tags": ["mid"], "output_tags": ["final"],
         "processor": {"model": "laptop", "prompt": "B {contents}"}},
    ],
}


def test_convert_shape(tmp_path):
    cfg = convert_v7(V7)
    assert cfg["seed_tag"] == "topic"
    assert cfg["seed_content"] == "hello"
    assert cfg["result_tag"] == "final"
    assert cfg["edges"][0]["model"] == "laptop/deepseek-flash"
    assert [e["layer"] for e in cfg["edges"]] == [0, 1]

    p = tmp_path / "v8.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    loaded = cfg_mod.load(p)
    assert [i for i in lint_config(loaded) if i.level == "error"] == []

    res = dag8.run(p, db_path=tmp_path / "v8.sqlite3", dry_run=True)
    assert res.status == "ok"


def test_convert_without_models():
    data = dict(V7, models={})
    cfg = convert_v7(data)
    assert "model" not in cfg["edges"][0]


def test_convert_result_override():
    assert convert_v7(V7, result_tag="mid")["result_tag"] == "mid"


def test_convert_cli(tmp_path):
    src = tmp_path / "v7.json"
    src.write_text(json.dumps(V7, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "v8.json"

    assert dag8_main(["convert", str(src), "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["edges"][0]["model"] == "laptop/deepseek-flash"
    assert cfg["result_tag"] == "final"


FANOUT = {
    "name": "fanout",
    "models": {"m": {"api": "http://unused.invalid/v1", "key": "k", "model": "mock"}},
    "contents": [{"tags": ["task"], "contents": "S"}],
    "edges": [
        {"id": "def", "layer": 0, "input_tags": ["task"], "output_tags": ["def"],
         "processor": {"model": "m", "prompt": "define {contents}"}},
        {"id": "benefit", "layer": 0, "input_tags": ["task"], "output_tags": ["benefit"],
         "processor": {"model": "m", "prompt": "benefit {contents}"}},
        {"id": "final", "layer": 1, "input_tags": ["def", "benefit"],
         "output_tags": ["out"],
         "processor": {"model": "m", "prompt": "merge {contents}"}},
    ],
}


def test_convert_graph_mode(tmp_path):
    parent, subs = convert_v7_to_graphs(FANOUT, subdir="graphs")
    assert len(subs) == 3
    assert all(e["kind"] == "graph" for e in parent["edges"])
    assert all(e["graph"]["seed_tag"] for e in parent["edges"])
    assert all(s["contents"] == [] for s in subs.values())

    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name, sub in subs.items():
        scaffold_write(sub, graphs / name)
    out = tmp_path / "v8.json"
    scaffold_write(parent, out)

    loaded = cfg_mod.load(out)
    assert [i for i in lint_config(loaded) if i.level == "error"] == []
    assert dag8.run(out, db_path=tmp_path / "r.sqlite3", dry_run=True).status == "ok"


def test_convert_cli_graph_mode(tmp_path):
    src = tmp_path / "v7.json"
    src.write_text(json.dumps(FANOUT, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "v8.json"
    assert dag8_main(["convert", str(src), "--mode", "graph",
                      "--out", str(out)]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert len(cfg["edges"]) == 3
    assert (tmp_path / "graphs" / "final.json").is_file()
