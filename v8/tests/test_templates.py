"""任务模板的离线测试：构建、契约、dry-run、CLI。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402
from dag8 import templates  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402


def test_list_templates():
    names = set(templates.list_templates())
    assert {"summarize", "extract_json", "classify", "translate", "review_diff",
            "bug_triage", "test_plan", "refactor_plan"} <= names


def test_all_templates_build_and_dry_run(tmp_path):
    for name in templates.TEMPLATES:
        cfg = templates.build(name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        cfg_mod.load(path)   # 配置本身必须合法
        res = dag8.run(path, db_path=tmp_path / f"{name}.sqlite3", dry_run=True)
        assert res.status == "ok", (name, res.summary())


def test_templates_carry_contracts():
    ex = templates.build("extract_json")
    assert ex["edges"][0]["expect"]["type"] == "json"
    assert ex["edges"][0]["repair"] == 1

    rd = templates.build("review_diff")
    assert rd["edges"][0]["kind"] == "command"
    assert "git" in rd["edges"][0]["command"]


def test_build_overrides_seed():
    cfg = templates.build("summarize", seed="hello", seed_tag="doc2")
    assert cfg["seed_content"] == "hello"
    assert cfg["seed_tag"] == "doc2"


def test_unknown_template():
    with pytest.raises(ValueError, match="未知模板"):
        templates.build("nope")


def test_cli_new(tmp_path):
    assert dag8_main(["new", "--list"]) == 0

    out = tmp_path / "s.json"
    assert dag8_main(["new", "summarize", "--out", str(out),
                      "--seed", "hello"]) == 0
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["seed_content"] == "hello"

    assert dag8_main(["new", "nope", "--out", str(out)]) == 1


def test_new_templates_carry_contracts():
    bt = templates.build("bug_triage")
    spec = bt["edges"][0]["expect"]
    assert spec["schema"]["required"] == ["severity", "component", "summary", "next_step"]
    assert spec["schema"]["properties"]["severity"]["enum"] == ["low", "medium", "high"]

    tp = templates.build("test_plan")
    assert tp["edges"][0]["expect"]["min_sections"] == 3

    rp = templates.build("refactor_plan")
    assert rp["edges"][0]["expect"]["min_chars"] == 100
