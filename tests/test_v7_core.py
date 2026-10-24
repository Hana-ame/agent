"""v7/ 作为 DAG 本体的最小验证：模块级 step/set/contents 可用。"""
from __future__ import annotations

import v7


def test_module_level_set_and_contents():
    v7._current_dag = v7.DAG()
    v7.add_content("输入", tags=["task"])
    v7.set("external", "外部结果")
    assert v7.get_result("external") == "外部结果"
    assert any("external" in c.tags for c in v7.contents)


def test_instance_set():
    d = v7.DAG()
    d.add_content("a", tags=["task"])
    d.set("b", "B")
    assert d.get_result("b") == "B"
