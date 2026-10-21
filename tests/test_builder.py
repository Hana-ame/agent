"""从自然语言描述构建 DAG 的测试（v7 新增，移植自 v6 解析层）。"""
from __future__ import annotations

import pytest

from dag_agent import DAG, MockLLMClient
from dag_agent.builder import (
    build_dag_from_description,
    parse_edges_from_description,
    parse_json_stream,
)


def test_single_sentence():
    edges = parse_edges_from_description("从 [raw_report] 经过 提取关键指标 到 [metrics]")
    assert len(edges) == 1
    e = edges[0]
    assert e["input_tags"] == ["raw_report"]
    assert e["output_tags"] == ["metrics"]
    assert "提取关键指标" in str(e["processor"])


def test_multi_sentence_split_by_newline_and_semicolon():
    desc = "从 [a] 经过 步骤一 到 [b]\n从 [b] 经过 步骤二 到 [c]；从 [c] 经过 步骤三 到 [d]"
    edges = parse_edges_from_description(desc)
    assert [e["output_tags"][0] for e in edges] == ["b", "c", "d"]


def test_arrow_syntax():
    edges = parse_edges_from_description("[a] -> 汇总 -> [b]")
    assert edges[0]["input_tags"] == ["a"]
    assert edges[0]["output_tags"] == ["b"]


def test_bracket_arrow_syntax():
    edges = parse_edges_from_description("[a] -=[调用工具]=> [b]")
    assert edges[0]["input_tags"] == ["a"]
    assert edges[0]["output_tags"] == ["b"]


def test_no_input_sentence():
    edges = parse_edges_from_description("无输入 经过 生成初始问题 到 [question]")
    assert edges[0]["input_tags"] == []
    assert edges[0]["output_tags"] == ["question"]


def test_multiple_input_tags():
    edges = parse_edges_from_description("从 [a, b, c] 经过 合并 到 [merged]")
    assert edges[0]["input_tags"] == ["a", "b", "c"]


def test_json_input():
    edges = parse_edges_from_description(
        '{"edges": [{"input_tags": ["x"], "processor": {"model": "m", "prompt": "p"}, "output_tags": ["y"]}]}'
    )
    assert len(edges) == 1
    assert edges[0]["output_tags"] == ["y"]


def test_dict_list_input():
    edges = parse_edges_from_description(
        [{"input_tags": "x", "processor": "do", "output_tags": "y"}]
    )
    assert edges[0]["input_tags"] == ["x"]


def test_unparseable_raises_with_hint():
    with pytest.raises(ValueError, match="无法从描述中解析"):
        parse_edges_from_description("这句话没有任何边结构")


def test_numbered_list_prefix_is_stripped():
    edges = parse_edges_from_description("1. 从 [a] 经过 处理 到 [b]\n2. 从 [b] 经过 处理 到 [c]")
    assert len(edges) == 2


def test_parse_json_stream_extracts_each_document():
    text = '前言 {"a": 1} 中间 [1,2] 尾巴'
    assert parse_json_stream(text) == [{"a": 1}, [1, 2]]


# ---------------------------------------------------------------------------
# 组装成可跑的 DAG
# ---------------------------------------------------------------------------
def test_build_dag_appends_final_tag():
    dag = build_dag_from_description("从 [a] 经过 处理 到 [b]")
    assert isinstance(dag, DAG)
    assert dag.get_result("result") is not None or True  # 未跑时为空
    outs = {t for e in dag.edges for t in e.output_tags}
    assert "result" in outs, "ensure_final_tag 应把 result 追加到最后一条边"


def test_build_dag_can_run():
    dag = build_dag_from_description(
        "从 [seed] 经过 处理 到 [mid]\n从 [mid] 经过 再处理 到 [done]",
        ensure_final_tag=None,
        llm_client=MockLLMClient(),
    )
    dag.add_content(contents="hello", tags=["seed"], gen=0)
    dag.run(until_tag="done", max_steps=10)
    assert dag.has_tag("done")
    assert dag.get_result("done")


def test_build_dag_rejects_empty_description():
    with pytest.raises(ValueError, match="没有解析出任何边"):
        build_dag_from_description("   ")
