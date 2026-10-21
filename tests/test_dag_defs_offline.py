"""dag_defs 内纯函数的离线测试：不联网、不调 LLM、不落盘。

这些函数决定了「爬回来的东西怎么解析」「模型输出怎么兜底」，是 DAG 里最容易
静默出错的部分，必须能在断网环境下验证。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import dags  # noqa: F401  (触发 dag_defs 自动加载)
from dags import registry

s1 = registry.get_definition_module("stage1st_ai")
stability = registry.get_definition_module("stage1st_stability")


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------
def test_parse_dt_valid():
    assert s1._parse_dt("2026-10-21 04:36").year == 2026
    assert s1._parse_dt("2026-10-21 04:36:07").second == 7


@pytest.mark.parametrize("bad", ["", "   ", "2026/10/21", "昨天", None])
def test_parse_dt_invalid(bad):
    assert s1._parse_dt(bad) is None


# ---------------------------------------------------------------------------
# 模型序号解析（LLM 输出 → 索引集合）
# ---------------------------------------------------------------------------
def test_parse_index_list_object_payload():
    assert s1._parse_index_list('{"picked": [1, 3, 99, 0, -2]}', 5) == [1, 3]


def test_parse_index_list_plain_array_dedup():
    assert s1._parse_index_list("[2, 2, 4]", 5) == [2, 4]


def test_parse_index_list_falls_back_to_digits():
    assert s1._parse_index_list("前 1、3 和后 7", 5) == [1, 3]


def test_parse_index_list_empty():
    assert s1._parse_index_list("[]", 5) == []


def test_parse_index_list_clamps_to_n():
    assert s1._parse_index_list("[1, 2, 3, 4]", 2) == [1, 2]


# ---------------------------------------------------------------------------
# 抽取式兜底摘要
# ---------------------------------------------------------------------------
def test_extractive_keeps_headings_and_links():
    digest = "## 模型\n- [A](http://a)\n- [B](http://b)\n## 芯片\n- [C](http://c)\n"
    out = s1._extractive(digest)
    assert "- **模型**" in out
    assert "- **芯片**" in out
    assert out.count("  - [") == 3


def test_extractive_marks_fallback_when_noted():
    out = s1._extractive("## A\n", note="超时")
    assert "抽取式摘要" in out


def test_extractive_without_note_has_no_fallback_banner():
    assert "抽取式摘要" not in s1._extractive("## A\n")


def test_extractive_truncates_long_lines():
    long_link = "- [" + "x" * 500 + "]"
    out = s1._extractive("## A\n" + long_link + "\n")
    assert max(len(line) for line in out.splitlines()) <= 400


# ---------------------------------------------------------------------------
# META 块
# ---------------------------------------------------------------------------
def test_digest_meta_roundtrip():
    assert s1._digest_meta('正文<!--META:{"a": 1}-->尾巴') == '{"a": 1}'


def test_digest_meta_default_is_valid_json():
    meta = json.loads(s1._digest_meta("没有元信息"))
    assert meta["threads"] == []
    assert meta["catalog_total"] == 0


# ---------------------------------------------------------------------------
# 入参归一化（引擎可能传 str / ContentItem / list）
# ---------------------------------------------------------------------------
def test_as_text_accepts_str_list_and_content_item():
    assert s1._as_text("hi") == "hi"
    assert s1._as_text(["a", "b"]) == "a\n\nb"
    assert s1._as_text(SimpleNamespace(contents="c")) == "c"
    assert s1._as_text(SimpleNamespace(contents={"k": 1})) == '{"k": 1}'


def test_as_text_never_raises_on_odd_input():
    assert isinstance(s1._as_text(None), str)
    assert isinstance(s1._as_text(123), str)


# ---------------------------------------------------------------------------
# 稳定性脚本的结构体检
# ---------------------------------------------------------------------------
GOOD_REPORT = "## 摘要\n今天有若干条 AI 讨论。\n\n## 主题索引\n- [x](http://x)\n"


def test_structure_report_accepts_complete_report():
    st = stability._structure_report(GOOD_REPORT)
    assert st["summary_ends_properly"] is True
    assert st["summary_fallback"] is False
    assert st["has_index_section"] is True
    assert st["index_items"] == 1
    assert st["sections_count"] >= 2


def test_structure_report_flags_fallback_summary():
    text = "## 摘要\n（LLM 摘要不可用，以下为抽取式摘要）\n- x\n\n## 主题索引\n"
    assert stability._structure_report(text)["summary_fallback"] is True


def test_structure_report_flags_truncated_summary():
    st = stability._structure_report("## 摘要\n说到一半就断\n\n## 主题索引\n")
    assert st["summary_ends_properly"] is False


# ---------------------------------------------------------------------------
# 自动发现契约
# ---------------------------------------------------------------------------
def test_builtin_definitions_are_loaded():
    assert {"stage1st_ai", "stage1st_stability"} <= set(registry.LOADED_MODULES)
