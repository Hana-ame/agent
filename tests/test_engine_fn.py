"""fn 边执行契约的回归测试。

背景（2026-10-22）：v7 引擎在单参数 fn 失败后会吞掉异常，改用 ContentItem 再调一次。
实际后果：CF 图里 cf_verify 拿到 WA 数据抛 KeyError('zones')，被换成
TypeError("not ContentItem")，真正的错误被藏了；有副作用的 fn 还会被执行两遍。
"""
from __future__ import annotations

import pytest

from dag import DAG
from dag.schema import Processor


def _dag_with(fn, *, input_tags=("in",), output_tags=("out",), layer=0):
    d = DAG(name="fn_contract")
    d.add_edge(
        input_tags=list(input_tags),
        processor=Processor(fn=fn),
        output_tags=list(output_tags),
        layer=layer,
    )
    return d


# ---- 异常不许藏 ------------------------------------------------------------
def test_fn_exception_propagates_unmasked():
    def boom(_text):
        raise KeyError("zones")

    d = _dag_with(boom)
    d.add_content(contents="payload", tags=["in"], gen=0)

    with pytest.raises(KeyError, match="zones"):
        d.step()


def test_fn_called_once_even_when_it_raises():
    calls = []

    def boom(_text):
        calls.append(1)
        raise RuntimeError("boom")

    d = _dag_with(boom)
    d.add_content(contents="payload", tags=["in"], gen=0)

    with pytest.raises(RuntimeError):
        d.step()
    assert calls == [1], "有副作用的 fn 不许被重试"


# ---- 参数形态 --------------------------------------------------------------
def test_single_param_fn_receives_text():
    seen = []

    def fn(text):
        seen.append(text)
        return "ok"

    d = _dag_with(fn)
    d.add_content(contents="hello", tags=["in"], gen=0)
    d.step()

    assert seen == ["hello"]


def test_two_param_fn_receives_all_contents_and_items():
    """契约：2 参数 fn 收到 (内容, 条目)；多输入时内容也是列表。

    - 1 个输入：("A", ContentItem)
    - N 个输入：(["A", "B"], [ContentItem, ContentItem])
    """
    seen = []

    def fn(first, items):
        seen.append((first, [(list(it.tags), it.contents) for it in items]))
        return "merged"

    d = DAG(name="fn_contract")
    d.matcher.allow_multi_item_direct_match = True
    d.add_edge(
        input_tags=["a", "b"],
        processor=Processor(fn=fn),
        output_tags=["out"],
        layer=0,
    )
    d.add_content(contents="A", tags=["a"], gen=0)
    d.add_content(contents="B", tags=["b"], gen=0)
    d.step()

    assert seen == [(["A", "B"], [(["a"], "A"), (["b"], "B")])]


def test_single_param_multi_input_receives_text_list():
    seen = []

    def fn(texts):
        seen.append(texts)
        return "joined"

    d = DAG(name="fn_contract")
    d.matcher.allow_multi_item_direct_match = True
    d.add_edge(
        input_tags=["a", "b"],
        processor=Processor(fn=fn),
        output_tags=["out"],
        layer=0,
    )
    d.add_content(contents="A", tags=["a"], gen=0)
    d.add_content(contents="B", tags=["b"], gen=0)
    d.step()

    assert seen == [["A", "B"]]


# ---- fn 边不许因模糊匹配重复执行 -------------------------------------------
def test_similar_task_tags_do_not_duplicate_fn_edges():
    """cf_task ↔ cf_wa_task 相似度 0.8044：修复前会把两条拉数边各跑两遍。"""
    calls = []

    def cf_query(_text):
        calls.append("cf")
        return "cf-data"

    def wa_query(_text):
        calls.append("wa")
        return "wa-data"

    d = DAG(name="fn_exact")
    d.add_edge(input_tags=["cf_task"], processor=Processor(fn=cf_query),
               output_tags=["cf_subdomain_daily"], layer=0)
    d.add_edge(input_tags=["cf_wa_task"], processor=Processor(fn=wa_query),
               output_tags=["cf_wa_daily"], layer=0)
    d.add_content(contents="拉流量", tags=["cf_task"], gen=0)
    d.add_content(contents="拉 WA", tags=["cf_wa_task"], gen=0)
    d.run(until_tag="never", max_steps=10)

    assert calls == ["cf", "wa"]
