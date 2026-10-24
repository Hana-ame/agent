"""标签匹配回归测试。

背景（2026-10-22）：CF 图里 `cf_verify` 的输入 tag 是 `cf_subdomain_daily`，
却把 `cf_wa_daily` 的 Web Analytics JSON 吃了进来。原因是离线确定性 embedding
把字符和 bigram 当成语义 token，两个 tag 余弦 0.7402 > 0.70 阈值。

这里锁死两件事：
1. 只共享词缀的标识符不许互相匹配；
2. 真正的语义/拼写变体仍然要能匹配（别把模糊匹配修死）。
"""
from __future__ import annotations

import pytest

from dag.matcher import TagMatcher
from dag.schema import ContentItem, EdgeSpec, Processor


def _sim(a: str, b: str) -> float:
    return TagMatcher().tag_similarity(a, b)


# ---- 不许误配 --------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("cf_subdomain_daily", "cf_wa_daily"),
    ("posts", "points"),
])
def test_similar_looking_identifiers_do_not_match(a: str, b: str):
    assert _sim(a, b) < 0.70, f"{a} ↔ {b} 相似度 {_sim(a, b)}，会串数据"


def test_exact_tag_is_always_1():
    assert _sim("cf_wa_daily", "cf_wa_daily") == 1.0


def test_verify_edge_does_not_consume_wa_content():
    """最小复现：verify 边只有一个输入 tag，工作区里只有 WA 内容。"""
    verify = EdgeSpec(
        input_tags=["cf_subdomain_daily"],
        processor=Processor(fn=lambda text: text),
        output_tags=["cf_verify_result"],
        layer=1,
        id="verify",
    )
    wa_item = ContentItem(tags=["cf_wa_daily"], contents='{"wa": true}', gen=0)
    assert TagMatcher().find_best_candidate([verify], [wa_item], set()) is None


# ---- 该匹配的仍然匹配 ------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("draft", "drafts"),          # 拼写变体（词法 n-gram）
    ("task", "tasks"),
    ("report", "报告"),            # 同义词 canonical（embedding）
    ("revenue", "income"),
])
def test_fuzzy_and_semantic_matches_still_work(a: str, b: str):
    assert _sim(a, b) >= 0.70, f"{a} ↔ {b} 相似度 {_sim(a, b)}，模糊匹配被修死了"


def test_fn_edge_still_matches_exact_tag():
    """精确相等时 fn 边必须照常命中（别把精确匹配也修死）。"""
    edge = EdgeSpec(
        input_tags=["draft"],
        processor=Processor(fn=lambda text: text),
        output_tags=["critique"],
        layer=0,
        id="critique",
    )
    item = ContentItem(tags=["draft"], contents="草稿正文", gen=0)
    cand = TagMatcher().find_best_candidate([edge], [item], set())
    assert cand is not None and cand.content_indices == (0,)
    assert cand.similarity == 1.0


# ---- fn 边只认精确 tag（LLM 边不受影响）------------------------------------
def test_fn_edge_does_not_match_similar_task_tag():
    """cf_task ↔ cf_wa_task 相似度 0.8044：旧行为会让拉数边各跑两遍。"""
    query = EdgeSpec(
        input_tags=["cf_task"],
        processor=Processor(fn=lambda text: text),
        output_tags=["cf_subdomain_daily"],
        layer=0,
        id="cf_query",
    )
    wa_seed = ContentItem(tags=["cf_wa_task"], contents="拉 WA 数据", gen=0)
    assert TagMatcher().find_best_candidate([query], [wa_seed], set()) is None


def test_llm_edge_still_allows_fuzzy_match():
    """自然语言边保留模糊匹配：draft ↔ drafts 仍应命中。"""
    edge = EdgeSpec(
        input_tags=["draft"],
        processor=Processor(model="some-model"),
        output_tags=["critique"],
        layer=0,
        id="critique",
    )
    item = ContentItem(tags=["drafts"], contents="草稿正文", gen=0)
    cand = TagMatcher().find_best_candidate([edge], [item], set())
    assert cand is not None and cand.content_indices == (0,)
