"""stage1st-ai-24h 的端到端离线验证：HTTP 与 LLM 都替掉，只验业务链路。

为什么必须有这个测试：外部站点 stage1st.com 实测 **TCP 三次握手成功、TLS 不响应**
（走本地代理也一样：`CONNECT ... 200` 之后没有下文）。没有这个测试时，站点一挂
整条 DAG 只能报「抓取失败」，**无法区分「代码坏了」和「数据源挂了」**。

这里把两处外部依赖都换成本地样本：
- `_http`     -> 返回构造好的板块目录页 / 帖子页 HTML
- `_llm_text` -> 按 prompt 特征返回分类 JSON 或摘要正文

于是「解析 → 判定 → 回溯 → 聚合 → 摘要 → 验收」整条链路可以在 1 秒内跑完。
"""
from __future__ import annotations

import json

import pytest

from dag_defs import stage1st_ai as s1


def _row(tid: str, title: str, last: str, replies: int) -> str:
    return (
        f'<tbody id="normalthread_{tid}">'
        f'<tr><td><a href="thread-{tid}-1-1.html" class="s xst">{title}</a></td>'
        f'<td><a href="forum.php?mod=redirect&tid={tid}&goto=lastpost#lastpost">{last}</a></td>'
        f'<td><a href="javascript:;" class="xi2">{replies}</a></td></tr>'
        f"</tbody>"
    )


CATALOG_HTML = "<html><body>" + "".join(
    [
        _row("100001", "本地部署大模型的显存优化实践", "2026-10-21 22:10", 18),
        _row("100002", "今天中午吃什么比较好", "2026-10-21 23:00", 42),
        _row("100003", "DeepSeek 新版本推理速度对比", "2026-10-22 01:20", 7),
        _row("100004", "谈一谈 Agent 框架的工程化", "2026-10-22 03:40", 11),
    ]
) + "</body></html>"


def _post(pid: str, ts: str, text: str) -> str:
    return (
        f'<div id="post_{pid}"><span>发表于 {ts}</span>'
        f'<td class="t_f" id="postmessage_{pid}">{text}</td></div>'
    )


THREAD_HTML = "<html><body>" + "".join(
    [
        _post("9001", "2026-10-22 01:10", "我用 int8 量化之后显存从 48G 降到 22G，吞吐基本没掉。"),
        _post("9002", "2026-10-22 02:05", "量化对长上下文影响比较大，超过 32k 之后质量明显下滑。"),
        _post("9003", "2026-10-22 05:30", "实测 vLLM 的 paged attention 配合量化效果最好。"),
    ]
) + "</body></html>"

SUMMARY_TEXT = (
    "### 一句话总览\n社区在讨论本地大模型部署的显存优化。\n\n"
    "### 核心议题\n1. **int8 量化**：显存 48G→22G。\n"
    "2. **长上下文质量**：超过 32k 后下滑。\n"
    "3. **vLLM**：Paged Attention 配合量化效果最好。\n"
)


def _fake_http(url: str, timeout: float = 15.0):
    if "forum-157-1.html" in url:
        return url, CATALOG_HTML
    if "mod=redirect" in url:
        return f"{s1.BASE}/thread-100001-1-1.html", ""
    if "thread-100001" in url:
        return url, THREAD_HTML
    if "thread-" in url:  # 其它帖子没有回帖内容
        return url, "<html><body></body></html>"
    return url, ""


def _fake_llm(prompt: str, **_kw) -> str:
    if "判断哪些帖子与 AI 相关" in prompt:
        return '{"ai": [1, 3, 4]}'  # 排除「今天中午吃什么」
    return SUMMARY_TEXT


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(s1, "_http", _fake_http)
    monkeypatch.setattr(s1, "_llm_text", _fake_llm)
    return s1


def _run(wired):
    dag = wired.build_stage1st_ai_dag()
    dag.add_content(contents=f"{wired.BASE}/forum-157-1.html", tags=["s1_board_url"], gen=0)
    dag.run(until_tag="s1_ai_report", max_steps=10)
    report = dag.get_result("s1_ai_report")
    assert report, "没有产出终版报告"
    return dag, report


def test_full_pipeline_produces_verifiable_report(wired):
    dag, report = _run(wired)
    assert "## 摘要" in report
    assert "## 主题索引" in report
    assert "目录 4 帖 / AI 相关 3 帖" in report
    assert report.count("\n- [") >= 3
    ok, why = wired.verify_report(report)
    assert ok, why


def test_classification_excludes_irrelevant_thread(wired):
    dag, report = _run(wired)
    assert "本地部署大模型的显存优化实践" in report
    assert "DeepSeek 新版本推理速度对比" in report
    assert "今天中午吃什么" not in report, "无关帖不该进索引"


def test_24h_backfill_collects_posts(wired):
    dag, report = _run(wired)
    assert "回帖 3 条" in report


def test_classify_falls_back_when_primary_model_fails(wired, monkeypatch):
    """主模型 401 时应回退到备用模型，而不是整条链崩掉。"""
    calls = []

    def llm(prompt, model_key=None, **_kw):
        calls.append(model_key)
        if model_key == wired.CLASSIFY_MODEL_KEY:
            raise RuntimeError("HTTPError: 401 Client Error: Unauthorized")
        return _fake_llm(prompt)

    monkeypatch.setattr(wired, "_llm_text", llm)
    dag, report = _run(wired)
    assert "目录 4 帖 / AI 相关 3 帖" in report
    assert calls[0] == wired.CLASSIFY_MODEL_KEY
    assert wired.SUMMARY_MODEL_KEY in calls, "应当回退到 laptop-6.8"


def test_empty_catalog_fails_fast(wired, monkeypatch):
    monkeypatch.setattr(wired, "_http", lambda url, timeout=15.0: (url, "<html></html>"))
    dag = wired.build_stage1st_ai_dag()
    dag.add_content(contents="x", tags=["s1_board_url"], gen=0)
    with pytest.raises(RuntimeError, match="目录抓取失败：0 帖"):
        dag.run(until_tag="s1_ai_report", max_steps=5)
