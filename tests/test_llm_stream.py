"""OpenAICompatibleClient 的流式与重试测试（全离线，用 httpx.MockTransport）。

这两项能力自 v6 线移植：
- SSE 流式：长输出非流式会被中间层掐断，实测稳定性差一个量级
- 重试白名单：403(1010) 是网关瞬时 WAF 封锁，不是鉴权失败，必须重试
"""
from __future__ import annotations

import json

import httpx
import pytest

from dag_agent.llm import OpenAICompatibleClient
from dag_agent.schema import ModelConfig


def _cfg() -> ModelConfig:
    return ModelConfig(key="unit", endpoint="http://unit.test/v1", apikey="not-needed", model="m")


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}" for c in chunks]
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _delta(**kw) -> dict:
    return {"choices": [{"delta": kw}]}


def _client(handler, **kw) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(stream=True, transport=httpx.MockTransport(handler), **kw)


def test_stream_concatenates_content_chunks():
    body = _sse(_delta(content="你好"), _delta(content="，世界"), _delta(content="！"))
    seen = {}

    def handler(request):
        seen["accept"] = request.headers.get("accept", "")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    out = _client(handler).complete("hi", model_config=_cfg())

    assert out == "你好，世界！"
    assert "text/event-stream" in seen["accept"]
    assert seen["payload"]["stream"] is True  # 流式时强制带 stream=true


def test_stream_falls_back_to_reasoning_when_content_empty():
    body = _sse(_delta(reasoning_content="思考中"), _delta(reasoning_content="最终答案"))
    out = _client(lambda r: httpx.Response(200, content=body)).complete("hi", model_config=_cfg())
    assert out == "思考中最终答案"


def test_stream_ignores_done_and_malformed_frames():
    raw = (
        b'data: {"choices":[{"delta":{"content":"A"}}]}\n'
        b"data: not-json\n"
        b": keep-alive comment\n"
        b'data: {"choices":[]}\n'
        b'data: {"choices":[{"delta":{"content":"B"}}]}\n'
        b"data: [DONE]\n"
    )
    out = _client(lambda r: httpx.Response(200, content=raw)).complete("hi", model_config=_cfg())
    assert out == "AB"


def test_403_is_retried_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, content=b'{"code":1010,"msg":"waf"}')
        return httpx.Response(200, content=_sse(_delta(content="ok")))

    client = _client(handler, max_retries=2)
    assert client.complete("hi", model_config=_cfg()) == "ok"
    assert calls["n"] == 2


def test_connection_reset_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("[WinError 10054] 远程主机强迫关闭了一个现有的连接")
        return httpx.Response(200, content=_sse(_delta(content="ok")))

    assert _client(handler, max_retries=2).complete("hi", model_config=_cfg()) == "ok"
    assert calls["n"] == 2


def test_400_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, content=b'{"error":"bad request"}')

    with pytest.raises(RuntimeError):
        _client(handler, max_retries=3).complete("hi", model_config=_cfg())
    assert calls["n"] == 1, "参数错误不该重试，浪费配额"


def test_retry_budget_is_bounded():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, content=b"upstream down")

    with pytest.raises(RuntimeError):
        _client(handler, max_retries=3).complete("hi", model_config=_cfg())
    assert calls["n"] == 3


def test_empty_stream_raises_without_retry():
    """空响应不在重试白名单里——与 v6 行为一致：不打转刷配额，让上层决定重发。"""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=_sse(_delta(content="")))

    with pytest.raises(RuntimeError, match="空响应"):
        _client(handler, max_retries=2).complete("hi", model_config=_cfg())
    assert calls["n"] == 1


def test_call_history_records_each_attempt_once():
    body = _sse(_delta(content="ok"))
    client = _client(lambda r: httpx.Response(200, content=body))
    client.complete("hi", model_config=_cfg())
    assert len(client.call_history) == 1
    assert client.call_history[0].is_clean_context is True
