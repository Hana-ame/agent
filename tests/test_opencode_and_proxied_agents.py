"""Tests for OpenCodeAgent, ProxiedLLMAgent and their shared HTTP base.

Covers the two new agent engines plus the ``_HTTPAgentBase`` plumbing they
both inherit (retry semantics, payload assembly, SSE streaming, lifecycle),
and the ``get_agent`` factory wiring for the new spec strings / config blocks.
"""

import asyncio
import json
import logging
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.agents import (
    DEFAULT_PROXY_BASE_URL,
    DEFAULT_ZEN_BASE_URL,
    DEFAULT_ZEN_MODEL,
    KNOWN_ZEN_MODELS,
    BaseAgent,
    HttpLLMAgent,
    MockAgent,
    NonRetryableHTTPError,
    OpenCodeAgent,
    ProxiedLLMAgent,
    ThrottleTimeoutError,
    get_agent,
)
from framework.agents._http_base import RETRYABLE_STATUS, _TokenBucket
from framework.utils.errors import ComputeError

PROXY_ENV_VARS = ("LLM_PROXY_BASE_URL", "OPENAI_BASE_URL", "LLM_PROXY_API_KEY", "OPENAI_API_KEY")


# ====================================================================
# Shared HTTP base plumbing
# ====================================================================
class _HTTPHelpers:
    """Response / stream fakes shared by every test class here."""

    @staticmethod
    def make_resp(status_code=200, content="ok"):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = json.dumps({"choices": [{"message": {"content": content}}]})
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"{status_code}",
                request=MagicMock(),
                response=resp,
            )
        return resp

    @staticmethod
    def make_stream(status_code=200, lines=None, body=b'{"error": "nope"}'):
        """Async-context-manager fake for ``client.stream()``."""
        if lines is None:
            lines = [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                "",
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                'data: [DONE]',
            ]

        stream = MagicMock()
        stream.status_code = status_code
        stream.aread = AsyncMock(return_value=body)

        async def _aiter_lines():
            for line in lines:
                yield line

        stream.aiter_lines = _aiter_lines
        stream.__aenter__ = AsyncMock(return_value=stream)
        stream.__aexit__ = AsyncMock(return_value=False)
        return stream


class TestHTTPAgentHierarchy:
    """All three HTTP agents share one implementation."""

    def test_http_agent_is_subclass(self):
        assert issubclass(HttpLLMAgent, BaseAgent)

    def test_opencode_agent_is_subclass(self):
        assert issubclass(OpenCodeAgent, BaseAgent)

    def test_proxied_agent_is_subclass(self):
        assert issubclass(ProxiedLLMAgent, BaseAgent)

    @pytest.mark.asyncio
    async def test_shared_client_config(self):
        agent = HttpLLMAgent(timeout=55.0, trust_env=False, extra_headers={"X-Title": "vea-test"})
        assert agent.base_url.rstrip("/") == agent.base_url
        assert agent.timeout == 55.0
        assert agent.trust_env is False
        assert agent.client.trust_env is False
        assert agent.headers["Content-Type"] == "application/json"
        assert agent.headers["X-Title"] == "vea-test"
        assert "Authorization" in agent.headers
        await agent.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        agent = HttpLLMAgent()
        await agent.close()
        assert agent._closed is True
        # Second close must be a no-op, not an error.
        await agent.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with HttpLLMAgent() as agent:
            assert agent.client is not None
        assert agent._closed is True

    @pytest.mark.asyncio
    async def test_headers_sent_on_request(self):
        agent = HttpLLMAgent(api_key="k", extra_headers={"X-Request-Id": "abc-123"})
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "m")
        headers = agent.client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer k"
        assert headers["X-Request-Id"] == "abc-123"
        await agent.close()

    @pytest.mark.asyncio
    async def test_dict_data_json_encoded(self):
        """Structured data is JSON-encoded so the LLM sees a stable shape."""
        agent = HttpLLMAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process({"a": 1, "中文": "值"}, "p", "m")
        payload = agent.client.post.call_args[1]["json"]
        user_msg = payload["messages"][-1]
        assert user_msg["role"] == "user"
        assert json.loads(user_msg["content"]) == {"a": 1, "中文": "值"}
        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_prompt_omits_system_message(self):
        agent = HttpLLMAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "", "m")
        payload = agent.client.post.call_args[1]["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        await agent.close()

    @pytest.mark.asyncio
    async def test_non_ascii_output_survives_json_encoding(self):
        agent = HttpLLMAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("café résumé", "p", "m")
        payload = agent.client.post.call_args[1]["json"]
        assert "café résumé" in payload["messages"][-1]["content"]
        await agent.close()


# ====================================================================
# Retry semantics (inherited by all HTTP agents)
# ====================================================================
class TestRetrySemantics:
    @pytest.mark.asyncio
    async def test_500_is_retried_then_raises(self):
        agent = HttpLLMAgent(api_key="test-key", max_retries=2)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(500))
        with pytest.raises(httpx.HTTPStatusError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2
        await agent.close()

    def test_retryable_status_partition(self):
        """Transient vs fatal is decided by one explicit set, not by feel."""
        assert RETRYABLE_STATUS == {408, 429, 500, 502, 503, 504}
        for fatal in (400, 401, 403, 404, 405, 409, 410, 413, 422):
            assert fatal not in RETRYABLE_STATUS

    @pytest.mark.asyncio
    async def test_throttle_timeout_is_a_compute_error(self):
        """Edges must treat a throttled agent like a failed computation."""
        assert issubclass(ThrottleTimeoutError, ComputeError)
        exc = ThrottleTimeoutError(waited=0.25, kind="concurrency slot")
        assert exc.waited == 0.25
        assert exc.kind == "concurrency slot"
        assert "0.25s" in str(exc)

    @pytest.mark.asyncio
    async def test_503_is_retryable_on_opencode_agent(self):
        agent = OpenCodeAgent(max_retries=2)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(503))
        with pytest.raises(httpx.HTTPStatusError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_504_is_retryable_on_proxied_agent(self):
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1", max_retries=2)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(504))
        with pytest.raises(httpx.HTTPStatusError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_400_is_fatal_no_retry(self):
        agent = HttpLLMAgent(max_retries=3)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(400))
        with pytest.raises(NonRetryableHTTPError) as excinfo:
            await agent.process("d", "p", "m")
        assert excinfo.value.status_code == 400
        assert agent.client.post.call_count == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_401_is_fatal(self):
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1", max_retries=3)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(401))
        with pytest.raises(NonRetryableHTTPError) as excinfo:
            await agent.process("d", "p", "m")
        assert excinfo.value.status_code == 401
        assert "401" in str(excinfo.value)
        assert agent.client.post.call_count == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_403_is_fatal_on_opencode_agent(self):
        agent = OpenCodeAgent(max_retries=3)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(403))
        with pytest.raises(NonRetryableHTTPError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_connection_error_retries_then_raises(self):
        agent = OpenCodeAgent(max_retries=2)
        agent.client.post = AsyncMock(side_effect=httpx.RequestError("boom"))
        with pytest.raises(httpx.RequestError, match="boom"):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_then_success_recovers(self):
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1", max_retries=3)
        agent.client.post = AsyncMock(
            side_effect=[_HTTPHelpers.make_resp(500), _HTTPHelpers.make_resp(200, "back")]
        )
        assert await agent.process("d", "p", "m") == "back"
        assert agent.client.post.call_count == 2
        await agent.close()


# ====================================================================
# SSE streaming (inherited by all HTTP agents)
# ====================================================================
class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_process_yields_deltas(self):
        agent = HttpLLMAgent()
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream())
        chunks = [c async for c in agent.stream_process("d", "p", "m")]
        assert chunks == ["Hel", "lo"]
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_payload_sets_stream_flag(self):
        agent = HttpLLMAgent()
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream())
        async for _ in agent.stream_process("d", "p", "m", settings={"llm_kwargs": {"temperature": 0.2}}):
            pass
        call = agent.client.stream.call_args
        # method and url are positional, json/headers are keyword-only.
        assert call[0][0] == "POST"
        assert call[0][1] == f"{agent.base_url}/chat/completions"
        assert call[1]["json"]["stream"] is True
        assert call[1]["json"]["temperature"] == 0.2
        assert call[1]["headers"] == agent.headers
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_ignores_blank_and_non_data_lines(self):
        lines = [
            ": keep-alive comment",
            "",
            'data: {"choices":[{"delta":{"content":"only"}}]}',
            'event: error',
            'data: not-json-at-all',
            'data: [DONE]',
        ]
        agent = OpenCodeAgent()
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream(lines=lines))
        assert [c async for c in agent.stream_process("d", "p", "m")] == ["only"]
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_skips_empty_deltas(self):
        lines = [
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[]}',
            'data: {"choices":[{"delta":{"content":"x"}}]}',
            'data: [DONE]',
        ]
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1")
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream(lines=lines))
        assert [c async for c in agent.stream_process("d", "p", "m")] == ["x"]
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_stops_at_done_sentinel(self):
        lines = [
            'data: [DONE]',
            'data: {"choices":[{"delta":{"content":"late"}}]}',
        ]
        agent = HttpLLMAgent()
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream(lines=lines))
        assert [c async for c in agent.stream_process("d", "p", "m")] == []
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_4xx_raises_nonretryable(self):
        agent = HttpLLMAgent()
        agent.client.stream = MagicMock(
            return_value=_HTTPHelpers.make_stream(status_code=401, body=b'{"error":"bad key"}')
        )
        with pytest.raises(NonRetryableHTTPError) as excinfo:
            [c async for c in agent.stream_process("d", "p", "m")]
        assert excinfo.value.status_code == 401
        assert "bad key" in str(excinfo.value)
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_transport_error_propagates(self):
        agent = OpenCodeAgent()
        agent.client.stream = MagicMock(side_effect=httpx.ReadError("socket closed"))
        with pytest.raises(httpx.ReadError, match="socket closed"):
            [c async for c in agent.stream_process("d", "p", "m")]
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_resolves_model_before_post(self):
        agent = OpenCodeAgent()
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream(lines=[]))
        async for _ in agent.stream_process("d", "p", "default"):
            pass
        assert agent.client.stream.call_args[1]["json"]["model"] == DEFAULT_ZEN_MODEL
        await agent.close()


# ====================================================================
# Transport-level HTTP proxy (the request tunnels through a proxy server)
# ====================================================================
class TestTransportProxy:
    """``proxy=`` makes the HTTP request itself go through a proxy server.

    Distinct from ``proxy_url`` on :class:`ProxiedLLMAgent` (which is the
    application-level gateway base URL): ``proxy`` is the transport tunnel
    — corporate egress / SOCKS / authenticated proxy.
    """

    @staticmethod
    def _mounts(agent):
        return getattr(agent.client, "_mounts", {})

    @pytest.mark.asyncio
    async def test_no_proxy_has_no_mount(self, monkeypatch):
        for var in ("HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.delenv(var, raising=False)
        agent = HttpLLMAgent(trust_env=False)
        assert agent.proxy is None
        assert len(self._mounts(agent)) == 0
        await agent.close()

    @pytest.mark.asyncio
    async def test_explicit_proxy_creates_a_mount(self):
        agent = HttpLLMAgent(proxy="http://corp-proxy:3128")
        assert agent.proxy == "http://corp-proxy:3128"
        assert len(self._mounts(agent)) == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_socks_proxy_accepted(self):
        agent = OpenCodeAgent(proxy="socks5://host:1080")
        assert agent.proxy == "socks5://host:1080"
        assert len(self._mounts(agent)) == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_authenticated_proxy_url_stored(self):
        url = "http://user:pa%40ss@corp-proxy:3128"
        agent = HttpLLMAgent(proxy=url)
        assert agent.proxy == url
        assert len(self._mounts(agent)) == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxy_and_trust_env_are_independent(self, monkeypatch):
        """``proxy`` wins regardless of ``trust_env``."""
        for var in ("HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.delenv(var, raising=False)
        agent = HttpLLMAgent(proxy="http://corp:3128", trust_env=False)
        assert agent.proxy == "http://corp:3128"
        assert agent.trust_env is False
        assert len(self._mounts(agent)) == 1  # explicit proxy still mounts
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxied_agent_gateway_and_transport_proxy_stack(self, monkeypatch):
        """The gateway URL and the transport proxy are two layers."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(
            proxy_url="http://litellm.internal:4000/v1",  # WHO
            proxy="http://corp-egress:3128",               # HOW
        )
        assert agent.base_url == "http://litellm.internal:4000/v1"  # gateway
        assert agent.proxy == "http://corp-egress:3128"             # transport
        assert len(self._mounts(agent)) == 1
        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_request_actually_tunnels_through_the_proxy(self):
        """End-to-end: the HTTP request must physically pass through the proxy.

        A real local proxy server relays the agent's POST to a local upstream
        (both on 127.0.0.1), and both sides record what they saw. If the
        proxy saw the absolute-form request *and* the upstream got the LLM
        payload, then ``proxy=`` truly routed the HTTP request — not just a
        stored attribute.
        """
        import http.client
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        upstream_payloads = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                upstream_payloads.append(self.rfile.read(length).decode())
                body = json.dumps({"choices": [{"message": {"content": "proxied-ok"}}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, fmt, *args):
                pass

        proxy_seen = []

        class Proxy(BaseHTTPRequestHandler):
            """Minimal HTTP forward proxy: relays absolute-form requests upstream."""
            def do_POST(self):
                proxy_seen.append(self.path)  # absolute-form URL, e.g. http://host:port/v1/...
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                parsed = urlparse(self.path)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80)
                conn.request(self.command, parsed.path or "/", body=body, headers=dict(self.headers))
                resp = conn.getresponse()
                data = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(data)
                conn.close()

            def log_message(self, fmt, *args):
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
            proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
            agent = HttpLLMAgent(base_url=upstream_url, proxy=proxy_url)
            result = await agent.process("hello", "be terse", "test-model")
            assert result == "proxied-ok"
            # The proxy saw the absolute-form request ...
            assert len(proxy_seen) == 1
            assert "/v1/chat/completions" in proxy_seen[0]
            # ... and the real LLM payload arrived at the upstream.
            assert len(upstream_payloads) == 1
            assert json.loads(upstream_payloads[0])["model"] == "test-model"
            await agent.close()
        finally:
            proxy.shutdown()
            upstream.shutdown()

    def test_https_proxy_key_alias(self, monkeypatch):
        """graph.json can set the transport proxy as `https_proxy` / `HTTPS_PROXY`."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        for key in ("proxy", "https_proxy", "HTTPS_PROXY"):
            agent = get_agent({"type": "http", key: "http://127.0.0.4:7890"})
            assert agent.proxy == "http://127.0.0.4:7890", key
            assert len(getattr(agent.client, "_mounts", {})) == 1, key

    @pytest.mark.asyncio
    async def test_config_proxy_overrides_env(self, monkeypatch):
        """A proxy set in the graph config overrides HTTPS_PROXY / HTTP_PROXY env.

        The environment points at a *drop* proxy (records and 502s), while the
        graph config points at a *real* proxy (forwards upstream). If the
        request went through the real proxy and the drop proxy saw nothing,
        the graph config wins — exactly the override semantics required.
        """
        import http.client
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        for var in (
            "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
            "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
        ):
            monkeypatch.delenv(var, raising=False)

        drop_seen = []
        real_seen = []
        upstream_payloads = []

        class DropProxy(BaseHTTPRequestHandler):
            """Env proxy: records and refuses — if hit, override failed."""
            def do_POST(self):
                drop_seen.append(self.path)
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        class RealProxy(BaseHTTPRequestHandler):
            """Config proxy: forwards upstream."""
            def do_POST(self):
                real_seen.append(self.path)
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                parsed = urlparse(self.path)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80)
                conn.request(self.command, parsed.path or "/", body=body, headers=dict(self.headers))
                resp = conn.getresponse()
                data = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(data)
                conn.close()

            def log_message(self, *a):
                pass

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                upstream_payloads.append(self.rfile.read(length).decode())
                body = json.dumps({"choices": [{"message": {"content": "config-proxy-wins"}}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *a):
                pass

        drop = ThreadingHTTPServer(("127.0.0.1", 0), DropProxy)
        real = ThreadingHTTPServer(("127.0.0.1", 0), RealProxy)
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=drop.serve_forever, daemon=True).start()
        threading.Thread(target=real.serve_forever, daemon=True).start()
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            # Env points at the drop proxy.
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{drop.server_address[1]}")
            monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{drop.server_address[1]}")
            # Graph config (agent dict, as it appears in graph.json) wins.
            agent = get_agent({
                "type": "http",
                "base_url": f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                "https_proxy": f"http://127.0.0.1:{real.server_address[1]}",
            })
            assert agent.proxy == f"http://127.0.0.1:{real.server_address[1]}"
            result = await agent.process("hello", "p", "m")
            assert result == "config-proxy-wins"
            assert len(real_seen) == 1      # went through the configured proxy
            assert len(drop_seen) == 0      # env proxy untouched → override works
            await agent.close()
        finally:
            drop.shutdown()
            real.shutdown()
            upstream.shutdown()


# ====================================================================
# Self-throttling: token bucket + bounded concurrency
# ====================================================================
class TestTokenBucket:
    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="rate and period must be positive"):
            _TokenBucket(rate=0, period=1.0)
        with pytest.raises(ValueError, match="rate and period must be positive"):
            _TokenBucket(rate=5, period=0)
        with pytest.raises(ValueError, match="rate and period must be positive"):
            _TokenBucket(rate=-1, period=1.0)

    @pytest.mark.asyncio
    async def test_first_token_is_free(self):
        bucket = _TokenBucket(rate=1.0, period=1.0)
        waited = await bucket.acquire()
        assert waited == pytest.approx(0.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_burst_capacity_allows_immediate_acquires(self):
        """Default capacity equals the rate, so a burst of N is instant."""
        bucket = _TokenBucket(rate=5.0, period=1.0)
        started = time.monotonic()
        for _ in range(5):
            await bucket.acquire()
        assert time.monotonic() - started < 0.1

    @pytest.mark.asyncio
    async def test_exhausted_bucket_waits_for_refill(self):
        bucket = _TokenBucket(rate=1.0, period=0.2)  # 1 token per 200 ms
        await bucket.acquire()
        started = time.monotonic()
        await bucket.acquire()
        waited = time.monotonic() - started
        assert 0.15 <= waited <= 0.6

    @pytest.mark.asyncio
    async def test_capacity_is_capped_at_burst(self):
        """Refilling must not accumulate credits past the burst cap."""
        bucket = _TokenBucket(rate=1.0, period=0.1, capacity=1.0)
        await bucket.acquire()
        await asyncio.sleep(0.3)  # would earn 3 tokens, cap is 1
        started = time.monotonic()
        await bucket.acquire()          # consumes the 1 available
        assert time.monotonic() - started < 0.05
        started = time.monotonic()
        await bucket.acquire()          # nothing left — must wait
        assert time.monotonic() - started >= 0.05

    @pytest.mark.asyncio
    async def test_timeout_raises_and_does_not_lose_tokens(self):
        bucket = _TokenBucket(rate=1.0, period=60.0)  # 1 token per minute
        await bucket.acquire()
        with pytest.raises(ThrottleTimeoutError) as excinfo:
            await bucket.acquire(timeout=0.05)
        assert excinfo.value.kind == "rate budget"
        assert excinfo.value.waited >= 0.04

    @pytest.mark.asyncio
    async def test_failed_timeout_does_not_consume_the_token(self):
        """A caller that timed out must not have stolen anyone's token."""
        bucket = _TokenBucket(rate=1.0, period=60.0)
        await bucket.acquire()
        with pytest.raises(ThrottleTimeoutError):
            await bucket.acquire(timeout=0.05)
        # The original token is already spent, so a third caller still waits.
        with pytest.raises(ThrottleTimeoutError):
            await bucket.acquire(timeout=0.05)

    @pytest.mark.asyncio
    async def test_concurrent_waiters_each_get_one_token(self):
        """N waiters against rate=N must all succeed roughly simultaneously."""
        bucket = _TokenBucket(rate=4.0, period=1.0)
        results = await asyncio.gather(*(bucket.acquire() for _ in range(4)))
        assert all(w < 0.1 for w in results)


class TestConcurrencyGate:
    """An agent with a wide graph in front of it must queue, not pile on."""

    @staticmethod
    async def _measure_parallelism(agent, calls=6, delay=0.1):
        depth = 0
        max_depth = 0
        resp = _HTTPHelpers.make_resp()

        async def slow_post(*args, **kwargs):
            nonlocal depth, max_depth
            depth += 1
            max_depth = max(max_depth, depth)
            await asyncio.sleep(delay)
            depth -= 1
            return resp

        agent.client.post = AsyncMock(side_effect=slow_post)
        await asyncio.gather(*(agent.process("d", "p", "m") for _ in range(calls)))
        return max_depth

    @pytest.mark.asyncio
    async def test_opencode_agent_bounds_in_flight_calls(self):
        agent = OpenCodeAgent(max_concurrency=2, requests_per_minute=None)
        assert await self._measure_parallelism(agent) == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxied_agent_bounds_in_flight_calls(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1", max_concurrency=3)
        assert await self._measure_parallelism(agent, calls=7) == 3
        await agent.close()

    @pytest.mark.asyncio
    async def test_concurrency_of_one_serialises(self):
        order = []
        resp = _HTTPHelpers.make_resp()
        agent = OpenCodeAgent(max_concurrency=1, requests_per_minute=None)

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.05)
            order.append(len(order))
            return resp

        agent.client.post = AsyncMock(side_effect=slow_post)
        await asyncio.gather(*(agent.process("d", "p", "m") for _ in range(3)))
        assert order == [0, 1, 2]
        await agent.close()

    @pytest.mark.asyncio
    async def test_plain_http_agent_is_unbounded(self):
        agent = HttpLLMAgent()
        assert getattr(agent, "max_concurrency", None) is None
        assert getattr(agent, "_rate_budget", None) is None
        assert await self._measure_parallelism(agent, calls=6, delay=0.02) == 6
        await agent.close()

    @pytest.mark.asyncio
    async def test_budget_paces_attempts(self):
        """The per-minute budget is charged per *attempt*, not per call."""
        agent = OpenCodeAgent(max_concurrency=10, requests_per_minute=1.0)
        # Rebind to a fast clock so the test does not sleep for 60 s.
        agent._rate_budget = _TokenBucket(rate=2.0, period=1.0)  # burst 2, then 2/s
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())

        started = time.monotonic()
        for _ in range(4):
            await agent.process("d", "p", "m")
        elapsed = time.monotonic() - started
        assert elapsed >= 0.9  # 2 immediate + ~0.5 s per remaining token
        await agent.close()

    @pytest.mark.asyncio
    async def test_disabled_budget_is_a_noop(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(proxy_url="http://proxy:4000/v1")
        assert agent.requests_per_minute is None
        assert agent._rate_budget is None
        started = time.monotonic()
        for _ in range(20):
            await agent._acquire_budget()
        assert time.monotonic() - started < 0.05

    @pytest.mark.asyncio
    async def test_queue_timeout_on_concurrency_slot(self):
        """A saturated gate must refuse instead of hanging the graph."""
        resp = _HTTPHelpers.make_resp()
        agent = OpenCodeAgent(max_concurrency=1, queue_timeout=0.05, requests_per_minute=None)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        with pytest.raises(ThrottleTimeoutError) as excinfo:
            async with agent._concurrency_gate():  # occupy the only slot
                await agent.process("d", "p", "m")
        assert excinfo.value.kind == "concurrency slot"
        await agent.close()

    @pytest.mark.asyncio
    async def test_queue_timeout_on_rate_budget(self):
        agent = OpenCodeAgent(max_concurrency=10, queue_timeout=0.05, requests_per_minute=1.0)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "m")  # spends the single token
        with pytest.raises(ThrottleTimeoutError) as excinfo:
            await agent.process("d", "p", "m")
        assert excinfo.value.kind == "rate budget"
        await agent.close()

    @pytest.mark.asyncio
    async def test_stream_holds_a_concurrency_slot(self):
        """Streams are long-lived, so they must count against the ceiling."""
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: [DONE]',
        ]
        agent = OpenCodeAgent(max_concurrency=1, queue_timeout=0.05, requests_per_minute=None)
        agent.client.stream = MagicMock(return_value=_HTTPHelpers.make_stream(lines=lines))
        with pytest.raises(ThrottleTimeoutError):
            async with agent._concurrency_gate():
                [c async for c in agent.stream_process("d", "p", "m")]
        await agent.close()

    @pytest.mark.parametrize("kwargs,match", [
        ({"max_concurrency": 0}, "max_concurrency must be >= 1"),
        ({"max_concurrency": -2}, "max_concurrency must be >= 1"),
        ({"requests_per_minute": 0}, "requests_per_minute must be > 0 or None"),
        ({"requests_per_minute": -5}, "requests_per_minute must be > 0 or None"),
        ({"queue_timeout": 0}, "queue_timeout must be > 0 or None"),
    ])
    def test_invalid_throttle_config_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OpenCodeAgent(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"max_concurrency": 1},
        {"requests_per_minute": None, "queue_timeout": None},
        {"max_concurrency": 64, "requests_per_minute": 120.0},
    ])
    def test_valid_throttle_config_accepted(self, kwargs):
        assert isinstance(OpenCodeAgent(**kwargs), OpenCodeAgent)

    @pytest.mark.asyncio
    async def test_throttle_timeout_is_logged(self, caplog):
        """A refused call must be diagnosable from the logs alone."""
        agent = OpenCodeAgent(max_concurrency=1, queue_timeout=0.05, requests_per_minute=None)
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        with caplog.at_level(logging.ERROR, logger="vertex_edge_agent.agents"):
            with pytest.raises(ThrottleTimeoutError):
                async with agent._concurrency_gate():
                    await agent.process("d", "p", "m")
        assert any("Throttled" in r.message for r in caplog.records)
        await agent.close()


# ====================================================================
# OpenCodeAgent
# ====================================================================
class TestOpenCodeAgentConstruction:
    def test_default_values(self):
        agent = OpenCodeAgent()
        assert agent.base_url == DEFAULT_ZEN_BASE_URL == "https://opencode.ai/zen/v1"
        assert agent.api_key == "public"
        assert agent.default_model == DEFAULT_ZEN_MODEL == "hy3-free"
        assert agent.max_retries == 3
        assert agent.timeout == 300.0
        assert agent.trust_env is True
        assert agent.NAME == "OpenCodeAgent"
        assert agent.client is not None
        # Self-throttling defaults — bounded by design for the free tier.
        assert agent.max_concurrency == 3
        assert agent.requests_per_minute == 20.0
        assert agent.queue_timeout == 60.0
        assert agent._in_flight_gate is not None
        assert agent._rate_budget is not None

    def test_custom_values(self):
        agent = OpenCodeAgent(
            base_url="http://zen.local:9000/v1/",
            api_key="sk-zen-123",
            default_model="deepseek-v4-flash",
            max_retries=5,
            timeout=60.0,
            trust_env=False,
        )
        assert agent.base_url == "http://zen.local:9000/v1"  # trailing slash stripped
        assert agent.api_key == "sk-zen-123"
        assert agent.default_model == "deepseek-v4-flash"
        assert agent.max_retries == 5
        assert agent.timeout == 60.0
        assert agent.trust_env is False
        assert agent.client.trust_env is False

    def test_throttle_knobs_overridable(self):
        agent = OpenCodeAgent(
            max_concurrency=12, requests_per_minute=90.0, queue_timeout=5.0
        )
        assert agent.max_concurrency == 12
        assert agent.requests_per_minute == 90.0
        assert agent.queue_timeout == 5.0
        assert agent._rate_budget is not None

    def test_throttle_knobs_overridable_on_proxied_agent(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(
            proxy_url="http://gw:4000/v1", max_concurrency=8, queue_timeout=10.0
        )
        assert agent.max_concurrency == 8
        assert agent.queue_timeout == 10.0

    def test_default_model_class_attr_matches_constant(self):
        assert OpenCodeAgent.DEFAULT_MODEL == DEFAULT_ZEN_MODEL


class TestOpenCodeAgentModelCatalog:
    def test_catalog_is_not_empty(self):
        assert KNOWN_ZEN_MODELS
        assert DEFAULT_ZEN_MODEL in KNOWN_ZEN_MODELS

    def test_available_models_returns_copy(self):
        catalog = OpenCodeAgent.available_models()
        catalog["hacked"] = "None"
        assert "hacked" not in KNOWN_ZEN_MODELS

    def test_is_known_model(self):
        assert OpenCodeAgent.is_known_model("hy3-free") is True
        assert OpenCodeAgent.is_known_model("definitely-not-a-real-model") is False
        assert OpenCodeAgent.is_known_model("") is False

    def test_unknown_model_warns(self, caplog):
        agent = OpenCodeAgent()
        with caplog.at_level(logging.WARNING, logger="vertex_edge_agent.agents"):
            assert agent.resolve_model("mystery-model") == "mystery-model"
        assert any("not in the known OpenCode Zen catalog" in r.message for r in caplog.records)

    def test_known_model_does_not_warn(self, caplog):
        agent = OpenCodeAgent()
        with caplog.at_level(logging.WARNING, logger="vertex_edge_agent.agents"):
            assert agent.resolve_model("hy3-free") == "hy3-free"
        assert not [r for r in caplog.records if "known OpenCode Zen catalog" in r.message]

    def test_default_fallback_is_not_warned(self, caplog):
        agent = OpenCodeAgent()
        with caplog.at_level(logging.WARNING, logger="vertex_edge_agent.agents"):
            assert agent.resolve_model("default") == DEFAULT_ZEN_MODEL
            assert agent.resolve_model("") == DEFAULT_ZEN_MODEL
        assert not [r for r in caplog.records if "known OpenCode Zen catalog" in r.message]

    def test_custom_default_model_outside_catalog_warns(self, caplog):
        agent = OpenCodeAgent(default_model="legacy-model")
        with caplog.at_level(logging.WARNING, logger="vertex_edge_agent.agents"):
            assert agent.resolve_model("default") == "legacy-model"
        assert any("legacy-model" in r.message for r in caplog.records)

    def test_custom_default_model_still_applies(self):
        agent = OpenCodeAgent(default_model="gemini-3.5-flash")
        assert agent.resolve_model("default") == "gemini-3.5-flash"


class TestOpenCodeAgentProcess:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        agent = OpenCodeAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(200, "from zen"))
        assert await agent.process("in", "sys", "deepseek-v4-flash") == "from zen"
        await agent.close()

    @pytest.mark.asyncio
    async def test_payload_structure(self):
        agent = OpenCodeAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("user msg", "be brief", "gpt-5.5", settings={"llm_kwargs": {"temperature": 0.7}})
        call = agent.client.post.call_args
        assert call[0][0].endswith("/chat/completions")
        payload = call[1]["json"]
        assert payload["model"] == "gpt-5.5"
        assert payload["messages"][0] == {"role": "system", "content": "be brief"}
        assert payload["messages"][1] == {"role": "user", "content": "user msg"}
        assert payload["temperature"] == 0.7
        await agent.close()

    @pytest.mark.asyncio
    async def test_public_auth_header(self):
        agent = OpenCodeAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "m")
        assert agent.client.post.call_args[1]["headers"]["Authorization"] == "Bearer public"
        await agent.close()

    @pytest.mark.asyncio
    async def test_settings_llm_kwargs_merged(self):
        agent = OpenCodeAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "m", settings={"llm_kwargs": {"top_p": 0.9, "max_tokens": 100}})
        payload = agent.client.post.call_args[1]["json"]
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 100
        await agent.close()


# ====================================================================
# ProxiedLLMAgent — configuration resolution
# ====================================================================
class TestProxiedLLMAgentConstruction:
    def test_default_values(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent()
        assert agent.base_url == DEFAULT_PROXY_BASE_URL == "http://localhost:8000/v1"
        assert agent.api_key == "public"
        assert agent.default_model == "gpt-4o-mini"
        assert agent.max_retries == 3
        assert agent.timeout == 300.0
        assert agent.trust_env is True
        assert agent.model_map == {}
        assert agent.NAME == "ProxiedLLMAgent"

    def test_explicit_proxy_url(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(proxy_url="http://litellm.internal:4000/v1/")
        assert agent.base_url == "http://litellm.internal:4000/v1"

    def test_base_url_kwarg_is_accepted_as_alias(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert ProxiedLLMAgent(base_url="http://a:1/v1").base_url == "http://a:1/v1"

    def test_proxy_url_wins_over_base_url(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(proxy_url="http://explicit:1/v1", base_url="http://ignored:2/v1")
        assert agent.base_url == "http://explicit:1/v1"

    @pytest.mark.parametrize("env_var", ["LLM_PROXY_BASE_URL", "OPENAI_BASE_URL"])
    def test_env_base_url_fallback(self, monkeypatch, env_var):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(env_var, "http://gateway.example:8080/v1")
        assert ProxiedLLMAgent().base_url == "http://gateway.example:8080/v1"

    def test_proxy_env_beats_openai_env(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROXY_BASE_URL", "http://proxy-wins:1/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-loser:2/v1")
        assert ProxiedLLMAgent().base_url == "http://proxy-wins:1/v1"

    def test_explicit_url_beats_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROXY_BASE_URL", "http://from-env:1/v1")
        assert ProxiedLLMAgent(proxy_url="http://from-code:2/v1").base_url == "http://from-code:2/v1"

    @pytest.mark.parametrize("env_var", ["LLM_PROXY_API_KEY", "OPENAI_API_KEY"])
    def test_env_api_key_fallback(self, monkeypatch, env_var):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(env_var, "sk-proxy-secret")
        assert ProxiedLLMAgent().api_key == "sk-proxy-secret"

    def test_proxy_key_env_beats_openai_key_env(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROXY_API_KEY", "sk-proxy")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        assert ProxiedLLMAgent().api_key == "sk-proxy"

    def test_empty_env_values_are_ignored(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROXY_BASE_URL", "")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://real:1/v1")
        monkeypatch.setenv("LLM_PROXY_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        agent = ProxiedLLMAgent()
        assert agent.base_url == "http://real:1/v1"
        assert agent.api_key == "sk-real"

    def test_defaults_log_when_unconfigured(self, monkeypatch, caplog):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level(logging.INFO, logger="vertex_edge_agent.agents"):
            ProxiedLLMAgent()
        assert any("no proxy configured" in r.message for r in caplog.records)

    def test_no_warning_when_explicitly_configured(self, monkeypatch, caplog):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level(logging.INFO, logger="vertex_edge_agent.agents"):
            ProxiedLLMAgent(proxy_url="http://explicit:1/v1")
        assert not [r for r in caplog.records if "no proxy configured" in r.message]

    def test_trust_env_forwarded_to_client(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert ProxiedLLMAgent(proxy_url="http://a:1/v1", trust_env=False).client.trust_env is False

    def test_default_model_custom(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert ProxiedLLMAgent(default_model="claude-3.5").default_model == "claude-3.5"

    def test_extra_headers_merged(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = ProxiedLLMAgent(proxy_url="http://a:1/v1", extra_headers={"X-Workspace": "team-a"})
        assert agent.headers["X-Workspace"] == "team-a"
        assert agent.headers["Authorization"] == "Bearer public"


class TestProxiedLLMAgentModelMap:
    """Alias -> upstream model routing through the gateway."""

    def _agent(self, monkeypatch, model_map, default_model=None):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        return ProxiedLLMAgent(
            proxy_url="http://proxy:4000/v1", model_map=model_map, default_model=default_model
        )

    def test_alias_is_rewritten(self, monkeypatch):
        agent = self._agent(monkeypatch, {"alias": "deepseek-v4-flash"})
        assert agent.resolve_model("alias") == "deepseek-v4-flash"

    def test_unmapped_model_passes_through(self, monkeypatch):
        agent = self._agent(monkeypatch, {"alias": "deepseek-v4-flash"})
        assert agent.resolve_model("gpt-5.5") == "gpt-5.5"

    def test_default_alias_resolves_before_map(self, monkeypatch):
        """``"default"`` falls back first, so the default model may be aliased."""
        agent = self._agent(monkeypatch, {"alias": "deepseek-v4-flash"}, default_model="alias")
        assert agent.resolve_model("default") == "deepseek-v4-flash"
        assert agent.resolve_model("") == "deepseek-v4-flash"

    def test_default_alias_resolves_when_no_map(self, monkeypatch):
        agent = self._agent(monkeypatch, {}, default_model="gpt-5.5")
        assert agent.resolve_model("default") == "gpt-5.5"

    def test_empty_map_is_inert(self, monkeypatch):
        agent = self._agent(monkeypatch, {})
        assert agent.resolve_model("any-model") == "any-model"

    def test_none_map_treated_as_empty(self, monkeypatch):
        assert self._agent(monkeypatch, None).model_map == {}

    def test_model_map_is_copied(self, monkeypatch):
        source = {"alias": "deepseek-v4-flash"}
        agent = self._agent(monkeypatch, source)
        source["alias"] = "mutated"
        assert agent.model_map["alias"] == "deepseek-v4-flash"

    def test_mapping_logs_debug(self, monkeypatch, caplog):
        agent = self._agent(monkeypatch, {"alias": "deepseek-v4-flash"})
        with caplog.at_level(logging.DEBUG, logger="vertex_edge_agent.agents"):
            agent.resolve_model("alias")
        assert any("via proxy" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_map_applied_in_request_payload(self, monkeypatch):
        agent = self._agent(monkeypatch, {"alias": "claude-opus-4.7"}, default_model="alias")
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "default")
        assert agent.client.post.call_args[1]["json"]["model"] == "claude-opus-4.7"
        await agent.close()

    @pytest.mark.asyncio
    async def test_request_goes_to_proxy_url(self, monkeypatch):
        agent = self._agent(monkeypatch, {})
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp())
        await agent.process("d", "p", "gpt-5.5")
        assert agent.client.post.call_args[0][0] == "http://proxy:4000/v1/chat/completions"
        assert agent.client.post.call_args[1]["json"]["model"] == "gpt-5.5"
        await agent.close()

    @pytest.mark.asyncio
    async def test_process_via_env_configured_proxy(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROXY_BASE_URL", "http://env-gateway:9090/v1")
        monkeypatch.setenv("LLM_PROXY_API_KEY", "sk-env")
        agent = ProxiedLLMAgent()
        agent.client.post = AsyncMock(return_value=_HTTPHelpers.make_resp(200, "proxied reply"))
        assert await agent.process("d", "p", "m") == "proxied reply"
        call = agent.client.post.call_args
        assert call[0][0] == "http://env-gateway:9090/v1/chat/completions"
        assert call[1]["headers"]["Authorization"] == "Bearer sk-env"
        await agent.close()


# ====================================================================
# get_agent factory wiring
# ====================================================================
class TestAgentFactory:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("opencode", OpenCodeAgent),
            ("proxy", ProxiedLLMAgent),
            ("proxied", ProxiedLLMAgent),
            ("http", HttpLLMAgent),
            ("mock", MockAgent),
        ],
    )
    def test_string_shorthands(self, monkeypatch, spec, expected):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent(spec)
        assert isinstance(agent, expected)

    def test_none_returns_none(self):
        assert get_agent(None) is None

    def test_instance_passes_through(self):
        agent = OpenCodeAgent()
        assert get_agent(agent) is agent

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent("does-not-exist")

    def test_unknown_dict_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported agent config type"):
            get_agent({"type": "quantum"})

    @pytest.mark.asyncio
    async def test_http_dict_config_regression(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({"type": "http", "api_key": "sk-http", "base_url": "http://h:1/v1/", "max_retries": 7})
        assert isinstance(agent, HttpLLMAgent)
        assert agent.api_key == "sk-http"
        assert agent.base_url == "http://h:1/v1"
        assert agent.max_retries == 7
        await agent.close()

    @pytest.mark.asyncio
    async def test_opencode_dict_config(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({
            "type": "opencode",
            "api_key": "sk-zen",
            "model": "deepseek-v4-flash",
            "max_retries": 5,
            "timeout": 45.0,
            "trust_env": False,
        })
        assert isinstance(agent, OpenCodeAgent)
        assert agent.api_key == "sk-zen"
        assert agent.default_model == "deepseek-v4-flash"
        assert agent.max_retries == 5
        assert agent.timeout == 45.0
        assert agent.client.trust_env is False
        await agent.close()

    @pytest.mark.asyncio
    async def test_opencode_dict_config_defaults(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({"type": "opencode"})
        assert agent.base_url == DEFAULT_ZEN_BASE_URL
        assert agent.default_model == DEFAULT_ZEN_MODEL
        await agent.close()

    @pytest.mark.parametrize("key", ["proxy_url", "base_url"])
    async def test_proxy_dict_config(self, monkeypatch, key):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({
            "type": "proxy",
            key: "http://gw:4000/v1/",
            "api_key": "sk-gw",
            "model": "alias",
            "model_map": {"alias": "deepseek-v4-flash"},
            "max_retries": 2,
        })
        assert isinstance(agent, ProxiedLLMAgent)
        assert agent.base_url == "http://gw:4000/v1"
        assert agent.api_key == "sk-gw"
        assert agent.default_model == "alias"
        assert agent.model_map == {"alias": "deepseek-v4-flash"}
        assert agent.resolve_model("default") == "deepseek-v4-flash"
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxied_dict_alias(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert isinstance(get_agent({"type": "proxied", "proxy_url": "http://a:1/v1"}), ProxiedLLMAgent)

    async def test_proxy_dict_without_api_key_uses_env(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        agent = get_agent({"type": "proxy", "proxy_url": "http://a:1/v1"})
        assert agent.api_key == "sk-from-env"

    @pytest.mark.asyncio
    async def test_opencode_dict_throttle_knobs(self, monkeypatch):
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({
            "type": "opencode",
            "max_concurrency": 9,
            "requests_per_minute": 45.0,
            "queue_timeout": 12.0,
        })
        assert agent.max_concurrency == 9
        assert agent.requests_per_minute == 45.0
        assert agent.queue_timeout == 12.0
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxy_dict_budget_off_by_default(self, monkeypatch):
        """The gateway governs its own rate, so the agent only bounds concurrency."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({"type": "proxy", "proxy_url": "http://a:1/v1", "max_concurrency": 4})
        assert agent.max_concurrency == 4
        assert agent.requests_per_minute is None
        assert agent._rate_budget is None
        await agent.close()

    @pytest.mark.asyncio
    async def test_http_dict_stays_unbounded(self, monkeypatch):
        """``http`` is the escape hatch for callers who drive their own concurrency."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({"type": "http", "max_concurrency": 9, "requests_per_minute": 100})
        assert getattr(agent, "max_concurrency", None) is None
        assert getattr(agent, "_rate_budget", None) is None
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxy_key_wired_for_http(self, monkeypatch):
        """``{"type": "http", "proxy": ...}`` makes the request tunnel through a proxy."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        agent = get_agent({"type": "http", "proxy": "http://corp:3128"})
        assert agent.proxy == "http://corp:3128"
        assert len(getattr(agent.client, "_mounts", {})) == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_proxy_key_wired_for_opencode_and_proxied(self, monkeypatch):
        """``proxy`` (transport) is accepted by every HTTP agent type."""
        for var in PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        zen = get_agent({"type": "opencode", "proxy": "http://corp:3128"})
        assert zen.proxy == "http://corp:3128"
        assert len(getattr(zen.client, "_mounts", {})) == 1
        await zen.close()

        gateway = get_agent({"type": "proxy", "proxy_url": "http://gw:4000/v1", "proxy": "http://corp:3128"})
        assert gateway.proxy == "http://corp:3128"
        assert gateway.base_url == "http://gw:4000/v1"
        assert len(getattr(gateway.client, "_mounts", {})) == 1
        await gateway.close()
