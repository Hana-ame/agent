"""Tests for framework.agents — HttpLLMAgent, HttpLLMAgent, PiAgentRunner.

Real unit tests with mocked external dependencies (httpx, subprocess).
"""

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.agents import BaseAgent, HttpLLMAgent, HttpLLMAgent, PiAgentRunner


# ====================================================================
# BaseAgent — cannot be instantiated
# ====================================================================
class TestBaseAgent:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_mock_agent_is_subclass(self):
        assert issubclass(HttpLLMAgent, BaseAgent)

    def test_http_agent_is_subclass(self):
        assert issubclass(HttpLLMAgent, BaseAgent)

    def test_pi_runner_is_subclass(self):
        assert issubclass(PiAgentRunner, BaseAgent)


# ====================================================================
# HttpLLMAgent
# ====================================================================
class TestHttpLLMAgentDefault:
    """Default HttpLLMAgent echoes data with model metadata."""

    @pytest.mark.asyncio
    async def test_string_echo(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process("hello", "prompt", "test-model")
        assert result == "[test-model] hello"

    @pytest.mark.asyncio
    async def test_dict_echo(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process({"key": "val"}, "prompt", "m")
        assert result["_processed"] is True
        assert result["_model"] == "m"
        assert result["input"] == {"key": "val"}
        assert "key" in result["output"]

    @pytest.mark.asyncio
    async def test_int_echo(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process(42, "p", "m")
        assert result == "[m] 42"

    @pytest.mark.asyncio
    async def test_none_echo(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process(None, "p", "m")
        assert "None" in result

    @pytest.mark.asyncio
    async def test_list_echo(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process([1, 2, 3], "p", "m")
        assert "[m]" in result

    @pytest.mark.asyncio
    async def test_settings_ignored_by_default(self):
        agent = HttpLLMAgent(mock=True)
        result = await agent.process("d", "p", "m", settings={"extra": True})
        assert result == "[m] d"


class TestHttpLLMAgentCustomFn:
    """HttpLLMAgent with custom response_fn."""

    @pytest.mark.asyncio
    async def test_sync_response_fn(self):
        fn = lambda d, p, m, s: f"echo:{d}:{p}"
        agent = HttpLLMAgent(mock=True, mock_handler=fn)
        result = await agent.process("data", "ask", "model")
        assert result == "echo:data:ask"

    @pytest.mark.asyncio
    async def test_async_response_fn(self):
        async def afn(d, p, m, s):
            await asyncio.sleep(0)
            return f"async:{d}"
        agent = HttpLLMAgent(mock=True, mock_handler=afn)
        result = await agent.process("x", "", "")
        assert result == "async:x"

    @pytest.mark.asyncio
    async def test_fn_receives_all_params(self):
        captured = {}
        def capture(d, p, m, s):
            captured["d"] = d
            captured["p"] = p
            captured["m"] = m
            captured["s"] = s
            return "ok"
        agent = HttpLLMAgent(mock=True, mock_handler=capture)
        await agent.process("data", "prompt", "model", {"k": "v"})
        assert captured == {"d": "data", "p": "prompt", "m": "model", "s": {"k": "v"}}

    @pytest.mark.asyncio
    async def test_fn_returning_dict(self):
        agent = HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: {"result": d})
        result = await agent.process("inp", "", "", None)
        assert result == {"result": "inp"}

    @pytest.mark.asyncio
    async def test_fn_returning_none(self):
        agent = HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: None)
        result = await agent.process("d", "", "", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_fn_returning_list(self):
        agent = HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: [d, d])
        result = await agent.process(1, "", "", None)
        assert result == [1, 1]

    @pytest.mark.asyncio
    async def test_fn_exception_propagates(self):
        def boom(d, p, m, s):
            raise ValueError("boom")
        agent = HttpLLMAgent(mock=True, mock_handler=boom)
        with pytest.raises(ValueError, match="boom"):
            await agent.process("d", "", "", None)


# ====================================================================
# HttpLLMAgent
# ====================================================================
class TestHttpLLMAgentConstruction:
    def test_default_values(self):
        agent = HttpLLMAgent()
        assert agent.base_url == "https://opencode.ai/zen/v1/chat/completions"
        assert agent.api_key == "public"
        assert agent.max_retries == 3
        assert agent.client is not None

    def test_custom_values(self):
        agent = HttpLLMAgent(api_key="sk-abc", base_url="http://localhost:8080/v1/", max_retries=5)
        assert agent.api_key == "sk-abc"
        assert agent.base_url == "http://localhost:8080/v1"  # trailing slash stripped
        assert agent.max_retries == 5

    @pytest.mark.asyncio
    async def test_close(self):
        agent = HttpLLMAgent()
        # close should not raise
        await agent.close()


class TestHttpLLMAgentProcess:
    """Test HttpLLMAgent.process with mocked httpx."""

    def _make_response(self, status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {
            "choices": [{"message": {"content": "LLM response"}}]
        }
        resp.text = json.dumps(json_data or {"error": "bad"})
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import httpx
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"{status_code}",
                request=MagicMock(),
                response=resp,
            )
        return resp

    @pytest.mark.asyncio
    async def test_successful_call(self):
        agent = HttpLLMAgent(api_key="test-key")
        mock_resp = self._make_response(200, {"choices": [{"message": {"content": "hello from llm"}}]})
        agent.client.post = AsyncMock(return_value=mock_resp)

        result = await agent.process("input data", "system prompt", "gpt-4o")
        assert result == "hello from llm"
        await agent.close()

    @pytest.mark.asyncio
    async def test_request_payload_structure(self):
        agent = HttpLLMAgent(api_key="k", base_url="http://x.com/v1")
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        await agent.process("user msg", "sys prompt", "model-1", settings={"llm_kwargs": {"temperature": 0.7}})
        call_args = agent.client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["model"] == "model-1"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "sys prompt"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "user msg"
        assert payload["temperature"] == 0.7
        await agent.close()

    @pytest.mark.asyncio
    async def test_default_model_fallback(self):
        agent = HttpLLMAgent()
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        await agent.process("d", "p", "default")
        payload = agent.client.post.call_args[1]["json"]
        assert payload["model"] == "hy3-free"  # fallback
        await agent.close()

    @pytest.mark.asyncio
    async def test_empty_model_fallback(self):
        agent = HttpLLMAgent()
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        await agent.process("d", "p", "")
        payload = agent.client.post.call_args[1]["json"]
        assert payload["model"] == "hy3-free"
        await agent.close()

    @pytest.mark.asyncio
    async def test_auth_header(self):
        agent = HttpLLMAgent(api_key="Bearer-token-123")
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        await agent.process("d", "p", "m")
        call_kwargs = agent.client.post.call_args
        headers = call_kwargs[1]["headers"] if "headers" in call_kwargs[1] else call_kwargs[1].get("headers", {})
        assert headers["Authorization"] == "Bearer Bearer-token-123"
        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        """500 errors should be retried up to max_retries then raise."""
        import httpx
        agent = HttpLLMAgent(max_retries=2)
        error_resp = self._make_response(500)
        agent.client.post = AsyncMock(return_value=error_resp)

        with pytest.raises(httpx.HTTPStatusError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2  # retried once
        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        """429 (rate limit) should be retried."""
        import httpx
        agent = HttpLLMAgent(max_retries=3)
        error_resp = self._make_response(429)
        agent.client.post = AsyncMock(return_value=error_resp)

        with pytest.raises(httpx.HTTPStatusError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 3
        await agent.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """400 errors should fail immediately without retries."""
        from framework.agents.http_llm_agent import NonRetryableHTTPError
        agent = HttpLLMAgent(max_retries=3)
        error_resp = self._make_response(400)
        agent.client.post = AsyncMock(return_value=error_resp)

        with pytest.raises(NonRetryableHTTPError):
            await agent.process("d", "p", "m")
        # Fatal: no retries — exactly 1 call
        assert agent.client.post.call_count == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self):
        """401 errors should fail immediately without retries."""
        from framework.agents.http_llm_agent import NonRetryableHTTPError
        agent = HttpLLMAgent(max_retries=3)
        error_resp = self._make_response(401)
        agent.client.post = AsyncMock(return_value=error_resp)

        with pytest.raises(NonRetryableHTTPError):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """First call fails 500, second succeeds."""
        import httpx
        agent = HttpLLMAgent(max_retries=3)
        fail_resp = self._make_response(500)
        ok_resp = self._make_response(200, {"choices": [{"message": {"content": "recovered"}}]})
        agent.client.post = AsyncMock(side_effect=[fail_resp, ok_resp])

        result = await agent.process("d", "p", "m")
        assert result == "recovered"
        assert agent.client.post.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_connection_error_retries(self):
        """httpx.RequestError (connection failure) should retry per tenacity config."""
        import httpx
        agent = HttpLLMAgent(max_retries=2)
        # Must use httpx.RequestError, not plain Exception — tenacity only
        # retries on (httpx.RequestError, httpx.HTTPStatusError)
        agent.client.post = AsyncMock(side_effect=httpx.RequestError("connection refused"))

        with pytest.raises(httpx.RequestError, match="connection refused"):
            await agent.process("d", "p", "m")
        assert agent.client.post.call_count == 2
        await agent.close()

    @pytest.mark.asyncio
    async def test_settings_llm_kwargs_merged(self):
        agent = HttpLLMAgent()
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        await agent.process("d", "p", "m", settings={"llm_kwargs": {"top_p": 0.9, "max_tokens": 100}})
        payload = agent.client.post.call_args[1]["json"]
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 100
        await agent.close()

    @pytest.mark.asyncio
    async def test_no_settings_no_error(self):
        agent = HttpLLMAgent()
        mock_resp = self._make_response(200)
        agent.client.post = AsyncMock(return_value=mock_resp)

        result = await agent.process("d", "p", "m", settings=None)
        assert result is not None
        await agent.close()


# ====================================================================
# PiAgentRunner
# ====================================================================
class TestPiAgentRunnerProcess:
    """Test PiAgentRunner with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_basic_string_invocation(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output text", b""))
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await runner.process("hello", "translate this", "gpt-4")
            assert result == "output text"
            args = mock_exec.call_args[0]
            # cmd should be: pi -p --model gpt-4 --system-prompt translate this -- hello
            assert args[0] == "pi"
            assert "-p" in args
            assert "--model" in args
            assert "gpt-4" in args
            assert "--system-prompt" in args
            assert "translate this" in args
            assert "--" in args
            assert "hello" in args

    @pytest.mark.asyncio
    async def test_dict_input_serialized(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b'{"ok": true}', b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await runner.process({"key": "val"}, "p", "m")
            args = mock_exec.call_args[0]
            msg = args[-1]
            assert json.loads(msg) == {"key": "val"}

    @pytest.mark.asyncio
    async def test_list_input_serialized(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await runner.process([1, 2], "p", "m")
            args = mock_exec.call_args[0]
            assert json.loads(args[-1]) == [1, 2]

    @pytest.mark.asyncio
    async def test_no_model_flag_when_default(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await runner.process("d", "p", "default")
            args = mock_exec.call_args[0]
            assert "--model" not in args

    @pytest.mark.asyncio
    async def test_no_prompt_flag_when_empty(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await runner.process("d", "", "m")
            args = mock_exec.call_args[0]
            assert "--system-prompt" not in args

    @pytest.mark.asyncio
    async def test_settings_mapped_to_cli_args(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        settings = {
            "mode": "json",
            "tools": "bash,read",
            "thinking": "high",
            "api_key": "sk-123",
            "provider": "anthropic",
        }
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await runner.process("d", "p", "m", settings=settings)
            args = mock_exec.call_args[0]
            assert "--mode" in args
            assert "json" in args
            assert "--tools" in args
            assert "bash,read" in args
            assert "--thinking" in args
            assert "high" in args
            assert "--api-key" in args
            assert "sk-123" in args
            assert "--provider" in args
            assert "anthropic" in args

    @pytest.mark.asyncio
    async def test_json_mode_parses_output(self):
        runner = PiAgentRunner()
        json_output = json.dumps({"answer": 42})
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json_output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.process("d", "p", "m", settings={"mode": "json"})
            assert result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_json_mode_invalid_falls_back_to_string(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not json at all", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.process("d", "p", "m", settings={"mode": "json"})
            assert result == "not json at all"

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"command not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Pi Agent CLI error"):
                await runner.process("d", "p", "m")

    @pytest.mark.asyncio
    async def test_command_not_found_raises(self):
        runner = PiAgentRunner()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                await runner.process("d", "p", "m")

    @pytest.mark.asyncio
    async def test_stderr_included_in_error(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 2
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some stderr error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="some stderr error"):
                await runner.process("d", "p", "m")

    @pytest.mark.asyncio
    async def test_message_is_last_arg_after_dash_dash(self):
        runner = PiAgentRunner()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await runner.process("my message", "p", "m")
            args = mock_exec.call_args[0]
            dd_idx = args.index("--")
            assert args[dd_idx + 1] == "my message"
            # nothing after message
            assert len(args) == dd_idx + 2


# ====================================================================
# PiAgentRunner — supplementary edge-case & branch-coverage tests
# ====================================================================
class TestPiAgentRunnerExtra:
    """Edge cases for PiAgentRunner: falsy branches, output handling,
    command structure, error boundaries, and settings coercion."""

    def _make_proc(self, stdout=b"ok", stderr=b"", returncode=0):
        proc = AsyncMock()
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        return proc

    # --- falsy model / data branches -------------------------------------
    @pytest.mark.asyncio
    async def test_empty_model_string_no_flag(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "")
            args = mock_exec.call_args[0]
            assert "--model" not in args

    @pytest.mark.asyncio
    async def test_none_data_converted_to_string(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"done")
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process(None, "p", "m")
            assert mock_exec.call_args[0][-1] == "None"

    @pytest.mark.asyncio
    async def test_int_data_converted_to_string(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process(42, "p", "m")
            assert mock_exec.call_args[0][-1] == "42"

    # --- output handling -------------------------------------------------
    @pytest.mark.asyncio
    async def test_stdout_trailing_whitespace_stripped(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"  hello\n\n  ")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m")
            assert result == "hello"

    @pytest.mark.asyncio
    async def test_json_mode_returns_list(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"[1, 2, 3]")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m", settings={"mode": "json"})
            assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_json_mode_returns_primitive_number(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"42")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m", settings={"mode": "json"})
            assert result == 42

    @pytest.mark.asyncio
    async def test_json_mode_empty_output_falls_back(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"   ")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m", settings={"mode": "json"})
            assert result == ""  # stripped empty -> JSONDecodeError -> raw ""

    # --- command structure ----------------------------------------------
    @pytest.mark.asyncio
    async def test_command_first_two_args(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "m")
            args = mock_exec.call_args[0]
            assert args[0] == "pi"
            assert args[1] == "-p"

    @pytest.mark.asyncio
    async def test_model_value_immediately_follows_flag(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "my-model")
            args = mock_exec.call_args[0]
            idx = args.index("--model")
            assert args[idx + 1] == "my-model"

    @pytest.mark.asyncio
    async def test_system_prompt_value_immediately_follows_flag(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "my prompt", "m")
            args = mock_exec.call_args[0]
            idx = args.index("--system-prompt")
            assert args[idx + 1] == "my prompt"

    @pytest.mark.asyncio
    async def test_subprocess_called_with_pipe(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "m")
            kwargs = mock_exec.call_args[1]
            assert kwargs["stdout"] == asyncio.subprocess.PIPE
            assert kwargs["stderr"] == asyncio.subprocess.PIPE

    # --- error boundaries -----------------------------------------------
    @pytest.mark.asyncio
    async def test_other_exception_propagates(self):
        runner = PiAgentRunner()
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                await runner.process("d", "p", "m")

    @pytest.mark.asyncio
    async def test_returncode_zero_with_stderr_still_succeeds(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"ok", stderr=b"warning text", returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m")
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_negative_returncode_raises(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout=b"", stderr=b"signal killed", returncode=-1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with pytest.raises(RuntimeError, match="signal killed"):
                await runner.process("d", "p", "m")

    # --- settings coercion ----------------------------------------------
    @pytest.mark.asyncio
    async def test_settings_none_defaults_to_empty(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m", settings=None)
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_partial_settings_only_mode(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "m", settings={"mode": "json"})
            args = mock_exec.call_args[0]
            assert "--mode" in args
            assert "--tools" not in args
            assert "--thinking" not in args
            assert "--api-key" not in args
            assert "--provider" not in args

    @pytest.mark.asyncio
    async def test_partial_settings_only_tools(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "m", settings={"tools": "bash"})
            args = mock_exec.call_args[0]
            assert "--tools" in args
            assert "bash" in args
            assert "--mode" not in args

    @pytest.mark.asyncio
    async def test_settings_value_int_converted_to_str(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "m", settings={"thinking": 5})
            args = mock_exec.call_args[0]
            idx = args.index("--thinking")
            assert args[idx + 1] == "5"  # str(5)

    @pytest.mark.asyncio
    async def test_model_default_with_settings_still_no_model_flag(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process("d", "p", "default", settings={"mode": "json"})
            args = mock_exec.call_args[0]
            assert "--model" not in args
            assert "--mode" in args

    # --- encoding --------------------------------------------------------
    @pytest.mark.asyncio
    async def test_dict_with_non_ascii_serialized(self):
        runner = PiAgentRunner()
        proc = self._make_proc()
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            await runner.process({"key": "caf\u00e9_\u00fcmlaut"}, "p", "m")
            msg = mock_exec.call_args[0][-1]
            assert "caf\u00e9_\u00fcmlaut" in msg  # ensure_ascii=False keeps non-ascii chars

    @pytest.mark.asyncio
    async def test_multibyte_output_decoded(self):
        runner = PiAgentRunner()
        proc = self._make_proc(stdout="caf\u00e9_resum\u00e9".encode("utf-8"))
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await runner.process("d", "p", "m")
            assert result == "caf\u00e9_resum\u00e9"


# ====================================================================
# HttpLLMAgent — lifecycle (defect 4 regression): async context manager,
# explicit close(), idempotence
# ====================================================================
class TestHttpLLMAgentLifecycle:
    """Lifecycle cleanup: __aenter__/__aexit__ + idempotent close()."""

    @pytest.mark.asyncio
    async def test_explicit_close_sets_closed_and_releases_client(self):
        agent = HttpLLMAgent()
        with patch.object(agent.client, "aclose", new=AsyncMock()) as mock_aclose:
            await agent.close()
            mock_aclose.assert_awaited_once()
        assert agent._closed is True

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        agent = HttpLLMAgent()
        with patch.object(agent.client, "aclose", new=AsyncMock()) as mock_aclose:
            await agent.close()
            await agent.close()  # second call must be a safe no-op
            mock_aclose.assert_awaited_once()  # exactly one teardown
        assert agent._closed is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        # ``close`` is mocked here so the real flag-setter does not run; the
        # assertion is that ``__aexit__`` awaits ``close`` exactly once.
        with patch.object(HttpLLMAgent, "close", new=AsyncMock()) as mock_close:
            async with HttpLLMAgent() as agent:
                mock_close.assert_not_awaited()
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_even_on_exception(self):
        with patch.object(HttpLLMAgent, "close", new=AsyncMock()) as mock_close:
            with pytest.raises(RuntimeError):
                async with HttpLLMAgent() as agent:
                    raise RuntimeError("boom inside context")
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_sets_closed_with_real_close(self):
        """With the real (unmocked) close(), exiting the context sets _closed."""
        async with HttpLLMAgent() as agent:
            assert not agent._closed
        assert agent._closed is True


# ====================================================================
# PiAgentRunner — subprocess timeout / cancellation cleanup
# ====================================================================
class TestPiAgentRunnerCleanup:
    """Verify settings['timeout'] kills the child and CancelledError
    repropagates cleanly (no orphaned subprocess)."""

    @staticmethod
    async def _hang():
        await asyncio.Event().wait()

    @pytest.mark.asyncio
    async def test_timeout_kills_process_and_raises(self):
        """settings['timeout'] bounds the CLI call; on timeout the child is
        killed (kill + wait) and a clear RuntimeError is raised."""
        runner = PiAgentRunner()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=self._hang)
        mock_proc.kill = MagicMock()  # real Popen.kill() is synchronous
        mock_proc.wait = AsyncMock()  # real Popen.wait() is awaited

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="timed out"):
                await runner.process("d", "p", "m", settings={"timeout": 0.05})

        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_error_kills_process_and_repropagates(self):
        """When the enclosing task is cancelled, kill + wait the child and
        re-raise CancelledError (no orphaned subprocess)."""
        runner = PiAgentRunner()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(asyncio.CancelledError):
                await runner.process("d", "p", "m")

        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()
