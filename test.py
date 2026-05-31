"""opencode 模块测试 — 使用 mock，无外部依赖"""
from unittest.mock import patch, Mock
import opencode


def test_json_output():
    print("\n测试: JSON 输出解析")
    mock = Mock(stdout='{"key": "value"}', stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        result = opencode.run("hi", agent="Null", timeout=30)
    print(f"  结果: {result}")
    assert result == {"output": {"key": "value"}, "json": True, "success": True}
    args, kwargs = p.call_args
    assert kwargs["timeout"] == 30
    assert args[0] == ["opencode", "--agent", "Null", "run", "hi"]


def test_non_json_output():
    print("\n测试: 非 JSON 输出")
    mock = Mock(stdout="你好世界", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock):
        result = opencode.run("hi", timeout=10)
    print(f"  结果: {result}")
    assert result == {"output": "你好世界", "json": False, "success": True}


def test_failure():
    print("\n测试: 命令失败")
    mock = Mock(stdout="error", stderr="timeout", returncode=1)
    with patch("subprocess.run", return_value=mock):
        result = opencode.run("hi")
    print(f"  结果: success={result['success']}")
    assert result["success"] is False


def test_command_no_agent_no_model():
    print("\n测试: 无 agent 无 model 的命令构造")
    mock = Mock(stdout="ok", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        opencode.run("hello")
    args = p.call_args[0][0]
    print(f"  命令: {args}")
    assert args == ["opencode", "run", "hello"]


def test_command_with_agent():
    print("\n测试: 带 agent 的命令构造")
    mock = Mock(stdout="ok", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        opencode.run("hello", agent="Null")
    args = p.call_args[0][0]
    print(f"  命令: {args}")
    assert args == ["opencode", "--agent", "Null", "run", "hello"]


def test_command_with_model():
    print("\n测试: 带 model 的命令构造")
    mock = Mock(stdout="ok", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        opencode.run("hello", model="gemma-4-31b-it")
    args = p.call_args[0][0]
    print(f"  命令: {args}")
    assert args == ["opencode", "--model", "gemma-4-31b-it", "run", "hello"]


def test_command_with_both():
    print("\n测试: 带 agent 和 model 的命令构造")
    mock = Mock(stdout="ok", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        opencode.run("hello", agent="Null", model="qwen3-8b")
    args = p.call_args[0][0]
    print(f"  命令: {args}")
    assert args == ["opencode", "--agent", "Null", "--model", "qwen3-8b", "run", "hello"]


def test_timeout_passthrough():
    print("\n测试: timeout 参数透传")
    mock = Mock(stdout="ok", stderr="", returncode=0)
    with patch("subprocess.run", return_value=mock) as p:
        opencode.run("hi", timeout=999)
    print(f"  timeout={p.call_args[1]['timeout']}")
    assert p.call_args[1]["timeout"] == 999
