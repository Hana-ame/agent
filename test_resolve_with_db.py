"""PromptResolver 类测试 — mock opencode.run，无外部依赖"""
import json
from unittest.mock import patch
import pytest

from resolve_with_db import PromptResolver
from prompt_db import PromptDB


@pytest.fixture
def db(tmp_path):
    return PromptDB(tmp_path / "test.db")


@pytest.fixture
def resolver(db):
    return PromptResolver(db, model="test-model", timeout=30)


def _mock_run(output, success=True, text_mode=True):
    """创建一个模拟 opencode_run 返回值"""
    if text_mode:
        return {"output": output, "json": False, "success": success, "error": ""}
    return {"output": output, "json": True, "success": success, "error": ""}


class TestPureStr:
    def test_str_direct_call(self, resolver):
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("4")):
            result = resolver.resolve("2+2=?")
        assert result == "4"

    def test_str_failure_returns_empty(self, resolver):
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("", success=False)):
            result = resolver.resolve("do-something")
        assert result == ""

    def test_str_dict_output(self, resolver):
        out = {"answer": 42}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run(out, text_mode=False)):
            result = resolver.resolve("meaning of life")
        assert result == json.dumps(out, indent=2, ensure_ascii=False)


class TestIntRef:
    def test_int_with_done_response(self, db, resolver):
        pid = db.add("color?", agent="Null")
        db.done(pid, "蓝色")
        result = resolver.resolve(pid)
        assert result == "蓝色"

    def test_int_nonexistent_returns_empty(self, resolver):
        result = resolver.resolve(99999)
        assert result == ""

    def test_int_empty_context(self, db, resolver):
        pid = db.add("", agent="Null")
        result = resolver.resolve(pid)
        assert result == ""

    def test_int_plain_text_context(self, db, resolver):
        pid = db.add("1+1=?", agent="Null")
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("2")):
            result = resolver.resolve(pid)
        assert result == "2"

    def test_int_plain_text_context_failure(self, db, resolver):
        pid = db.add("1+1=?", agent="Null")
        with patch("resolve_with_db.opencode_run",
                   return_value=_mock_run("error", success=False)):
            result = resolver.resolve(pid)
        assert result == "error"
        row = db.get(pid)
        assert row["status"] == "failed"

    def test_int_json_array_context(self, db, resolver):
        pid1 = db.add("color?", agent="Null")
        db.done(pid1, "红色")
        pid2 = db.add([pid1], agent="Null")
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("最终答案")):
            result = resolver.resolve(pid2)
        assert result == "最终答案"

    def test_int_json_non_list_parsed_as_text(self, db, resolver):
        pid = db.add('{"key": "value"}', agent="Null")
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("parsed")):
            result = resolver.resolve(pid)
        assert result == "parsed"

    def test_int_uses_instance_model_fallback(self, db):
        pid = db.add("hello", agent="Null", model="")
        r = PromptResolver(db, model="fallback-model", timeout=30)
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("ok")) as m:
            r.resolve(pid)
        assert m.call_args[1]["model"] == "fallback-model"


class TestDictPrompt:
    def test_dict_str_context(self, resolver):
        prompt = {"agent": "Null", "context": "2+3=?"}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("5")):
            result = resolver.resolve(prompt)
        assert result == "5"

    def test_dict_int_context(self, db, resolver):
        pid = db.add("color?", agent="Null")
        db.done(pid, "绿色")
        prompt = {"agent": "Null", "context": pid}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("绿色")):
            result = resolver.resolve(prompt)
        assert result == "绿色"

    def test_dict_list_context_mixed(self, db, resolver):
        pid = db.add("food?", agent="Null")
        db.done(pid, "面条")
        prompt = {"agent": "Null", "context": ["水果", pid]}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("综合回答")):
            result = resolver.resolve(prompt)
        assert result == "综合回答"

    def test_dict_list_nested_dict(self, resolver):
        prompt = {
            "agent": "Null",
            "context": [
                "开头",
                {"agent": "Null", "context": "嵌套"},
            ],
        }
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("结果")):
            result = resolver.resolve(prompt)
        assert result == "结果"

    def test_dict_non_str_int_context(self, resolver):
        """context 是 dict 等类型 → str() 转换"""
        prompt = {"agent": "Null", "context": {"nested": "obj"}}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("converted")) as m:
            result = resolver.resolve(prompt)
        assert result == "converted"
        # 验证拼接的文本是 str(context)
        assert m.call_args[0][0] == str({"nested": "obj"})

    def test_dict_context_list_nested_int(self, db, resolver):
        pid_inner = db.add("inner", agent="Null")
        db.done(pid_inner, "inner_result")
        prompt = {"agent": "Null", "context": [{"agent": "Null", "context": pid_inner}]}
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("outer_result")):
            result = resolver.resolve(prompt)
        assert result == "outer_result"

    def test_dict_failure_returns_empty(self, resolver):
        prompt = {"agent": "Null", "context": "hi"}
        with patch("resolve_with_db.opencode_run",
                   return_value=_mock_run("", success=False)):
            result = resolver.resolve(prompt)
        assert result == ""


class TestCaching:
    def test_cache_str_avoids_second_call(self, resolver):
        resolver.use_cache = True
        with patch("resolve_with_db.opencode_run",
                   return_value=_mock_run("42")) as m:
            r1 = resolver.resolve("meaning?")
            r2 = resolver.resolve("meaning?")
        assert r1 == r2 == "42"
        assert m.call_count == 1

    def test_cache_int_avoids_second_call(self, db, resolver):
        resolver.use_cache = True
        pid = db.add("q?", agent="Null")
        db.done(pid, "cached_answer")
        r1 = resolver.resolve(pid)
        r2 = resolver.resolve(pid)
        assert r1 == r2 == "cached_answer"

    def test_cache_dict_avoids_second_call(self, resolver):
        resolver.use_cache = True
        prompt = {"agent": "Null", "context": "hello"}
        with patch("resolve_with_db.opencode_run",
                   return_value=_mock_run("world")) as m:
            r1 = resolver.resolve(prompt)
            r2 = resolver.resolve(prompt)
        assert r1 == r2 == "world"
        assert m.call_count == 1

    def test_cache_disabled_makes_multiple_calls(self, resolver):
        resolver.use_cache = False
        with patch("resolve_with_db.opencode_run",
                   return_value=_mock_run("42")) as m:
            resolver.resolve("same?")
            resolver.resolve("same?")
        assert m.call_count == 2


class TestResolveIntEdgeCases:
    def test_int_db_returns_none(self, resolver):
        """_resolve_int 遇到 None row → ''"""
        result = resolver.resolve(0)
        assert result == ""

    def test_int_parsed_json_ctx_nested_recursive(self, db, resolver):
        pid_a = db.add("a", agent="Null")
        db.done(pid_a, "A_res")
        pid_b = db.add([pid_a], agent="Null")
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("B_res")):
            result = resolver.resolve(pid_b)
        assert result == "B_res"
        row = db.get(pid_b)
        assert row["status"] == "done"

    def test_int_json_context_non_list_non_str(self, db):
        """JSON 解析结果是数字 → str() 作为 prompt"""
        pid = db.add("42", agent="Null")
        r = PromptResolver(db, model="m", timeout=30)
        with patch("resolve_with_db.opencode_run", return_value=_mock_run("ok")) as m:
            r.resolve(pid)
        assert m.call_args[0][0] == "42"
