"""
Tests for loop.py — covers config loading, token estimation, history building,
compact logic, and loop1/loop2 with mocked opencode calls.
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database import DataBase
from loop import (
    estimate_tokens,
    load_config,
    build_history,
    format_conversation_context,
    compact_history,
    loop1_abstract,
    loop2_jielong,
    loop666_auto666,
    call_opencode,
    run_loop_instance,
    ABSTRACT_MODELS,
    JIELONG_MODELS,
    MAX_CONTEXT_TOKENS,
    DEFAULT_CONFIG,
)

TEST_DB = "test_loop.db"


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def seed_test_data(db: DataBase):
    id1 = db.prompts.Insert({
        "prompt": "你好，请问你是谁？",
        "agent": "user",
        "model": "",
        "response": "我是一个 AI 助手，很高兴为你服务！",
        "abstract": "用户打招呼，AI 自我介绍",
        "should_end": 0,
    })
    id2 = db.prompts.Insert({
        "prompt": "你能帮我写代码吗？",
        "agent": "user",
        "model": "",
        "response": "当然可以！请告诉我你需要什么类型的代码？",
        "abstract": "用户询问编程能力，AI 表示愿意帮助",
        "should_end": 0,
        "previous_id": id1,
    })
    id3 = db.prompts.Insert({
        "prompt": "写一个 Python 斐波那契函数",
        "agent": "user",
        "model": "siliconflow-cn/Qwen/Qwen3-8B",
        "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "abstract": "用户要求写斐波那契函数，AI 给出递归实现",
        "should_end": 0,
        "previous_id": id2,
    })
    id4 = db.prompts.Insert({
        "prompt": "测试没有 abstract 的 prompt",
        "agent": "user",
        "model": "",
        "response": "这是测试回复",
        "abstract": "",
        "should_end": 0,
    })
    id5 = db.prompts.Insert({
        "prompt": "再见！",
        "agent": "user",
        "model": "",
        "response": "再见，祝您愉快！",
        "abstract": "道别",
        "should_end": 1,
        "previous_id": id3,
    })
    return {"id1": id1, "id2": id2, "id3": id3, "id4": id4, "id5": id5}


# ── 配置加载 ──────────────────────────────────────────────────────


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text(json.dumps([
            {"name": "a", "type": "abstract", "count": 2, "interval_seconds": 10},
            {"name": "b", "type": "jielong"},
        ]), encoding="utf-8")
        result = load_config(str(cfg))
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[0]["count"] == 2
        assert result[0]["enabled"] is True
        assert result[1]["type"] == "jielong"
        assert result[1]["count"] == 1  # default
        assert result[1]["interval_seconds"] == 60  # default

    def test_load_config_defaults(self, tmp_path):
        cfg = tmp_path / "minimal.json"
        cfg.write_text(json.dumps([
            {"type": "abstract"},
        ]), encoding="utf-8")
        result = load_config(str(cfg))
        assert result[0]["count"] == 1
        assert result[0]["interval_seconds"] == 60
        assert result[0]["enabled"] is True
        assert result[0]["name"] == "abstract-0"

    def test_load_config_invalid_type(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text(json.dumps([
            {"type": "invalid"},
        ]), encoding="utf-8")
        with pytest.raises(ValueError, match="无效"):
            load_config(str(cfg))

    def test_load_config_not_array(self, tmp_path):
        cfg = tmp_path / "obj.json"
        cfg.write_text(json.dumps({"type": "abstract"}), encoding="utf-8")
        with pytest.raises(ValueError, match="必须是 JSON 数组"):
            load_config(str(cfg))

    def test_load_config_loop666_type(self, tmp_path):
        cfg = tmp_path / "loop666.json"
        cfg.write_text(json.dumps([
            {"name": "loop666", "type": "loop666", "count": 1, "interval_seconds": 0},
        ]), encoding="utf-8")
        result = load_config(str(cfg))
        assert len(result) == 1
        assert result[0]["type"] == "loop666"

    def test_load_config_not_list_item(self, tmp_path):
        cfg = tmp_path / "bad_item.json"
        cfg.write_text(json.dumps(["string"]), encoding="utf-8")
        with pytest.raises(ValueError, match="必须是对象"):
            load_config(str(cfg))

    def test_default_config_path(self):
        assert DEFAULT_CONFIG.endswith("loop.json")

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json")


# ── Token 估算 ──────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_none(self):
        assert estimate_tokens(None) == 0

    def test_english_only(self):
        result = estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result == 14

    def test_chinese_only(self):
        assert estimate_tokens("你好世界") == 8

    def test_mixed(self):
        result = estimate_tokens("hello你好")
        assert isinstance(result, int)
        assert result == 10

    def test_long_text(self):
        text = "a" * 1000
        result = estimate_tokens(text)
        assert result > 0
        assert isinstance(result, int)


# ── 历史构建 ────────────────────────────────────────────────────


class TestBuildHistory:
    def test_build_history_single(self):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)
        history = build_history(db, ids["id5"])
        assert len(history) == 4
        assert history[0]["id"] == ids["id1"]
        assert history[1]["id"] == ids["id2"]
        assert history[2]["id"] == ids["id3"]
        assert history[3]["id"] == ids["id5"]
        db.close()

    def test_build_history_chain(self):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)
        history = build_history(db, ids["id3"])
        assert len(history) == 3
        assert [h["id"] for h in history] == [ids["id1"], ids["id2"], ids["id3"]]
        db.close()

    def test_build_history_no_previous(self):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)
        history = build_history(db, ids["id1"])
        assert len(history) == 1
        assert history[0]["id"] == ids["id1"]
        db.close()

    def test_build_history_invalid_id(self):
        db = DataBase(TEST_DB)
        history = build_history(db, 9999)
        assert history == []
        db.close()


# ── 上下文格式化 ────────────────────────────────────────────────


class TestFormatConversationContext:
    def test_format_single(self):
        history = [{"id": 1, "previous_id": None, "prompt": "你好", "agent": "user",
                     "model": "", "response": "你好！", "abstract": "问候", "should_end": 0}]
        result = format_conversation_context(history)
        assert "--- 第 1 轮 ---" in result
        assert "用户: 你好" in result
        assert "助手: 你好！" in result
        assert "摘要: 问候" in result

    def test_format_multiple(self):
        history = [
            {"id": 1, "previous_id": None, "prompt": "你好", "agent": "user",
             "model": "", "response": "你好！", "abstract": "问候", "should_end": 0},
            {"id": 2, "previous_id": 1, "prompt": "再见", "agent": "user",
             "model": "", "response": "再见！", "abstract": "道别", "should_end": 1},
        ]
        result = format_conversation_context(history)
        assert "--- 第 1 轮 ---" in result
        assert "--- 第 2 轮 ---" in result

    def test_format_empty(self):
        result = format_conversation_context([])
        assert result == ""


# ── 历史压缩（async） ──────────────────────────────────────────


class TestCompactHistory:
    @pytest.mark.asyncio
    async def test_compact_no_history(self):
        db = DataBase(TEST_DB)
        result = await compact_history(db, [])
        assert result == []
        db.close()

    @pytest.mark.asyncio
    async def test_compact_single(self):
        db = DataBase(TEST_DB)
        history = [{"id": 1, "previous_id": None, "prompt": "你好", "agent": "user",
                     "model": "", "response": "你好！", "abstract": "问候", "should_end": 0}]
        result = await compact_history(db, history)
        assert len(result) == 1
        db.close()

    @pytest.mark.asyncio
    async def test_compact_two(self):
        db = DataBase(TEST_DB)
        history = [
            {"id": 1, "previous_id": None, "prompt": "你好", "agent": "user",
             "model": "", "response": "你好！", "abstract": "问候", "should_end": 0},
            {"id": 2, "previous_id": 1, "prompt": "再见", "agent": "user",
             "model": "", "response": "再见！", "abstract": "道别", "should_end": 1},
        ]
        result = await compact_history(db, history)
        assert len(result) == 2
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_compact_more_than_two(self, mock_call):
        db = DataBase(TEST_DB)
        mock_call.return_value = {"success": True, "output": "压缩后的摘要", "usage": {}}
        history = [
            {"id": 1, "previous_id": None, "prompt": "第一轮", "agent": "user",
             "model": "", "response": "回复1", "abstract": "摘要1", "should_end": 0},
            {"id": 2, "previous_id": 1, "prompt": "第二轮", "agent": "user",
             "model": "", "response": "回复2", "abstract": "摘要2", "should_end": 0},
            {"id": 3, "previous_id": 2, "prompt": "第三轮", "agent": "user",
             "model": "", "response": "回复3", "abstract": "摘要3", "should_end": 0},
        ]
        result = await compact_history(db, history)
        assert len(result) == 3
        assert result[0].get("_compressed") == True
        assert result[0]["summary"].startswith("压缩后的摘要")
        assert result[1].get("_compressed") is not True
        assert result[2].get("_compressed") is not True
        db.close()


# ── Loop 1: Abstract 生成（async） ─────────────────────────────


class TestLoop1Abstract:
    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop1_processes_without_abstract(self, mock_call):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"abstract": "测试摘要内容", "should_end": 0}',
            "usage": {"input": 50, "output": 100, "total": 150},
        }

        results = await loop1_abstract(db, count=10)
        assert len(results) >= 1
        rows = db.prompts.Read(condition=f"id={ids['id4']}")
        assert rows[0][6] == "测试摘要内容"
        assert rows[0][7] == 0
        req_rows = db.requests.Read()
        assert len(req_rows) >= 1
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop1_sets_should_end(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"abstract": "结束对话", "should_end": 1}',
            "usage": {"input": 30, "output": 50, "total": 80},
        }

        results = await loop1_abstract(db, count=10)
        found = [r for r in results if r["should_end"] == 1]
        assert len(found) >= 1
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop1_records_request(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"abstract": "测试", "should_end": 0}',
            "usage": {"input": 20, "output": 40, "total": 60},
        }

        results = await loop1_abstract(db, count=10)
        assert len(results) >= 1
        for r in results:
            assert r["request_id"] is not None
            assert r["request_id"] > 0
            assert r["success"] == True
        db.close()

    @pytest.mark.asyncio
    async def test_loop1_no_candidates(self):
        db = DataBase(TEST_DB)
        results = await loop1_abstract(db, count=10)
        assert results == []
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop1_count_negative_one(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"abstract": "摘要", "should_end": 0}',
            "usage": {"input": 10, "output": 20, "total": 30},
        }

        results = await loop1_abstract(db, count=-1)
        assert len(results) == 1  # only id4 has no abstract
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop1_custom_models(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"abstract": "摘要", "should_end": 0}',
            "usage": {},
        }

        custom_models = ["my-custom-model/v1"]
        results = await loop1_abstract(db, count=10, models=custom_models)
        assert len(results) >= 1
        db.close()


# ── Loop 2: 对话接龙（async） ──────────────────────────────────


class TestLoop2Jielong:
    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop2_processes_with_abstract_no_end(self, mock_call):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"next_prompt": "那递归实现有什么缺点？", "response": "递归实现虽然简洁，但有栈溢出风险，建议用迭代或动态规划。"}',
            "usage": {"input": 200, "output": 150, "total": 350},
        }

        results = await loop2_jielong(db, count=10)
        assert len(results) >= 1
        r = results[0]
        assert r["new_prompt_id"] is not None
        assert r["request_id"] is not None
        assert r["success"] == True
        new_rows = db.prompts.Read(condition=f"id={r['new_prompt_id']}")
        assert len(new_rows) == 1
        assert new_rows[0][2] == "那递归实现有什么缺点？"
        assert new_rows[0][5] == "递归实现虽然简洁，但有栈溢出风险，建议用迭代或动态规划。"
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop2_excludes_should_end_1(self, mock_call):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"next_prompt": "下一个问题", "response": "下一个回答"}',
            "usage": {"input": 100, "output": 50, "total": 150},
        }

        results = await loop2_jielong(db, count=10)
        for r in results:
            assert r["prompt_id"] != ids["id5"]
        db.close()

    @pytest.mark.asyncio
    async def test_loop2_no_candidates(self):
        db = DataBase(TEST_DB)
        results = await loop2_jielong(db, count=10)
        assert results == []
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop2_count_negative_one(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"next_prompt": "Q", "response": "A"}',
            "usage": {},
        }

        results = await loop2_jielong(db, count=-1)
        # should process all 3 eligible prompts (id1, id2, id3)
        assert len(results) == 3
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop2_custom_models(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": '{"next_prompt": "Q", "response": "A"}',
            "usage": {},
        }

        custom = ["my-model/v2"]
        results = await loop2_jielong(db, count=1, models=custom)
        assert len(results) >= 1
        db.close()


# ── Call opencode（async） ─────────────────────────────────────


class TestCallOpencode:
    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_success(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "output": "Hello world",
            "usage": {"input": 10, "output": 20, "total": 30},
        }

        result = await call_opencode("test prompt", model="test-model")
        assert result["success"] == True
        assert result["output"] == "Hello world"
        assert result["usage"]["input"] == 10
        assert result["usage"]["output"] == 20
        assert result["usage"]["total"] == 30
        mock_run.assert_called_once_with("test prompt", "test-model", "")

    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_no_json_output(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "output": "plain text output",
            "usage": {"input": 0, "output": 0, "total": 0},
        }

        result = await call_opencode("test")
        assert result["success"] == True
        assert result["output"] == "plain text output"

    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_with_agent(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "output": "Agent result",
            "usage": {"input": 5, "output": 10, "total": 15},
        }

        result = await call_opencode("prompt", agent="AbstractAgent")
        mock_run.assert_called_once_with("prompt", "", "AbstractAgent")
        assert result["success"] == True
        assert result["output"] == "Agent result"

    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_failure(self, mock_run):
        mock_run.return_value = {"success": False, "error": "找不到 opencode 命令，请确认已安装"}

        result = await call_opencode("test")
        assert result["success"] == False
        assert "找不到" in result["error"]

    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_nonzero_returncode(self, mock_run):
        mock_run.return_value = {"success": False, "error": "opencode 调用失败: error: something failed"}

        result = await call_opencode("test")
        assert result["success"] == False
        assert "失败" in result["error"]

    @pytest.mark.asyncio
    @patch("loop._opencode_client.run_prompt_json")
    async def test_call_opencode_timeout(self, mock_run):
        mock_run.return_value = {"success": False, "error": "opencode 超时 (3600s)"}

        result = await call_opencode("test")
        assert result["success"] == False
        assert "超时" in result["error"]


# ── Loop 666: Auto666（async） ───────────────────────────────


class TestLoop666Auto666:
    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop666_processes_latest_prompt(self, mock_call):
        db = DataBase(TEST_DB)
        ids = seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": "任务完成",
            "usage": {"input": 100, "output": 50, "total": 150},
        }

        results = await loop666_auto666(db, count=1)
        assert len(results) == 1
        assert results[0]["prompt_id"] == ids["id5"]  # latest
        assert results[0]["request_id"] is not None
        assert results[0]["success"] is True
        req_rows = db.requests.Read()
        assert len(req_rows) == 1
        assert req_rows[0][2] == "Auto666"  # agent_name
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop666_handles_failure(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {"success": False, "error": "执行失败"}

        results = await loop666_auto666(db, count=1)
        assert len(results) == 1
        assert results[0]["success"] is False
        db.close()

    @pytest.mark.asyncio
    async def test_loop666_no_prompts(self):
        db = DataBase(TEST_DB)
        results = await loop666_auto666(db, count=1)
        assert results == []
        db.close()

    @pytest.mark.asyncio
    @patch("loop.call_opencode")
    async def test_loop666_count_negative_one(self, mock_call):
        db = DataBase(TEST_DB)
        seed_test_data(db)

        mock_call.return_value = {
            "success": True,
            "output": "ok",
            "usage": {},
        }

        results = await loop666_auto666(db, count=-1)
        assert len(results) == 5
        db.close()


# ── 模型列表 ───────────────────────────────────────────────────


class TestModelLists:
    def test_abstract_models_not_empty(self):
        assert len(ABSTRACT_MODELS) > 0

    def test_jielong_models_not_empty(self):
        assert len(JIELONG_MODELS) > 0

    def test_abstract_models_are_subset_of_jielong(self):
        for m in ABSTRACT_MODELS:
            assert m in JIELONG_MODELS


# ── run_loop_instance ──────────────────────────────────────────


class TestRunLoopInstance:
    @pytest.mark.asyncio
    @patch("loop.loop1_abstract")
    async def test_run_loop_one_shot(self, mock_loop1):
        db = DataBase(TEST_DB)
        mock_loop1.return_value = [{"prompt_id": 1, "success": True}]

        config = {
            "name": "test", "type": "abstract",
            "count": 1, "interval_seconds": 0, "enabled": True, "models": None,
        }
        await run_loop_instance(db, config)
        mock_loop1.assert_awaited_once_with(db, count=1, models=None)
        db.close()

    @pytest.mark.asyncio
    @patch("loop.loop2_jielong")
    async def test_run_loop_with_interval_cancels(self, mock_loop2):
        db = DataBase(TEST_DB)
        mock_loop2.return_value = []

        config = {
            "name": "test2", "type": "jielong",
            "count": -1, "interval_seconds": 0, "enabled": True, "models": ["m1"],
        }
        await run_loop_instance(db, config)
        mock_loop2.assert_awaited_once_with(db, count=-1, models=["m1"])
        db.close()

    @pytest.mark.asyncio
    @patch("loop.loop666_auto666")
    async def test_run_loop_loop666_type(self, mock_loop666):
        db = DataBase(TEST_DB)
        mock_loop666.return_value = [{"prompt_id": 1, "success": True}]

        config = {
            "name": "test666", "type": "loop666",
            "count": 1, "interval_seconds": 0, "enabled": True, "models": None,
        }
        await run_loop_instance(db, config)
        mock_loop666.assert_awaited_once_with(db, count=1, models=None)
        db.close()

    @pytest.mark.asyncio
    @patch("loop.loop1_abstract")
    async def test_run_loop_handles_exception(self, mock_loop1):
        db = DataBase(TEST_DB)
        mock_loop1.side_effect = RuntimeError("boom")

        config = {
            "name": "err", "type": "abstract",
            "count": 1, "interval_seconds": 0, "enabled": True, "models": None,
        }
        await run_loop_instance(db, config)
        mock_loop1.assert_awaited_once()
        db.close()


# ── main ───────────────────────────────────────────────────────


class TestMain:
    @pytest.mark.asyncio
    @patch("loop.run_loop_instance")
    @patch("loop.load_config")
    async def test_main_loads_config_and_runs(self, mock_load, mock_run):
        mock_load.return_value = [
            {"name": "a", "type": "abstract", "enabled": True},
            {"name": "b", "type": "jielong", "enabled": True},
        ]
        mock_run.return_value = None

        from loop import main as loop_main
        with patch.object(sys, "argv", ["loop.py", "dummy.json"]):
            await loop_main()

        assert mock_run.await_count == 2

    @pytest.mark.asyncio
    @patch("loop.run_loop_instance")
    @patch("loop.load_config")
    async def test_main_skips_disabled(self, mock_load, mock_run):
        mock_load.return_value = [
            {"name": "a", "type": "abstract", "enabled": True},
            {"name": "b", "type": "jielong", "enabled": False},
        ]
        mock_run.return_value = None

        from loop import main as loop_main
        with patch.object(sys, "argv", ["loop.py", "dummy.json"]):
            await loop_main()

        assert mock_run.await_count == 1
