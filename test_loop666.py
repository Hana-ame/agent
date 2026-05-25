"""loop666.py 单元测试

注：loop666.py 顶层包含无限循环，无法安全 import，
因此将核心纯函数 find_restart_command 直接内联于测试文件中。
"""
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


# ── 核心逻辑函数（与 loop666.py 保持一致） ─────────────────────────


def find_restart_command(data, after_ts):
    """遍历板块内容，查找包含 [restart 666] 且时间戳晚于 after_ts 的帖子。
    
    返回: (no, id, tid) 或 None
    """
    for thread in data:
        ts = thread.get("ts", "")
        no = thread.get("no", 0)
        tid = no
        txt = thread.get("txt") or ""

        if ts > after_ts and "[restart 666]" in txt:
            return (no, thread.get("id"), tid)

        for reply in thread.get("list", []):
            rts = reply.get("ts", "")
            rtxt = reply.get("txt") or ""
            if rts > after_ts and "[restart 666]" in rtxt:
                return (reply.get("no"), reply.get("id"), tid)

    return None


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_board_data():
    return [
        {"ts": "2026-05-25T01:00:00Z", "id": "abc123", "no": 190200, "txt": "普通帖子内容", "num": 0, "list": []},
        {"ts": "2026-05-25T02:00:00Z", "id": "def456", "no": 190201, "txt": "又一个普通帖子", "num": 2,
         "list": [
             {"ts": "2026-05-25T02:30:00Z", "id": "ghi789", "no": 190202, "txt": "一条普通回复"},
             {"ts": "2026-05-25T02:35:00Z", "id": "jkl012", "no": 190203, "txt": "另一条回复"},
         ]},
    ]


@pytest.fixture
def board_with_restart_in_thread():
    return [
        {"ts": "2026-05-25T03:00:00Z", "id": "mno345", "no": 190210,
         "txt": "帮我执行一下 [restart 666] 谢谢", "num": 0, "list": []},
    ]


@pytest.fixture
def board_with_restart_in_reply():
    return [
        {"ts": "2026-05-25T03:00:00Z", "id": "pqr678", "no": 190220, "txt": "一些内容", "num": 1,
         "list": [
             {"ts": "2026-05-25T04:00:00Z", "id": "stu901", "no": 190221, "txt": "回复 [restart 666]"},
         ]},
    ]


@pytest.fixture
def board_multiple_threads():
    return [
        {"ts": "2026-05-25T01:00:00Z", "id": "aaa111", "no": 190230, "txt": "普通帖子A", "list": []},
        {"ts": "2026-05-25T05:00:00Z", "id": "bbb222", "no": 190231, "txt": "[restart 666]", "list": []},
        {"ts": "2026-05-25T03:00:00Z", "id": "ccc333", "no": 190232, "txt": "普通帖子B", "list": []},
    ]


# ── find_restart_command 测试 ──────────────────────────────────────


class TestFindRestartCommand:

    def test_no_restart_found(self, sample_board_data):
        assert find_restart_command(sample_board_data, "2026-05-25T00:00:00Z") is None

    def test_restart_in_thread_txt(self, board_with_restart_in_thread):
        result = find_restart_command(board_with_restart_in_thread, "2026-05-25T00:00:00Z")
        assert result == (190210, "mno345", 190210)

    def test_restart_in_reply_txt(self, board_with_restart_in_reply):
        result = find_restart_command(board_with_restart_in_reply, "2026-05-25T00:00:00Z")
        assert result == (190221, "stu901", 190220)

    def test_before_timestamp(self, board_with_restart_in_thread):
        assert find_restart_command(board_with_restart_in_thread, "2026-05-25T06:00:00Z") is None

    def test_exact_timestamp_boundary(self):
        data = [{"ts": "2026-05-25T03:00:00Z", "id": "x", "no": 190240, "txt": "[restart 666]", "list": []}]
        assert find_restart_command(data, "2026-05-25T03:00:00Z") is None

    def test_partial_match_not_triggered(self):
        data = [{"ts": "2026-05-25T04:00:00Z", "id": "x", "no": 190250,
                 "txt": "restart 666 但没方括号", "list": []}]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_multiple_threads(self, board_multiple_threads):
        result = find_restart_command(board_multiple_threads, "2026-05-25T00:00:00Z")
        assert result == (190231, "bbb222", 190231)

    def test_empty_data(self):
        assert find_restart_command([], "2026-05-25T00:00:00Z") is None

    def test_thread_missing_ts(self):
        data = [{"no": 190260, "txt": "[restart 666]", "list": []}]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_reply_missing_ts(self):
        data = [{"ts": "2026-05-25T03:00:00Z", "no": 190270, "txt": "正文",
                 "list": [{"no": 190271, "txt": "[restart 666]"}]}]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_multiple_restart_commands(self):
        """多个 [restart 666] 时返回第一个遇到的。"""
        data = [
            {"ts": "2026-05-25T02:00:00Z", "id": "a", "no": 1, "txt": "第一个 [restart 666]", "list": []},
            {"ts": "2026-05-25T03:00:00Z", "id": "b", "no": 2, "txt": "第二个 [restart 666]", "list": []},
        ]
        result = find_restart_command(data, "2026-05-25T00:00:00Z")
        assert result == (1, "a", 1)  # 返回第一个

    def test_empty_txt_field(self):
        """空 txt 字段不应报错。"""
        data = [{"ts": "2026-05-25T03:00:00Z", "id": "x", "no": 190280, "txt": "", "list": []}]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_none_txt_field(self):
        """None txt 字段不应报错。"""
        data = [{"ts": "2026-05-25T03:00:00Z", "id": "x", "no": 190290, "txt": None, "list": []}]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_mixed_case(self):
        """大小写敏感 - 应完全匹配 [restart 666]。"""
        data = [
            {"ts": "2026-05-25T03:00:00Z", "id": "x", "no": 1, "txt": "[Restart 666]", "list": []},
            {"ts": "2026-05-25T04:00:00Z", "id": "y", "no": 2, "txt": "[RESTART 666]", "list": []},
        ]
        assert find_restart_command(data, "2026-05-25T00:00:00Z") is None

    def test_embedded_restart(self):
        """[restart 666] 嵌在文本中间应被识别。"""
        data = [{"ts": "2026-05-25T03:00:00Z", "id": "x", "no": 190300,
                 "txt": "前面内容 [restart 666] 后面内容", "list": []}]
        result = find_restart_command(data, "2026-05-25T00:00:00Z")
        assert result is not None


# ── 时间戳比较逻辑 ─────────────────────────────────────────────────


class TestTimestampComparison:

    def test_iso8601_string_comparison(self):
        """ISO 8601 字符串字典序与时间序一致。"""
        assert "2026-05-25T01:00:00Z" < "2026-05-25T02:00:00Z"

        data = [{"ts": "2026-05-25T02:00:00Z", "no": 1, "txt": "[restart 666]", "list": []}]
        assert find_restart_command(data, "2026-05-25T01:00:00Z") is not None
        assert find_restart_command(data, "2026-05-25T02:00:00Z") is None
        assert find_restart_command(data, "2026-05-25T03:00:00Z") is None

    def test_different_days(self):
        """跨天比较应正确。"""
        data = [{"ts": "2026-05-26T00:00:00Z", "no": 1, "txt": "[restart 666]", "list": []}]
        assert find_restart_command(data, "2026-05-25T23:59:59Z") is not None
        assert find_restart_command(data, "2026-05-26T00:00:00Z") is None


# ── board_api 集成测试 ─────────────────────────────────────────────


class TestBoardAPI:

    def test_request_board_returns_string(self):
        from board_api import request_board
        result = request_board(bid=666, timeout=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_request_board_valid_json(self):
        from board_api import request_board
        data = json.loads(request_board(bid=666, timeout=10))
        assert isinstance(data, list)
        assert len(data) > 0

    def test_request_board_item_structure(self):
        from board_api import request_board
        for item in json.loads(request_board(bid=666, timeout=10)):
            assert "ts" in item
            assert "no" in item
            assert "txt" in item
            for reply in item.get("list", []):
                assert "ts" in reply
                assert "txt" in reply

    def test_request_board_timeout(self):
        """超短 timeout 应抛出异常。"""
        from board_api import request_board
        with pytest.raises(RuntimeError):
            request_board(bid=666, timeout=1, retries=1)


# ── 真实数据端到端测试 ─────────────────────────────────────────────


class TestEndToEnd:

    def test_real_data_flow(self):
        from board_api import request_board
        data = json.loads(request_board(bid=666, timeout=10))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        result = find_restart_command(data, today)
        # 不应报错，result 可能为 None（当前无指令）
        if result is not None:
            no, _, tid = result
            assert isinstance(no, int)
            assert isinstance(tid, int)
