"""
Edge 子类测试模板
================

使用方法：
1. 复制此文件到你的项目 tests/ 目录
2. 按照 TODO 注释填入你的 Edge 子类
3. 运行 pytest tests/test_my_edge.py -v

框架假设：
- 你的 Edge 子类继承自 framework.edge.Edge
- 你实现了 condition / pre_process / post_process 中的某些方法
- 你的 Edge 有不同的 settings 配置场景
"""

import asyncio
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.edge import Edge
from framework.vertex import Vertex, VertexState, EdgeSignal
from framework.agents import MockAgent


# ═══════════════════════════════════════════════════════════════════
# TODO: 填入你的 Edge 子类
# ═══════════════════════════════════════════════════════════════════

# from my_module import MyCustomEdge


# ═══════════════════════════════════════════════════════════════════
# 完整使用示例
# ═══════════════════════════════════════════════════════════════════

"""
示例：假设你有一个 Edge 子类

class MyGuardEdge(Edge):
    def condition(self, data, settings):
        threshold = settings.get("threshold", 0)
        return isinstance(data, (int, float)) and data >= threshold

    def pre_process(self, data, settings):
        prefix = settings.get("prefix", "")
        return f"{prefix}{data}" if prefix else data

测试代码：

@pytest.mark.asyncio
async def test_my_guard_edge():
    # 方式1：单数据
    src = make_source_vertex(90, channel="score")
    dst = make_dest_vertex(incoming_edges=["e1"])
    edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                     settings={"threshold": 80, "prefix": "[OK]"})
    result = await edge.execute(src, dst, echo_agent())
    assert result == "[OK]90"

    # 方式2：多数据
    src2 = make_source_vertex(initial_data=[
        {"data_id": "score", "value": 95},
        {"data_id": "user", "value": "Alice"},
    ])
    dst2 = make_dest_vertex(incoming_edges=["e2"])
    edge2 = make_edge(MyGuardEdge, edge_id="e2", channel="score",
                      settings={"threshold": 80})
    result2 = await edge2.execute(src2, dst2, echo_agent())
    assert result2 == 95

    # 方式3：手动设置
    src3 = make_source_vertex(vertex_id="src3")
    await src3.set_data("score", 75)
    dst3 = make_dest_vertex(incoming_edges=["e3"])
    edge3 = make_edge(MyGuardEdge, edge_id="e3", channel="score")
    result3 = await edge3.execute(src3, dst3, echo_agent())
    assert result3 == 75
"""


# ═══════════════════════════════════════════════════════════════════
# 测试数据工厂 - 根据你的 Edge 需求修改
# ═══════════════════════════════════════════════════════════════════

class EdgeTestData:
    """集中管理测试数据，方便维护和扩展。"""

    # TODO: 定义你的 Edge 需要的输入数据场景
    INPUT_SCENARIOS: Dict[str, Any] = {
        "basic": "hello",
        "empty": "",
        "number": 42,
        "dict_data": {"key": "value", "score": 90},
        "list_data": [1, 2, 3],
        # 添加更多场景...
    }

    # TODO: 定义你的 Edge 的 settings 配置场景
    SETTINGS_SCENARIOS: Dict[str, Dict] = {
        "default": {},
        "with_threshold": {"threshold": 80, "operator": ">="},
        "with_prefix": {"prefix": "[PROCESSED]"},
        # 添加更多配置...
    }

    # TODO: 定义期望的输出结果 (input_key, settings_key) -> expected
    EXPECTED_OUTPUTS: Dict[Tuple[str, str], Any] = {
        # ("basic", "default"): "expected_output",
        # ("dict_data", "with_threshold"): True,
    }


# ═══════════════════════════════════════════════════════════════════
# 测试基础设施（不需要修改）
# ═══════════════════════════════════════════════════════════════════

def make_edge(
    edge_class: type,
    edge_id: str = "test_edge",
    source_id: str = "src",
    dest_id: str = "dst",
    channel: str = "default",
    settings: Optional[Dict] = None,
    **kwargs,
) -> Edge:
    """工厂函数：创建 Edge 实例。"""
    return edge_class(
        edge_id=edge_id,
        source_id=source_id,
        destination_id=dest_id,
        channel=channel,
        settings=settings or {},
        **kwargs,
    )


def make_source_vertex(
    data: Any = None,
    channel: str = "default",
    vertex_id: str = "src",
    initial_data: Optional[List[Dict]] = None,
) -> Vertex:
    """工厂函数：创建带数据的源 Vertex。

    用法：
        # 单数据
        src = make_source_vertex(90, channel="score")

        # 多数据
        src = make_source_vertex(initial_data=[
            {"data_id": "score", "value": 90},
            {"data_id": "name", "value": "Alice"},
        ])

        # 空节点（手动设置）
        src = make_source_vertex()
        await src.set_data("score", 90)
    """
    if initial_data:
        items = initial_data
    elif data is not None:
        items = [{"data_id": channel, "value": data}]
    else:
        items = []
    return Vertex(vertex_id=vertex_id, initial_data=items)


def make_dest_vertex(
    vertex_id: str = "dst",
    incoming_edges: Optional[List[str]] = None,
    required_input_count: int = 1,
) -> Vertex:
    """工厂函数：创建目标 Vertex。"""
    v = Vertex(vertex_id=vertex_id)
    v.incoming_edges = incoming_edges or []
    v.required_input_count = required_input_count
    return v


def echo_agent():
    """返回原样数据的 Agent。"""
    return MockAgent(response_fn=lambda d, p, m, s: d)


# ═══════════════════════════════════════════════════════════════════
# 条件守卫测试（如果你的 Edge 有 condition 方法）
# ═══════════════════════════════════════════════════════════════════

class TestCondition:
    """测试 condition() / evaluate_condition() 的各种场景。"""

    def test_condition_basic_pass(self):
        """基本场景：条件满足时返回 True。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge, settings={"threshold": 80})
        # assert edge.evaluate_condition(90, edge.settings) is True
        pass

    def test_condition_basic_fail(self):
        """基本场景：条件不满足时返回 False。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge, settings={"threshold": 80})
        # assert edge.evaluate_condition(50, edge.settings) is False
        pass

    def test_condition_edge_cases(self):
        """边界情况：空值、None、类型错误等。"""
        # TODO: 根据你的 condition 逻辑编写
        # edge = make_edge(MyCustomEdge)
        # assert edge.evaluate_condition(None, {}) is False
        # assert edge.evaluate_condition("", {}) is ...
        # assert edge.evaluate_condition([], {}) is ...
        pass

    @pytest.mark.parametrize("input_data,expected", [
        # TODO: 参数化测试多个输入场景
        # (90, True),
        # (80, True),
        # (79, False),
        # (0, False),
    ])
    def test_condition_parametrized(self, input_data, expected):
        """参数化测试：批量验证条件判断。"""
        # edge = make_edge(MyCustomEdge, settings={"threshold": 80})
        # assert edge.evaluate_condition(input_data, edge.settings) is expected
        pass


# ═══════════════════════════════════════════════════════════════════
# Hook 方法测试（如果你的 Edge 有 pre_process / post_process）
# ═══════════════════════════════════════════════════════════════════

class TestHooks:
    """测试 pre_process / post_process 的数据转换逻辑。"""

    def test_pre_process_transforms_data(self):
        """pre_process 正确转换输入数据。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge)
        # result = edge.pre_process("input", edge.settings)
        # assert result == "expected_output"
        pass

    def test_post_process_transforms_result(self):
        """post_process 正确处理 LLM 输出。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge)
        # result = edge.post_process("llm_output", edge.settings)
        # assert result == "expected_output"
        pass

    def test_hooks_preserve_type(self):
        """Hook 保持数据类型不变。"""
        # TODO: 验证输入输出类型一致
        # edge = make_edge(MyCustomEdge)
        # input_data = {"key": "value"}
        # result = edge.pre_process(input_data, edge.settings)
        # assert isinstance(result, dict)
        pass

    def test_hooks_handle_none(self):
        """Hook 处理 None 输入。"""
        # TODO: 根据你的逻辑决定是否允许 None
        # edge = make_edge(MyCustomEdge)
        # result = edge.pre_process(None, edge.settings)
        # assert result is None  # 或者 assert result == "default"
        pass

    def test_hooks_with_settings(self):
        """Hook 根据 settings 行为变化。"""
        # TODO: 测试不同 settings 下的行为
        # edge1 = make_edge(MyCustomEdge, settings={"mode": "upper"})
        # edge2 = make_edge(MyCustomEdge, settings={"mode": "lower"})
        # assert edge1.pre_process("hello", edge1.settings) == "HELLO"
        # assert edge2.pre_process("hello", edge2.settings) == "hello"
        pass


# ═══════════════════════════════════════════════════════════════════
# 完整执行测试（端到端）
# ═══════════════════════════════════════════════════════════════════

class TestExecution:
    """测试 Edge 完整执行流程：source -> edge -> dest。"""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """基本执行：数据从 source 流向 dest。"""
        # TODO: 替换为你的 Edge 类和测试数据
        # src = make_source_vertex("hello", channel="default")
        # dst = make_dest_vertex(incoming_edges=["e1"])
        # edge = make_edge(MyCustomEdge, channel="default")
        #
        # result = await edge.execute(src, dst, echo_agent())
        #
        # assert edge.completed is True
        # assert edge.aborted is False
        # assert result is not None
        pass

    @pytest.mark.asyncio
    async def test_execute_with_settings(self):
        """不同 settings 配置下的执行。"""
        # TODO: 测试关键配置
        # src = make_source_vertex(90, channel="score")
        # dst = make_dest_vertex(incoming_edges=["e1"])
        # edge = make_edge(
        #     MyCustomEdge,
        #     channel="score",
        #     settings={"threshold": 80}
        # )
        #
        # result = await edge.execute(src, dst, echo_agent())
        #
        # assert edge.completed is True
        # assert result == 90
        pass

    @pytest.mark.asyncio
    async def test_execute_guard_abort(self):
        """守卫条件不满足时中止。"""
        # TODO: 测试 condition 返回 False 的场景
        # src = make_source_vertex(50, channel="score")
        # dst = make_dest_vertex(incoming_edges=["e1"])
        # edge = make_edge(
        #     MyCustomEdge,
        #     channel="score",
        #     settings={"threshold": 80}
        # )
        #
        # result = await edge.execute(src, dst, echo_agent())
        #
        # assert result is None
        # assert edge.aborted is True
        # assert edge.completed is False
        # assert dst.state == VertexState.ABORTED
        pass

    @pytest.mark.asyncio
    async def test_execute_data_in_dest(self):
        """执行后数据正确写入目标 Vertex。"""
        # TODO: 验证数据写入
        # src = make_source_vertex("test", channel="d")
        # dst = make_dest_vertex(incoming_edges=["e1"])
        # edge = make_edge(MyCustomEdge, channel="d")
        #
        # await edge.execute(src, dst, echo_agent())
        #
        # stored = await dst.fetch_data(channel="d")
        # assert stored == "expected_value"
        pass

    @pytest.mark.asyncio
    async def test_execute_agent_exception(self):
        """Agent 异常时 Edge 正确记录错误。"""
        def failing_agent(d, p, m, s):
            raise RuntimeError("agent failed")

        # TODO: 替换为你的 Edge 类
        # src = make_source_vertex("test")
        # dst = make_dest_vertex(incoming_edges=["e1"])
        # edge = make_edge(MyCustomEdge)
        # agent = MockAgent(response_fn=failing_agent)
        #
        # with pytest.raises(RuntimeError, match="agent failed"):
        #     await edge.execute(src, dst, agent)
        #
        # assert edge.error is not None
        # assert "agent failed" in edge.error
        pass


# ═══════════════════════════════════════════════════════════════════
# Settings 组合测试
# ═══════════════════════════════════════════════════════════════════

class TestSettingsCombinations:
    """测试不同 settings 组合的行为。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("settings,expected_behavior", [
        # TODO: 参数化测试关键配置组合
        # ({"threshold": 80, "operator": ">="}, "pass"),
        # ({"threshold": 100, "operator": ">="}, "abort"),
        # ({"prefix": "[A]"}, "prefixed"),
    ])
    async def test_settings_combination(self, settings, expected_behavior):
        """验证特定 settings 组合产生预期行为。"""
        # TODO: 根据 expected_behavior 断言
        pass


# ═══════════════════════════════════════════════════════════════════
# Reset 和 Repr 测试
# ═══════════════════════════════════════════════════════════════════

class TestResetAndRepr:
    """测试 reset() 和 __repr__()。"""

    def test_reset_clears_state(self):
        """reset 清除执行状态。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge)
        # edge.completed = True
        # edge.result = "data"
        # edge.error = "err"
        #
        # edge.reset()
        #
        # assert edge.completed is False
        # assert edge.result is None
        # assert edge.error is None
        pass

    def test_repr_includes_class_name(self):
        """repr 包含类名。"""
        # TODO: 替换为你的 Edge 类
        # edge = make_edge(MyCustomEdge, edge_id="e1")
        # assert "MyCustomEdge" in repr(edge)
        # assert "e1" in repr(edge)
        pass


# ═══════════════════════════════════════════════════════════════════
# 快速验证脚本（可选）
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("运行测试: pytest tests/test_my_edge.py -v")
    print("跳过未实现的测试: pytest tests/test_my_edge.py -v -k 'not TODO'")
