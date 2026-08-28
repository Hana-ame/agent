"""
Edge Subclass Test Template
===========================

Usage:
1. Copy this file to your project's tests/ directory
2. Swap ``MyGuardEdge`` / ``TaggingEdge`` below for your own Edge subclass
3. Adjust the settings scenarios to match your configuration surface
4. Run pytest tests/test_my_edge.py -v

This file is deliberately self-contained: it runs green as-is against the
framework's own example subclasses, so you can use it both as documentation
and as a copy-paste starting point. Every test here asserts something real —
do not leave a bare ``pass`` when you copy it.

Framework assumptions:
- Your Edge subclass inherits from framework.edge.Edge
- You implement some methods of condition / pre_process / post_process / compute
- Your Edge has different settings configuration scenarios

Key semantics these tests rely on (see framework/edge.py):
- ``execute`` runs fetch → guard → pre_process → compute → post_process → deliver
- A failed guard sets ``aborted=True``, returns ``None``, and delivers nothing
- ``compute`` calls the agent only when settings carries ``prompt`` or ``model``;
  otherwise it passes the pre-processed data straight through
- An agent exception sets ``error``, sends a FAILED signal, and re-raises
- Hook methods may be sync or async — ``_run_pre_process`` / ``_run_post_process``
  await coroutines automatically, so subclasses can define plain ``def``
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.edge import Edge
from framework.vertex import Vertex
from framework.agents import MockAgent


# ===================================================================
# TODO: Import your Edge subclass, then delete the examples below
# ===================================================================

# from my_module import MyCustomEdge


class MyGuardEdge(Edge):
    """Example Edge subclass: numeric threshold guard + optional prefix.

    Replace this with your own subclass when you copy the file.
    """

    def condition(self, data, settings):
        threshold = settings.get("threshold", 0)
        return isinstance(data, (int, float)) and not isinstance(data, bool) and data >= threshold

    def pre_process(self, data, settings):
        prefix = settings.get("prefix", "")
        return f"{prefix}{data}" if prefix else data


class TaggingEdge(Edge):
    """Example Edge subclass: post_process appends a tag to the result."""

    def post_process(self, result, settings):
        tag = settings.get("tag", "")
        return f"{result}{tag}" if tag else result


# ===================================================================
# Test Data Factory
# ===================================================================

class EdgeTestData:
    """Centralised management for test data scenarios."""

    INPUT_SCENARIOS: Dict[str, Any] = {
        "basic": "hello",
        "empty": "",
        "number": 42,
        "dict_data": {"key": "value", "score": 90},
        "list_data": [1, 2, 3],
    }

    SETTINGS_SCENARIOS: Dict[str, Dict] = {
        "default": {},
        "with_threshold": {"threshold": 80},
        "with_prefix": {"prefix": "[TEST]"},
        "with_retry": {
            "retry_policy": {
                "max_retries": 3,
                "backoff_factor": 0.01,
                "retry_on": ["KeyError", "ValueError"],
            }
        },
        "with_timeout": {"timeout": 5.0},
    }

    # (data, settings, expected condition result)
    GUARD_TEST_CASES: List[Tuple[Any, Dict, bool]] = [
        (90, {"threshold": 80}, True),
        (80, {"threshold": 80}, True),        # boundary: equal to threshold passes
        (79, {"threshold": 80}, False),
        (1, {"threshold": 0}, True),
        (3.5, {"threshold": 3}, True),        # floats count as numeric
        ("hello", {"threshold": 0}, False),   # non-numeric is rejected by the guard
        (None, {}, False),                    # None does not raise, just returns False
        (True, {"threshold": 0}, False),      # bool is not treated as numeric
    ]

    # (settings, payload, expected execute() result)
    EXECUTION_CASES: List[Tuple[Dict, Any, Any]] = [
        ({}, 42, 42),                                        # pure passthrough
        ({"prefix": "[OK]"}, 42, "[OK]42"),                  # prefix applied
        ({"prefix": ""}, 42, 42),                            # empty prefix = none
        ({"prompt": "run"}, 7, 7),                           # prompt triggers the (echo) agent
        ({"threshold": 0, "prefix": "[OK]"}, 1, "[OK]1"),
    ]


# ===================================================================
# Helper Fixtures & Functions
# ===================================================================

def make_source_vertex(
    data: Any = None,
    channel: str = "default",
    vertex_id: str = "src",
    initial_data: Optional[List[Dict]] = None,
) -> Vertex:
    """Create a source Vertex with initial data."""
    if initial_data is not None:
        return Vertex(vertex_id, initial_data=initial_data)
    elif data is not None:
        return Vertex(vertex_id, initial_data=[{"data_id": channel, "value": data}])
    else:
        return Vertex(vertex_id)


def make_dest_vertex(
    vertex_id: str = "dst",
    incoming_edges: Optional[List[str]] = None,
    required_input_count: int = 1,
) -> Vertex:
    """Create a destination Vertex."""
    v = Vertex(vertex_id)
    v.required_input_count = required_input_count
    v.incoming_edges = incoming_edges or ["e1"]
    return v


def make_edge(
    edge_cls: type = Edge,
    edge_id: str = "e1",
    source_id: str = "src",
    destination_id: str = "dst",
    channel: str = "default",
    settings: Optional[Dict] = None,
    **kwargs,
) -> Edge:
    """Create an Edge instance.

    Compute-layer config (prompt/model/agent/retry_policy/...) belongs inside
    ``settings`` — top-level keys are rejected by Graph.from_dict.
    """
    return edge_cls(
        edge_id=edge_id,
        source_id=source_id,
        destination_id=destination_id,
        channel=channel,
        settings=settings or {},
        **kwargs,
    )


def echo_agent() -> MockAgent:
    """Return an agent that echoes data back."""
    return MockAgent(response_fn=lambda d, p, m, s: d)


def fixed_response_agent(response: Any) -> MockAgent:
    """Return an agent that produces a fixed response."""
    return MockAgent(response_fn=lambda d, p, m, s: response)


def failing_agent(exc: Exception) -> MockAgent:
    """Return an agent that always raises ``exc``."""
    def _raise(d, p, m, s):
        raise exc

    return MockAgent(response_fn=_raise)


# ===================================================================
# Unit Tests for Hooks
# ===================================================================

class TestHooksDirectly:
    """Unit test hook methods directly without executing full Edge."""

    def test_condition_default(self):
        """Base Edge with no guard config passes everything through."""
        edge = make_edge(Edge, settings={})
        for data in EdgeTestData.INPUT_SCENARIOS.values():
            assert edge.condition(data, {}) is True

    def test_pre_process_transforms_data(self):
        """pre_process applies the configured prefix."""
        edge = make_edge(MyGuardEdge, settings={"prefix": "[OK]"})
        assert edge.pre_process(90, edge.settings) == "[OK]90"
        assert edge.pre_process("abc", edge.settings) == "[OK]abc"
        # No prefix configured -> data untouched
        bare = make_edge(MyGuardEdge, settings={})
        assert bare.pre_process(90, bare.settings) == 90

    def test_post_process_transforms_result(self):
        """post_process transforms the compute output."""
        tagging = make_edge(TaggingEdge, settings={"tag": " [done]"})
        assert tagging.post_process("report", tagging.settings) == "report [done]"
        # Base post_process is a coroutine; go through the framework's normaliser
        assert asyncio.run(make_edge(Edge, settings={})._run_post_process("x")) == "x"

    def test_hooks_preserve_type(self):
        """Without a prefix the hook returns the very same object."""
        edge = make_edge(MyGuardEdge, settings={})
        for name, data in EdgeTestData.INPUT_SCENARIOS.items():
            assert edge.pre_process(data, {}) is data, name

    def test_hooks_handle_none(self):
        """Hooks tolerate None without raising."""
        edge = make_edge(MyGuardEdge, settings={"prefix": "[OK]"})
        assert edge.pre_process(None, edge.settings) == "[OK]None"
        # The guard reports False instead of blowing up
        assert make_edge(MyGuardEdge, settings={}).condition(None, {}) is False

    def test_hooks_with_settings(self):
        """Different settings produce different hook behaviour."""
        prefixed = make_edge(MyGuardEdge, settings={"prefix": "[A]"})
        assert prefixed.pre_process(1, prefixed.settings) == "[A]1"
        empty = make_edge(MyGuardEdge, settings={"prefix": ""})
        assert empty.pre_process(1, empty.settings) == 1
        # threshold shifts the guard verdict
        assert make_edge(MyGuardEdge, settings={"threshold": 0}).condition(1, {"threshold": 0}) is True
        assert make_edge(MyGuardEdge, settings={"threshold": 5}).condition(1, {"threshold": 5}) is False

    def test_evaluate_condition_proxy(self):
        """evaluate_condition is the public alias; settings may be omitted."""
        edge = make_edge(MyGuardEdge, settings={"threshold": 80})
        assert edge.evaluate_condition(90) is True
        assert edge.evaluate_condition(10) is False
        # explicit settings override the instance settings
        assert edge.evaluate_condition(10, {"threshold": 0}) is True

    @pytest.mark.parametrize("data,settings,expected", EdgeTestData.GUARD_TEST_CASES)
    def test_guard_matrix(self, data, settings, expected):
        """Guard verdicts across the settings/data matrix."""
        edge = make_edge(MyGuardEdge, settings=settings)
        assert edge.condition(data, settings) is expected


# ===================================================================
# End-to-End Execution Tests
# ===================================================================

class TestExecution:
    """Test complete Edge execution flow: source -> edge -> dest."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Data flows from source through the edge to the destination."""
        src = make_source_vertex(90, channel="score")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                         settings={"threshold": 80, "prefix": "[OK]"})
        result = await edge.execute(src, dst, echo_agent())
        assert result == "[OK]90"
        assert edge.completed is True
        assert edge.aborted is False
        assert edge.error is None

    @pytest.mark.asyncio
    async def test_execute_with_settings(self):
        """A prompt routes through the agent; without one, data passes through."""
        # No prompt/model -> compute passes the pre-processed data straight through
        src = make_source_vertex(90, channel="score")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                         settings={"threshold": 80, "prefix": "[OK]"})
        assert await edge.execute(src, dst, echo_agent()) == "[OK]90"

        # With a prompt -> the agent's response wins, however it transforms the data
        src2 = make_source_vertex(90, channel="score")
        dst2 = make_dest_vertex(vertex_id="dst2", incoming_edges=["e2"])
        edge2 = make_edge(MyGuardEdge, edge_id="e2", channel="score",
                          settings={"threshold": 80, "prompt": "rate it"})
        assert await edge2.execute(src2, dst2, fixed_response_agent("FIXED")) == "FIXED"

    @pytest.mark.asyncio
    async def test_execute_guard_abort(self):
        """A failed guard prunes the edge and delivers nothing downstream."""
        src = make_source_vertex(50, channel="score")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                         settings={"threshold": 80})
        result = await edge.execute(src, dst, echo_agent())
        assert result is None
        assert edge.aborted is True
        assert edge.completed is False
        assert "Guard condition" in edge.abort_reason
        # The destination never received this channel
        assert await dst.fetch_data("score") is None

    @pytest.mark.asyncio
    async def test_execute_data_in_dest(self):
        """The delivered payload lands in the destination's channel store."""
        payload = {"score": 95, "user": "Alice"}
        src = make_source_vertex(payload, channel="payload")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(Edge, edge_id="e1", channel="payload")
        assert await edge.execute(src, dst, echo_agent()) == payload
        assert await dst.fetch_data("payload") == payload

    @pytest.mark.asyncio
    async def test_execute_agent_exception(self):
        """An agent failure records the error, signals FAILED, and re-raises."""
        src = make_source_vertex(1, channel="x")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(Edge, edge_id="e1", channel="x", settings={"prompt": "to int"})
        with pytest.raises(ValueError, match="invalid literal"):
            await edge.execute(src, dst, failing_agent(ValueError("invalid literal for int(): 'abc'")))
        assert edge.error is not None
        assert "invalid literal" in edge.error
        assert edge.completed is False
        assert edge.aborted is False


# ===================================================================
# Settings Combination Tests
# ===================================================================

class TestSettingsCombinations:
    """Test edge behavior across settings combinations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("settings,payload,expected", EdgeTestData.EXECUTION_CASES)
    async def test_settings_combination(self, settings, payload, expected):
        """Same edge class, different settings, different end result."""
        src = make_source_vertex(payload, channel="v")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="v", settings=settings)
        assert await edge.execute(src, dst, echo_agent()) == expected


# ===================================================================
# Reset and Repr Tests
# ===================================================================

class TestResetAndRepr:
    """Test reset() and __repr__()."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """reset() wipes execution state so the edge is reusable."""
        src = make_source_vertex(50, channel="score")
        dst = make_dest_vertex(incoming_edges=["e1"])
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                         settings={"threshold": 80, "prompt": "score"})
        assert await edge.execute(src, dst, echo_agent()) is None   # pruned by the guard
        assert edge.aborted is True
        assert edge.abort_reason is not None
        assert edge.prompt.startswith("score")

        edge.reset()
        assert edge.aborted is False
        assert edge.abort_reason is None
        assert edge.error is None
        assert edge.completed is False
        assert edge.result is None

    def test_repr_includes_class_name(self):
        """__repr__ carries the class name, edge id, direction and status mark."""
        edge = make_edge(MyGuardEdge, edge_id="e1", channel="score", settings={})
        r = repr(edge)
        assert "MyGuardEdge" in r
        assert "e1" in r
        assert "src->dst" in r
        assert "·" in r      # not yet executed


if __name__ == "__main__":
    print("Run tests: pytest tests/test_edge_template.py -v")
