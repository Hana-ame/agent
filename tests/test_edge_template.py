"""
Edge Subclass Test Template
===========================

Usage:
1. Copy this file to your project's tests/ directory
2. Fill in your Edge subclass according to the TODO comments
3. Run pytest tests/test_my_edge.py -v

Framework assumptions:
- Your Edge subclass inherits from framework.edge.Edge
- You implement some methods of condition / pre_process / post_process / compute
- Your Edge has different settings configuration scenarios
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


# ===================================================================
# TODO: Import your Edge subclass
# ===================================================================

# from my_module import MyCustomEdge


# ===================================================================
# Usage Example
# ===================================================================

"""
Example: Suppose you have an Edge subclass

class MyGuardEdge(Edge):
    def condition(self, data, settings):
        threshold = settings.get("threshold", 0)
        return isinstance(data, (int, float)) and data >= threshold

    def pre_process(self, data, settings):
        prefix = settings.get("prefix", "")
        return f"{prefix}{data}" if prefix else data

Test code:

@pytest.mark.asyncio
async def test_my_guard_edge():
    # Case 1: Single data
    src = make_source_vertex(90, channel="score")
    dst = make_dest_vertex(incoming_edges=["e1"])
    edge = make_edge(MyGuardEdge, edge_id="e1", channel="score",
                     settings={"threshold": 80, "prefix": "[OK]"})
    result = await edge.execute(src, dst, echo_agent())
    assert result == "[OK]90"

    # Case 2: Multi-data
    src2 = make_source_vertex(initial_data=[
        {"data_id": "score", "value": 95},
        {"data_id": "user", "value": "Alice"},
    ])
    dst2 = make_dest_vertex(incoming_edges=["e2"])
    edge2 = make_edge(MyGuardEdge, edge_id="e2", channel="score",
                      settings={"threshold": 80})
    result2 = await edge2.execute(src2, dst2, echo_agent())
    assert result2 == 95
"""


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

    GUARD_TEST_CASES: List[Tuple[Any, Dict, bool]] = [
        # (data, settings, expected_result)
        # TODO: Add guard test cases here
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
    """Create an Edge instance."""
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


# ===================================================================
# Unit Tests for Hooks
# ===================================================================

class TestHooksDirectly:
    """Unit test hook methods directly without executing full Edge."""

    def test_condition_default(self):
        """Default condition logic."""
        pass

    def test_pre_process_transforms_data(self):
        """pre_process correctly transforms input data."""
        pass

    def test_post_process_transforms_result(self):
        """post_process correctly transforms LLM / compute output."""
        pass

    def test_hooks_preserve_type(self):
        """Hook preserves expected data types."""
        pass

    def test_hooks_handle_none(self):
        """Hook handles None input gracefully."""
        pass

    def test_hooks_with_settings(self):
        """Hook behaviour changes according to settings."""
        pass


# ===================================================================
# End-to-End Execution Tests
# ===================================================================

class TestExecution:
    """Test complete Edge execution flow: source -> edge -> dest."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Basic execution: data flows from source to destination."""
        pass

    @pytest.mark.asyncio
    async def test_execute_with_settings(self):
        """Execution under different settings configurations."""
        pass

    @pytest.mark.asyncio
    async def test_execute_guard_abort(self):
        """Abort when guard condition is not satisfied."""
        pass

    @pytest.mark.asyncio
    async def test_execute_data_in_dest(self):
        """Data is correctly delivered to destination Vertex."""
        pass

    @pytest.mark.asyncio
    async def test_execute_agent_exception(self):
        """Edge records error on agent exceptions."""
        pass


# ===================================================================
# Settings Combination Tests
# ===================================================================

class TestSettingsCombinations:
    """Test edge behavior across settings combinations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("settings,expected_behavior", [
        # TODO: Parameterize key configuration combinations
    ])
    async def test_settings_combination(self, settings, expected_behavior):
        pass


# ===================================================================
# Reset and Repr Tests
# ===================================================================

class TestResetAndRepr:
    """Test reset() and __repr__()."""

    def test_reset_clears_state(self):
        """reset clears execution state."""
        pass

    def test_repr_includes_class_name(self):
        """repr includes the class name."""
        pass


if __name__ == "__main__":
    print("Run tests: pytest tests/test_edge_template.py -v")
