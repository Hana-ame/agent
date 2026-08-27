"""Shared pytest fixtures for vertex-edge-agent tests."""

import asyncio
import json
import os
import sys
import tempfile
from typing import Dict

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Vertex, Edge, Graph, Executor, MockPIAgent
from framework.vertex import VertexState


# ------------------------------------------------------------------
# Fixtures: vertices
# ------------------------------------------------------------------
@pytest.fixture
def source_vertex():
    """A source vertex with initial data."""
    return Vertex(
        vertex_id="src",
        settings={"type": "source"},
        initial_data=[
            {"data_id": "text", "tags": ["en"], "value": "Hello world"},
        ],
    )


@pytest.fixture
def empty_vertex():
    """A bare vertex with no data or settings."""
    return Vertex(vertex_id="empty")


@pytest.fixture
def sink_vertex():
    """A sink vertex (no outgoing edges)."""
    return Vertex(vertex_id="sink", settings={"type": "sink"})


# ------------------------------------------------------------------
# Fixtures: mock agent
# ------------------------------------------------------------------
@pytest.fixture
def mock_agent():
    return MockPIAgent()


@pytest.fixture
def echo_agent():
    """Agent that returns data unchanged."""
    return MockPIAgent(response_fn=lambda d, p, m, s: d)


@pytest.fixture
def upper_agent():
    """Agent that uppercases string data."""
    return MockPIAgent(
        response_fn=lambda d, p, m, s: d.upper() if isinstance(d, str) else d
    )


# ------------------------------------------------------------------
# Fixtures: graph configs
# ------------------------------------------------------------------
@pytest.fixture
def linear_config() -> Dict:
    """Minimal linear graph: A → B → C."""
    return {
        "metadata": {"name": "linear"},
        "vertices": [
            {
                "id": "A",
                "initial_data": [
                    {"data_id": "x", "tags": [], "value": "hello"},
                ],
            },
            {"id": "B"},
            {"id": "C"},
        ],
        "edges": [
            {
                "id": "e1",
                "source": "A",
                "destination": "B",
                "data_id": "x",
                "tags": [],
                "prompt": "process",
                "model": "mock",
            },
            {
                "id": "e2",
                "source": "B",
                "destination": "C",
                "data_id": "x",
                "tags": [],
                "prompt": "finalize",
                "model": "mock",
            },
        ],
    }


@pytest.fixture
def diamond_config() -> Dict:
    """Diamond graph: A → B, A → C, B → D, C → D."""
    return {
        "metadata": {"name": "diamond"},
        "vertices": [
            {
                "id": "A",
                "initial_data": [
                    {"data_id": "v", "tags": ["t1"], "value": "start"},
                    {"data_id": "v", "tags": ["t2"], "value": "start"},
                ],
            },
            {"id": "B"},
            {"id": "C"},
            {"id": "D"},
        ],
        "edges": [
            {
                "id": "ab",
                "source": "A",
                "destination": "B",
                "data_id": "v",
                "tags": ["t1"],
                "prompt": "branch-1",
                "model": "mock",
            },
            {
                "id": "ac",
                "source": "A",
                "destination": "C",
                "data_id": "v",
                "tags": ["t2"],
                "prompt": "branch-2",
                "model": "mock",
            },
            {
                "id": "bd",
                "source": "B",
                "destination": "D",
                "data_id": "v",
                "tags": ["t1"],
                "prompt": "merge-1",
                "model": "mock",
            },
            {
                "id": "cd",
                "source": "C",
                "destination": "D",
                "data_id": "v",
                "tags": ["t2"],
                "prompt": "merge-2",
                "model": "mock",
            },
        ],
    }


@pytest.fixture
def cycle_config() -> Dict:
    """Invalid graph with a cycle: A → B → A."""
    return {
        "vertices": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {
                "id": "e1",
                "source": "A",
                "destination": "B",
                "data_id": "x",
                "prompt": "",
                "model": "m",
            },
            {
                "id": "e2",
                "source": "B",
                "destination": "A",
                "data_id": "x",
                "prompt": "",
                "model": "m",
            },
        ],
    }


@pytest.fixture
def tmp_json(tmp_path):
    """Factory that writes a config dict to a temp JSON file and returns the path."""
    def _write(config: Dict) -> str:
        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            json.dump(config, f)
        return path
    return _write
