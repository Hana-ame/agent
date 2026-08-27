"""Integration tests - full end-to-end pipeline tests.

Tests the framework with external scripts, complex DAGs,
and real JSON config files.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockAgent
from framework.vertex import VertexState, DataRejectedError


# ── simple example ───────────────────────────────────────────────
class TestSimpleExample:
    """Test the simple linear pipeline from examples/simple/config.json."""

    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "simple", "config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("simple example config not found")

        g = Graph.from_json(config_path)
        result = await Executor(g, MockAgent(), timeout=10).run()

        assert result.success
        assert "e1" in result.edge_results
        assert "e2" in result.edge_results
        assert g.vertices["input"].state == VertexState.DONE
        assert g.vertices["processor"].state == VertexState.DONE
        assert g.vertices["output"].state == VertexState.DONE


# ── complex example ──────────────────────────────────────────────
class TestComplexExample:
    """Test the complex DAG from examples/complex/config.json."""

    @pytest.mark.asyncio
    async def test_complex_pipeline(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "complex", "config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("complex example config not found")

        g = Graph.from_json(config_path)
        agent = MockAgent(
            response_fn=lambda d, p, m, s: f"[{m}] {d}" if isinstance(d, str) else d
        )
        result = await Executor(g, agent, timeout=15).run()

        assert result.success
        assert len(result.edge_results) == 5
        assert g.vertices["output"].state == VertexState.DONE

    @pytest.mark.asyncio
    async def test_complex_scripts_run(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "complex", "config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("complex example config not found")

        g = Graph.from_json(config_path)
        # The uppercase handler should uppercase data on receive
        assert g.vertices["transform"]._script_module is not None
        assert g.vertices["merge"]._script_module is not None


# ── script-heavy pipeline ────────────────────────────────────────
class TestScriptPipeline:
    """Test a pipeline that exercises all script hooks."""

    @pytest.mark.asyncio
    async def test_full_script_lifecycle(self, tmp_path):
        # Vertex script: transforms and consolidates
        v_script = tmp_path / "v_hook.py"
        v_script.write_text(
            "def on_receive(data, data_id, tags, settings):\n"
            "    return data.strip() if isinstance(data, str) else data\n"
            "\n"
            "def on_ready(all_data, settings):\n"
            "    vals = [str(v) for v in all_data.values()]\n"
            "    return {('out', ('final',)): ' + '.join(vals)}\n"
        )

        # Edge script: wraps
        e_script = tmp_path / "e_hook.py"
        e_script.write_text(
            "def pre_process(data, settings):\n"
            "    return f'<{data}>'\n"
            "\n"
            "def post_process(data, settings):\n"
            "    return f'({data})'\n"
        )

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "d", "tags": [], "value": " hello "}]},
                {"id": "B", "script": str(v_script)},
                {"id": "C"},
            ],
            "edges": [
                {"id": "e1", "source": "A", "destination": "B",
                 "data_id": "d", "prompt": "p", "model": "m"},
                {"id": "e2", "source": "B", "destination": "C",
                 "data_id": "out", "tags": ["final"],
                 "prompt": "p", "model": "m",
                 "script": str(e_script)},
            ],
        }

        g = Graph.from_dict(config)
        echo = MockAgent(response_fn=lambda d, p, m, s: d)
        result = await Executor(g, echo, timeout=10).run()

        assert result.success
        # B received " hello " → on_receive strips → "hello"
        # B.on_ready merges → out:final = "hello"
        # e2: pre_process("<hello>") → echo → post_process("(<hello>)")
        c_data = result.vertex_results["C"]["data"]
        assert any("(<hello>)" in str(v) for v in c_data.values())


# ── rejection pipeline ───────────────────────────────────────────
class TestRejectionPipeline:
    """Test that data rejection in a vertex stops the pipeline gracefully."""

    @pytest.mark.asyncio
    async def test_rejection_causes_error(self, tmp_path):
        reject_script = tmp_path / "reject.py"
        reject_script.write_text(
            "def on_receive(data, data_id, tags, settings):\n"
            "    if isinstance(data, str) and 'bad' in data:\n"
            "        raise ValueError('contains bad word')\n"
            "    return data\n"
        )

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "d", "value": "bad data"}]},
                {"id": "B", "script": str(reject_script)},
            ],
            "edges": [
                {"id": "e", "source": "A", "destination": "B",
                 "data_id": "d", "prompt": "", "model": "m"},
            ],
        }

        g = Graph.from_dict(config)
        echo = MockAgent(response_fn=lambda d, p, m, s: d)
        result = await Executor(g, echo, timeout=10).run()

        assert not result.success
        assert any("bad word" in e for e in result.errors)


# ── multi-source fan-in ──────────────────────────────────────────
class TestMultiSourceFanIn:
    @pytest.mark.asyncio
    async def test_three_sources_one_sink(self):
        config = {
            "vertices": [
                {"id": "s1", "initial_data": [{"data_id": "d", "tags": ["1"], "value": "one"}]},
                {"id": "s2", "initial_data": [{"data_id": "d", "tags": ["2"], "value": "two"}]},
                {"id": "s3", "initial_data": [{"data_id": "d", "tags": ["3"], "value": "three"}]},
                {"id": "sink"},
            ],
            "edges": [
                {"id": "e1", "source": "s1", "destination": "sink",
                 "data_id": "d", "tags": ["1"], "prompt": "", "model": "m"},
                {"id": "e2", "source": "s2", "destination": "sink",
                 "data_id": "d", "tags": ["2"], "prompt": "", "model": "m"},
                {"id": "e3", "source": "s3", "destination": "sink",
                 "data_id": "d", "tags": ["3"], "prompt": "", "model": "m"},
            ],
        }

        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), timeout=10).run()

        assert result.success
        sink_data = result.vertex_results["sink"]["data"]
        assert len(sink_data) == 3


# ── deeply chained pipeline ──────────────────────────────────────
class TestDeepChain:
    @pytest.mark.asyncio
    async def test_10_vertex_chain(self):
        """Chain of 10 vertices, each transforming data."""
        n = 10
        config = {
            "vertices": [
                {"id": "v0", "initial_data": [{"data_id": "d", "value": "start"}]},
            ] + [{"id": f"v{i}"} for i in range(1, n)],
            "edges": [
                {"id": f"e{i}", "source": f"v{i}", "destination": f"v{i+1}",
                 "data_id": "d", "prompt": f"step-{i}", "model": "m"}
                for i in range(n - 1)
            ],
        }

        g = Graph.from_dict(config)
        counter = {"n": 0}

        def counting_fn(d, p, m, s):
            counter["n"] += 1
            return f"({d})"

        result = await Executor(g, MockAgent(response_fn=counting_fn), timeout=10).run()

        assert result.success
        assert counter["n"] == n - 1  # 9 edges

        # Final vertex should have deeply nested result
        last_data = result.vertex_results[f"v{n-1}"]["data"]
        val = list(last_data.values())[0]
        assert val.count("(") == n - 1
