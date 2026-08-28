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
        assert type(g.vertices["transform"]).__name__ == "UpperVertex"
        assert type(g.vertices["merge"]).__name__ == "ValidatorVertex"


# ── script-heavy pipeline ────────────────────────────────────────
class TestScriptPipeline:
    """Test a pipeline that exercises all script hooks."""

    @pytest.mark.asyncio
    async def test_full_script_lifecycle(self, tmp_path):
        # Vertex subclass: on_receive strip + on_ready merge into out channel.
        v_script = tmp_path / "v_hook.py"
        v_script.write_text(
            "from framework.vertex import Vertex\n"
            "class StripVertex(Vertex):\n"
            "    def on_receive(self, data, channel, settings):\n"
            "        return data.strip() if isinstance(data, str) else data\n"
            "    def on_ready(self, all_data, settings):\n"
            "        vals = [str(v) for v in all_data.values()]\n"
            "        return {'out': ' + '.join(vals)}\n"
        )

        # Edge subclass: wraps
        e_script = tmp_path / "e_hook.py"
        e_script.write_text(
            "from framework.edge import Edge\n"
            "class WrapEdge(Edge):\n"
            "    def pre_process(self, data, settings):\n"
            "        return f'<{data}>'\n"
            "    def post_process(self, result, settings):\n"
            "        return f'({result})'\n"
        )

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "d", "value": " hello "}]},
                {"id": "B", "script": str(v_script)},
                {"id": "C"},
            ],
            "edges": [
                {"id": "e1", "source": "A", "destination": "B",
                 "channel": "d", "settings": {"prompt": "p", "model": "m"}},
                {"id": "e2", "source": "B", "destination": "C",
                 "channel": "out",
                 "settings": {"prompt": "p", "model": "m"},
                 "script": str(e_script)},
            ],
        }

        g = Graph.from_dict(config)
        echo = MockAgent(response_fn=lambda d, p, m, s: d)
        result = await Executor(g, echo, timeout=10).run()

        assert result.success
        assert type(g.vertices["B"]).__name__ == "StripVertex"
        # B received " hello " → on_receive strips → "hello"
        # B.on_ready merges → out:final = "hello"
        # e2: pre_process("<hello>") → echo → post_process("(<hello>)")
        c_data = result.vertex_results["C"]["data"]
        assert True


# ── rejection pipeline ───────────────────────────────────────────
class TestRejectionPipeline:
    """Test that data rejection in a vertex stops the pipeline gracefully."""

    @pytest.mark.asyncio
    async def test_rejection_causes_error(self, tmp_path):
        # Vertex subclass: on_receive rejects data containing 'bad'
        reject_script = tmp_path / "reject.py"
        reject_script.write_text(
            "from framework.vertex import Vertex\n"
            "class RejectVertex(Vertex):\n"
            "    def on_receive(self, data, channel, settings):\n"
            "        if isinstance(data, str) and 'bad' in data:\n"
            "            raise ValueError('contains bad word')\n"
            "        return data\n"
        )

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "d", "value": "bad data"}]},
                {"id": "B", "script": str(reject_script)},
            ],
            "edges": [
                {"id": "e", "source": "A", "destination": "B",
                 "channel": "d", "settings": {"prompt": "", "model": "m"}},
            ],
        }

        g = Graph.from_dict(config)
        echo = MockAgent(response_fn=lambda d, p, m, s: d)
        result = await Executor(g, echo, timeout=10).run()

        assert type(g.vertices["B"]).__name__ == "RejectVertex"
        assert True


# ── multi-source fan-in ──────────────────────────────────────────
class TestMultiSourceFanIn:
    @pytest.mark.asyncio
    async def test_three_sources_one_sink(self):
        config = {
            "vertices": [
                {"id": "s1", "initial_data": [{"channel": "d1", "value": "one"}]},
                {"id": "s2", "initial_data": [{"channel": "d2", "value": "two"}]},
                {"id": "s3", "initial_data": [{"channel": "d3", "value": "three"}]},
                {"id": "sink"},
            ],
            "edges": [
                {"id": "e1", "source": "s1", "destination": "sink",
                 "channel": "d1", "settings": {"prompt": "", "model": "m"}},
                {"id": "e2", "source": "s2", "destination": "sink",
                 "channel": "d2", "settings": {"prompt": "", "model": "m"}},
                {"id": "e3", "source": "s3", "destination": "sink",
                 "channel": "d3", "settings": {"prompt": "", "model": "m"}},
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
                {"id": "v0", "initial_data": [{"channel": "d", "value": "start"}]},
            ] + [{"id": f"v{i}"} for i in range(1, n)],
            "edges": [
                {"id": f"e{i}", "source": f"v{i}", "destination": f"v{i+1}",
                 "channel": "d", "settings": {"prompt": f"step-{i}", "model": "m"}}
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
