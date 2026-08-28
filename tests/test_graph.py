"""Tests for framework.graph."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.graph import Graph


# ── loading ──────────────────────────────────────────────────────
class TestGraphLoading:
    def test_from_dict_linear(self, linear_config):
        g = Graph.from_dict(linear_config)
        assert len(g.vertices) == 3
        assert len(g.edges) == 2

    def test_from_dict_diamond(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        assert len(g.vertices) == 4
        assert len(g.edges) == 4

    def test_from_json_file(self, linear_config, tmp_json):
        path = tmp_json(linear_config)
        g = Graph.from_json(path)
        assert len(g.vertices) == 3

    def test_metadata(self, linear_config):
        g = Graph.from_dict(linear_config)
        assert g.metadata["name"] == "linear"

    def test_edge_registration(self, linear_config):
        g = Graph.from_dict(linear_config)
        assert "e1" in g.vertices["A"].outgoing_edges
        assert "e1" in g.vertices["B"].incoming_edges


# ── validation ───────────────────────────────────────────────────
class TestGraphValidation:
    def test_cycle_rejected(self, cycle_config):
        with pytest.raises(ValueError, match="cycle"):
            Graph.from_dict(cycle_config)

    def test_missing_source_vertex(self):
        config = {
            "vertices": [{"id": "B"}],
            "edges": [
                {"id": "e", "source": "MISSING", "destination": "B",
                 "data_id": "x", "prompt": "", "model": "m"},
            ],
        }
        with pytest.raises(ValueError, match="source"):
            Graph.from_dict(config)

    def test_missing_dest_vertex(self):
        config = {
            "vertices": [{"id": "A"}],
            "edges": [
                {"id": "e", "source": "A", "destination": "MISSING",
                 "data_id": "x", "prompt": "", "model": "m"},
            ],
        }
        with pytest.raises(ValueError, match="destination"):
            Graph.from_dict(config)

    def test_valid_dag_passes(self, diamond_config):
        # Should not raise
        g = Graph.from_dict(diamond_config)
        assert g is not None


# ── queries ──────────────────────────────────────────────────────
class TestGraphQueries:
    def test_source_vertices(self, linear_config):
        g = Graph.from_dict(linear_config)
        sources = g.get_source_vertices()
        assert len(sources) == 1
        assert sources[0].id == "A"

    def test_sink_vertices(self, linear_config):
        g = Graph.from_dict(linear_config)
        sinks = g.get_sink_vertices()
        assert len(sinks) == 1
        assert sinks[0].id == "C"

    def test_diamond_sources_and_sinks(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        assert len(g.get_source_vertices()) == 1  # A
        assert len(g.get_sink_vertices()) == 1     # D

    def test_outgoing_edges(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        out_a = g.get_outgoing_edges("A")
        assert len(out_a) == 2

    def test_incoming_edges(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        in_d = g.get_incoming_edges("D")
        assert len(in_d) == 2

    def test_required_input_count(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        assert g.vertices["D"].required_input_count == 2
        assert g.vertices["A"].required_input_count == 0


# ── scripts ──────────────────────────────────────────────────────
class TestGraphScripts:
    def test_vertex_script_loaded(self, tmp_path):
        script = tmp_path / "vs.py"
        script.write_text(
            "from framework.vertex import Vertex\n"
            "class MyVertex(Vertex):\n"
            "    def on_receive(self, data, channel, settings):\n"
            "        return data\n"
        )

        config = {
            "vertices": [{"id": "A", "script": str(script)}],
            "edges": [],
        }
        g = Graph.from_dict(config)
        assert type(g.vertices["A"]).__name__ == "MyVertex"

    def test_edge_script_loaded(self, tmp_path):
        script = tmp_path / "es.py"
        script.write_text(
            "from framework.edge import Edge\n"
            "class MyEdge(Edge):\n"
            "    def pre_process(self, data, settings):\n"
            "        return data\n"
        )

        config = {
            "vertices": [{"id": "A"}, {"id": "B"}],
            "edges": [
                {"id": "e", "source": "A", "destination": "B",
                 "data_id": "x", "prompt": "", "model": "m",
                 "pipeline": str(script)},
            ],
        }
        g = Graph.from_dict(config)
        assert type(g.edges["e"]).__name__ == "MyEdge"

    def test_missing_script_does_not_crash(self):
        config = {
            "vertices": [{"id": "A", "script": "/nonexistent/path.py"}],
            "edges": [],
        }
        # Based on user requirement: "如果有问题直接error", we now expect RuntimeError
        import pytest
        with pytest.raises(RuntimeError):
            g = Graph.from_dict(config)
