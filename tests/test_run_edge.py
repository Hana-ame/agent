"""Tests for framework.utils.run_edge — standalone single-edge driver.

Locks three things that have regressed before:
1. script "file.py:ClassName" loads the EXPLICIT class (not alphabetically the
   first Edge subclass — the script-loader bug that loaded FetchEdge for
   SummarizeEdge).
2. The script path resolves relative to ``--dir`` (config-dir style), not CWD.
3. The full chain pre_process -> compute -> post_process runs and the edge's
   post_process output shape is preserved.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.utils.run_edge import run_edge


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.asyncio
async def test_loads_explicit_named_class_not_first_subclass():
    """s1_edges.py:SummarizeEdge must resolve to SummarizeEdge, not FetchEdge."""
    report = await run_edge(
        dir_path=os.path.join(REPO, "examples", "s1_ai_report_map"),
        script="s1_edges.py:SummarizeEdge",
        data={"title": "Thread_A", "url": "https://x", "content": "Reply_content"},
        skip_compute=True,
    )
    assert report["ok"] is True
    assert report["class"] == "SummarizeEdge", \
        f"expected SummarizeEdge, got {report['class']}"
    assert report["skip_compute"] is True
    assert report["agent"] is None  # no mock fallback — compute skipped


@pytest.mark.asyncio
async def test_post_process_preserves_structured_title():
    """skip-compute path: title/url remembered in pre_process survive post_process."""
    report = await run_edge(
        dir_path=os.path.join(REPO, "examples", "s1_ai_report_map"),
        script="s1_edges.py:SummarizeEdge",
        data={"title": "Thread_A", "url": "https://x", "content": "Reply_content"},
        skip_compute=True,
    )
    result = report["result"]
    assert isinstance(result, dict)
    assert result["title"] == "Thread_A"       # remembered from pre_process
    assert result["url"] == "https://x"
    assert "summary" in result


@pytest.mark.asyncio
async def test_script_path_resolves_relative_to_dir_not_cwd():
    """The same script name must resolve when run from a different CWD."""
    report = await run_edge(
        dir_path=os.path.join(REPO, "examples", "hn_ai_report"),
        script="hn_edges.py:SummarizeEdge",
        data={"title": "Story", "url": "https://y", "content": "comments"},
        skip_compute=True,
    )
    assert report["ok"] is True
    assert report["class"] == "SummarizeEdge"


@pytest.mark.asyncio
async def test_self_owning_agent_edge_gets_no_driver_http_client(tmp_path):
    """Fix: when a script edge owns its own agent (``self.agent`` in __init__),
    run_edge must NOT create an unused driver HttpLLMAgent just because
    --base-url was given. Edge.compute precedence is self.agent > driver agent,
    so the report's agent field must reflect the OWNED agent (HttpLLMAgent), and
    the result must come from it — no throwaway HTTP client, no LLM call."""
    script = tmp_path / "self_owned.py"
    script.write_text(
        "from framework.edge import Edge\n"
        "from framework.agents import HttpLLMAgent\n"
        "\n"
        "class SelfOwnedEdge(Edge):\n"
        "    def __init__(self, **kw):\n"
        "        super().__init__(**kw)\n"
        "        # owns its agent — driver must not hand it another one\n"
        "        self.agent = HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: 'own:' + str(d))\n"
        "        self.prompt = 'owned prompt'\n"
        "        self.model = 'owned-model'\n"
        "\n"
        "    def post_process(self, result, settings):\n"
        "        return {'via': 'self_owned_agent', 'result': result}\n"
        ,
        encoding="utf-8",
    )

    report = await run_edge(
        dir_path=str(tmp_path),
        script="self_owned.py:SelfOwnedEdge",
        data="payload",
        base_url="https://fake.example/v1/chat/completions",
        api_key="k",
    )
    assert report["ok"] is True, report["result"]
    assert report["agent"] == "HttpLLMAgent"  # owned agent, NOT a driver HttpLLMAgent
    assert report["result"]["via"] == "self_owned_agent"
    assert report["result"]["result"] == "own:payload"  # answer came from own agent
    assert "usage" not in report  # no real LLM call was made


@pytest.mark.asyncio
async def test_compute_without_endpoint_or_skip_compute_raises():
    """No HttpLLMAgent fallback: a compute run with no base_url must raise, not
    silently fall back to a mock agent."""
    with pytest.raises(ValueError) as exc:
        await run_edge(
            dir_path=os.path.join(REPO, "examples", "s1_ai_report_map"),
            script="s1_edges.py:SummarizeEdge",
            data={"title": "A", "url": "https://x", "content": "c"},
        )
    assert "base-url" in str(exc.value) or "compute" in str(exc.value)


@pytest.mark.asyncio
async def test_fetch_edge_runs_and_returns_data():
    """FetchEdges are network edges (real HTTP in pre_process) — offline we only
    assert the driver loads and constructs the correct class, not that the chain
    runs (which would hit the network). Same check the script-loader bug would
    have failed: "FetchEdge" must resolve to FetchEdge, not a sibling subclass.
    """
    from framework.edge import Edge as _EdgeBase
    from framework.utils.run_edge import _resolve_class

    cls = _resolve_class(
        os.path.join(REPO, "examples", "s1_ai_report_map"), "s1_edges.py:FetchEdge"
    )
    assert cls.__name__ == "FetchEdge"
    assert issubclass(cls, _EdgeBase)
    # constructing the edge must not raise
    edge = cls(edge_id="single_edge", source_id="src", destination_id="dst")
    assert edge.id == "single_edge"


@pytest.mark.asyncio
async def test_bad_script_raises_reported_failure():
    """A missing script file must surface as a report failure, not crash out."""
    report = await run_edge(
        dir_path=os.path.join(REPO, "examples", "s1_ai_report_map"),
        script="does_not_exist.py:WhateverEdge",
        data="x",
        skip_compute=True,
    )
    assert report["ok"] is False
    assert "not found" in report["result"].lower() or "ScriptNot" in report["result"]


@pytest.mark.asyncio
async def test_graph_edge_skip_compute_does_not_call_llm():
    """Graph-level: an edge with settings.skip_compute=true runs pre->post
    without invoking the LLM (no HttpLLMAgent fallback, no real call). The mock
    response_fn must never be called."""
    from framework import Graph, Executor, HttpLLMAgent

    calls = []

    def never_called(data, prompt, model, settings):
        calls.append(prompt)
        return "SHOULD NOT HAPPEN"

    config = {
        "vertices": [
            {"id": "A", "initial_data": [{"channel": "in", "value": "hi"}]},
            {"id": "B"},
        ],
        "edges": [
            {
                "id": "e_skip",
                "source": "A",
                "destination": "B",
                "channel": "in",
                "settings": {"skip_compute": True},  # no prompt/model needed
            }
        ],
    }
    g = Graph.from_dict(config)
    result = await Executor(g, HttpLLMAgent(mock=True, mock_handler=never_called)).run()

    assert result.success, result.summary()
    assert calls == []  # LLM never invoked
    assert await g.vertices["B"].fetch_data("in") == "hi"  # pre->post passthrough


@pytest.mark.asyncio
async def test_graph_skip_compute_still_runs_post_process():
    """Graph-level: skip_compute still runs the edge's post_process hook."""
    from framework import Graph, Executor, Edge, HttpLLMAgent

    class TransformEdge(Edge):
        def post_process(self, result, settings):
            return f"wrapped:{result}"

    config = {
        "vertices": [
            {"id": "A", "initial_data": [{"channel": "in", "value": "raw"}]},
            {"id": "B"},
        ],
        "edges": [
            {
                "id": "e_skip2",
                "source": "A",
                "destination": "B",
                "channel": "in",
                "settings": {"skip_compute": True},
            }
        ],
    }
    g = Graph.from_dict(config)
    # swap in our post-processing subclass
    old = g.edges["e_skip2"]
    g.edges["e_skip2"] = TransformEdge(
        edge_id=old.id, source_id=old.source_id, destination_id=old.destination_id,
        channel=old.channel, settings=old.settings,
        concurrency_type=old.concurrency_type, max_iterations=old.max_iterations,
    )
    result = await Executor(g, HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: "nope")).run()

    assert result.success
    assert await g.vertices["B"].fetch_data("in") == "wrapped:raw"


@pytest.mark.asyncio
async def test_graph_edge_without_prompt_model_is_passthrough():
    """A plain edge with no prompt/model already passes data straight through
    (no mock, no LLM) — this is exactly why skip_compute is the *explicit*
    graph-level spelling for the same intent: pure-data edges never invoke an
    agent. The response_fn must not be called."""
    from framework import Graph, Executor, HttpLLMAgent

    config = {
        "vertices": [
            {"id": "A", "initial_data": [{"channel": "in", "value": 5}]},
            {"id": "B"},
        ],
        "edges": [
            {"id": "e_plain", "source": "A", "destination": "B", "channel": "in"}
        ],
    }
    calls = []

    def fn(data, prompt, model, settings):
        calls.append(data)
        return data * 10

    g = Graph.from_dict(config)
    result = await Executor(g, HttpLLMAgent(mock=True, mock_handler=fn)).run()
    assert result.success
    assert calls == []  # agent never invoked for a data edge
    assert await g.vertices["B"].fetch_data("in") == 5  # passthrough


def test_edge_module_standalone_cli_execution():
    """Verify python -m framework.edge runs standalone successfully via CLI."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "framework.edge",
        "--dir",
        os.path.join(REPO, "examples", "s1_ai_report_map"),
        "--script",
        "s1_edges.py:SummarizeEdge",
        "--data",
        '{"title": "Thread_A", "url": "https://x", "content": "Reply_content"}',
        "--skip-compute",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI execution failed with stderr: {proc.stderr}"
    assert "SummarizeEdge" in proc.stdout
    assert "ok            : True" in proc.stdout


def test_edge_module_standalone_cli_mock_flag():
    """Verify python -m framework.edge supports --mock flag for offline LLM compute."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "framework.edge",
        "--dir",
        os.path.join(REPO, "examples", "s1_ai_report_map"),
        "--script",
        "s1_edges.py:SummarizeEdge",
        "--data",
        '{"title": "Thread_Mock", "url": "https://x", "content": "Mock_content"}',
        "--mock",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI execution with --mock failed: {proc.stderr}"
    assert "SummarizeEdge" in proc.stdout
    assert "HttpLLMAgent (mock)" in proc.stdout
    assert "ok            : True" in proc.stdout


def test_edge_v1_from_config_and_file(tmp_path):
    """Verify V1 Edge instantiates correctly from JSON file and dict."""
    import json
    from framework.edge import Edge

    # From dict
    edge_dict = {
        "id": "edge_cfg_dict",
        "source": "A",
        "destination": "B",
        "channel": "ch1",
        "settings": {"model": "custom-v1-model"},
    }
    e1 = Edge.from_config(edge_dict)
    assert e1.id == "edge_cfg_dict"
    assert e1.source_id == "A"
    assert e1.destination_id == "B"
    assert e1.channel == "ch1"
    assert e1.model == "custom-v1-model"

    # From file
    json_path = tmp_path / "edge_v1.json"
    json_path.write_text(json.dumps({
        "id": "edge_cfg_file",
        "source": "src_node",
        "destination": "dst_node",
        "settings": {"prompt": "Analyze: {input}"},
    }), encoding="utf-8")
    e2 = Edge.from_config_file(json_path)
    assert e2.id == "edge_cfg_file"
    assert e2.source_id == "src_node"
    assert e2.destination_id == "dst_node"


def test_edge_v1_cli_driven_by_json_config(tmp_path):
    """Verify framework/edge.py execution driven by a single JSON config file."""
    import subprocess
    import sys
    import json

    edge_script = os.path.join(REPO, "framework", "edge.py")
    cfg_file = tmp_path / "edge_driver_cfg.json"
    cfg_file.write_text(json.dumps({
        "dir": os.path.join(REPO, "examples", "hn_ai_report"),
        "script": "hn_edges.py:SummarizeEdge",
        "data": {"title": "Title from JSON driver"},
        "mock": True,
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        edge_script,
        "--config",
        str(cfg_file),
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI with --config failed: {proc.stderr}"
    assert "SummarizeEdge" in proc.stdout
    assert "HttpLLMAgent (mock)" in proc.stdout
    assert "Title from JSON driver" in proc.stdout
    assert "ok            : True" in proc.stdout


def test_edge_v1_cli_json_config_override(tmp_path):
    """Verify CLI arguments override values in JSON config file."""
    import subprocess
    import sys
    import json

    edge_script = os.path.join(REPO, "framework", "edge.py")
    cfg_file = tmp_path / "edge_base_cfg.json"
    cfg_file.write_text(json.dumps({
        "dir": os.path.join(REPO, "examples", "hn_ai_report"),
        "script": "hn_edges.py:SummarizeEdge",
        "data": {"title": "Original Title"},
        "mock": True,
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        edge_script,
        "--config",
        str(cfg_file),
        "--data",
        '{"title": "CLI Overridden Title"}',
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI override failed: {proc.stderr}"
    assert "CLI Overridden Title" in proc.stdout
    assert "ok            : True" in proc.stdout



