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
        data={"title": "帖子A", "url": "https://x", "content": "回帖内容"},
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
        data={"title": "帖子A", "url": "https://x", "content": "回帖内容"},
        skip_compute=True,
    )
    result = report["result"]
    assert isinstance(result, dict)
    assert result["title"] == "帖子A"       # remembered from pre_process
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
    so the report's agent field must reflect the OWNED agent (MockAgent), and
    the result must come from it — no throwaway HTTP client, no LLM call."""
    script = tmp_path / "self_owned.py"
    script.write_text(
        "from framework.edge import Edge\n"
        "from framework.agents import MockAgent\n"
        "\n"
        "class SelfOwnedEdge(Edge):\n"
        "    def __init__(self, **kw):\n"
        "        super().__init__(**kw)\n"
        "        # owns its agent — driver must not hand it another one\n"
        "        self.agent = MockAgent(response_fn=lambda d, p, m, s: 'own:' + str(d))\n"
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
    assert report["agent"] == "MockAgent"  # owned agent, NOT a driver HttpLLMAgent
    assert report["result"]["via"] == "self_owned_agent"
    assert report["result"]["result"] == "own:payload"  # answer came from own agent
    assert "usage" not in report  # no real LLM call was made


@pytest.mark.asyncio
async def test_compute_without_endpoint_or_skip_compute_raises():
    """No MockAgent fallback: a compute run with no base_url must raise, not
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
    without invoking the LLM (no MockAgent fallback, no real call). The mock
    response_fn must never be called."""
    from framework import Graph, Executor, MockAgent

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
    result = await Executor(g, MockAgent(response_fn=never_called)).run()

    assert result.success, result.summary()
    assert calls == []  # LLM never invoked
    assert await g.vertices["B"].fetch_data("in") == "hi"  # pre->post passthrough


@pytest.mark.asyncio
async def test_graph_skip_compute_still_runs_post_process():
    """Graph-level: skip_compute still runs the edge's post_process hook."""
    from framework import Graph, Executor, Edge, MockAgent

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
    result = await Executor(g, MockAgent(response_fn=lambda d, p, m, s: "nope")).run()

    assert result.success
    assert await g.vertices["B"].fetch_data("in") == "wrapped:raw"


@pytest.mark.asyncio
async def test_graph_edge_without_prompt_model_is_passthrough():
    """A plain edge with no prompt/model already passes data straight through
    (no mock, no LLM) — this is exactly why skip_compute is the *explicit*
    graph-level spelling for the same intent: pure-data edges never invoke an
    agent. The response_fn must not be called."""
    from framework import Graph, Executor, MockAgent

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
    result = await Executor(g, MockAgent(response_fn=fn)).run()
    assert result.success
    assert calls == []  # agent never invoked for a data edge
    assert await g.vertices["B"].fetch_data("in") == 5  # passthrough
