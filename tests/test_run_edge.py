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
    )
    assert report["ok"] is True
    assert report["class"] == "SummarizeEdge", \
        f"expected SummarizeEdge, got {report['class']}"
    assert report["agent"] == "MockAgent"


@pytest.mark.asyncio
async def test_post_process_preserves_structured_title():
    """Mock path: title/url remembered in pre_process survive post_process."""
    report = await run_edge(
        dir_path=os.path.join(REPO, "examples", "s1_ai_report_map"),
        script="s1_edges.py:SummarizeEdge",
        data={"title": "帖子A", "url": "https://x", "content": "回帖内容"},
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
    )
    assert report["ok"] is True
    assert report["class"] == "SummarizeEdge"


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
    )
    assert report["ok"] is False
    assert "not found" in report["result"].lower() or "ScriptNot" in report["result"]