"""Unit and manual tests for S1 custom edges."""
import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework import Vertex, Edge, MockAgent
from examples.s1_ai_report.s1_edges import (
    FetchThreadsEdge, SelectEdge, FetchEdge, SummarizeEdge
)

MOCK_SELECTED = [
    {"tid": "2275806", "title": "GLM-5.3-Flash Released | LLM Discussion Megathread", "url": "https://stage1st.com/2b/thread-2275806-1-1.html"},
    {"tid": "2288716", "title": "Debating Theology and Philosophy with AI: Experiences and Insights", "url": "https://stage1st.com/2b/thread-2288716-1-1.html"},
]

@pytest.mark.asyncio
async def test_s1_select_edge():
    v_router = Vertex("v_router")
    await v_router.set_data("default", MOCK_SELECTED)
    edge = SelectEdge("e_sel1", "v_router", "v_sel1", settings={"index": 0})
    result = await edge.execute(v_router, Vertex("v_sel1"), MockAgent())
    assert result == MOCK_SELECTED[0]

@pytest.mark.asyncio
async def test_s1_summarize_edge():
    v_t = Vertex("v_t")
    thread_data = {"title": "Sample Title", "url": "https://example.com", "content": "Sample discussion content"}
    await v_t.set_data("default", thread_data)
    edge = SummarizeEdge(
        "e_sum", "v_t", "v_report",
        settings={
            "prompt": "Summarize: {data}",
            "model": "mock",
        }
    )
    result = await edge.execute(v_t, Vertex("v_report"), MockAgent(response_fn=lambda d, p, m, s: f"PROCESSED: {d}"))
    # SummarizeEdge attaches the original title/url structurally; the LLM body
    # goes into "summary".
    assert isinstance(result, dict)
    assert result["title"] == "Sample Title"
    assert result["url"] == "https://example.com"
    assert "Thread Title: Sample Title" in result["summary"]
    assert "PROCESSED" in result["summary"]
    assert edge.completed is True


async def run_fetch_threads():
    v = Vertex("v_forum", initial_data=[{"channel": "default", "value": "https://stage1st.com/2b/forum-157-1.html"}])
    edge = FetchThreadsEdge("e_fetch_threads", "v_forum", "v_threads")
    result = await edge.execute(v, Vertex("v_threads"), MockAgent())
    print(f"[e_fetch_threads] Got {len(result)} threads")
    for t in result[:3]:
        print(f"  - {t.get("title")}")
    return result

async def run_select(threads, index):
    v = Vertex("v_router")
    await v.set_data("default", threads)
    edge = SelectEdge(f"e_sel{index+1}", "v_router", f"v_sel{index+1}", settings={"index": index})
    result = await edge.execute(v, Vertex(f"v_sel{index+1}"), MockAgent())
    print(f"[e_sel{index+1}] {result['title']}")
    return result

async def run_fetch(thread):
    v = Vertex("v_sel")
    await v.set_data("default", thread)
    edge = FetchEdge("e_fetch", "v_sel", "v_t")
    result = await edge.execute(v, Vertex("v_t"), MockAgent())
    print(f"[e_fetch] {result['title']}: {len(result['content'])} chars")
    return result

async def run_summarize(thread_data):
    v = Vertex("v_t")
    await v.set_data("default", thread_data)
    edge = SummarizeEdge(
        "e_sum", "v_t", "v_report",
        settings={
            "prompt": "Here is the thread discussion from the last 24 hours:\n{data}\nPlease summarize the AI/LLM trends, user opinions, and key arguments in markdown.",
            "model": "hy3-free",
        }
    )
    result = await edge.execute(v, Vertex("v_report"), MockAgent())
    print(f"[e_sum] Summary: {len(result)} chars")
    print(result[:300])
    return result

async def main():
    print("=== Step 1: Fetch thread list ===")
    threads = await run_fetch_threads()

    print("\n=== Step 2: Select Nth thread ===")
    sel = await run_select(MOCK_SELECTED, 0)

    print("\n=== Step 3: Fetch thread content ===")
    fetched = await run_fetch(sel)

    print("\n=== Step 4: Summarize ===")
    await run_summarize(fetched)

if __name__ == "__main__":
    asyncio.run(main())
