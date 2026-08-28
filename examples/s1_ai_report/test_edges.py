"""Test individual edges for debugging."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework import Vertex, Edge, MockAgent
from examples.s1_ai_report.s1_pipelines import (
    FetchThreadsPipeline, SelectPipeline, FetchPipeline, SummarizePipeline
)

MOCK_SELECTED = [
    {"tid": "2275806", "title": "Ox Alpha被认领，GLM-5.3-Flash上线|大模型讨论专楼", "url": "https://stage1st.com/2b/thread-2275806-1-1.html"},
    {"tid": "2288716", "title": "和AI辩（圣）经，其乐无穷，收获颇深", "url": "https://stage1st.com/2b/thread-2288716-1-1.html"},
]

async def test_fetch_threads():
    v = Vertex("v_forum", initial_data=[{"channel": "default", "value": "https://stage1st.com/2b/forum-157-1.html"}])
    edge = Edge("e_fetch_threads", "v_forum", "v_threads")
    edge.set_pipeline_module(FetchThreadsPipeline)
    result = await edge.execute(v, Vertex("v_threads"), MockAgent())
    print(f"[e_fetch_threads] Got {len(result)} threads")
    for t in result[:3]:
        print(f"  - {t['title']}")
    return result

async def test_select(threads, index):
    v = Vertex("v_router")
    await v.set_data("default", threads)
    edge = Edge(f"e_sel{index+1}", "v_router", f"v_sel{index+1}", settings={"index": index})
    edge.set_pipeline_module(SelectPipeline)
    result = await edge.execute(v, Vertex(f"v_sel{index+1}"), MockAgent())
    print(f"[e_sel{index+1}] {result['title']}")
    return result

async def test_fetch(thread):
    v = Vertex("v_sel")
    await v.set_data("default", thread)
    edge = Edge("e_fetch", "v_sel", "v_t")
    edge.set_pipeline_module(FetchPipeline)
    result = await edge.execute(v, Vertex("v_t"), MockAgent())
    print(f"[e_fetch] {result['title']}: {len(result['content'])} chars")
    return result

async def test_summarize(thread_data):
    v = Vertex("v_t")
    await v.set_data("default", thread_data)
    edge = Edge("e_sum", "v_t", "v_report",
                prompt="以下是论坛帖子最近24小时的讨论内容：\n{data}\n请总结AI/LLM趋势、用户观点和关键论点。输出markdown。",
                model="hy3-free")
    edge.set_pipeline_module(SummarizePipeline)
    result = await edge.execute(v, Vertex("v_report"), MockAgent())
    print(f"[e_sum] Summary: {len(result)} chars")
    print(result[:300])
    return result

async def main():
    print("=== Step 1: Fetch thread list ===")
    threads = await test_fetch_threads()

    print("\n=== Step 2: Select Nth thread ===")
    sel = await test_select(MOCK_SELECTED, 0)

    print("\n=== Step 3: Fetch thread content ===")
    fetched = await test_fetch(sel)

    print("\n=== Step 4: Summarize ===")
    await test_summarize(fetched)

if __name__ == "__main__":
    asyncio.run(main())
