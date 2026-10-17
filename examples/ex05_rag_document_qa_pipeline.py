"""Example 05: RAG Document Chunking & Question Answering Pipeline DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 05: RAG Document QA Pipeline DAG ===")
    dag = DAG("rag_pipeline_dag", models="models.json", llm_client=MockLLMClient())

    # Edge 0: Chunk document into semantic segments
    dag.add_edge(
        input_tags=["raw_document"],
        processor=lambda doc: f"Chunk1: SLA guarantees 99.99% uptime. Chunk2: Penalty is 10% credit. Source: {doc[:30]}",
        output_tags=["doc_chunks"],
        layer=0,
    )

    # Edge 1: Query retrieval match
    dag.add_edge(
        input_tags=["user_query", "doc_chunks"],
        processor=lambda inputs: f"RetrievedChunk: SLA guarantees 99.99% uptime for query {inputs}",
        output_tags=["retrieved_context"],
        layer=1,
    )

    # Edge 2: Synthesis QA
    dag.add_edge(
        input_tags=["retrieved_context"],
        processor=lambda ctx: f"FinalAnswer: 服务等级协议保证全年 99.99% 高可用性。依据: {ctx[:50]}",
        output_tags=["final_qa_response"],
        layer=2,
    )

    dag.add_content("企业级云服务客户服务协议与违约条款...", tags=["raw_document"], gen=0)
    dag.add_content("我们的服务可用性指标承诺是多少？", tags=["user_query"], gen=0)

    dag.run(until_tag="final_qa_response", max_steps=6)
    ans = dag.get_result("final_qa_response")
    print(f"RAG QA Response: {ans}")
    assert "99.99%" in ans
    print("Example 05 Passed Successfully!")


if __name__ == "__main__":
    main()
