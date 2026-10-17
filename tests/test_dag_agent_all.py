"""Comprehensive Test Suite for DAG Agent Framework & Versions A through G.

Tests:
1. JSON Loading & Python Context Read/Modify/Write (`models.json`, `contents.json`, `edges.json`)
2. Version A: Exact Tag Matching, `gen + 1` progression, `layer` (0-first) & `gen` (0-first) priority, `count` increment
3. Version B: Semantic / Embedding Threshold Matching (`cosine_similarity >= threshold`)
4. Version C (Mode 1): `default_processor` N-in-1-out Targeted Puzzle Assembly prioritizing smaller `edge.count`
5. Version C (Mode 2): `default_processor` Autonomous Free Routing (choosing N inputs + output_tags) with strictly Clean Context on formal output request
6. Version D: Meta-DAG (`create_meta_dag` / `modify_with_prompt`) modifying target DAG JSON & self-modification
7. Version E: Async Step & Parallel Frontier Execution (`step_parallel_frontier`)
8. Version F: Transactional Checkpoints, Rollback on Failure, and `max_edge_count` Guard
9. Version G: Full Production End-to-End Pipeline, Trace Telemetry & Mermaid Export
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from dag_agent import (
    DAG,
    ContentItem,
    DefaultProcessor,
    EdgeSpec,
    MockLLMClient,
    Processor,
    VersionA_MinimalDAG,
    VersionB_SemanticEmbeddingDAG,
    VersionC_CleanContextDefaultDAG,
    VersionD_MetaSelfModifyingDAG,
    VersionE_AsyncParallelDAG,
    VersionF_TransactionalReactiveDAG,
    VersionG_ProductionCompleteDAG,
    create_meta_dag,
    transform_dag_with_meta_dag,
)


class TestDAGAgentFramework(unittest.TestCase):
    """End-to-end verification of all core requirements and Versions A through G."""

    def test_01_json_files_and_python_context_read_modify_write(self) -> None:
        """Verify `models.json`, `contents.json`, and `edges.json` loading, Python context mutation, and saving."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            models_data = {
                "fast_llm": {
                    "endpoint": "https://api.example.com/v1",
                    "apikey": "sk-test-123",
                    "model": "deepseek-v4-flash",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
            }
            contents_data = [
                {"tags": ["raw_doc"], "contents": "Initial text", "gen": 0},
            ]
            edges_data = [
                {
                    "input_tags": ["raw_doc"],
                    "processor": {"model": "fast_llm", "prompt": "Summarize: {contents}"},
                    "output_tags": ["summary"],
                    "layer": 0,
                    "count": 0,
                }
            ]

            (tmp_dir / "models.json").write_text(json.dumps(models_data), encoding="utf-8")
            (tmp_dir / "contents.json").write_text(json.dumps(contents_data), encoding="utf-8")
            (tmp_dir / "edges.json").write_text(json.dumps(edges_data), encoding="utf-8")

            dag = DAG.from_directory(tmp_dir)
            self.assertIn("fast_llm", dag.models)
            self.assertEqual(dag.models["fast_llm"].endpoint, "https://api.example.com/v1")
            self.assertEqual(dag.models["fast_llm"].apikey, "sk-test-123")
            self.assertEqual(dag.models["fast_llm"].params["temperature"], 0.1)

            # Read, modify, and append in Python context
            with dag.contents.edit(auto_save=True) as store:
                self.assertEqual(store[0]["contents"], "Initial text")
                store[0].contents = "Updated text in Python context"
                store[0].tags.append("verified")
                store.add({"tags": ["extra"], "contents": "Second item", "gen": 0})

            reloaded = json.loads((tmp_dir / "contents.json").read_text(encoding="utf-8"))
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded[0]["contents"], "Updated text in Python context")
            self.assertIn("verified", reloaded[0]["tags"])

    def test_02_version_a_minimal_priority_layer_gen_and_count(self) -> None:
        """Version A: Verify `layer` (0 highest priority), `gen` (smaller first), `gen + 1`, and `edge.count`."""
        dag = VersionA_MinimalDAG()
        # Add edges with different layers (layer 0 must run before layer 2)
        dag.add_edge(["task"], lambda c: f"L2:{c}", ["out_l2"], layer=2)
        dag.add_edge(["task"], lambda c: f"L0:{c}", ["out_l0"], layer=0)
        # Edge on ["item"] to test `gen` ordering (gen=0 must run before gen=3)
        dag.add_edge(["item"], lambda c: f"processed:{c}", ["item_done"], layer=1)

        dag.add_content("Task payload", tags=["task"], gen=0)
        dag.add_content("Late gen item", tags=["item"], gen=3)
        dag.add_content("Early gen item", tags=["item"], gen=0)

        # Step 1: Must pick layer=0 edge on ["task"]
        r1 = dag.step()
        self.assertEqual(r1.edge_layer, 0)
        self.assertEqual(r1.produced_items[0].tags, ["out_l0"])
        self.assertEqual(r1.produced_items[0].gen, 1)
        self.assertEqual(dag.edges[1].count, 1)

        # Step 2: Next best layer is layer=1 on ["item"]; must pick gen=0 item before gen=3 item!
        r2 = dag.step()
        self.assertEqual(r2.edge_layer, 1)
        self.assertEqual(r2.produced_items[0].contents, "processed:Early gen item")
        self.assertEqual(r2.produced_items[0].gen, 1)  # 0 + 1
        self.assertEqual(dag.edges[2].count, 1)

        # Step 3: Still on layer=1 ["item"], now processes the gen=3 item
        r3 = dag.step()
        self.assertEqual(r3.edge_layer, 1)
        self.assertEqual(r3.produced_items[0].contents, "processed:Late gen item")
        self.assertEqual(r3.produced_items[0].gen, 4)  # 3 + 1
        self.assertEqual(dag.edges[2].count, 2)  # Executed twice!

        # Step 4: Finally runs layer=2 edge on ["task"]
        r4 = dag.step()
        self.assertEqual(r4.edge_layer, 2)
        self.assertEqual(r4.produced_items[0].tags, ["out_l2"])
        self.assertEqual(dag.edges[0].count, 1)

    def test_03_version_b_semantic_and_embedding_threshold_matching(self) -> None:
        """Version B: Verify semantic/embedding similarity threshold matching when tags are close but not identical."""
        mock = MockLLMClient(
            custom_embeddings={
                "financial_revenue": [0.95, 0.05, 0.0],
                "sales_income": [0.92, 0.08, 0.0],
                "unrelated_weather": [0.0, 0.0, 1.0],
            }
        )
        dag = VersionB_SemanticEmbeddingDAG(llm_client=mock, similarity_threshold=0.85)
        dag.add_edge(
            ["financial_revenue"],
            Processor(model="default", prompt="Analyze revenue: {contents}"),
            ["result"],
            layer=0,
        )

        # Feed an item with tag "sales_income" (close in embedding space to "financial_revenue", cosine > 0.99)
        dag.add_content("Q3 Revenue = $12M", tags=["sales_income"], gen=0)

        explanation = dag.explain_tag_match("financial_revenue", "sales_income")
        self.assertTrue(explanation["matched"])
        self.assertGreaterEqual(explanation["similarity"], 0.85)

        step_res = dag.step()
        self.assertEqual(step_res.action, "edge")
        self.assertTrue(dag.has_tag("result"))
        self.assertEqual(dag.get_latest("result").gen, 1)

    def test_04_version_c_default_processor_targeted_puzzle_prioritizes_low_count(self) -> None:
        """Version C (Mode 1): `default_processor` N-in-1-out assembly prioritizes edges with smaller `count`."""
        dag = VersionC_CleanContextDefaultDAG(similarity_threshold=0.95)

        # Edge 0 already has count=3; Edge 1 has count=0. Both need ["rev", "cost"] which must be pieced together.
        dag.add_edge(["rev", "cost"], lambda c: f"Edge0_Out({c})", ["e0_done"], layer=0, count=3)
        dag.add_edge(["rev", "cost"], lambda c: f"Edge1_Out({c})", ["result"], layer=1, count=0)

        # Provide separate pieces: neither single item matches ["rev", "cost"] directly
        dag.add_content('{"rev": 500}', tags=["rev"], gen=0)
        dag.add_content('{"cost": 300}', tags=["cost"], gen=1)

        step_res = dag.step()
        self.assertEqual(step_res.action, "default_targeted_assembly")
        # Must prioritize Edge 1 because Edge 1 has count=0 < Edge 0's count=3!
        self.assertEqual(step_res.edge_id, "edge_1")
        self.assertEqual(dag.edges[1].count, 1)
        self.assertTrue(dag.has_tag("result"))
        # Gen of synthesized item is max(0, 1) + 1 = 2; gen of Edge 1 output is 2 + 1 = 3
        result_item = dag.get_latest("result")
        self.assertEqual(result_item.gen, 3)
        self.assertIn("500", result_item.contents)
        self.assertIn("300", result_item.contents)

    def test_05_version_c_autonomous_default_processor_and_clean_context_guarantee(self) -> None:
        """Version C (Mode 2): When no edge can run, `default_processor` selects inputs & output_tags,
        and the formal output LLM call is guaranteed to have a 100% clean context!
        """
        mock = MockLLMClient(
            rules={
                # Phase 1 routing response selects indices [0, 2] and decides output_tags ["custom_insight"]
                "selected_indices": '{"selected_indices": [0, 2], "output_tags": ["custom_insight"]}',
                # Phase 2 formal synthesis response
                "请将以下输入信息整合": "Cleanly synthesized from item 0 and item 2",
            }
        )
        dag = VersionC_CleanContextDefaultDAG(llm_client=mock, similarity_threshold=0.99)
        # No edges added (or only an unreachable edge), forcing Mode 2 autonomous default_processor
        dag.add_content("Selected Alpha Data", tags=["alpha"], gen=0)
        dag.add_content("NOISE_UNSELECTED_DISTRACTOR_DATA", tags=["noise"], gen=0)
        dag.add_content("Selected Beta Data", tags=["beta"], gen=2)

        step_res = dag.step()
        self.assertEqual(step_res.action, "default_autonomous")
        produced = step_res.produced_items[0]
        self.assertEqual(produced.tags, ["custom_insight"])
        self.assertEqual(produced.gen, 3)  # max(0, 2) + 1 = 3
        self.assertEqual(produced.contents, "Cleanly synthesized from item 0 and item 2")

        # Verify Phase 1 (routing) vs Phase 2 (clean output) call isolation!
        routing_calls = dag.get_routing_calls()
        clean_calls = dag.get_clean_output_calls()
        self.assertEqual(len(routing_calls), 1)
        self.assertEqual(len(clean_calls), 1)

        # Formal output request MUST contain selected items and MUST NOT contain unselected noise or routing prompt!
        formal_prompt = clean_calls[0].prompt
        self.assertIn("Selected Alpha Data", formal_prompt)
        self.assertIn("Selected Beta Data", formal_prompt)
        self.assertNotIn("NOISE_UNSELECTED_DISTRACTOR_DATA", formal_prompt)
        self.assertNotIn("selected_indices", formal_prompt)
        self.assertTrue(clean_calls[0].is_clean_context)

    def test_06_version_d_meta_dag_modifies_target_dag_and_itself(self) -> None:
        """Version D: Verify Meta-DAG takes a target DAG (as JSON) + prompt and outputs the modified DAG."""
        # 1. Create an initial simple business DAG
        target_dag = VersionD_MetaSelfModifyingDAG("order_pipeline")
        target_dag.models.register("gpt4", endpoint="https://api.openai.com/v1", apikey="sk-xxx")
        target_dag.add_edge(["raw_order"], "提取金额: {contents}", ["amount"], layer=0)

        # 2. Modify `target_dag` via Meta-DAG using a prompt
        prompt = (
            "添加边: [amount] -> (gpt4: 计算税费与最终应付: {contents}) -> [result] layer=1;\n"
            "添加内容: tags=[raw_order] contents=Order#1001 Total=$200"
        )
        modified_dag = target_dag.modify_with_prompt(prompt)

        self.assertEqual(len(modified_dag.edges), 2)
        self.assertEqual(modified_dag.edges[1].input_tags, ["amount"])
        self.assertEqual(modified_dag.edges[1].output_tags, ["result"])
        self.assertEqual(modified_dag.edges[1].layer, 1)
        self.assertEqual(len(modified_dag.contents), 1)

        # 3. Run the modified DAG to completion
        modified_dag.run(until_tag="result")
        self.assertTrue(modified_dag.has_tag("result"))
        self.assertIn("Order#1001", modified_dag.get_result("result"))

        # 4. Meta-DAG self-modification: use `meta_dag` to modify another `meta_dag`!
        meta_1 = create_meta_dag()
        meta_2 = transform_dag_with_meta_dag(
            meta_1,
            "添加边: [result] -> (default: 审计输出JSON: {contents}) -> [audited_result] layer=3",
        )
        self.assertEqual(len(meta_2.edges), len(meta_1.edges) + 1)
        self.assertEqual(meta_2.edges[-1].output_tags, ["audited_result"])

    def test_07_version_e_async_parallel_frontier_execution(self) -> None:
        """Version E: Verify `step_parallel_frontier()` concurrently runs all independent edges in the top frontier."""
        dag = VersionE_AsyncParallelDAG(max_concurrency=3)
        dag.add_edge(["doc"], lambda c: f"Keywords({c})", ["keywords"], layer=0)
        dag.add_edge(["doc"], lambda c: f"Sentiment({c})", ["sentiment"], layer=0)
        dag.add_edge(["doc"], lambda c: f"Entities({c})", ["entities"], layer=0)
        dag.add_edge(["keywords", "sentiment", "entities"], lambda c: f"Final({c})", ["result"], layer=1)

        dag.add_content("AI Agent Report 2026", tags=["doc"], gen=0)

        async def _run_pipeline() -> None:
            # First parallel frontier runs all 3 layer=0 edges concurrently
            batch_1 = await dag.step_parallel_frontier()
            self.assertEqual(len(batch_1), 3)
            self.assertTrue(dag.has_tag("keywords"))
            self.assertTrue(dag.has_tag("sentiment"))
            self.assertTrue(dag.has_tag("entities"))
            # Second step uses N-in-1-out assembly to combine the 3 outputs into ["result"]
            batch_2 = await dag.step_parallel_frontier()
            self.assertEqual(len(batch_2), 1)
            self.assertTrue(dag.has_tag("result"))

        asyncio.run(_run_pipeline())

    def test_08_version_f_transactional_rollback_and_max_count_guard(self) -> None:
        """Version F: Verify checkpoint/rollback, `with dag.transaction():`, and `max_edge_count` loop guard."""
        dag = VersionF_TransactionalReactiveDAG(max_edge_count=2)
        dag.add_edge(["ping"], lambda c: f"{c}+1", ["ping"], layer=0)
        dag.add_content("start", tags=["ping"], gen=0)

        # Save a checkpoint before running
        snap = dag.checkpoint("initial")

        # Run transactional block that raises an error -> must automatically roll back!
        try:
            with dag.transaction():
                dag.step()
                self.assertEqual(len(dag.contents), 2)
                raise RuntimeError("Simulated mid-flight failure")
        except RuntimeError:
            pass

        # Verify rollback restored state cleanly
        self.assertEqual(len(dag.contents), 1)
        self.assertEqual(dag.edges[0].count, 0)

        # Now run with `max_edge_count=2` cap on cyclic edge `["ping"] -> ["ping"]`
        dag.step()
        dag.step()
        self.assertEqual(dag.edges[0].count, 2)
        # Third step will not re-run the cyclic edge because `edge.count == max_edge_count`
        r3 = dag.step()
        self.assertNotEqual(r3.action, "edge")

        # Explicit rollback to "initial"
        dag.rollback(snap)
        self.assertEqual(len(dag.contents), 1)
        self.assertEqual(dag.edges[0].count, 0)

    def test_09_version_g_complete_production_trace_and_mermaid(self) -> None:
        """Version G: Verify unified production workflow, trace summary export, and Mermaid diagram generation."""
        dag = VersionG_ProductionCompleteDAG("finance_agent")
        dag.models.register("reasoner", endpoint="https://api.example.com/v1", apikey="sk-demo")
        dag.add_edge(["raw"], lambda c: '{"rev": 1000}', ["rev_data"], layer=0)
        dag.add_edge(["raw"], lambda c: '{"cost": 600}', ["cost_data"], layer=0)
        dag.add_edge(["rev_data", "cost_data"], lambda c: f"NetProfit=400 from {c}", ["result"], layer=1)

        dag.add_content("2026 Q4 Raw Financial Statement", tags=["raw"], gen=0)
        dag.run(until_tag="result")

        self.assertTrue(dag.has_tag("result"))
        self.assertIn("NetProfit=400", dag.get_result("result"))

        trace = dag.export_trace_summary()
        self.assertEqual(len(trace), 3)
        mermaid = dag.to_mermaid()
        self.assertIn("flowchart LR", mermaid)
        self.assertIn("count=1", mermaid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
