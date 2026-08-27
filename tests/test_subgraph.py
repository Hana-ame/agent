"""Unit and boundary mapping tests for SubgraphVertex (Phase 1)."""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Vertex, SubgraphVertex, MockAgent, Executor


class TestSubgraphBoundaryMapping:
    def test_subgraph_vertex_instantiation_from_dict(self):
        config = {
            "vertices": [
                {
                    "id": "NestedTeam",
                    "type": "subgraph",
                    "settings": {
                        "graph_config": {
                            "vertices": [{"id": "Worker1"}, {"id": "Worker2"}],
                            "edges": [
                                {"id": "e_inner", "source": "Worker1", "destination": "Worker2", "channel": "task"}
                            ],
                        },
                        "input_map": {"incoming_task": "Worker1.task"},
                        "output_map": {"Worker2.task": "final_summary"},
                    },
                }
            ],
            "edges": [],
        }
        g = Graph.from_dict(config)
        v = g.vertices["NestedTeam"]
        assert isinstance(v, SubgraphVertex)
        assert v.id == "NestedTeam"
        assert "incoming_task" in v.input_map
        assert "Worker2.task" in v.output_map

    @pytest.mark.asyncio
    async def test_stage_inner_inputs_routes_data_to_inner_sources(self):
        inner_config = {
            "vertices": [
                {"id": "SearchAgent"},
                {"id": "WriterAgent"},
            ],
            "edges": [
                {"id": "e1", "source": "SearchAgent", "destination": "WriterAgent", "channel": "query"}
            ],
        }

        subgraph_v = SubgraphVertex(
            vertex_id="ResearchTeam",
            settings={
                "graph_config": inner_config,
                "input_map": {
                    "topic": "SearchAgent.query",
                },
                "output_map": {
                    "WriterAgent.query": "report",
                },
            },
        )
        
        # Inject parent data into SubgraphVertex
        await subgraph_v.set_data("topic", "AI Agent Architecture")
        
        inner_g = subgraph_v.initialize_inner_graph()
        await subgraph_v.stage_inner_inputs(inner_g)

        # Check inner source vertex received data on channel 'query'
        search_agent = inner_g.vertices["SearchAgent"]
        staged_val = await search_agent.fetch_data("query")
        assert staged_val == "AI Agent Architecture"

    @pytest.mark.asyncio
    async def test_collect_inner_outputs_hoists_data_to_parent_vertex(self):
        inner_config = {
            "vertices": [
                {"id": "WorkerA"},
                {"id": "WorkerB"},
            ],
            "edges": [],
        }

        subgraph_v = SubgraphVertex(
            vertex_id="CompositeNode",
            settings={
                "graph_config": inner_config,
                "input_map": {},
                "output_map": {
                    "WorkerB.final_score": "aggregated_score",
                },
            },
        )

        inner_g = subgraph_v.initialize_inner_graph()
        # Simulate inner execution result in WorkerB
        await inner_g.vertices["WorkerB"].set_data("final_score", 98.5)

        # Hoist back
        await subgraph_v.collect_inner_outputs(inner_g)

        hoisted_val = await subgraph_v.fetch_data("aggregated_score")
        assert hoisted_val == 98.5


class TestSubgraphEndToEndExecution:
    @pytest.mark.asyncio
    async def test_parent_graph_executes_nested_subgraph(self):
        """
        Verify end-to-end flow:
        InputNode -> SubgraphVertex (SearchAgent -> SummaryAgent) -> OutputNode
        """
        inner_config = {
            "vertices": [
                {"id": "SearchAgent"},
                {"id": "SummaryAgent"},
            ],
            "edges": [
                {
                    "id": "e_inner_search_to_summary",
                    "source": "SearchAgent",
                    "destination": "SummaryAgent",
                    "channel": "query",
                    "prompt": "Find top insights on {query}",
                }
            ],
        }

        parent_config = {
            "vertices": [
                {"id": "Trigger", "initial_data": [{"channel": "topic", "value": "Autonomous Agents"}]},
                {
                    "id": "ResearchSubsystem",
                    "type": "subgraph",
                    "settings": {
                        "graph_config": inner_config,
                        "input_map": {"topic": "SearchAgent.query"},
                        "output_map": {"SummaryAgent.query": "report"},
                    },
                },
                {"id": "Publisher"},
            ],
            "edges": [
                {
                    "id": "e1_trigger_to_subgraph",
                    "source": "Trigger",
                    "destination": "ResearchSubsystem",
                    "channel": "topic",
                    "prompt": "Pass topic to research",
                },
                {
                    "id": "e2_subgraph_to_pub",
                    "source": "ResearchSubsystem",
                    "destination": "Publisher",
                    "channel": "report",
                    "prompt": "Publish report",
                },
            ],
        }

        def mock_llm_fn(data, prompt, model, settings):
            return f"Processed[{data}]"

        parent_graph = Graph.from_dict(parent_config)
        agent = MockAgent(response_fn=mock_llm_fn)
        executor = Executor(parent_graph, agents=agent)

        events = []
        async for event in executor.stream():
            events.append(event)

        assert executor._result.success is True
        
        # Verify event bubbling with namespaced vertex_id
        subgraph_event_types = [e.event_type for e in events if "subgraph_" in e.event_type]
        assert len(subgraph_event_types) > 0
        assert any(e.vertex_id == "ResearchSubsystem.SearchAgent" for e in events)
        assert any(e.vertex_id == "ResearchSubsystem.SummaryAgent" for e in events)

        # Check publisher received final hoisted report
        pub_v = parent_graph.vertices["Publisher"]
        final_output = await pub_v.fetch_data("report")
        assert "Autonomous Agents" in final_output

    @pytest.mark.asyncio
    async def test_checkpointed_subgraph_persists_with_namespaced_run_id(self):
        """Verify that CheckpointedExecutor saves inner subgraph snapshots with run_id::<subgraph_id>."""
        from framework import CheckpointedExecutor, SQLiteStateStore

        store = SQLiteStateStore(":memory:")
        inner_config = {
            "vertices": [{"id": "Step1"}, {"id": "Step2"}],
            "edges": [{"id": "e_in", "source": "Step1", "destination": "Step2", "channel": "val"}],
        }

        parent_config = {
            "vertices": [
                {"id": "Start", "initial_data": [{"channel": "val", "value": 100}]},
                {
                    "id": "NestedBox",
                    "type": "subgraph",
                    "settings": {
                        "graph_config": inner_config,
                        "input_map": {"val": "Step1.val"},
                        "output_map": {"Step2.val": "result"},
                    },
                },
            ],
            "edges": [
                {"id": "e_parent", "source": "Start", "destination": "NestedBox", "channel": "val"}
            ],
        }

        parent_graph = Graph.from_dict(parent_config)
        agent = MockAgent(response_fn=lambda d, p, m, s: d * 2)
        executor = CheckpointedExecutor(
            parent_graph,
            agents=agent,
            store=store,
            run_id="parent_run_42",
            graph_config=parent_config,
        )

        result = await executor.run()
        assert result.success is True

        # Verify snapshots were created for parent run
        parent_snapshots = store.load_all_snapshots("parent_run_42")
        assert len(parent_snapshots) > 0

        # Verify snapshots were created for namespaced child run
        child_run_id = "parent_run_42::NestedBox"
        child_snapshots = store.load_all_snapshots(child_run_id)
        assert len(child_snapshots) > 0
        assert any("Step2" in s.vertex_states for s in child_snapshots)
