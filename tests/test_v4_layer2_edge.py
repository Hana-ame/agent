"""Layer 2: Edge Runtime Layer (EdgeV4, CodeEdgeV4, LLMEdgeV4, ReflexiveEdgeV4) Comprehensive Test Suite.

Validates:
1. Pure text processing edges (text transformations, formatting, extraction, word counts).
2. Pure JSON processing edges (parsing, schema validation, transformation, error diagnostics).
3. LLMEdgeV4 multi-protocol agent dispatch (chat, generate, process, callable).
4. LLMEdgeV4 prompt template rendering and staging scratchpad persistence.
5. LLMEdgeV4 JSON attribute enforcement (markdown fence stripping, syntax validation, failure fallback).
6. ReflexiveEdgeV4 self-loop recovery and circuit breaker thresholds.
7. Two-sided handshake protocol and state-based execution guards.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import pytest

from framework.edge_v4 import (
    CallableLLMEdgeV4,
    ChatLLMEdgeV4,
    CodeEdgeV4,
    EdgeResultV4,
    EdgeV4,
    GenerateLLMEdgeV4,
    LLMEdgeV4,
    ProcessLLMEdgeV4,
    ReflexiveEdgeV4,
)
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import GraphV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
    VertexV4,
)


@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


class MockChatAgent:
    """Mock agent supporting agent.chat(messages, model, temperature)."""

    def __init__(self, response: str = "mock chat output", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.received_messages: List[Dict[str, str]] = []

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.received_messages = messages
        if self.should_fail:
            raise RuntimeError("Chat inference service unavailable")
        return self.response


class MockGenerateAgent:
    """Mock agent supporting agent.generate(prompt, model, temperature)."""

    def __init__(self, response: str = "mock generate output"):
        self.response = response
        self.received_prompt: Optional[str] = None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.received_prompt = prompt
        return self.response


class MockProcessAgent:
    """Mock agent supporting agent.process(input, prompt, model, settings)."""

    def __init__(self, response: str = "mock process output"):
        self.response = response

    def process(self, raw_input: str, prompt: str, **kwargs: Any) -> str:
        return f"{self.response}: {raw_input}"


# ---------------------------------------------------------------------------
# 1. Pure Text Processing Edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pure_text_transformation(mem_store: VertexStoreV4) -> None:
    """Verify a CodeEdge performing pure text manipulation (case conversion, cleanup)."""
    sess = "text_transform_session"
    mem_store.save_vertex(sess, "raw_text", "  hello world! this is a test.  ", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "cleaned_text", "", state=VertexStateV4.TODO)

    def clean_text_fn(content: str, settings: Dict[str, Any], staging: Any) -> str:
        return content.strip().title()

    edge = CodeEdgeV4("edge_text_clean", "raw_text", "cleaned_text", script=clean_text_fn)
    res = await edge.run(sess, mem_store)

    assert res.success is True
    assert res.output == "Hello World! This Is A Test."
    out_v = mem_store.get_vertex(sess, "cleaned_text")
    assert out_v.content == "Hello World! This Is A Test."
    assert out_v.state == VertexStateV4.DATA_READY.value
    assert out_v.processed_count == 1


@pytest.mark.asyncio
async def test_pure_text_extraction_and_metrics(mem_store: VertexStoreV4) -> None:
    """Verify pure text extraction with intermediate metrics recorded to staging."""
    sess = "text_metrics_session"
    sample_doc = (
        "Report 2026: Revenue grew by 15% year-over-year. "
        "Operating margins reached 24.5%. Key driver was AI automation."
    )
    mem_store.save_vertex(sess, "doc_in", sample_doc, state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "summary_out", "", state=VertexStateV4.TODO)

    def text_analytics_fn(content: str, settings: Dict[str, Any], staging: Any) -> str:
        percentages = re.findall(r"\d+(?:\.\d+)?%", content)
        summary = f"Identified percentages: {', '.join(percentages)}"
        return summary

    edge = CodeEdgeV4("edge_text_analytics", "doc_in", "summary_out", script=text_analytics_fn)
    res = await edge.run(sess, mem_store)

    assert res.success is True
    assert "15%, 24.5%" in res.output
    out_v = mem_store.get_vertex(sess, "summary_out")
    assert out_v.content == "Identified percentages: 15%, 24.5%"
    assert out_v.state == VertexStateV4.DATA_READY.value


# ---------------------------------------------------------------------------
# 2. Pure JSON Processing Edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pure_json_parsing_and_transformation(mem_store: VertexStoreV4) -> None:
    """Verify a CodeEdge parsing JSON input, modifying fields, and outputting formatted JSON."""
    sess = "json_transform_session"
    input_payload = json.dumps({
        "order_id": "ORD-9901",
        "items": [
            {"sku": "A1", "price": 10.0, "qty": 2},
            {"sku": "B2", "price": 25.0, "qty": 1},
        ],
        "discount_code": "SAVE10",
    })
    mem_store.save_vertex(sess, "order_raw", input_payload, state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "order_processed", "", state=VertexStateV4.TODO)

    def process_order_json(content: str, settings: Dict[str, Any], staging: Any) -> str:
        data = json.loads(content)
        subtotal = sum(item["price"] * item["qty"] for item in data["items"])
        discount = 0.10 if data.get("discount_code") == "SAVE10" else 0.0
        total = subtotal * (1.0 - discount)

        output_data = {
            "order_id": data["order_id"],
            "item_count": len(data["items"]),
            "subtotal": subtotal,
            "discount": discount,
            "total": round(total, 2),
            "status": "calculated",
        }
        return json.dumps(output_data, indent=2)

    edge = CodeEdgeV4("edge_json_calc", "order_raw", "order_processed", script=process_order_json)
    res = await edge.run(sess, mem_store)

    assert res.success is True
    result_dict = json.loads(res.output)
    assert result_dict["order_id"] == "ORD-9901"
    assert result_dict["subtotal"] == 45.0
    assert result_dict["total"] == 40.5
    assert result_dict["status"] == "calculated"


@pytest.mark.asyncio
async def test_pure_json_validation_failure_transitions_to_reject(mem_store: VertexStoreV4) -> None:
    """Verify that a JSON processing edge fails gracefully on malformed JSON and sets REJECT."""
    sess = "json_fail_session"
    malformed_json = "{ invalid_key: missing_quotes, 123 "
    mem_store.save_vertex(sess, "json_bad", malformed_json, state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "json_out", "", state=VertexStateV4.TODO)

    def strict_json_parser(content: str, settings: Dict[str, Any], staging: Any) -> str:
        data = json.loads(content)
        return json.dumps({"status": "ok", "keys": list(data.keys())})

    edge = CodeEdgeV4("edge_json_validate", "json_bad", "json_out", script=strict_json_parser)
    res = await edge.run(sess, mem_store)

    assert res.success is False
    assert "JSONDecodeError" in str(res.error) or "Expecting property name" in str(res.error)

    # Verify downstream transition to REJECT
    out_v = mem_store.get_vertex(sess, "json_out")
    assert out_v.state == VertexStateV4.REJECT.value

    # Verify error diagnostic saved to staging
    staged_error = mem_store.get_latest_staged_for_vertex(sess, "json_out", key="error_feedback")
    assert staged_error is not None
    assert "JSONDecodeError" in staged_error.metadata.get("exception_type", "") or "json" in staged_error.value.lower()


# ---------------------------------------------------------------------------
# 3. LLMEdgeV4 Comprehensive Suite (Addressing Audit Gap)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_edge_with_chat_protocol(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 dispatching to agent.chat with prompt template rendering."""
    sess = "llm_chat_session"
    mem_store.save_vertex(sess, "topic_in", "Quantum Computing", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "analysis_out", "", state=VertexStateV4.TODO)

    mock_agent = MockChatAgent(response="Quantum computing leverages superposition and entanglement.")
    edge = LLMEdgeV4(
        edge_id="llm_edge_chat",
        input_vertex="topic_in",
        output_vertex="analysis_out",
        prompt_template="Explain the core principles of {input} in one sentence.",
        model="sensenova-6.8-flash-lite",
        settings={"temperature": 0.3},
    )

    res = await edge.run(sess, mem_store, agent=mock_agent)

    assert res.success is True
    assert "superposition" in res.output
    # Check rendered prompt was staged before execution
    staged_prompt = mem_store.get_latest_staged_for_vertex(sess, "analysis_out", key="rendered_prompt")
    assert staged_prompt is not None
    assert staged_prompt.value == "Explain the core principles of Quantum Computing in one sentence."

    # Check downstream vertex updated
    out_v = mem_store.get_vertex(sess, "analysis_out")
    assert out_v.state == VertexStateV4.DATA_READY.value
    assert out_v.content == res.output
    assert out_v.processed_count == 1


@pytest.mark.asyncio
async def test_llm_edge_with_generate_protocol(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 dispatching to agent.generate(prompt, ...)."""
    sess = "llm_gen_session"
    mem_store.save_vertex(sess, "prompt_v", "Summarize DAG benefits", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "summary_v", "", state=VertexStateV4.TODO)

    agent = MockGenerateAgent(response="DAGs guarantee deterministic ordering and parallel execution.")
    edge = LLMEdgeV4("edge_gen", "prompt_v", "summary_v", prompt_template="{input}")
    res = await edge.run(sess, mem_store, agent=agent)

    assert res.success is True
    assert agent.received_prompt == "Summarize DAG benefits"
    out_v = mem_store.get_vertex(sess, "summary_v")
    assert out_v.state == VertexStateV4.DATA_READY.value
    assert out_v.content == "DAGs guarantee deterministic ordering and parallel execution."


@pytest.mark.asyncio
async def test_llm_edge_with_callable_agent(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 dispatching to a simple callable/lambda."""
    sess = "llm_callable_session"
    mem_store.save_vertex(sess, "q_in", "What is 2+2?", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "a_out", "", state=VertexStateV4.TODO)

    callable_agent = lambda prompt: f"Calculated: 4 for '{prompt}'"
    edge = LLMEdgeV4("edge_call", "q_in", "a_out", prompt_template="{input}")
    res = await edge.run(sess, mem_store, agent=callable_agent)

    assert res.success is True
    assert "Calculated: 4" in res.output


@pytest.mark.asyncio
async def test_llm_edge_json_enforcement_and_fence_stripping(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 cleans markdown json code fences when downstream expects JSON."""
    sess = "llm_json_clean_session"
    mem_store.save_vertex(sess, "input_v", "Extract person details", state=VertexStateV4.DATA_READY)
    # Mark downstream with JSON attribute
    mem_store.save_vertex(
        sess,
        "json_out",
        "",
        attributes=[VertexAttributeV4.JSON],
        state=VertexStateV4.TODO,
    )

    fenced_output = "```json\n{\n  \"name\": \"Alice\",\n  \"role\": \"Engineer\"\n}\n```"
    mock_agent = MockChatAgent(response=fenced_output)

    edge = LLMEdgeV4("edge_json_clean", "input_v", "json_out")
    res = await edge.run(sess, mem_store, agent=mock_agent)

    assert res.success is True
    # Output should be valid, cleanly stripped JSON
    parsed = json.loads(res.output)
    assert parsed["name"] == "Alice"
    assert parsed["role"] == "Engineer"


@pytest.mark.asyncio
async def test_llm_edge_json_enforcement_rejects_malformed_json(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 transitions to REJECT when downstream expects JSON but LLM outputs plain text."""
    sess = "llm_json_fail_session"
    mem_store.save_vertex(sess, "input_v", "Generate JSON", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(
        sess,
        "json_target",
        "",
        attributes=[VertexAttributeV4.JSON],
        state=VertexStateV4.TODO,
    )

    plain_text_response = "Sorry, I am an AI and cannot generate JSON at this time."
    mock_agent = MockChatAgent(response=plain_text_response)

    edge = LLMEdgeV4("edge_json_strict", "input_v", "json_target")
    res = await edge.run(sess, mem_store, agent=mock_agent)

    assert res.success is False
    assert "Malformed JSON output" in (res.error or "")

    out_v = mem_store.get_vertex(sess, "json_target")
    assert out_v.state == VertexStateV4.REJECT.value

    staged_err = mem_store.get_latest_staged_for_vertex(sess, "json_target", key="error_feedback")
    assert staged_err is not None
    assert "Malformed JSON output" in staged_err.value


@pytest.mark.asyncio
async def test_llm_edge_agent_failure_stages_diagnostic(mem_store: VertexStoreV4) -> None:
    """Verify LLMEdgeV4 captures model exceptions, stages diagnostics, and transitions to REJECT."""
    sess = "llm_fail_session"
    mem_store.save_vertex(sess, "in_v", "test query", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "out_v", "", state=VertexStateV4.TODO)

    failing_agent = MockChatAgent(should_fail=True)
    edge = LLMEdgeV4("edge_failing", "in_v", "out_v")
    res = await edge.run(sess, mem_store, agent=failing_agent)

    assert res.success is False
    assert "Chat inference service unavailable" in (res.error or "")

    out_v = mem_store.get_vertex(sess, "out_v")
    assert out_v.state == VertexStateV4.REJECT.value


# ---------------------------------------------------------------------------
# 4. ReflexiveEdgeV4 Self-Healing & Circuit Breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflexive_edge_self_healing_reset(mem_store: VertexStoreV4) -> None:
    """Verify ReflexiveEdge reads error feedback, applies fix, and resets vertex to TODO_URGENT."""
    sess = "reflexive_heal_session"
    # Vertex in REJECT state with staged error
    mem_store.save_vertex(sess, "flaky_node", "initial draft", state=VertexStateV4.REJECT)
    mem_store.stage_output(
        session_id=sess,
        edge_id="code_edge_1",
        key="error_feedback",
        value="SyntaxError at line 4: unexpected indent",
        vertex_name="flaky_node",
    )

    def recovery_script(target_v: VertexRecordV4, error_diag: str, settings: Dict[str, Any]) -> str:
        return f"Repaired payload based on diagnostic: {error_diag}"

    reflexive = ReflexiveEdgeV4(
        edge_id="recover_flaky_node",
        vertex_name="flaky_node",
        trigger_state=VertexStateV4.REJECT.value,
        target_state=VertexStateV4.TODO_URGENT.value,
        max_retries=3,
        script=recovery_script,
    )

    res = await reflexive.run(sess, mem_store)

    assert res.success is True
    out_v = mem_store.get_vertex(sess, "flaky_node")
    # Vertex state reset to TODO_URGENT
    assert out_v.state == VertexStateV4.TODO_URGENT.value
    assert "Repaired payload" in out_v.content


@pytest.mark.asyncio
async def test_reflexive_edge_circuit_breaker_locks_to_forbidden(mem_store: VertexStoreV4) -> None:
    """Verify ReflexiveEdge locks vertex to FORBIDDEN when max_retries is reached."""
    sess = "circuit_breaker_session"
    # Vertex in REJECT with processed_count equal to max_retries
    mem_store.save_vertex(sess, "stubborn_node", "bad code", state=VertexStateV4.REJECT)
    # Manually advance processed count to threshold
    for _ in range(3):
        mem_store.increment_processed_count(sess, "stubborn_node")

    reflexive = ReflexiveEdgeV4(
        edge_id="circuit_breaker_edge",
        vertex_name="stubborn_node",
        max_retries=3,
    )

    res = await reflexive.run(sess, mem_store)

    assert res.success is False
    assert res.error == "Max retries exceeded"
    assert "Attempted 3 retries out of 3" in (res.reason or "")

    out_v = mem_store.get_vertex(sess, "stubborn_node")
    assert out_v.state == VertexStateV4.FORBIDDEN.value

    # Check staging recorded retry_exhausted
    staged = mem_store.get_latest_staged_for_vertex(sess, "stubborn_node", key="retry_exhausted")
    assert staged is not None
    assert "Exceeded max retries: 3" in staged.value


# ---------------------------------------------------------------------------
# 5. Handshake Protocol & Execution Guards
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_skipped_when_upstream_not_ready(mem_store: VertexStoreV4) -> None:
    """Verify Edge skips execution when upstream is in IDLE instead of DATA_READY."""
    sess = "handshake_guard_session"
    mem_store.save_vertex(sess, "up", "draft", state=VertexStateV4.IDLE)
    mem_store.save_vertex(sess, "down", "", state=VertexStateV4.TODO)

    edge = CodeEdgeV4("test_guard", "up", "down", script=lambda c, s, st: "result")
    res = await edge.run(sess, mem_store)

    assert res.success is False
    assert res.skipped is True
    assert "required 'data ready'" in (res.reason or "")


@pytest.mark.asyncio
async def test_edge_skipped_when_downstream_in_pruning_or_forbidden(mem_store: VertexStoreV4) -> None:
    """Verify Edge skips execution when downstream vertex is PRUNING or FORBIDDEN."""
    sess = "prune_guard_session"
    mem_store.save_vertex(sess, "src", "ready data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "pruned", "", state=VertexStateV4.PRUNING)

    edge = CodeEdgeV4("edge_pruned", "src", "pruned", script=lambda c, s, st: "ignored")
    res = await edge.run(sess, mem_store)

    assert res.success is False
    assert res.skipped is True
    assert "required 'todo'" in (res.reason or "")


# ---------------------------------------------------------------------------
# 6. Specialized LLMEdge Subclasses (Chat, Generate, Process, Callable)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_llm_edge_subclass_direct(mem_store: VertexStoreV4) -> None:
    """Verify ChatLLMEdgeV4 invokes agent.chat directly and updates store."""
    sess = "subclass_chat_session"
    mem_store.save_vertex(sess, "v_in", "Prompt Input", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "v_out", "", state=VertexStateV4.TODO)

    agent = MockChatAgent(response="Chat response successfully generated")
    edge = ChatLLMEdgeV4(
        edge_id="edge_sub_chat",
        input_vertex="v_in",
        output_vertex="v_out",
        prompt_template="Tell: {input}",
    )
    res = await edge.run(sess, mem_store, agent=agent)

    assert res.success is True
    assert res.output == "Chat response successfully generated"
    assert mem_store.get_vertex(sess, "v_out").state == VertexStateV4.DATA_READY.value
    assert agent.received_messages[0]["content"] == "Tell: Prompt Input"


@pytest.mark.asyncio
async def test_generate_llm_edge_subclass_direct(mem_store: VertexStoreV4) -> None:
    """Verify GenerateLLMEdgeV4 invokes agent.generate directly and updates store."""
    sess = "subclass_gen_session"
    mem_store.save_vertex(sess, "v_in", "Topic for generation", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "v_out", "", state=VertexStateV4.TODO)

    agent = MockGenerateAgent(response="Generated deep analysis content")
    edge = GenerateLLMEdgeV4(
        edge_id="edge_sub_gen",
        input_vertex="v_in",
        output_vertex="v_out",
        prompt_template="Generate article about {input}",
    )
    res = await edge.run(sess, mem_store, agent=agent)

    assert res.success is True
    assert agent.received_prompt == "Generate article about Topic for generation"
    assert mem_store.get_vertex(sess, "v_out").content == "Generated deep analysis content"


@pytest.mark.asyncio
async def test_process_llm_edge_subclass_direct(mem_store: VertexStoreV4) -> None:
    """Verify ProcessLLMEdgeV4 invokes agent.process with input content and template."""
    sess = "subclass_proc_session"
    mem_store.save_vertex(sess, "v_in", "Input Data 42", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "v_out", "", state=VertexStateV4.TODO)

    agent = MockProcessAgent(response="ProcessResult")
    edge = ProcessLLMEdgeV4(
        edge_id="edge_sub_proc",
        input_vertex="v_in",
        output_vertex="v_out",
        prompt_template="Extract number from {input}",
    )
    res = await edge.run(sess, mem_store, agent=agent)

    assert res.success is True
    assert "ProcessResult: Input Data 42" in res.output
    assert mem_store.get_vertex(sess, "v_out").state == VertexStateV4.DATA_READY.value


@pytest.mark.asyncio
async def test_callable_llm_edge_subclass_direct(mem_store: VertexStoreV4) -> None:
    """Verify CallableLLMEdgeV4 invokes a simple sync/async callable directly."""
    sess = "subclass_call_session"
    mem_store.save_vertex(sess, "v_in", "Calculate 10*10", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "v_out", "", state=VertexStateV4.TODO)

    edge = CallableLLMEdgeV4(
        edge_id="edge_sub_call",
        input_vertex="v_in",
        output_vertex="v_out",
        prompt_template="Evaluate {input}",
    )
    res = await edge.run(sess, mem_store, agent=lambda p: f"Answer: 100 for '{p}'")

    assert res.success is True
    assert res.output == "Answer: 100 for 'Evaluate Calculate 10*10'"


@pytest.mark.asyncio
async def test_subclass_protocol_mismatch_fails_gracefully(mem_store: VertexStoreV4) -> None:
    """Verify passing an agent lacking the required method causes a clean failure and REJECT state."""
    sess = "mismatch_session"
    mem_store.save_vertex(sess, "v_in", "test data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(sess, "v_out", "", state=VertexStateV4.TODO)

    # Chat edge passed an agent that only has generate()
    edge = ChatLLMEdgeV4(edge_id="mismatched_chat", input_vertex="v_in", output_vertex="v_out")
    gen_agent = MockGenerateAgent(response="gen only")

    res = await edge.run(sess, mem_store, agent=gen_agent)

    assert res.success is False
    assert "does not implement callable 'chat'" in (res.error or "")
    assert mem_store.get_vertex(sess, "v_out").state == VertexStateV4.REJECT.value


def test_from_base_cloning_and_attribute_preservation() -> None:
    """Verify LLMEdgeV4.from_base preserves all settings, priority, timeout, and concurrency attributes."""
    base = LLMEdgeV4(
        edge_id="base_llm_1",
        input_vertex="node_a",
        output_vertex="node_b",
        model="custom-gemini-pro",
        prompt_template="Format {input} as JSON",
        settings={"temperature": 0.2, "custom_flag": True},
        concurrency_limit=3,
        concurrency_group="gemini_group",
        priority=8,
        timeout=45.0,
    )

    specialized = ChatLLMEdgeV4.from_base(base)

    assert isinstance(specialized, ChatLLMEdgeV4)
    assert specialized.id == "base_llm_1"
    assert specialized.input_vertex == "node_a"
    assert specialized.output_vertex == "node_b"
    assert specialized.model == "custom-gemini-pro"
    assert specialized.prompt_template == "Format {input} as JSON"
    assert specialized.temperature == 0.2
    assert specialized.settings["custom_flag"] is True
    assert specialized.concurrency_limit == 3
    assert specialized.concurrency_group == "gemini_group"
    assert specialized.priority == 8
    assert specialized.timeout == 45.0


# ---------------------------------------------------------------------------
# 7. Executor Runtime Subclass Designation & Execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_specifies_chat_llm_subclass(mem_store: VertexStoreV4) -> None:
    """Verify ExecutorV4 designates ChatLLMEdgeV4 for base LLM edges in graph execution."""
    sess = "exec_chat_session"
    graph = GraphV4(session_id=sess)
    graph.add_vertex(VertexRecordV4(0, sess, "src", "AI trends", [], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, sess, "dst", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))

    # Base LLMEdgeV4 in graph manifest
    graph.add_edge(LLMEdgeV4("llm_edge", "src", "dst", prompt_template="Outline: {input}"))

    chat_agent = MockChatAgent(response="AI Trends Outline: Agents, RAG, DAGs")
    executor = ExecutorV4(
        graph=graph,
        store=mem_store,
        agent=chat_agent,
        llm_edge_cls=ChatLLMEdgeV4,
    )

    # Verify edge was specialized in executor graph
    assert isinstance(executor.graph.edges["llm_edge"], ChatLLMEdgeV4)

    result = await executor.run()
    assert result.success is True
    assert "Agents, RAG, DAGs" in mem_store.get_vertex(sess, "dst").content


@pytest.mark.asyncio
async def test_executor_specifies_generate_llm_subclass(mem_store: VertexStoreV4) -> None:
    """Verify ExecutorV4 designates GenerateLLMEdgeV4 for base LLM edges in graph execution."""
    sess = "exec_gen_session"
    graph = GraphV4(session_id=sess)
    graph.add_vertex(VertexRecordV4(0, sess, "v_a", "Raw input", [], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, sess, "v_b", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))

    graph.add_edge(LLMEdgeV4("edge_gen_run", "v_a", "v_b", prompt_template="Process {input}"))

    gen_agent = MockGenerateAgent(response="Generated text payload")
    executor = ExecutorV4(
        graph=graph,
        store=mem_store,
        agent=gen_agent,
        llm_edge_cls=GenerateLLMEdgeV4,
    )

    assert isinstance(executor.graph.edges["edge_gen_run"], GenerateLLMEdgeV4)

    result = await executor.run()
    assert result.success is True
    assert mem_store.get_vertex(sess, "v_b").content == "Generated text payload"


# ---------------------------------------------------------------------------
# 8. Granular Concurrency, Priority Scheduling, and Timeout Enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edge_attributes_granular_concurrency(mem_store: VertexStoreV4) -> None:
    """Verify group_concurrency throttles edges in group 'llm' while allowing 'code' edges concurrently."""
    sess = "group_concurrency_session"
    graph = GraphV4(session_id=sess)
    graph.add_vertex(VertexRecordV4(0, sess, "input_v", "data", [], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, sess, "llm_out_1", "", [], VertexStateV4.TODO.value))
    graph.add_vertex(VertexRecordV4(0, sess, "llm_out_2", "", [], VertexStateV4.TODO.value))
    graph.add_vertex(VertexRecordV4(0, sess, "code_out", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))

    concurrent_llm_tasks = 0
    max_observed_concurrent_llm = 0

    async def slow_chat(messages, **kwargs):
        nonlocal concurrent_llm_tasks, max_observed_concurrent_llm
        concurrent_llm_tasks += 1
        max_observed_concurrent_llm = max(max_observed_concurrent_llm, concurrent_llm_tasks)
        await asyncio.sleep(0.08)
        concurrent_llm_tasks -= 1
        return "slow chat done"

    agent = MockChatAgent()
    agent.chat = slow_chat

    # 2 LLM edges in group "llm", 1 Code edge in group "code"
    graph.add_edge(ChatLLMEdgeV4("llm_1", "input_v", "llm_out_1", concurrency_group="llm"))
    graph.add_edge(ChatLLMEdgeV4("llm_2", "input_v", "llm_out_2", concurrency_group="llm"))
    graph.add_edge(CodeEdgeV4("code_1", "input_v", "code_out", script=lambda c, s, st: "fast code done", concurrency_group="code"))

    executor = ExecutorV4(
        graph=graph,
        store=mem_store,
        agent=agent,
        max_concurrency=4,
        group_concurrency={"llm": 1},  # Limit LLM group to at most 1 at a time!
    )

    result = await executor.run()
    assert result.success is True
    # Group limit of 1 must be strictly enforced
    assert max_observed_concurrent_llm == 1


@pytest.mark.asyncio
async def test_edge_priority_tie_breaking(mem_store: VertexStoreV4) -> None:
    """Verify higher priority edge is dispatched before lower priority edge in the same DAG tier."""
    sess = "priority_test_session"
    graph = GraphV4(session_id=sess)
    graph.add_vertex(VertexRecordV4(0, sess, "root", "start", [], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, sess, "out_low", "", [], VertexStateV4.TODO.value))
    graph.add_vertex(VertexRecordV4(0, sess, "out_high", "", [], VertexStateV4.TODO.value))

    execution_order = []

    def make_script(name: str):
        def _fn(c, s, st):
            execution_order.append(name)
            return f"done {name}"
        return _fn

    graph.add_edge(CodeEdgeV4("edge_low", "root", "out_low", script=make_script("low"), priority=1))
    graph.add_edge(CodeEdgeV4("edge_high", "root", "out_high", script=make_script("high"), priority=100))

    # max_concurrency=1 to verify strict scheduling order
    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=1)
    result = await executor.run()

    assert result.success is True
    assert execution_order == ["high", "low"]


@pytest.mark.asyncio
async def test_per_edge_timeout_triggers_rejection(mem_store: VertexStoreV4) -> None:
    """Verify edge execution exceeding per-edge timeout transitions output to REJECT."""
    sess = "timeout_test_session"
    graph = GraphV4(session_id=sess)
    graph.add_vertex(VertexRecordV4(0, sess, "src", "data", [], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, sess, "dst", "", [], VertexStateV4.TODO.value))

    async def hanging_script(c, s, st):
        await asyncio.sleep(1.0)
        return "should not finish"

    # Edge timeout is set to 0.05 seconds
    graph.add_edge(CodeEdgeV4("hanging_edge", "src", "dst", script=hanging_script, timeout=0.05))

    executor = ExecutorV4(graph=graph, store=mem_store, timeout=5.0)
    result = await executor.run()

    dst_v = mem_store.get_vertex(sess, "dst")
    assert dst_v.state == VertexStateV4.REJECT.value
    # Staged error diagnostics present
    err_record = mem_store.get_latest_staged_for_vertex(sess, "dst", key="error_feedback")
    assert err_record is not None
    assert "timed out" in err_record.value.lower()


def test_edge_v4_standalone_arbitrary_cwd(tmp_path) -> None:
    """Verify edge_v4.py runs cleanly when invoked from any arbitrary working directory."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge_v4.py"

    cmd = [
        sys.executable,
        str(edge_script),
        "--session",
        "arbitrary_cwd_sess",
        "--type",
        "code",
        "--input",
        "in_node",
        "--output",
        "out_node",
        "--seed-input",
        "payload from arbitrary cwd",
    ]
    # Execute with cwd set to a temporary folder completely outside the repo
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert data["output"] == "payload from arbitrary cwd"


def test_subclass_edges_standalone_arbitrary_cwd(tmp_path) -> None:
    """Verify specialized edge subclass files execute standalone from any working directory."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    subclass_scripts = [
        repo_root / "framework" / "chat_llm_edge_v4.py",
        repo_root / "framework" / "generate_llm_edge_v4.py",
        repo_root / "framework" / "process_llm_edge_v4.py",
        repo_root / "framework" / "callable_llm_edge_v4.py",
    ]

    for script in subclass_scripts:
        cmd = [
            sys.executable,
            str(script),
            "--session",
            "subclass_cwd_sess",
            "--type",
            "code",
            "--input",
            "in_node",
            "--output",
            "out_node",
            "--seed-input",
            f"hello from {script.name}",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name} failed with stderr: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["success"] is True
        assert data["output"] == f"hello from {script.name}"


def test_edge_v1_standalone_arbitrary_cwd(tmp_path) -> None:
    """Verify framework/edge.py runs standalone from any arbitrary working directory."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge.py"

    cmd = [
        sys.executable,
        str(edge_script),
        "--dir",
        str(repo_root / "examples" / "hn_ai_report"),
        "--script",
        "hn_edges.py:SummarizeEdge",
        "--data",
        '{"title": "Arbitrary CWD Story"}',
        "--skip-compute",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"edge.py CLI execution failed with stderr: {proc.stderr}"
    assert "SummarizeEdge" in proc.stdout
    assert "ok            : True" in proc.stdout


def test_edge_v4_standalone_mock_flag(tmp_path) -> None:
    """Verify edge_v4.py --mock runs standalone with simulated LLM compute."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge_v4.py"

    cmd = [
        sys.executable,
        str(edge_script),
        "--session",
        "mock_sess_v4",
        "--type",
        "llm",
        "--input",
        "in_node",
        "--output",
        "out_node",
        "--seed-input",
        "prompt to mock",
        "--mock",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"edge_v4.py --mock failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert "mock" in data["output"]


def test_subclass_edges_standalone_mock_flag(tmp_path) -> None:
    """Verify subclass edge runners support --mock flag for offline testing."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    cases = [
        (repo_root / "framework" / "chat_llm_edge_v4.py", "llm_chat"),
        (repo_root / "framework" / "generate_llm_edge_v4.py", "llm_generate"),
        (repo_root / "framework" / "process_llm_edge_v4.py", "llm_process"),
        (repo_root / "framework" / "callable_llm_edge_v4.py", "llm_callable"),
    ]

    for script, edge_type in cases:
        cmd = [
            sys.executable,
            str(script),
            "--session",
            f"mock_{edge_type}_sess",
            "--type",
            edge_type,
            "--input",
            "in_node",
            "--output",
            "out_node",
            "--seed-input",
            f"input for {edge_type}",
            "--mock",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name} --mock failed with stderr: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["success"] is True
        assert "mock" in data["output"]


def test_edge_v1_standalone_mock_flag(tmp_path) -> None:
    """Verify framework/edge.py supports --mock flag from arbitrary working directory."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge.py"

    cmd = [
        sys.executable,
        str(edge_script),
        "--dir",
        str(repo_root / "examples" / "hn_ai_report"),
        "--script",
        "hn_edges.py:SummarizeEdge",
        "--data",
        '{"title": "Offline Mock Story"}',
        "--mock",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"edge.py --mock failed with stderr: {proc.stderr}"
    assert "SummarizeEdge" in proc.stdout
    assert "HttpLLMAgent (mock)" in proc.stdout
    assert "ok            : True" in proc.stdout


def test_edge_v4_from_config_and_file(tmp_path) -> None:
    """Verify EdgeV4 and subclasses instantiate correctly from JSON config file or dict."""
    from framework.edge_v4 import EdgeV4, CodeEdgeV4, ChatLLMEdgeV4, ReflexiveEdgeV4

    # 1. Instantiate CodeEdgeV4 from dict
    code_edge = EdgeV4.from_config({
        "id": "code_from_dict",
        "type": "code",
        "input": "src_v",
        "output": "dst_v",
        "priority": 15,
        "concurrency_limit": 2,
    })
    assert isinstance(code_edge, CodeEdgeV4)
    assert code_edge.id == "code_from_dict"
    assert code_edge.priority == 15
    assert code_edge.concurrency_limit == 2

    # 2. Instantiate ChatLLMEdgeV4 from JSON file
    cfg_file = tmp_path / "chat_edge_cfg.json"
    cfg_file.write_text(json.dumps({
        "id": "chat_from_json",
        "type": "llm_chat",
        "input": "prompt_in",
        "output": "response_out",
        "model": "test-chat-model",
        "priority": 20,
        "settings": {"prompt": "Answer: {input}"},
    }), encoding="utf-8")

    chat_edge = EdgeV4.from_config_file(cfg_file)
    assert isinstance(chat_edge, ChatLLMEdgeV4)
    assert chat_edge.id == "chat_from_json"
    assert chat_edge.model == "test-chat-model"
    assert chat_edge.priority == 20

    # 3. Instantiate ReflexiveEdgeV4 from JSON file
    refl_file = tmp_path / "refl_edge_cfg.json"
    refl_file.write_text(json.dumps({
        "id": "refl_from_json",
        "type": "reflexive",
        "input": "err_node",
        "trigger_state": "reject",
        "target_state": "todo_urgent",
        "max_retries": 4,
    }), encoding="utf-8")

    refl_edge = ReflexiveEdgeV4.from_config_file(refl_file)
    assert isinstance(refl_edge, ReflexiveEdgeV4)
    assert refl_edge.trigger_state == "reject"
    assert refl_edge.max_retries == 4


def test_edge_v4_cli_driven_by_json_config(tmp_path) -> None:
    """Verify edge_v4.py execution completely driven by a single JSON file."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge_v4.py"

    cfg_file = tmp_path / "standalone_edge_cfg.json"
    cfg_file.write_text(json.dumps({
        "session": "json_cli_session",
        "edge_id": "cfg_cli_edge",
        "type": "code",
        "input": "input_node",
        "output": "output_node",
        "seed_input": "driven by json file",
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        str(edge_script),
        "--config",
        str(cfg_file),
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert data["edge_id"] == "cfg_cli_edge"
    assert data["output"] == "driven by json file"


def test_edge_v4_cli_json_config_with_cli_overrides(tmp_path) -> None:
    """Verify CLI arguments override values in JSON config file."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    edge_script = repo_root / "framework" / "edge_v4.py"

    cfg_file = tmp_path / "base_cfg.json"
    cfg_file.write_text(json.dumps({
        "session": "original_session",
        "edge_id": "original_edge",
        "type": "code",
        "input": "orig_in",
        "output": "orig_out",
        "seed_input": "original input",
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        str(edge_script),
        "--config",
        str(cfg_file),
        "--session",
        "overridden_session",
        "--seed-input",
        "overridden input",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert data["output"] == "overridden input"


def test_subclass_cli_driven_by_json_config(tmp_path) -> None:
    """Verify chat_llm_edge_v4.py CLI execution driven by JSON file."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    subclass_script = repo_root / "framework" / "chat_llm_edge_v4.py"

    cfg_file = tmp_path / "chat_subclass_cfg.json"
    cfg_file.write_text(json.dumps({
        "session": "chat_subclass_session",
        "edge_id": "chat_subclass_edge",
        "type": "llm_chat",
        "input": "in_topic",
        "output": "out_resp",
        "seed_input": "chat question",
        "mock": True,
    }), encoding="utf-8")

    cmd = [
        sys.executable,
        str(subclass_script),
        "--config",
        str(cfg_file),
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, f"Subclass failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["success"] is True
    assert "mock" in data["output"]



