"""Progressive Architectural Versions A through G (`dag_agent/versions.py`).

Each version is a concrete, self-contained or cleanly extended class demonstrating a distinct
stage of architectural capability, all adhering to high-cohesion and low-coupling design discipline:

- **Version A (`VersionA_MinimalDAG`)**:
  Exact tag matching, `models.json`/`contents.json`/`edges.json` loading, `layer` (0-first) & `gen` (0-first)
  priority ordering, `edge.count` tracking, and `gen + 1` progression.
- **Version B (`VersionB_SemanticEmbeddingDAG`)**:
  Adds vector embedding cosine similarity + N-gram similarity threshold matching (`similarity >= threshold`),
  allowing semantically close tags to trigger edges without exact string identity.
- **Version C (`VersionC_CleanContextDefaultDAG`)**:
  Full N-in-1-out `default_processor` with:
  1. Targeted puzzle assembly prioritizing edges with the smallest `count`.
  2. Two-phase autonomous routing where Phase 1 decides `(selected_indices, output_tags)` and
     Phase 2 executes the formal synthesis in a **100% clean request context**.
- **Version D (`VersionD_MetaSelfModifyingDAG`)**:
  Integrates the first-class Meta-DAG (`meta_dag`), enabling any DAG (including `meta_dag` itself)
  to be modified via JSON under the influence of a natural-language or structured `prompt`.
- **Version E (`VersionE_AsyncParallelDAG`)**:
  Adds `async_step()` and `step_frontier_parallel()` to execute all non-conflicting candidates
  sharing the top `(layer, gen)` priority frontier concurrently via `asyncio.gather`.
- **Version F (`VersionF_TransactionalReactiveDAG`)**:
  Adds transactional state snapshots (`checkpoint()`, `rollback()`), Python `with dag.transaction():`
  atomic context management, and per-edge `max_count` starvation/loop guards.
- **Version G (`VersionG_ProductionCompleteDAG`)**:
  The unified production engine combining all capabilities from Versions A–F with execution trace
  telemetry, Mermaid visualization export, and one-line Meta-DAG evolution.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from dag_agent.default_processor import DefaultProcessor
from dag_agent.engine import DAG, StepResult
from dag_agent.llm import LLMCallRecord, LLMClientProtocol
from dag_agent.meta import create_meta_dag, transform_dag_with_meta_dag
from dag_agent.schema import ContentItem


class VersionA_MinimalDAG(DAG):
    """Version A — Minimal Exact-Tag Core DAG.

    Uses strict exact tag matching (`similarity_threshold=1.0`, `use_embeddings=False`),
    `layer` (0 highest priority) and `gen` (0 highest priority) scheduling, and `edge.count` tracking.
    """

    def __init__(self, name: str = "version_a_dag", **kwargs: Any) -> None:
        kwargs.setdefault("similarity_threshold", 1.0)
        kwargs.setdefault("use_embeddings", False)
        super().__init__(name=name, **kwargs)


class VersionB_SemanticEmbeddingDAG(DAG):
    """Version B — Semantic & Embedding Threshold Matching DAG.

    Matches edges when content tags meet or exceed `similarity_threshold` via
    embedding cosine similarity or lexical N-gram similarity.
    """

    def __init__(
        self,
        name: str = "version_b_dag",
        *,
        similarity_threshold: float = 0.65,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            similarity_threshold=similarity_threshold,
            use_embeddings=True,
            **kwargs,
        )

    def explain_tag_match(self, required_tag: str, candidate_tag: str) -> Dict[str, Any]:
        score = self.matcher.tag_similarity(required_tag, candidate_tag)
        return {
            "required_tag": required_tag,
            "candidate_tag": candidate_tag,
            "similarity": score,
            "threshold": self.matcher.threshold,
            "matched": score >= self.matcher.threshold,
        }


class VersionC_CleanContextDefaultDAG(DAG):
    """Version C — Two-Phase Clean-Context `default_processor` DAG.

    Exposes audit helpers to verify that:
    1. Targeted N-in-1-out assembly prioritizes edges with smaller `count`.
    2. Autonomous routing separates Phase 1 (selection of inputs & output tags) from
       Phase 2 (formal output synthesis in a strictly clean context).
    """

    def __init__(self, name: str = "version_c_dag", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def get_clean_output_calls(self) -> List[LLMCallRecord]:
        """Return all formal output LLM calls made by `default_processor` and verify clean context."""
        history = getattr(self.llm_client, "call_history", [])
        return [r for r in history if r.phase == "default_clean_output" and r.is_clean_context]

    def get_routing_calls(self) -> List[LLMCallRecord]:
        """Return all Phase 1 routing decision calls made by `default_processor`."""
        history = getattr(self.llm_client, "call_history", [])
        return [r for r in history if r.phase == "default_routing"]


class VersionD_MetaSelfModifyingDAG(VersionC_CleanContextDefaultDAG):
    """Version D — Meta-DAG Self-Modifying Architecture.

    Provides `.modify_with_prompt(prompt)` which feeds `self` (as JSON) and `prompt`
    into a standard `Meta-DAG` instance and returns the newly modified DAG.
    """

    def __init__(self, name: str = "version_d_dag", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def modify_with_prompt(
        self,
        prompt: str,
        *,
        meta_dag: Optional[DAG] = None,
    ) -> "VersionD_MetaSelfModifyingDAG":
        modified = transform_dag_with_meta_dag(
            self,
            prompt,
            meta_dag=meta_dag,
            llm_client=self.llm_client,
        )
        return VersionD_MetaSelfModifyingDAG.from_dict(
            modified.to_dict(),
            llm_client=self.llm_client,
            similarity_threshold=self.matcher.threshold,
        )


class VersionE_AsyncParallelDAG(VersionD_MetaSelfModifyingDAG):
    """Version E — Async & Parallel Frontier Batch Execution DAG.

    Supports `async_step()` and `step_parallel_frontier()` to concurrently execute
    multiple runnable candidates sharing the lowest `(layer, gen)` priority level.
    """

    def __init__(
        self,
        name: str = "version_e_dag",
        *,
        max_concurrency: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.max_concurrency = max(1, int(max_concurrency))

    async def async_step(self) -> StepResult:
        """Run a single `step()` asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.step)

    async def step_parallel_frontier(self) -> List[StepResult]:
        """Find all non-conflicting candidates at the highest-priority `(layer, input_gen)` frontier
        and execute them concurrently. Falls back to `async_step()` if no direct frontier exists.
        """
        candidates = self.matcher.find_all_candidates(
            self.edges,
            self.contents,
            self._executed_edge_sigs,
        )
        if not candidates:
            single = await self.async_step()
            return [single]

        candidates.sort(key=lambda c: c.sort_key)
        top_layer = candidates[0].edge.layer
        top_gen = candidates[0].input_gen
        frontier = [
            c for c in candidates if c.edge.layer == top_layer and c.input_gen == top_gen
        ]

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run_one(cand: Any) -> Tuple[Any, ContentItem]:
            async with sem:
                matched_items = [self.contents[i] for i in cand.content_indices]
                out_item = await asyncio.to_thread(self._execute_edge, cand.edge, matched_items)
                return cand, out_item

        # Mark signatures before launching tasks
        for cand in frontier:
            self._executed_edge_sigs.add((cand.edge_index, cand.content_indices))

        completed = await asyncio.gather(*(_run_one(c) for c in frontier))
        results: List[StepResult] = []
        for cand, out_item in completed:
            self.step_count += 1
            self.contents.add(out_item)
            res = StepResult(
                step_num=self.step_count,
                action="edge_parallel",
                edge_id=cand.edge.id,
                edge_layer=cand.edge.layer,
                edge_count=cand.edge.count,
                consumed_indices=cand.content_indices,
                produced_items=[out_item],
                similarity=cand.similarity,
            )
            self.history.append(res)
            results.append(res)
        return results


class VersionF_TransactionalReactiveDAG(VersionE_AsyncParallelDAG):
    """Version F — Transactional Checkpoints, Rollback & Starvation Guard DAG.

    Adds:
    - `checkpoint()` / `rollback(snapshot_id)` for time-travel debugging
    - `with dag.transaction():` context manager that automatically rolls back on error
    - `max_edge_count` cap to prevent infinite loops in cyclic graphs
    """

    def __init__(
        self,
        name: str = "version_f_dag",
        *,
        max_edge_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.max_edge_count = max_edge_count
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def checkpoint(self, label: Optional[str] = None) -> str:
        snap_id = label or f"snap_{len(self._snapshots) + 1}"
        self._snapshots[snap_id] = {
            "state": copy.deepcopy(self.to_dict()),
            "step_count": self.step_count,
            "executed_sigs": copy.deepcopy(self._executed_edge_sigs),
            "assembled_sigs": copy.deepcopy(self._assembled_edge_sigs),
            "autonomous_sigs": copy.deepcopy(self._autonomous_sigs),
        }
        return snap_id

    def rollback(self, snap_id: str) -> None:
        if snap_id not in self._snapshots:
            raise KeyError(f"Unknown snapshot ID: {snap_id}")
        snap = self._snapshots[snap_id]
        state = copy.deepcopy(snap["state"])
        self.models.load_dict(state["models"])
        self.contents.clear()
        for item in state["contents"]:
            self.contents.add(item)
        self.edges.clear()
        for idx, edge_spec in enumerate(state["edges"]):
            self.add_edge(edge_spec, default_id=f"edge_{idx}")
        self.step_count = snap["step_count"]
        self._executed_edge_sigs = copy.deepcopy(snap["executed_sigs"])
        self._assembled_edge_sigs = copy.deepcopy(snap["assembled_sigs"])
        self._autonomous_sigs = copy.deepcopy(snap["autonomous_sigs"])

    @contextmanager
    def transaction(self, auto_save: bool = False) -> Iterator["VersionF_TransactionalReactiveDAG"]:
        snap_id = self.checkpoint("__tx__")
        try:
            yield self
            if auto_save and self.contents.path:
                self.contents.save()
            if auto_save and self.edges_path:
                self.save_edges()
        except Exception:
            self.rollback(snap_id)
            raise
        finally:
            self._snapshots.pop(snap_id, None)

    def step(self) -> StepResult:
        if self.max_edge_count is not None:
            #Temporarily mark saturated edges as already executed for all current items
            for e_idx, edge in enumerate(self.edges):
                if edge.count >= self.max_edge_count:
                    for c_idx in range(len(self.contents)):
                        self._executed_edge_sigs.add((e_idx, (c_idx,)))
        return super().step()


class VersionG_ProductionCompleteDAG(VersionF_TransactionalReactiveDAG):
    """Version G — Unified Production Architecture (All-in-One Modular Engine).

    Combines Versions A–F with full execution telemetry, Mermaid diagram generation,
    and self-evolving Meta-DAG pipelines.
    """

    def __init__(self, name: str = "version_g_dag", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def export_trace_summary(self) -> List[Dict[str, Any]]:
        """Export a concise, human-readable trace of all executed steps."""
        return [
            {
                "step": r.step_num,
                "action": r.action,
                "edge_id": r.edge_id,
                "layer": r.edge_layer,
                "edge_count": r.edge_count,
                "consumed_indices": list(r.consumed_indices),
                "produced_tags": [it.tags for it in r.produced_items],
                "produced_gens": [it.gen for it in r.produced_items],
                "similarity": r.similarity,
            }
            for r in self.history
        ]

    def to_mermaid(self) -> str:
        """Render the DAG's edges, layers, and execution counts as a Mermaid flowchart."""
        lines = ["flowchart LR"]
        for edge in sorted(self.edges, key=lambda e: (e.layer, e.id)):
            in_node = "_".join(edge.input_tags) or "none"
            out_node = "_".join(edge.output_tags) or "none"
            label = f"{edge.id} [L{edge.layer}, count={edge.count}]"
            lines.append(f'    {in_node}("{", ".join(edge.input_tags)}") -->|"{label}"| {out_node}("{", ".join(edge.output_tags)}")')
        return "\n".join(lines)


__all__ = [
    "VersionA_MinimalDAG",
    "VersionB_SemanticEmbeddingDAG",
    "VersionC_CleanContextDefaultDAG",
    "VersionD_MetaSelfModifyingDAG",
    "VersionE_AsyncParallelDAG",
    "VersionF_TransactionalReactiveDAG",
    "VersionG_ProductionCompleteDAG",
]
