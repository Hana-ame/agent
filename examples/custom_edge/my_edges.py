"""Example custom edge — no framework registration required.

Reference it from any graph/edge JSON with::

    "type": "my_edges.py:WordCountEdge"

The loader resolves the class, maps the JSON fields onto its constructor (dropping
kwargs it does not accept) and instantiates it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from framework.edges.base import EdgeResultV4, EdgeV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4


class WordCountEdge(EdgeV4):
    """Counts whitespace-separated words in the upstream vertex content."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        settings: Optional[Dict[str, Any]] = None,
    ):
        # A deliberately minimal constructor: no concurrency/timeout kwargs.
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="word_count",
            settings=settings,
        )

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Any = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(edge_id=self.id, success=False, skipped=True, reason=reason)

        count = len(str(in_v.content).split())
        label = self.settings.get("label", "words")
        payload = f"{count} {label}"

        store.apply_merge_strategy(
            session_id=session_id,
            name=self.output_vertex,
            incoming_content=payload,
        )
        if auto_transition:
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
        return EdgeResultV4(edge_id=self.id, success=True, output=payload)


class UpperCaseEdge(EdgeV4):
    """Uppercases the upstream content (shows a second class in the same file)."""

    def __init__(self, edge_id: str, input_vertex: str, output_vertex: str, settings: Optional[Dict[str, Any]] = None):
        super().__init__(edge_id=edge_id, input_vertex=input_vertex, output_vertex=output_vertex,
                         edge_type="upper", settings=settings)

    async def run(self, session_id: str, store: VertexStoreV4, agent: Any = None,
                  auto_transition: bool = True, **kwargs: Any) -> EdgeResultV4:
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(edge_id=self.id, success=False, skipped=True, reason=reason)
        payload = str(in_v.content).upper()
        store.apply_merge_strategy(session_id=session_id, name=self.output_vertex, incoming_content=payload)
        if auto_transition:
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
        return EdgeResultV4(edge_id=self.id, success=True, output=payload)
