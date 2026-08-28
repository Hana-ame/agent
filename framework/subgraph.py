"""Nested Sub-Graph Vertex module.

Allows a Vertex to encapsulate an entire independent child Graph ("Box-in-a-Box").
Translates input channel data from the parent graph to the inner graph's sources,
and hoists output data from the inner graph's sinks back to the parent vertex.
"""

import logging
from typing import Any, Dict, List, Optional

from .vertex import Vertex
from .graph import Graph

logger = logging.getLogger("vertex_edge_agent.subgraph")


class SubgraphVertex(Vertex):
    """A composite Vertex that executes an internal sub-graph.

    Configuration settings expected in ``settings``:
    - ``graph_config`` (Dict or str): The sub-graph configuration dict or path to JSON.
    - ``input_map`` (Dict[str, str]): Maps parent input channel/key to inner vertex channel
      (e.g., ``{"parent_channel": "InnerVertex.inner_channel"}`` or ``{"parent_channel": "InnerVertex"}``).
    - ``output_map`` (Dict[str, str]): Maps inner vertex channel to parent output channel
      (e.g., ``{"InnerVertex.inner_channel": "parent_channel"}`` or ``{"InnerVertex": "parent_channel"}``).
    """

    def __init__(
        self,
        vertex_id: str,
        settings: Optional[Dict] = None,
        initial_data: Optional[List[Dict]] = None,
    ):
        super().__init__(
            vertex_id=vertex_id,
            settings=settings,
            initial_data=initial_data,
        )
        self.graph_config: Dict = self.settings.get("graph_config", {})
        self.input_map: Dict[str, str] = self.settings.get("input_map", {})
        self.output_map: Dict[str, str] = self.settings.get("output_map", {})
        self.inner_graph: Optional[Graph] = None

    def initialize_inner_graph(self) -> Graph:
        """Parse and instantiate the inner Graph from ``graph_config``."""
        if isinstance(self.graph_config, str):
            self.inner_graph = Graph.from_json(self.graph_config)
        elif isinstance(self.graph_config, dict):
            self.inner_graph = Graph.from_dict(self.graph_config)
        else:
            raise ValueError(
                f"[SubgraphVertex:{self.id}] Invalid graph_config type: {type(self.graph_config)}"
            )
        return self.inner_graph

    async def stage_inner_inputs(self, inner_graph: Optional[Graph] = None) -> None:
        """Route incoming data in this SubgraphVertex to the inner graph's source vertices."""
        target_graph = inner_graph or self.inner_graph
        if target_graph is None:
            target_graph = self.initialize_inner_graph()

        parent_data = await self.get_all_data()
        logger.debug(
            "[SubgraphVertex:%s] Staging inputs to inner graph: parent_data=%s, input_map=%s",
            self.id, list(parent_data.keys()), self.input_map,
        )

        for parent_key, inner_target in self.input_map.items():
            if parent_key not in parent_data:
                logger.warning(
                    "[SubgraphVertex:%s] parent_key '%s' specified in input_map not found in vertex data",
                    self.id, parent_key,
                )
                continue

            val = parent_data[parent_key]

            # Parse "InnerVertex.inner_channel" or "InnerVertex"
            if "." in inner_target:
                inner_vid, inner_channel = inner_target.split(".", 1)
            else:
                inner_vid = inner_target
                inner_channel = parent_key

            if inner_vid not in target_graph.vertices:
                raise KeyError(
                    f"[SubgraphVertex:{self.id}] Inner vertex '{inner_vid}' target of input_map not found in inner graph"
                )

            inner_v = target_graph.vertices[inner_vid]
            # Inject directly into inner vertex store
            await inner_v.set_data(inner_channel, val)
            logger.debug(
                "[SubgraphVertex:%s] Mapped parent key '%s' -> Inner [%s] channel '%s'",
                self.id, parent_key, inner_vid, inner_channel,
            )

    async def collect_inner_outputs(self, inner_graph: Optional[Graph] = None) -> None:
        """Hoist final data from inner graph sink vertices back into this SubgraphVertex's store."""
        target_graph = inner_graph or self.inner_graph
        if target_graph is None:
            return

        logger.debug(
            "[SubgraphVertex:%s] Collecting outputs from inner graph: output_map=%s",
            self.id, self.output_map,
        )

        for inner_source, parent_channel in self.output_map.items():
            if "." in inner_source:
                inner_vid, inner_channel = inner_source.split(".", 1)
            else:
                inner_vid = inner_source
                inner_channel = None

            if inner_vid not in target_graph.vertices:
                logger.warning(
                    "[SubgraphVertex:%s] Inner vertex '%s' in output_map not found in inner graph",
                    self.id, inner_vid,
                )
                continue

            inner_v = target_graph.vertices[inner_vid]
            inner_data = await inner_v.get_all_data()

            if inner_channel:
                if inner_channel in inner_data:
                    await self.set_data(parent_channel, inner_data[inner_channel])
                    logger.debug(
                        "[SubgraphVertex:%s] Hoisted Inner [%s].%s -> parent channel '%s'",
                        self.id, inner_vid, inner_channel, parent_channel,
                    )
            else:
                # If inner_channel not specified, copy entire dict or first value
                if len(inner_data) == 1:
                    first_val = next(iter(inner_data.values()))
                    await self.set_data(parent_channel, first_val)
                else:
                    await self.set_data(parent_channel, inner_data)
                logger.debug(
                    "[SubgraphVertex:%s] Hoisted Inner [%s] -> parent channel '%s'",
                    self.id, inner_vid, parent_channel,
                )
