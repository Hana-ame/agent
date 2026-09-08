"""VEA V4 Modular Graph Package.

Provides discrete graph topology specification, DAG validation,
dynamic subgraph operations, and storage synchronization:
- core: GraphV4, NodeColor
- validation: detect_cycles_and_order, validate_topology, compute_dag_tiers
- subgraph_ops: splice, insert, add subgraph, reenter vertex, reset affected vertices
- loader: DiscreteGraphLoaderV4, load_from_store_fn
- exceptions: GraphTopologyError
"""

from framework.graphs.core import GraphV4, NodeColor
from framework.graphs.exceptions import GraphTopologyError
from framework.graphs.loader import DiscreteGraphLoaderV4

__all__ = [
    "GraphV4",
    "DiscreteGraphLoaderV4",
    "GraphTopologyError",
    "NodeColor",
]
