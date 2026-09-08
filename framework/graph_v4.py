"""V4 Discrete Graph Manifest and Topology Specification (Façade).

This module re-exports all graph classes and exceptions from the modular
`framework.graphs` package to preserve 100% backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap repository root into sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.graphs import (
    DiscreteGraphLoaderV4,
    GraphTopologyError,
    GraphV4,
    NodeColor,
)

__all__ = [
    "GraphV4",
    "DiscreteGraphLoaderV4",
    "GraphTopologyError",
    "NodeColor",
]
