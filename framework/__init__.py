"""Vertex-Edge Agent Framework - Non-interactive graph execution engine."""

from .vertex import Vertex, VertexState, DataRejectedError
from .edge import Edge
from .graph import Graph
from .executor import Executor, ExecutionResult
from .pi_agent import PIAgent, MockPIAgent, HttpPIAgent, ExternalPIAgent
from .script_loader import load_script

__all__ = [
    'VertexState',
    'Vertex',
    'EdgeSignal',
    'DataRejectedError',
    'Edge',
    'Graph',
    'Executor', 'ExecutionResult',
    'PIAgent', 'MockPIAgent', 'HttpPIAgent', 'ExternalPIAgent',
    'load_script',
]

__version__ = "1.0.0"
