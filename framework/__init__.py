import importlib

from .executor import Executor, ExecutionResult, GraphEvent, CheckpointedExecutor, HumanGateVertex, ExecutorHooks
from .agents import (
    BaseAgent,
    MockAgent,
    HttpLLMAgent,
    NonRetryableHTTPError,
    ThrottleTimeoutError,
    PiAgentRunner,
    get_agent,
)
from .vertex import Vertex, VertexState, EdgeSignal, DataRejectedError
from .pipeline import Pipeline
from .edge import Edge
from .subgraph import SubgraphVertex
from .graph import Graph
from .utils.store import BaseStateStore, SQLiteStateStore, GraphSnapshot
from .utils.script_loader import load_script
from .utils.errors import (
    FrameworkError, ExecutionError, GuardAbortError, AbortPipeline,
    HookError, ComputeError, SubgraphError,
)

from .utils.memory import MemoryStore
from .utils.telemetry import TelemetryTracker, UsageMetrics, DEFAULT_PRICING, calculate_cost, estimate_tokens
from .utils.schema import SchemaRegistry, SchemaMismatchError
from .builders.chain import LinearChain
from .builders.builder import GraphBuilder

# V4 Core Architecture
from .vertex_v4 import (
    VertexStateV4,
    TraversalColor,
    VertexAttributeV4,
    VertexRecordV4,
    VertexV4,
    EdgeRecordV4,
    EdgeMetricRecordV4,
    EdgeMetricV4,
    StagingRecordV4,
    VertexStoreV4,
    MergeStrategyV4,
)
from .edge_v4 import (
    EdgeV4,
    CodeEdgeV4,
    ToolEdgeV4,
    LLMToolEdgeV4,
    LLMEdgeV4,
    MockAgentV4,
    ChatLLMEdgeV4,
    GenerateLLMEdgeV4,
    ProcessLLMEdgeV4,
    CallableLLMEdgeV4,
    ReflexiveEdgeV4,
    EdgeResultV4,
)
from .sensenova_edge_v4 import SensenovaEdgeV4, SenseNovaEdgeV4
from .edges.registry import register_edge_type, edge_type_choices
from .snapshot_v4 import GraphSnapshotManagerV4
from .graph_v4 import (
    GraphV4,
    DiscreteGraphLoaderV4,
    GraphTopologyError,
    NodeColor,
)
from .executor_v4 import (
    ExecutorV4,
    OrchestratorV4,
    ExecutionResultV4,
    GraphEventV4,
)
from .worker_queue_v4 import (
    BaseWorkerQueueV4,
    InMemoryWorkerQueueV4,
    EdgeTaskPayload,
    EdgeTaskResult,
)
from .workflow_executor_v4 import (
    WorkflowExecutorV4,
    WorkflowEventV4,
    WorkflowResultV4,
)
from .http_executor_v4 import (
    HttpHarnessExecutorV4,
)

__all__ = [
    'VertexState', 'Vertex', 'EdgeSignal', 'DataRejectedError',
    'Pipeline', 'Edge', 'SubgraphVertex', 'Graph', 'Executor', 'ExecutionResult', 'GraphEvent', 'ExecutorHooks',
    'MemoryStore', 'TelemetryTracker', 'UsageMetrics', 'DEFAULT_PRICING', 'calculate_cost', 'estimate_tokens',
    'SchemaRegistry', 'SchemaMismatchError',
    'FrameworkError', 'ExecutionError', 'GuardAbortError', 'AbortPipeline',
    'HookError', 'ComputeError', 'SubgraphError',
    'LinearChain', 'GraphBuilder',
    'BaseStateStore', 'SQLiteStateStore', 'GraphSnapshot',
    'CheckpointedExecutor', 'HumanGateVertex',
    'BaseAgent', 'MockAgent', 'HttpLLMAgent', 'NonRetryableHTTPError',
    'ThrottleTimeoutError',
    'load_script',
    # V4 exports
    'VertexStateV4', 'TraversalColor', 'VertexAttributeV4', 'VertexRecordV4', 'VertexV4',
    'EdgeRecordV4', 'EdgeMetricRecordV4', 'EdgeMetricV4',
    'StagingRecordV4', 'VertexStoreV4', 'MergeStrategyV4',
    'EdgeV4', 'CodeEdgeV4', 'ToolEdgeV4', 'LLMToolEdgeV4', 'LLMEdgeV4', 'MockAgentV4',
    'ChatLLMEdgeV4', 'GenerateLLMEdgeV4', 'ProcessLLMEdgeV4', 'CallableLLMEdgeV4',
    'ReflexiveEdgeV4', 'EdgeResultV4', 'SensenovaEdgeV4', 'SenseNovaEdgeV4',
    'register_edge_type', 'edge_type_choices',
    'GraphV4', 'DiscreteGraphLoaderV4', 'GraphTopologyError', 'NodeColor',
    'GraphSnapshotManagerV4',
    'ExecutorV4', 'OrchestratorV4', 'ExecutionResultV4', 'GraphEventV4',
    'BaseWorkerQueueV4', 'InMemoryWorkerQueueV4', 'EdgeTaskPayload', 'EdgeTaskResult',
    'create_v4_server', 'SessionGraphManagerV4',
    'load_v4_manifest', 'run_from_manifest', 'run_manifest_async',
    'SSEExecutorV4', 'ToolCallEcho', 'HttpHarnessExecutorV4',
    'WorkflowExecutorV4', 'WorkflowEventV4', 'WorkflowResultV4',
]

# Exported lazily. Three reasons:
# - ``.server_v4`` and ``.sse_executor_v4`` import FastAPI, so resolving them only
#   on demand keeps the core engine usable without a web framework installed
#   (``import framework`` must not require one).
# - ``.run_v4`` is a CLI runner; pulling it in eagerly would slow every consumer.
# - An eagerly imported submodule makes ``python -m framework.run_v4`` print
#   runpy's "already in sys.modules" warning.
# Symbols resolve to the same objects as before; only *when* they load changed.
_LAZY_EXPORTS = {
    # module (relative to this package), attribute
    "create_v4_server": (".server_v4", "create_v4_server"),
    "SessionGraphManagerV4": (".server_v4", "SessionGraphManagerV4"),
    "SSEExecutorV4": (".sse_executor_v4", "SSEExecutorV4"),
    "ToolCallEcho": (".sse_executor_v4", "ToolCallEcho"),
    "load_v4_manifest": (".run_v4", "load_v4_manifest"),
    "run_from_manifest": (".run_v4", "run_from_manifest"),
    "run_manifest_async": (".run_v4", "run_manifest_async"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    return getattr(importlib.import_module(module, __name__), attribute)

