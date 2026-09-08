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
from .server_v4 import (
    create_v4_server,
    SessionGraphManagerV4,
)
from .sse_executor_v4 import (
    SSEExecutorV4,
    ToolCallEcho,
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

# Exported lazily so that ``python -m framework.run_v4`` does not trip runpy's
# "already in sys.modules" warning, and so importing the package does not eagerly
# pull the CLI runner into every consumer.
_RUN_V4_EXPORTS = ("load_v4_manifest", "run_from_manifest", "run_manifest_async")


def __getattr__(name):
    if name in _RUN_V4_EXPORTS:
        from .run_v4 import (
            load_v4_manifest,
            run_from_manifest,
            run_manifest_async,
        )
        return {
            "load_v4_manifest": load_v4_manifest,
            "run_from_manifest": run_from_manifest,
            "run_manifest_async": run_manifest_async,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

