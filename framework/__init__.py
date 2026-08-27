"""Vertex-Edge Agent Framework - Non-interactive graph execution engine.

顶点-边(Vertex-Edge) Agent 框架 —— 一个非交互式、由 JSON 配置驱动的图执行引擎，
用于编排 AI Agent 流水线。

本包对外暴露以下公共 API：
    Vertex / VertexState / DataRejectedError  —— 顶点、状态机、数据拒绝异常
    Edge                                      —— 边(连接顶点，经 AI 处理数据)
    Graph                                     —— 从 JSON 加载并校验的 DAG
    Executor / ExecutionResult                —— 异步执行器与执行结果
    PIAgent / MockPIAgent / ExternalPIAgent   —— AI 处理接口及实现
    load_script                               —— 动态加载外部 Python 脚本
    build_archive / save_archive              —— 把一次图运行存档为 JSON 落盘
"""

# 导出顶点相关类型
from .vertex import Vertex, VertexState, DataRejectedError
from .edge import Edge
from .graph import Graph
from .executor import Executor, ExecutionResult
from .pi_agent import (
    PIAgent, MockPIAgent, ExternalPIAgent, PICLIPIAgent, OpenCodeAgent, OpenAIAgent,
    OPENCODE_PROXIES,
)
from .signal import AbortSignal, is_abort, abort_reason
from .script_loader import load_script
from .archive import (
    snapshot_initial_data,
    build_archive,
    save_archive,
)

__all__ = [
    'Vertex', 'VertexState', 'DataRejectedError',
    'Edge',
    'Graph',
    'Executor', 'ExecutionResult',
    'PIAgent', 'MockPIAgent', 'ExternalPIAgent', 'PICLIPIAgent', 'OpenCodeAgent', 'OpenAIAgent',
    'OPENCODE_PROXIES',
    'AbortSignal', 'is_abort', 'abort_reason',
    'load_script',
    'snapshot_initial_data', 'build_archive', 'save_archive',
]

__version__ = "1.0.0"
