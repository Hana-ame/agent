"""Modular server package for VEA v4."""

from framework.server.app import create_v4_server, parse_server_args, main, app
from framework.server.manager import SessionGraphManagerV4
from framework.server.schemas import (
    VertexCreateOrUpdateRequest,
    EdgeCreateOrUpdateRequest,
    WorkflowRunRequest,
    VertexReentryRequest,
    EdgeReconnectRequest,
    SubgraphSpliceRequest,
    SubgraphInsertRequest,
    SubgraphAddRequest,
    RouteAndRunRequest,
)

__all__ = [
    "create_v4_server",
    "SessionGraphManagerV4",
    "parse_server_args",
    "main",
    "app",
    "VertexCreateOrUpdateRequest",
    "EdgeCreateOrUpdateRequest",
    "WorkflowRunRequest",
    "VertexReentryRequest",
    "EdgeReconnectRequest",
    "SubgraphSpliceRequest",
    "SubgraphInsertRequest",
    "SubgraphAddRequest",
    "RouteAndRunRequest",
]
