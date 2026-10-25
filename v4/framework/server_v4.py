"""V4 Standalone Server and Database Viewer (Modularized Façade).

All internal server components have been refactored into modular packages under ``framework.server``:
- Schemas: ``framework.server.schemas``
- Session Manager: ``framework.server.manager``
- Routes: ``framework.server.routes.*``
- Application Factory: ``framework.server.app``

This module preserves 100% backward-compatible imports and CLI entrypoints.
"""

from __future__ import annotations

from framework.server import (
    EdgeCreateOrUpdateRequest,
    EdgeReconnectRequest,
    RouteAndRunRequest,
    SessionGraphManagerV4,
    SubgraphAddRequest,
    SubgraphInsertRequest,
    SubgraphSpliceRequest,
    VertexCreateOrUpdateRequest,
    VertexReentryRequest,
    WorkflowRunRequest,
    app,
    create_v4_server,
    main,
    parse_server_args,
)
from framework.server.helpers import (
    DASHBOARD_HTML,
    _TEMPLATE_DIR,
    _get_effective_catalog_dir,
    _list_tool_catalog,
    _load_dashboard_html,
)
from framework.server.security import _validate_path_security, validate_path_security

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
    "DASHBOARD_HTML",
    "_load_dashboard_html",
    "_TEMPLATE_DIR",
    "_validate_path_security",
    "validate_path_security",
    "_get_effective_catalog_dir",
    "_list_tool_catalog",
]

if __name__ == "__main__":
    main()
