"""Server API route modules."""

from framework.server.routes.dashboard import router as dashboard_router
from framework.server.routes.db import router as db_router
from framework.server.routes.graph import router as graph_router
from framework.server.routes.snapshots import router as snapshots_router
from framework.server.routes.execution import router as execution_router
from framework.server.routes.tools import router as tools_router
from framework.server.routes.openai import router as openai_router, _graph_templates, register_graph_template

__all__ = [
    "dashboard_router",
    "db_router",
    "graph_router",
    "snapshots_router",
    "execution_router",
    "tools_router",
    "openai_router",
    "_graph_templates",
    "register_graph_template",
]
