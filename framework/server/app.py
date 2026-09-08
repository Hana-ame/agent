"""FastAPI Application Factory and Server Entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from framework.server.manager import SessionGraphManagerV4
from framework.server.routes import (
    dashboard_router,
    db_router,
    execution_router,
    graph_router,
    openai_router,
    snapshots_router,
    tools_router,
    _graph_templates,
    register_graph_template,
)
from framework.vertex_v4 import VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.server_v4")


def create_v4_server(
    store_or_db: Union[VertexStoreV4, str, Path] = ":memory:",
    manager: Optional[SessionGraphManagerV4] = None,
    agent: Optional[Any] = None,
    allowed_origins: List[str] = ["*"],
    manifest_base_dir: Optional[Union[str, Path]] = None,
    snapshot_dir: Optional[Union[str, Path]] = "snapshots",
    tool_catalog_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """Create a FastAPI application powering online graph APIs and database dashboard."""
    if isinstance(store_or_db, VertexStoreV4):
        store = store_or_db
    else:
        store = VertexStoreV4(str(store_or_db))

    if manager is None:
        manager = SessionGraphManagerV4(store=store, snapshot_dir=snapshot_dir)
    else:
        store = manager.store

    app = FastAPI(
        title="VEA v4 Online Graph & Database Server",
        version="4.0.0",
        description="Online graph modification API and standalone database inspector.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared instances to app state
    app.state.store = store
    app.state.manager = manager
    app.state.agent = agent
    app.state.manifest_base_dir = manifest_base_dir
    app.state.tool_catalog_dir = tool_catalog_dir
    app.state.graph_templates = _graph_templates
    app.state.register_graph_template = register_graph_template

    # Register modular routers
    app.include_router(dashboard_router)
    app.include_router(db_router)
    app.include_router(graph_router)
    app.include_router(snapshots_router)
    app.include_router(execution_router)
    app.include_router(tools_router)
    app.include_router(openai_router)

    return app


def parse_server_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone server."""
    parser = argparse.ArgumentParser(description="VEA v4 Online Graph & Database Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=11434, help="Port to bind (default: 11434, same as Ollama)")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    parser.add_argument("--snapshot-dir", default="snapshots", help="Directory for complete graph JSON snapshots")
    parser.add_argument("--tool-catalog-dir", default=None, help="Directory for dynamic tool catalog JSON manifests")
    return parser.parse_args()


def main() -> None:
    """Run server directly from CLI."""
    import uvicorn
    args = parse_server_args()
    server_app = create_v4_server(
        store_or_db=args.db,
        snapshot_dir=args.snapshot_dir,
        tool_catalog_dir=args.tool_catalog_dir,
    )
    uvicorn.run(server_app, host=args.host, port=args.port)


# Module-level app for ``uvicorn framework.server.app:app``
app = create_v4_server()

if __name__ == "__main__":
    main()
