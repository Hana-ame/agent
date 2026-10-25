"""FastAPI Application Factory and Server Entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from fastapi import Depends, FastAPI
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
from framework.server.security import (
    API_KEY_ENV,
    enforce_session_id,
    get_configured_api_key,
    install_api_key_middleware,
    is_loopback_host,
)
from framework.utils.paths import default_manifest_base_dir
from framework.utils.script_loader import SCRIPT_ROOTS_ENV
from framework.vertex_v4 import VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.server_v4")


def create_v4_server(
    store_or_db: Union[VertexStoreV4, str, Path] = ":memory:",
    manager: Optional[SessionGraphManagerV4] = None,
    agent: Optional[Any] = None,
    allowed_origins: Optional[List[str]] = None,
    manifest_base_dir: Optional[Union[str, Path]] = None,
    snapshot_dir: Optional[Union[str, Path]] = "snapshots",
    tool_catalog_dir: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> FastAPI:
    """Create a FastAPI application powering online graph APIs and database dashboard.

    Security defaults:

    * ``allowed_origins`` is **empty** (no CORS) unless the caller opts in.
    * When ``api_key`` is provided (or ``VEA_API_KEY`` is set) every ``/api/*``
      and ``/v1/*`` route requires ``X-API-Key`` / ``Authorization: Bearer``.
    * Client-supplied manifest paths are confined to ``manifest_base_dir``.
    * Client-supplied edge script specs (``"my_edge.py:MyEdge"``) are confined to
      ``script_roots`` when provided, otherwise to the repository root plus
      ``VEA_SCRIPT_ROOTS``.
    """
    if isinstance(store_or_db, VertexStoreV4):
        store = store_or_db
    else:
        store = VertexStoreV4(str(store_or_db))

    if manager is None:
        manager = SessionGraphManagerV4(
            store=store,
            snapshot_dir=snapshot_dir,
            script_roots=script_roots,
        )
    else:
        store = manager.store

    app = FastAPI(
        title="VEA v4 Online Graph & Database Server",
        version="4.0.0",
        description="Online graph modification API and standalone database inspector.",
    )

    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Attach shared instances to app state
    app.state.store = store
    app.state.manager = manager
    app.state.agent = agent
    app.state.manifest_base_dir = (
        Path(manifest_base_dir).resolve() if manifest_base_dir else default_manifest_base_dir()
    )
    app.state.tool_catalog_dir = tool_catalog_dir
    app.state.graph_templates = _graph_templates
    app.state.register_graph_template = register_graph_template
    app.state.script_roots = [Path(p).resolve() for p in (script_roots or [])]

    # Register modular routers. Session-scoped routers validate the
    # ``session_id`` path parameter (it becomes a directory name on disk).
    _session_guard = [Depends(enforce_session_id)]
    app.include_router(dashboard_router)
    app.include_router(db_router, dependencies=_session_guard)
    app.include_router(graph_router, dependencies=_session_guard)
    app.include_router(snapshots_router, dependencies=_session_guard)
    app.include_router(execution_router, dependencies=_session_guard)
    app.include_router(tools_router, dependencies=_session_guard)
    app.include_router(openai_router)

    # Auto-register graph templates from examples/ directory
    _auto_register_graph_templates(app)

    # Authentication boundary (no-op unless a key is configured)
    install_api_key_middleware(app, get_configured_api_key(api_key))

    return app


def _auto_register_graph_templates(app: FastAPI) -> None:
    """Scan examples/ for V4 manifests and register them as model templates.

    Only manifests that actually parse as V4 graphs are registered, so legacy V1
    ``config.json`` files are not advertised as unusable models.
    """
    import json

    from framework.graph_v4 import DiscreteGraphLoaderV4

    examples_dir = Path(__file__).resolve().parent.parent.parent / "examples"
    if not examples_dir.is_dir():
        return

    for subdir in sorted(examples_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Prefer manifest.json, fall back to config.json
        for filename in ("manifest.json", "config.json"):
            fpath = subdir / filename
            if not fpath.is_file():
                continue
            try:
                cfg = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping unreadable template %s: %s", fpath, exc)
                break
            if not _is_v4_manifest(cfg):
                logger.debug("Skipping non-V4 template %s", fpath)
                break
            try:
                DiscreteGraphLoaderV4.load_from_dict(cfg, base_dir=fpath.parent)
            except Exception as exc:
                logger.debug("Skipping unloadable template %s: %s", fpath, exc)
                break
            register_graph_template(
                name=subdir.name,
                manifest_path=str(fpath),
                config=cfg,
            )
            logger.info("Registered graph template '%s' from %s", subdir.name, fpath)
            break  # only use the first matching file


def _is_v4_manifest(cfg: Any) -> bool:
    """Return True when ``cfg`` looks like a V4 graph manifest (dict vertices with names)."""
    if not isinstance(cfg, dict):
        return False
    vertices = cfg.get("vertices")
    if not isinstance(vertices, list):
        return False
    dict_vertices = [v for v in vertices if isinstance(v, dict)]
    if not dict_vertices:
        return False
    return all("name" in v for v in dict_vertices)


def parse_server_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone server."""
    parser = argparse.ArgumentParser(description="VEA v4 Online Graph & Database Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: loopback only)")
    parser.add_argument("--port", type=int, default=11434, help="Port to bind (default: 11434, same as Ollama)")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    parser.add_argument("--snapshot-dir", default="snapshots", help="Directory for complete graph JSON snapshots")
    parser.add_argument("--tool-catalog-dir", default=None, help="Directory for dynamic tool catalog JSON manifests")
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"Require this key on /api/* and /v1/* routes (default: ${API_KEY_ENV}). "
        "Mandatory when binding a non-loopback --host.",
    )
    parser.add_argument(
        "--manifest-base-dir",
        default=None,
        help="Root directory that client-supplied subgraph manifest paths must stay inside "
        "(default: repository root)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        dest="cors_origins",
        help="Allow a browser origin (repeatable). CORS is disabled when omitted.",
    )
    parser.add_argument(
        "--script-root",
        action="append",
        default=None,
        dest="script_roots",
        help="Directory that client-supplied edge script specs may load from (repeatable). "
        f"Defaults to the repository root plus ${SCRIPT_ROOTS_ENV}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run server directly from CLI."""
    import uvicorn

    args = parse_server_args()
    api_key = get_configured_api_key(args.api_key)

    if not is_loopback_host(args.host) and not api_key:
        raise SystemExit(
            f"Refusing to bind {args.host} without authentication.\n"
            f"Set --api-key or export {API_KEY_ENV}=<secret> first, or use --host 127.0.0.1."
        )

    server_app = create_v4_server(
        store_or_db=args.db,
        snapshot_dir=args.snapshot_dir,
        tool_catalog_dir=args.tool_catalog_dir,
        manifest_base_dir=args.manifest_base_dir,
        allowed_origins=args.cors_origins,
        api_key=api_key,
        script_roots=args.script_roots,
    )
    uvicorn.run(server_app, host=args.host, port=args.port)


# Module-level app for ``uvicorn framework.server.app:app``
app = create_v4_server()

if __name__ == "__main__":
    main()
