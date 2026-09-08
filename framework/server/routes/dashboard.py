"""Dashboard UI routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

from framework.edges.registry import edge_type_choices
from framework.server.helpers import load_dashboard_html, _TEMPLATE_DIR

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serve live single-page dashboard."""
    return HTMLResponse(content=load_dashboard_html())


@router.get("/api/edge-types")
async def get_edge_types(request: Request) -> Dict[str, Any]:
    """List edge types the graph editor may use.

    Registered types come from :data:`framework.edges.registry.EDGE_REGISTRY`, so
    a custom edge registered with ``@register_edge_type`` shows up here without
    any UI change. Script specs (``"my_edge.py:MyEdge"``) are accepted as free
    text in the editor, subject to the server's script-root confinement.
    """
    roots = getattr(request.app.state, "script_roots", None) or []
    return {
        "types": edge_type_choices(),
        "script_spec_supported": True,
        "script_spec_example": "my_edge.py:MyEdge",
        "script_roots_configured": bool(roots),
    }


@router.get("/dashboard/app.js")
async def get_dashboard_js() -> Response:
    """Serve dashboard JavaScript."""
    p = _TEMPLATE_DIR / "dashboard.js"
    return Response(
        content=p.read_text(encoding="utf-8") if p.exists() else "",
        media_type="application/javascript",
    )


@router.get("/dashboard/app.css")
async def get_dashboard_css() -> Response:
    """Serve dashboard CSS stylesheet."""
    p = _TEMPLATE_DIR / "dashboard.css"
    return Response(
        content=p.read_text(encoding="utf-8") if p.exists() else "",
        media_type="text/css",
    )
