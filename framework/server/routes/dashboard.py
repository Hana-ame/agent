"""Dashboard UI routes."""

from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

from framework.server.helpers import load_dashboard_html, _TEMPLATE_DIR

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serve live single-page dashboard."""
    return HTMLResponse(content=load_dashboard_html())


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
