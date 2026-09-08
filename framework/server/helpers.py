"""Helpers for dashboard assets and dynamic tool catalog discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Template directory resides in framework/templates
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_dashboard_html() -> str:
    """Load dashboard HTML template from file."""
    template_path = _TEMPLATE_DIR / "dashboard.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<html><body><h1>Dashboard template not found</h1></body></html>"


DASHBOARD_HTML = load_dashboard_html()


def get_effective_catalog_dir(catalog_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolve effective tool catalog directory, falling back to repository examples directory."""
    if catalog_dir:
        p = Path(catalog_dir).resolve()
        if p.exists() and p.is_dir():
            return p
    repo_default = Path(__file__).resolve().parent.parent.parent / "examples" / "dynamic_tool_library" / "tools"
    if repo_default.exists() and repo_default.is_dir():
        return repo_default
    return None


def list_tool_catalog(catalog_dir: Path) -> List[Dict[str, Any]]:
    """Enumerate and summarize all JSON tool manifests within catalog directory."""
    tools: List[Dict[str, Any]] = []
    if not catalog_dir.exists() or not catalog_dir.is_dir():
        return tools
    for f in sorted(catalog_dir.glob("*.json")):
        tool_id = f.stem
        try:
            manifest = json.loads(f.read_text(encoding="utf-8"))
            meta = manifest.get("metadata", {})
            vertices = manifest.get("vertices", [])
            edges = manifest.get("edges", [])
            tools.append({
                "id": tool_id,
                "name": meta.get("name", tool_id),
                "description": meta.get("description", ""),
                "category": meta.get("category", "general"),
                "entry_vertex": meta.get("entry_vertex", "sub_in"),
                "exit_vertex": meta.get("exit_vertex", "sub_out"),
                "vertex_count": len(vertices),
                "edge_count": len(edges),
                "manifest_path": str(f.resolve()),
            })
        except Exception as exc:
            tools.append({
                "id": tool_id,
                "name": tool_id,
                "description": f"Failed reading manifest: {exc}",
                "manifest_path": str(f.resolve()),
            })
    return tools


# Backward compatibility aliases
_load_dashboard_html = load_dashboard_html
_get_effective_catalog_dir = get_effective_catalog_dir
_list_tool_catalog = list_tool_catalog
