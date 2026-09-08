"""Security helpers: API-key authentication and path-traversal validation."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from framework.utils.paths import (
    REPO_ROOT,
    PathNotAllowedError,
    default_manifest_base_dir,
    resolve_confined_path,
    snapshot_session_dir,
    validate_session_id as _validate_session_id_plain,
)

logger = logging.getLogger("vertex_edge_agent.server.security")

#: Environment variable holding the API key that protects ``/api/*`` and ``/v1/*``.
API_KEY_ENV = "VEA_API_KEY"

#: Route prefixes that require the API key when one is configured.
DEFAULT_PROTECTED_PREFIXES = ("/api/", "/v1/")

#: Paths that stay reachable without a key so the dashboard shell can load.
DEFAULT_EXEMPT_PATHS = ("/dashboard", "/dashboard/", "/health", "/docs", "/openapi.json", "/redoc")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "localhost.localdomain"})


def get_configured_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return the API key to enforce, or ``None`` when auth is disabled.

    An explicit argument wins; otherwise ``VEA_API_KEY`` is consulted. An empty
    string is treated as "not configured" so that a stray export cannot lock
    everyone out.
    """
    if explicit is not None:
        key = explicit.strip()
        return key or None
    return (os.environ.get(API_KEY_ENV) or "").strip() or None


def is_loopback_host(host: str) -> bool:
    """Return True when ``host`` only exposes the server to the local machine."""
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def _extract_presented_key(request: Request) -> Optional[str]:
    raw = request.headers.get("x-api-key")
    if raw:
        return raw.strip()
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def install_api_key_middleware(
    app: FastAPI,
    api_key: Optional[str],
    protected_prefixes: Sequence[str] = DEFAULT_PROTECTED_PREFIXES,
    exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
) -> None:
    """Reject unauthenticated requests to protected routes when a key is set.

    When ``api_key`` is ``None`` the middleware is not installed at all, which
    keeps the library/embedded and test use-cases unchanged. Production
    deployments are expected to set ``VEA_API_KEY`` (see ``parse_server_args``).
    """
    if not api_key:
        return

    exempt = set(exempt_paths)

    @app.middleware("http")
    async def _api_key_guard(request: Request, call_next):
        path = request.url.path
        if path in exempt or not any(path.startswith(p) for p in protected_prefixes):
            return await call_next(request)

        presented = _extract_presented_key(request)
        if not presented or not secrets.compare_digest(presented, api_key):
            logger.warning(
                "Rejected unauthenticated request: %s %s from %s",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing or invalid API key",
                    "error_code": "UNAUTHORIZED",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    logger.info("API-key authentication enabled for prefixes: %s", ", ".join(protected_prefixes))


def validate_path_security(path_str: str, base_dir: Path) -> Path:
    """Validate that a path doesn't escape the allowed base directory.

    Raises HTTPException(400) with structured error if path is invalid.
    """
    try:
        return resolve_confined_path(path_str, base_dir)
    except PathNotAllowedError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"Invalid path: {exc}",
                "error_code": "PATH_TRAVERSAL_REJECTED",
            },
        ) from exc


def resolve_manifest_path(
    path_str: str,
    base_dir: Optional[Union[str, Path]] = None,
    must_be_json: bool = True,
) -> Path:
    """Resolve and confine a client-supplied manifest path (HTTP 400 on failure)."""
    if must_be_json and not str(path_str).endswith(".json"):
        # Plain-string detail kept for backward compatibility with existing clients.
        raise HTTPException(status_code=400, detail="Manifest path must end in .json")
    try:
        return resolve_confined_path(path_str, base_dir, must_be_json=False)
    except PathNotAllowedError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"Invalid manifest path: {exc}",
                "error_code": "PATH_TRAVERSAL_REJECTED",
            },
        ) from exc


def validate_session_id(session_id: str) -> str:
    """Reject session identifiers that could escape the snapshot directory."""
    try:
        return _validate_session_id_plain(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"Invalid session_id: {exc}",
                "error_code": "INVALID_SESSION_ID",
            },
        ) from exc


async def enforce_session_id(request: Request) -> None:
    """FastAPI dependency: validate the ``session_id`` path parameter when present.

    Session ids become directory names under ``snapshot_dir``, so traversal-shaped
    values (``..``, slashes, spaces) must be rejected before any handler runs.
    """
    raw = request.path_params.get("session_id")
    if raw is not None:
        validate_session_id(raw)


def get_session_snapshot_dir(base_dir: Union[str, Path], session_id: str) -> Path:
    """Confined, validated equivalent of ``base_dir / session_id`` (creates it)."""
    try:
        return snapshot_session_dir(base_dir, session_id)
    except PathNotAllowedError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"Invalid session_id: {exc}",
                "error_code": "INVALID_SESSION_ID",
            },
        ) from exc


# Backward compatibility aliases
_validate_path_security = validate_path_security
_REPO_ROOT = REPO_ROOT
