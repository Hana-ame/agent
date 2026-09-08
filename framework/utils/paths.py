"""Path confinement helpers shared by runtime and server layers.

Client-supplied file references (manifests, scripts, snapshot session ids) must
never escape a configured root. The helpers here raise plain exceptions so the
runtime layer can stay independent of FastAPI; ``framework.server.security``
translates them into HTTP 400 responses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

#: Repository root, used as the default confinement base.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Session identifiers become directory names under ``snapshot_dir``.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

#: Longest accepted session identifier.
MAX_SESSION_ID_LEN = 128


class PathNotAllowedError(PermissionError):
    """Raised when a client-supplied path escapes its allowed root."""


def default_manifest_base_dir() -> Path:
    """Return the default root that client-supplied manifests must stay inside."""
    return REPO_ROOT


def is_within(path: Union[str, Path], root: Union[str, Path]) -> bool:
    """Return True when ``path`` resolves to ``root`` or a descendant of it."""
    try:
        target = Path(path).resolve()
        base = Path(root).resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    return target == base or base in target.parents


def resolve_confined_path(
    path_str: str,
    base_dir: Optional[Union[str, Path]] = None,
    must_be_json: bool = False,
) -> Path:
    """Resolve ``path_str`` and assert it stays inside ``base_dir``.

    Relative paths resolve against ``base_dir`` (default: repository root).

    Raises:
        PathNotAllowedError: The path is empty, escapes the base, or is not a
            ``.json`` file when ``must_be_json`` is set.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise PathNotAllowedError("Path must be a non-empty string")

    base = Path(base_dir).resolve() if base_dir else default_manifest_base_dir()

    if must_be_json and not path_str.endswith(".json"):
        raise PathNotAllowedError("Manifest path must end in .json")

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = base / candidate

    resolved = candidate.resolve()
    if not is_within(resolved, base):
        raise PathNotAllowedError(f"Path is outside the allowed directory: {path_str}")
    return resolved


def validate_session_id(session_id: str) -> str:
    """Return a safe session identifier or raise ``ValueError``.

    Rejects empty, over-long, and traversal-shaped values (``.``/``..`` are
    rejected explicitly even though they match the character class).
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if len(session_id) > MAX_SESSION_ID_LEN:
        raise ValueError(f"session_id too long (max {MAX_SESSION_ID_LEN} characters)")
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("session_id may only contain letters, digits, '.', '_' and '-'")
    if set(session_id) <= {".", "_", "-"}:
        raise ValueError("session_id must contain at least one letter or digit")
    return session_id


def is_safe_session_id(session_id: str) -> bool:
    """Non-raising form of :func:`validate_session_id`."""
    try:
        validate_session_id(session_id)
        return True
    except ValueError:
        return False


def snapshot_session_dir(base_dir: Union[str, Path], session_id: str) -> Path:
    """Return ``base_dir/session_id``, creating it, after validating the id.

    Raises:
        PathNotAllowedError: The session id is unsafe or the resulting path
            escapes ``base_dir``.
    """
    try:
        safe = validate_session_id(session_id)
    except ValueError as exc:
        raise PathNotAllowedError(str(exc)) from exc

    base = Path(base_dir).resolve()
    target = (base / safe).resolve()
    if not is_within(target, base):
        raise PathNotAllowedError(f"Session path escapes the snapshot directory: {session_id}")
    target.mkdir(parents=True, exist_ok=True)
    return target
