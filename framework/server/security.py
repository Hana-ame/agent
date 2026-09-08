"""Security and path traversal validation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from fastapi import HTTPException


def validate_path_security(path_str: str, base_dir: Path) -> Path:
    """Validate that a path doesn't escape the allowed base directory.
    
    Raises HTTPException(400) with structured error if path is invalid.
    """
    resolved = Path(path_str).resolve()
    base_resolved = base_dir.resolve()
    if not str(resolved).startswith(str(base_resolved) + os.sep) and resolved != base_resolved:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Invalid path: target is outside allowed directory",
                "error_code": "PATH_TRAVERSAL_REJECTED",
            },
        )
    return resolved


# Backward compatibility alias
_validate_path_security = validate_path_security
