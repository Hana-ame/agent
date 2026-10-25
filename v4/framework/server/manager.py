"""Backward-compatible re-export.

``SessionGraphManagerV4`` now lives in ``framework.graph_manager_v4`` so that
runtime modules can depend on it without importing from the server package.
This module preserves all existing imports.
"""

from __future__ import annotations

from framework.graph_manager_v4 import SessionGraphManagerV4

__all__ = ["SessionGraphManagerV4"]
