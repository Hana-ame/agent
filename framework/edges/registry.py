"""Single source of truth for edge-type registration and construction.

Before this module existed, adding an edge type required editing five separate
dispatch tables (``EdgeV4.from_config``, ``graphs/loader.py`` twice,
``graph_manager_v4.add_or_update_edge``, ``edges/cli.py`` and
``server/routes/openai.py``). ``LLMToolEdgeV4`` was added and broke in two of
them. New edge types now register once::

    @register_edge_type("http_fetch", "fetch")
    class HttpFetchEdgeV4(EdgeV4):
        ...

Registration is idempotent per class (importing a module twice does not raise),
but a *different* class claiming an existing name is an error unless
``replace=True`` is passed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from framework.edges.base import EdgeV4

logger = logging.getLogger("vertex_edge_agent.edges.registry")

#: edge type string -> edge class
EDGE_REGISTRY: Dict[str, Type["EdgeV4"]] = {}

#: Order in which types are advertised (CLI choices, docs).
_REGISTRATION_ORDER: List[str] = []


def register_edge_type(*names: str, replace: bool = False):
    """Class decorator registering one or more type aliases for an edge class."""
    if not names:
        raise ValueError("register_edge_type requires at least one type name")

    def decorator(cls: Type["EdgeV4"]) -> Type["EdgeV4"]:
        existing_names = {n for n, c in EDGE_REGISTRY.items() if c is cls}
        for name in names:
            current = EDGE_REGISTRY.get(name)
            if current is not None and current is not cls and not replace:
                # Re-executing the same module under a different name (e.g. running
                # a façade file as __main__) creates an equal-but-distinct class;
                # treat that as the same registration rather than a collision.
                if getattr(current, "__name__", None) != getattr(cls, "__name__", None):
                    raise ValueError(
                        f"Edge type '{name}' is already registered to {current.__name__}; "
                        "pass replace=True to override"
                    )
                logger.debug(
                    "Re-registering edge type '%s' for %s (module executed twice)",
                    name,
                    cls.__name__,
                )
            if current is None:
                _REGISTRATION_ORDER.append(name)
            EDGE_REGISTRY[name] = cls
        cls.edge_type_names = tuple(sorted(existing_names | set(names)))
        return cls

    return decorator


def get_edge_class(edge_type: Optional[str]) -> Optional[Type["EdgeV4"]]:
    """Return the class registered for ``edge_type``, or ``None``."""
    if not edge_type:
        return None
    return EDGE_REGISTRY.get(str(edge_type))


def is_registered_edge_type(edge_type: Optional[str]) -> bool:
    """Return True when ``edge_type`` has a registered implementation."""
    return get_edge_class(edge_type) is not None


def edge_type_choices() -> List[str]:
    """Return the registered type names in registration order."""
    return list(_REGISTRATION_ORDER)


def is_tool_edge(edge: object) -> bool:
    """Return True for edges that emit OpenAI tool calls instead of computing."""
    return bool(getattr(edge, "EMITS_TOOL_CALL", False))


def is_recovery_edge(edge: object) -> bool:
    """Return True for self-loop recovery edges (reflexive)."""
    return bool(getattr(edge, "IS_RECOVERY", False)) or bool(getattr(edge, "is_reflexive", False))
