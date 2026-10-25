"""Global Memory & Shared Context Management module.

Provides a decoupled, thread-safe ``MemoryStore`` for multi-agent workflows,
enabling shared state across vertices and edges without bloating point-to-point edge channels.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("vertex_edge_agent.memory")


class MemoryStore:
    """Thread-safe global and scoped key-value store for agent workflows.

    Features:
    - Global shared context (accessible across all vertices and edges).
    - TTL-based temporary values.
    - Scoped child views (e.g. ``memory.scope("user_123")``).
    - Thread-safe async locking.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None, prefix: str = ""):
        self._prefix = prefix
        self._data: Dict[str, Any] = dict(initial_data or {})
        self._ttls: Dict[str, float] = {}  # key -> expiry monotonic timestamp
        self._lock = asyncio.Lock()

    def _qualify_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key, honoring TTL expiry."""
        qkey = self._qualify_key(key)
        async with self._lock:
            if qkey in self._ttls and time.monotonic() > self._ttls[qkey]:
                del self._data[qkey]
                del self._ttls[qkey]
                return default
            return self._data.get(qkey, default)

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value with optional TTL (in seconds)."""
        qkey = self._qualify_key(key)
        async with self._lock:
            self._data[qkey] = value
            if ttl is not None and ttl > 0:
                self._ttls[qkey] = time.monotonic() + ttl
            elif qkey in self._ttls:
                del self._ttls[qkey]
            logger.debug("[MemoryStore] SET %s = %s", qkey, repr(value)[:100])

    async def delete(self, key: str) -> bool:
        """Delete a key from memory store."""
        qkey = self._qualify_key(key)
        async with self._lock:
            if qkey in self._data:
                del self._data[qkey]
                self._ttls.pop(qkey, None)
                return True
            return False

    async def update(self, mapping: Dict[str, Any]) -> None:
        """Bulk update multiple keys."""
        async with self._lock:
            for k, v in mapping.items():
                qkey = self._qualify_key(k)
                self._data[qkey] = v
                self._ttls.pop(qkey, None)

    async def get_all(self) -> Dict[str, Any]:
        """Return a snapshot dictionary of all non-expired key-values in this scope."""
        now = time.monotonic()
        async with self._lock:
            # Clean expired
            expired = [k for k, exp in self._ttls.items() if now > exp]
            for k in expired:
                self._data.pop(k, None)
                self._ttls.pop(k, None)

            if not self._prefix:
                return dict(self._data)

            # Strip prefix for scoped views
            scoped_len = len(self._prefix)
            return {
                k[scoped_len:]: v
                for k, v in self._data.items()
                if k.startswith(self._prefix)
            }

    def scope(self, namespace: str) -> "MemoryStore":
        """Create a scoped view of this store with a sub-namespace prefix."""
        new_prefix = f"{self._prefix}{namespace}:"
        child = MemoryStore(prefix=new_prefix)
        # Share underlying dict and locks
        child._data = self._data
        child._ttls = self._ttls
        child._lock = self._lock
        return child
