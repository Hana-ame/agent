"""v8 节点结果缓存（文件式、content-addressed）。

key = sha256(节点指纹 + 输入 + 模型/参数)。命中就直接复用输出，跳过 `pi` 或 v7 调用。

为什么 key 里要带模型：**切换模型不丢结果**——旧模型的输出仍可复现；换回来也直接命中。
为什么默认关闭：生成任务通常希望每次有新结果；要复用/评测/重跑时用 `--cache` 显式打开。

布局：

    data/cache/<前两位>/<sha256>.json   {"output": "...", "meta": {...}, "created_at": "..."}

文件式而不是 sqlite：量小、可手工检查/删除、避免和运行库抢锁。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["DEFAULT_CACHE_DIR", "fingerprint", "NodeCache"]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def fingerprint(payload: Dict[str, Any]) -> str:
    """把任意 JSON 可序列化载荷压成稳定的 sha256。"""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NodeCache:
    def __init__(self, path: Optional[str | Path] = None,
                 ttl: Optional[float] = None) -> None:
        self.dir = Path(path) if path else DEFAULT_CACHE_DIR
        self.ttl = ttl
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        out = data.get("output")
        if not isinstance(out, str):
            return None
        if self.ttl is not None:
            ts = data.get("created_ts")
            # 老条目没有时间戳：在要求 TTL 时按过期处理，避免"永远新鲜"
            if not isinstance(ts, (int, float)) or time.time() - ts > self.ttl:
                return None
        return out

    def put(self, key: str, output: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "output": output,
            "meta": meta or {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "created_ts": time.time(),
        }
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def stats(self) -> Dict[str, int]:
        n = sum(1 for _ in self.dir.rglob("*.json"))
        return {"entries": n}

    def info(self) -> Dict[str, Any]:
        files = list(self.dir.rglob("*.json"))
        total = 0
        oldest = newest = None
        for p in files:
            try:
                total += p.stat().st_size
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ts = data.get("created_ts")
            if isinstance(ts, (int, float)):
                oldest = ts if oldest is None else min(oldest, ts)
                newest = ts if newest is None else max(newest, ts)
        return {"entries": len(files), "bytes": total,
                "oldest": oldest, "newest": newest}

    def evict(self, ttl: float) -> int:
        """删掉 created_ts 早于 ttl 秒的条目；无时间戳的保守保留。"""
        removed = 0
        for p in list(self.dir.rglob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ts = data.get("created_ts")
            if isinstance(ts, (int, float)) and time.time() - ts > ttl:
                p.unlink(missing_ok=True)
                removed += 1
        return removed

    def clear(self) -> int:
        removed = 0
        for p in list(self.dir.rglob("*.json")):
            p.unlink(missing_ok=True)
            removed += 1
        return removed
