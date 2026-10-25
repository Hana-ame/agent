"""v9 知识库：实体、关系、来源、推理链。

和 v7/v8 的 contents 池不同：
- 每条 claim 必须有 source_url（外部证据）或 derived_from（推理来源）
- sources 表缓存外部抓取，避免重复请求
- explored 表记录“试过但没有关系”的实体对，避免死循环
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL DEFAULT 'concept',
    source_url TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_url TEXT,
    derived_from TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    url TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    text TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS explored (
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    outcome TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (a, b)
);
"""


class Knowledge:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_entity(self, name: str, *, kind: str = "concept",
                   source_url: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO entities (name, kind, source_url, created_at) "
            "VALUES (?, ?, ?, ?)", (name, kind, source_url, time.time()))
        self.conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.conn.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        return int(row["id"])

    def add_claim(self, subject: str, predicate: str, obj: str, statement: str,
                  *, confidence: float = 0.5, source_url: Optional[str] = None,
                  derived_from: Optional[List[int]] = None) -> int:
        if not source_url and not derived_from:
            raise ValueError("claim 必须有 source_url 或 derived_from")
        cur = self.conn.execute(
            "INSERT INTO claims (subject, predicate, object, statement, confidence, "
            "source_url, derived_from, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (subject, predicate, obj, statement, max(0.0, min(1.0, confidence)),
             source_url, json.dumps(derived_from or [], ensure_ascii=False),
             time.time()))
        self.conn.commit()
        return int(cur.lastrowid)

    def put_source(self, url: str, text: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sources (url, fetched_at, text, hash) "
            "VALUES (?, ?, ?, ?)",
            (url, time.time(), text,
             hashlib.sha256(text.encode("utf-8")).hexdigest()))
        self.conn.commit()

    def get_source(self, url: str) -> Optional[str]:
        row = self.conn.execute("SELECT text FROM sources WHERE url=?", (url,)).fetchone()
        return row["text"] if row else None

    def is_explored(self, a: str, b: str) -> bool:
        x, y = sorted((a, b))
        return self.conn.execute(
            "SELECT 1 FROM explored WHERE a=? AND b=?", (x, y)
        ).fetchone() is not None

    def mark_explored(self, a: str, b: str, outcome: str) -> None:
        x, y = sorted((a, b))
        self.conn.execute(
            "INSERT OR REPLACE INTO explored (a, b, outcome, created_at) "
            "VALUES (?, ?, ?, ?)", (x, y, outcome, time.time()))
        self.conn.commit()

    def entities(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM entities ORDER BY name")]

    def claims(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM claims ORDER BY id")]

    def neighbors(self, name: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM claims WHERE subject=? OR object=? ORDER BY id",
            (name, name))]

    def unexplored_pairs(self, limit: int = 10) -> List[Tuple[str, str]]:
        names = [r["name"] for r in self.entities()]
        linked = set()
        for c in self.claims():
            linked.add(tuple(sorted((c["subject"], c["object"]))))
        tried = {tuple(r) for r in self.conn.execute(
            "SELECT a, b FROM explored").fetchall()}
        out = []
        for a, b in combinations(names, 2):
            key = tuple(sorted((a, b)))
            if key in linked or key in tried:
                continue
            out.append((a, b))
            if len(out) >= limit:
                break
        return out

    def stats(self) -> Dict[str, int]:
        return {
            "entities": self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "claims": self.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
            "sources": self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "explored": self.conn.execute("SELECT COUNT(*) FROM explored").fetchone()[0],
        }

    def close(self) -> None:
        self.conn.close()
