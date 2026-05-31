"""Free models tracker with dynamic discovery for opencode and nvidia."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Literal, overload
from urllib.request import urlopen, Request
from urllib.error import URLError
import json
import time
import db

SILICONFLOW_MODELS = [
    "siliconflow-cn/Qwen/Qwen3-8B",
    "siliconflow-cn/THUDM/GLM-Z1-9B-0414",
    "siliconflow-cn/THUDM/GLM-4-9B-0414",
]

OPENCODE_API = "https://opencode.ai/zen/v1/models"
NVIDIA_API = "https://integrate.api.nvidia.com/v1/models"


def _get_conn():
    conn = db.get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model TEXT PRIMARY KEY,
            provider TEXT,
            discovered_at REAL,
            last_seen REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            model TEXT PRIMARY KEY,
            calls INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            good INTEGER DEFAULT 0,
            bad INTEGER DEFAULT 0,
            FOREIGN KEY (model) REFERENCES models(model)
        )
    """)
    # 兼容旧表：添加 good/bad 列
    for col in ["good", "bad"]:
        try:
            conn.execute(f"ALTER TABLE usage ADD COLUMN {col} INTEGER DEFAULT 0")
        except Exception:
            pass
    conn.commit()
    return conn


def _fetch_json(url: str, timeout: int = 10) -> dict[str, Any] | None:
    try:
        req = Request(url, headers={"User-Agent": "free-models-tracker/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, json.JSONDecodeError):
        return None


def fetch_opencode_models() -> list[str]:
    """Fetch free models from OpenCode Zen API."""
    data = _fetch_json(OPENCODE_API)
    if not data or "data" not in data:
        return []
    models = []
    for m in data["data"]:
        mid = m.get("id", "")
        if "free" in mid.lower() or mid == "big-pickle":
            models.append(f"opencode/{mid}")
    return models


def fetch_nvidia_models() -> list[str]:
    """Fetch free models from NVIDIA NIM API."""
    data = _fetch_json(NVIDIA_API)
    if not data or "data" not in data:
        return []
    models = []
    for m in data["data"]:
        mid = m.get("id", "")
        models.append(f"nvidia/{mid}")
    return models


def sync_models() -> list[str]:
    """Fetch dynamic models and update database. Returns all free models."""
    conn = _get_conn()
    now = time.time()

    all_models = list(SILICONFLOW_MODELS)
    for provider, models in [("opencode", fetch_opencode_models()),
                             ("nvidia", fetch_nvidia_models())]:
        all_models.extend(models)
        for model in models:
            conn.execute("""
                INSERT INTO models (model, provider, discovered_at, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model) DO UPDATE SET last_seen = ?
            """, (model, provider, now, now, now))

    # Add siliconflow models
    for model in SILICONFLOW_MODELS:
        conn.execute("""
            INSERT INTO models (model, provider, discovered_at, last_seen)
            VALUES (?, 'siliconflow-cn', ?, ?)
            ON CONFLICT(model) DO UPDATE SET last_seen = ?
        """, (model, now, now, now))

    conn.commit()
    conn.close()
    return all_models


def list_free_models() -> list[str]:
    """Return all known free models from database."""
    conn = _get_conn()
    rows = conn.execute("SELECT model FROM models").fetchall()
    models = [r[0] for r in rows]
    for m in models:
        conn.execute("""
            INSERT OR IGNORE INTO usage (model, calls, successes, failures, good, bad)
            VALUES (?, 0, 0, 0, 0, 0)
        """, (m,))
    conn.commit()
    conn.close()
    return models

def record_call(model: str, success: bool = True, good: int = 0, bad: int = 0) -> None:
    """Record a call for a model.
    success: 调用是否成功（网络/API 层面）
    good: 好的回答数量（内容质量好）
    bad: 坏的回答数量（内容质量差）
    """
    conn = _get_conn()
    # Ensure model exists in models table
    conn.execute("""
        INSERT OR IGNORE INTO models (model, provider, discovered_at, last_seen)
        VALUES (?, 'manual', ?, ?)
    """, (model, time.time(), time.time()))

    if success:
        conn.execute("""
            INSERT INTO usage (model, calls, successes, failures, good, bad)
            VALUES (?, 1, 1, 0, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
                calls = calls + 1, successes = successes + 1,
                good = good + ?, bad = bad + ?
        """, (model, good, bad, good, bad))
    else:
        conn.execute("""
            INSERT INTO usage (model, calls, successes, failures, good, bad)
            VALUES (?, 1, 0, 1, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
                calls = calls + 1, failures = failures + 1,
                good = good + ?, bad = bad + ?
        """, (model, good, bad, good, bad))
    conn.commit()
    conn.close()


@overload
def get_stats(model: str) -> dict[str, Any] | None: ...

@overload
def get_stats(model: None = ...) -> list[dict[str, Any]]: ...

def get_stats(model: str | None = None) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Get usage stats. If model is None, return all."""
    conn = _get_conn()
    if model:
        row = conn.execute("""
            SELECT m.model, COALESCE(u.calls, 0), COALESCE(u.successes, 0),
                   COALESCE(u.failures, 0), COALESCE(u.good, 0), COALESCE(u.bad, 0)
            FROM models m LEFT JOIN usage u ON m.model = u.model
            WHERE m.model = ?
        """, (model,)).fetchone()
        conn.close()
        if row:
            return {"model": row[0], "calls": row[1], "successes": row[2],
                    "failures": row[3], "good": row[4], "bad": row[5]}
        return None

    rows = conn.execute("""
        SELECT m.model, COALESCE(u.calls, 0), COALESCE(u.successes, 0),
               COALESCE(u.failures, 0), COALESCE(u.good, 0), COALESCE(u.bad, 0)
        FROM models m LEFT JOIN usage u ON m.model = u.model
    """).fetchall()
    conn.close()
    return [{"model": r[0], "calls": r[1], "successes": r[2],
             "failures": r[3], "good": r[4], "bad": r[5]} for r in rows]


if __name__ == "__main__":
    print("Syncing models...")
    models = sync_models()
    print(f"Found {len(models)} free models:")
    for m in sorted(models):
        print(f"  - {m}")
