import sqlite3
import os
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")


def get_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS prompt (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prev_id     INTEGER,
            tag         TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")


app = FastAPI(title="Prompt Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ── Ping ───────────────────────────────────────────────

@app.get("/ping")
def ping():
    return PlainTextResponse("pong")


# ── Prompt ─────────────────────────────────────────────

class PromptForm(BaseModel):
    tag: str = ""
    prompt: str
    prev_id: Optional[int] = None


@app.post("/api/prompts")
def create_prompt(data: PromptForm):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO prompt (prev_id, tag, prompt) VALUES (?, ?, ?)",
            (data.prev_id, data.tag, data.prompt),
        )
    return {"status": "ok"}


@app.get("/api/prompts")
def list_prompts(tag: Optional[str] = Query(None),
                 id_gt: Optional[int] = Query(None),
                 id_lt: Optional[int] = Query(None),
                 limit: int = Query(50, ge=1, le=200)):
    with get_db() as conn:
        conditions = []
        params = []
        if tag:
            conditions.append("tag=?")
            params.append(tag)
        if id_gt is not None:
            conditions.append("id>?")
            params.append(id_gt)
        if id_lt is not None:
            conditions.append("id<?")
            params.append(id_lt)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        order = "ORDER BY id DESC" if id_gt is None else "ORDER BY id ASC"
        rows = conn.execute(
            f"SELECT id, prev_id, tag, prompt, created_at FROM prompt {where} {order} LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
    result = [
        {"id": r[0], "prev_id": r[1], "tag": r[2], "prompt": r[3], "created_at": r[4]}
        for r in rows[:limit]
    ]
    has_more = len(rows) > limit
    if rows:
        ids = [r[0] for r in rows[:limit]]
        max_id = max(ids)
        min_id = min(ids)
    else:
        max_id = min_id = None
    going_newer = id_gt is not None
    if going_newer:
        result.reverse()
    return {
        "items": result,
        "max_id": max_id,
        "min_id": min_id,
        "has_older": has_more if not going_newer else True,
        "has_newer": (going_newer and has_more) or (id_lt is not None),
    }
