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
            processed   INTEGER NOT NULL DEFAULT 0,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS node (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            accept_tags TEXT NOT NULL DEFAULT '',
            output_tag  TEXT NOT NULL DEFAULT '',
            model       TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            interval    INTEGER NOT NULL DEFAULT 5,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS node_exec (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            input_ids       TEXT NOT NULL DEFAULT '',
            output_id       INTEGER,
            node_name       TEXT NOT NULL,
            model           TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'success',
            error           TEXT NOT NULL DEFAULT '',
            elapsed         REAL NOT NULL DEFAULT 0,
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            total_tokens    INTEGER NOT NULL DEFAULT 0,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        # 兼容旧表
        for col, tbl in [("model", "node"), ("processed", "prompt")]:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0" if col == "processed" else f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
            except Exception:
                pass


app = FastAPI(title="Node + Prompt Manager")

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


# ── Node ─────────────────────────────────────────────

class NodeForm(BaseModel):
    name: str
    accept_tags: str = ""
    output_tag: str = ""
    model: str = ""
    prompt: str = ""
    interval: int = 5


@app.post("/api/nodes")
def create_node(data: NodeForm):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO node (name, accept_tags, output_tag, model, prompt, interval) VALUES (?, ?, ?, ?, ?, ?)",
            (data.name, data.accept_tags, data.output_tag, data.model, data.prompt, data.interval),
        )
    return {"status": "ok"}


@app.get("/api/nodes")
def list_nodes():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, accept_tags, output_tag, model, prompt, interval, created_at FROM node ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r[0], "name": r[1], "accept_tags": r[2], "output_tag": r[3],
         "model": r[4], "prompt": r[5], "interval": r[6], "created_at": r[7]}
        for r in rows
    ]


@app.get("/api/nodes/{node_id}")
def get_node(node_id: int):
    with get_db() as conn:
        r = conn.execute(
            "SELECT id, name, accept_tags, output_tag, model, prompt, interval, created_at FROM node WHERE id=?",
            (node_id,),
        ).fetchone()
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": r[0], "name": r[1], "accept_tags": r[2], "output_tag": r[3],
            "model": r[4], "prompt": r[5], "interval": r[6], "created_at": r[7]}


@app.put("/api/nodes/{node_id}")
def update_node(node_id: int, data: NodeForm):
    with get_db() as conn:
        conn.execute(
            "UPDATE node SET name=?, accept_tags=?, output_tag=?, model=?, prompt=?, interval=? WHERE id=?",
            (data.name, data.accept_tags, data.output_tag, data.model, data.prompt, data.interval, node_id),
        )
    return {"status": "ok"}


@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM node WHERE id=?", (node_id,))
    return {"status": "ok"}


# ── Node Exec ──────────────────────────────────────

@app.get("/api/execs")
def list_execs(node_name: Optional[str] = Query(None),
               id_lt: Optional[int] = Query(None),
               id_gt: Optional[int] = Query(None),
               limit: int = Query(50, ge=1, le=200)):
    with get_db() as conn:
        conditions = []
        params = []
        if node_name:
            conditions.append("node_name=?")
            params.append(node_name)
        if id_gt is not None:
            conditions.append("id>?")
            params.append(id_gt)
        if id_lt is not None:
            conditions.append("id<?")
            params.append(id_lt)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        order = "ORDER BY id DESC" if id_gt is None else "ORDER BY id ASC"
        rows = conn.execute(
            f"SELECT id, input_ids, output_id, node_name, model, status, error, "
            f"elapsed, input_tokens, output_tokens, total_tokens, created_at "
            f"FROM node_exec {where} {order} LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
    result = [{
        "id": r[0], "input_ids": r[1], "output_id": r[2], "node_name": r[3],
        "model": r[4], "status": r[5], "error": r[6], "elapsed": r[7],
        "input_tokens": r[8], "output_tokens": r[9], "total_tokens": r[10],
        "created_at": r[11],
    } for r in rows[:limit]]
    has_more = len(rows) > limit
    return {"items": result, "has_more": has_more}
