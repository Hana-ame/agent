import sqlite3
import os
import json
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
        conn.execute("""CREATE TABLE IF NOT EXISTS model_availability (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model           TEXT NOT NULL UNIQUE,
            total_calls     INTEGER NOT NULL DEFAULT 0,
            success_calls   INTEGER NOT NULL DEFAULT 0,
            failed_calls    INTEGER NOT NULL DEFAULT 0,
            availability    REAL NOT NULL DEFAULT 100.0,
            last_error      TEXT NOT NULL DEFAULT '',
            last_checked    DATETIME DEFAULT CURRENT_TIMESTAMP,
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
381: 
382: 
383: # ── Model Availability ─────────────────────────────
384: 
385: @app.get("/api/model-availability")
386: def list_model_availability():
387:     with get_db() as conn:
388:         rows = conn.execute(
389:             "SELECT model, total_calls, success_calls, failed_calls, availability, last_error, last_checked "
390:             "FROM model_availability ORDER BY availability ASC, total_calls DESC"
391:         ).fetchall()
392:     return [{
393:         "model": r[0], "total_calls": r[1], "success_calls": r[2],
394:         "failed_calls": r[3], "availability": r[4], "last_error": r[5],
395:         "last_checked": r[6],
396:     } for r in rows]


# ── Pipeline / Prompt Chain ─────────────────────────

class PipelineForm(BaseModel):
    name: str
    definition: str


class PipelineRunForm(BaseModel):
    input_data: dict = {}


@app.post("/api/pipelines")
def create_pipeline(data: PipelineForm):
    # validate JSON
    try:
        parsed = json.loads(data.definition)
        from pipeline_exec import validate_definition
        valid, msg = validate_definition(parsed)
        if not valid:
            return JSONResponse({"error": msg}, status_code=400)
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO pipeline (name, definition) VALUES (?, ?)",
            (data.name, data.definition),
        )
    return {"status": "ok"}


@app.get("/api/pipelines")
def list_pipelines():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, definition, created_at, updated_at FROM pipeline ORDER BY id DESC"
        ).fetchall()
    return [{
        "id": r[0], "name": r[1], "definition": json.loads(r[2]),
        "created_at": r[3], "updated_at": r[4],
    } for r in rows]


@app.get("/api/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: int):
    with get_db() as conn:
        r = conn.execute(
            "SELECT id, name, definition, created_at, updated_at FROM pipeline WHERE id=?",
            (pipeline_id,),
        ).fetchone()
    if not r:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "id": r[0], "name": r[1], "definition": json.loads(r[2]),
        "created_at": r[3], "updated_at": r[4],
    }


@app.put("/api/pipelines/{pipeline_id}")
def update_pipeline(pipeline_id: int, data: PipelineForm):
    try:
        parsed = json.loads(data.definition)
        from pipeline_exec import validate_definition
        valid, msg = validate_definition(parsed)
        if not valid:
            return JSONResponse({"error": msg}, status_code=400)
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)

    with get_db() as conn:
        conn.execute(
            "UPDATE pipeline SET name=?, definition=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (data.name, data.definition, pipeline_id),
        )
    return {"status": "ok"}


@app.delete("/api/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM pipeline WHERE id=?", (pipeline_id,))
    return {"status": "ok"}


@app.post("/api/pipelines/{pipeline_id}/run")
def run_pipeline(pipeline_id: int, data: PipelineRunForm):
    from pipeline_exec import run_pipeline as exec_pipeline
    result = exec_pipeline(pipeline_id, data.input_data)
    if result["status"] == "error":
        return JSONResponse(result, status_code=500)
    return result


@app.get("/api/pipelines/{pipeline_id}/execs")
def list_pipeline_execs(pipeline_id: int, limit: int = Query(50, ge=1, le=200)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, pipeline_id, input_data, output_data, status, error, steps_log, created_at "
            "FROM pipeline_exec WHERE pipeline_id=? ORDER BY id DESC LIMIT ?",
            (pipeline_id, limit),
        ).fetchall()
    return [{
        "id": r[0], "pipeline_id": r[1],
        "input_data": json.loads(r[2]), "output_data": json.loads(r[3]),
        "status": r[4], "error": r[5], "steps_log": json.loads(r[6]),
        "created_at": r[7],
    } for r in rows]


@app.get("/api/pipeline-execs")
def list_all_pipeline_execs(limit: int = Query(50, ge=1, le=200)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, pipeline_id, input_data, output_data, status, error, steps_log, created_at "
            "FROM pipeline_exec ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{
        "id": r[0], "pipeline_id": r[1],
        "input_data": json.loads(r[2]), "output_data": json.loads(r[3]),
        "status": r[4], "error": r[5], "steps_log": json.loads(r[6]),
        "created_at": r[7],
    } for r in rows]
