from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json

from cell_core.db import get_connection, init_db, seed_default_data, DB_PATH

app = FastAPI(title="CellAgent API", description="CellAgent system status interface")


class TaskCreate(BaseModel):
    dna_id: int
    input_json: dict


@app.on_event("startup")
def startup():
    init_db()
    seed_default_data()


@app.get("/tasks")
def list_tasks(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, dna_id, input_json, status, current_step, total_steps, "
        "quality_score, total_cost, total_time_ms, error, created_at, updated_at "
        "FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Task not found")
    task = dict(row)
    steps = conn.execute(
        "SELECT * FROM step_results WHERE task_id = ? ORDER BY step_index",
        (task_id,)
    ).fetchall()
    conn.close()
    task["steps"] = [dict(s) for s in steps]
    return task


@app.get("/organelles")
def list_organelles():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM organelles ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/mrna")
def list_mrna():
    conn = get_connection()
    rows = conn.execute(
        "SELECT m.id, m.organelle_id, o.name as organelle_name, "
        "m.template, m.version, m.quality_score, m.usage_count, m.created_at "
        "FROM mrna m JOIN organelles o ON m.organelle_id = o.id "
        "ORDER BY m.quality_score DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/dna")
def list_dna():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM dna ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/tasks")
def create_task(data: TaskCreate):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO tasks (dna_id, input_json, status) VALUES (?, ?, 'pending')",
        (data.dna_id, json.dumps(data.input_json, ensure_ascii=False)),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"task_id": task_id, "status": "pending"}
