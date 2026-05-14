import sqlite3
import os
import json
import time
import threading
import subprocess
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


# ── Opencode 调用 ────────────────────────────────────

def call_opencode(prompt: str, model: str = None) -> dict:
    """
    调用 opencode CLI 处理一段文本。

    参数:
        prompt: 输入文本
        model:  模型名（如 opencode/deepseek-v4-flash-free），可选

    返回:
        {"success": True, "output": "生成的文本", "usage": {"input": N, "output": N, "total": N}}
        或 {"success": False, "error": "错误信息"}
    """
    cmd = ["opencode", "run", "--format", "json"]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, encoding="utf-8", timeout=3600)
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"opencode 调用失败: {e.stderr[:500]}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "opencode 超时 (3600s)"}
    except FileNotFoundError:
        return {"success": False, "error": "找不到 opencode 命令，请确认已安装"}

    output_text = ""
    usage = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                output_text += event["part"].get("text", "")
            elif event.get("type") == "step_finish":
                usage = event["part"].get("tokens", {})
        except json.JSONDecodeError:
            continue

    return {
        "success": True,
        "output": output_text,
        "usage": {
            "input": usage.get("input", 0),
            "output": usage.get("output", 0),
            "total": usage.get("total", 0),
        },
    }


# ── Node 轮询引擎 ──────────────────────────────────

_poll_stop = threading.Event()
_last_checks: dict[int, float] = {}  # node_id → last check timestamp


def _try_execute_node(node: dict) -> Optional[dict]:
    tags = [t.strip() for t in node["accept_tags"].split(",") if t.strip()]
    if not tags:
        return None

    db = get_db()
    try:
        # 每个 tag 取一条未处理 prompt（最旧优先）
        selected = {}
        selected_ids = []
        for tag in tags:
            row = db.execute(
                "SELECT id, prompt FROM prompt WHERE tag=? AND processed=0 ORDER BY id ASC LIMIT 1",
                (tag,),
            ).fetchone()
            if not row:
                return None  # AND 逻辑：缺一个 tag 就不执行
            selected[tag] = row[1]
            selected_ids.append(str(row[0]))

        # 变量替换 {tag_name} → 对应文本
        filled = node["prompt"]
        for tag, text in selected.items():
            filled = filled.replace(f"{{{tag}}}", text)

        # 调用 opencode
        t0 = time.time()
        result = call_opencode(filled, node.get("model") or None)
        elapsed = time.time() - t0

        input_ids = ",".join(selected_ids)
        if result["success"]:
            # 写回输出
            cur = db.execute(
                "INSERT INTO prompt (prev_id, tag, prompt) VALUES (?, ?, ?)",
                (int(selected_ids[0]), node["output_tag"], result["output"]),
            )
            output_id = cur.lastrowid
            db.execute(
                "INSERT INTO node_exec (input_ids, output_id, node_name, model, status,"
                " elapsed, input_tokens, output_tokens, total_tokens)"
                " VALUES (?, ?, ?, ?, 'success', ?, ?, ?, ?)",
                (input_ids, output_id, node["name"], node.get("model") or "",
                 round(elapsed, 3), result["usage"]["input"],
                 result["usage"]["output"], result["usage"]["total"]),
            )
        else:
            output_id = None
            db.execute(
                "INSERT INTO node_exec (input_ids, output_id, node_name, model, status,"
                " error, elapsed, input_tokens, output_tokens, total_tokens)"
                " VALUES (?, ?, ?, ?, 'error', ?, ?, 0, 0, 0)",
                (input_ids, output_id, node["name"], node.get("model") or "",
                 result["error"][:500], round(elapsed, 3)),
            )

        # 标记输入为已处理
        for sid in selected_ids:
            db.execute("UPDATE prompt SET processed=1 WHERE id=?", (int(sid),))

        db.commit()
        return {"node_name": node["name"], "status": "success" if result["success"] else "error"}
    except Exception as e:
        db.rollback()
        return {"node_name": node["name"], "status": "error", "error": str(e)[:500]}
    finally:
        db.close()


def _poll_loop():
    while not _poll_stop.is_set():
        db = get_db()
        try:
            nodes = db.execute(
                "SELECT id, name, accept_tags, output_tag, model, prompt, interval FROM node"
            ).fetchall()
        finally:
            db.close()

        now = time.time()
        for n in nodes:
            if _poll_stop.is_set():
                break
            node_id = n[0]
            interval = n[6]
            if now - _last_checks.get(node_id, 0) < interval:
                continue
            _last_checks[node_id] = now
            node_dict = {
                "id": n[0], "name": n[1], "accept_tags": n[2],
                "output_tag": n[3], "model": n[4], "prompt": n[5],
                "interval": n[6],
            }
            _try_execute_node(node_dict)

        # 每秒检查一次
        _poll_stop.wait(1)


@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=_poll_loop, daemon=True, name="node-poll").start()


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
