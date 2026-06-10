"""Prompt DB 查看服务 — FastAPI

启动:  python3 prompt_db_server.py
端口:  8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from prompt_db import PromptDB, parse_log

app = FastAPI(title="Prompt DB", version="1.0")

# CORS: 最大权限，允许所有
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db = PromptDB()


# ── 查询 ─────────────────────────────────────────────────────────────────

@app.get("/prompts")
def list_prompts(status: str = "", max_id: int = 0, min_id: int = 0, size: int = Query(50, ge=1, le=200)):
    """列出记录，支持游标分页（max_id/min_id）和状态过滤。"""
    if status:
        rows = db.list_by_status(status)
    else:
        rows = db.list_all()

    rows = list(reversed(rows))

    if max_id:
        rows = [r for r in rows if r["id"] < max_id]
    if min_id:
        rows = [r for r in rows if r["id"] > min_id]

    rows = rows[:size]
    return {
        "items": rows,
        "min_id": rows[0]["id"] if rows else 0,
        "max_id": rows[-1]["id"] if rows else 0,
    }


@app.get("/prompts/{pid}")
def get_prompt(pid: int):
    """获取单条记录。"""
    row = db.get(pid)
    if not row:
        return JSONResponse({"error": f"id={pid} not found"}, 404)
    return row


@app.get("/stats")
def stats():
    """统计信息。"""
    all_rows = db.list_all()
    done = [r for r in all_rows if r["status"] == "done"]
    pending = [r for r in all_rows if r["status"] == "pending"]
    failed = [r for r in all_rows if r["status"] == "failed"]
    scores = [r["score"] for r in all_rows if r["score"] > 0]
    elos = [r["elo"] for r in all_rows]

    return {
        "total": len(all_rows),
        "done": len(done),
        "pending": len(pending),
        "failed": len(failed),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "avg_elo": round(sum(elos) / len(elos), 2) if elos else 0,
        "max_elo": max(elos) if elos else 0,
        "min_elo": min(elos) if elos else 0,
    }


@app.get("/leaderboard")
def leaderboard(limit: int = Query(20, ge=1, le=100)):
    """ELO 排行榜。"""
    rows = db.list_all()
    rows.sort(key=lambda r: r["elo"], reverse=True)
    return rows[:limit]


# ── 操作 ─────────────────────────────────────────────────────────────────

@app.post("/prompts")
def add_prompt(
    text: str = Query(None),
    context: str = Query(None),
    prompt: str = Query(None),
    response: str = Query(None),
    agent: str = "",
    model: str = "",
):
    """
    添加一条新记录。

    四种模式:
      - text: 纯文本输入（context 为文本）
      - context + prompt: 同时写入 context 和 prompt
      - context: 只写 context（数组或文本）
      - response: 直接写入空的 context 并指定 response 内容

    如果提供 response，状态自动设为 done。
    """
    if context is not None and prompt is not None:
        import json as _json
        try:
            ctx = _json.loads(context)
        except _json.JSONDecodeError:
            ctx = context
        pid = db.add(ctx, agent=agent, model=model)
        # 把 prompt 存到 response 字段作为暂存
        with db._conn() as conn:
            conn.execute("UPDATE prompts SET response=? WHERE id=?", (prompt, pid))
            conn.commit()
        # 如果提供了 response，标记为 done
        if response:
            db.done(pid, response, {"source": "api_upload"})
        return {"ok": True, "id": pid, "context": context, "prompt": prompt}
    elif text is not None:
        pid = db.add(text, agent=agent, model=model)
        # 如果提供了 response，标记为 done
        if response:
            db.done(pid, response, {"source": "api_upload"})
        return {"ok": True, "id": pid}
    elif context is not None:
        import json as _json
        try:
            ctx = _json.loads(context)
        except _json.JSONDecodeError:
            ctx = context
        pid = db.add(ctx, agent=agent, model=model)
        # 如果提供了 response，标记为 done
        if response:
            db.done(pid, response, {"source": "api_upload"})
        return {"ok": True, "id": pid}

    # --- 新增分支: 直接指定 Response，空 Context ---
    elif response is not None:
        pid = db.add("", agent=agent, model=model)
        db.done(pid, response, {"source": "api_upload_direct_response"})
        return {"ok": True, "id": pid}

    else:
        return JSONResponse({"error": "需要 text、context 或 response 参数"}, 400)


@app.post("/prompts/{pid}/score")
def set_score(pid: int, score: float = Query(..., ge=0.0, le=1.0)):
    """设置质量评分。"""
    row = db.get(pid)
    if not row:
        return JSONResponse({"error": f"id={pid} not found"}, 404)
    db.update_score(pid, score)
    return {"ok": True, "pid": pid, "score": score}


@app.post("/prompts/{pid}/elo")
def set_elo(pid: int, elo: float = Query(...)):
    """设置 ELO 分数。"""
    row = db.get(pid)
    if not row:
        return JSONResponse({"error": f"id={pid} not found"}, 404)
    db.update_elo(pid, elo)
    return {"ok": True, "pid": pid, "elo": elo}


@app.post("/prompts/match")
def elo_match(winner_id: int = Query(...), loser_id: int = Query(...), k: int = Query(32)):
    """ELO 对战：winner 击败 loser。"""
    result = db.elo_match(winner_id, loser_id, k)
    if result is None:
        return JSONResponse({"error": "invalid ids"}, 400)
    return {"ok": True, **result}


@app.delete("/prompts/{pid}")
def delete_prompt(pid: int):
    """删除记录。"""
    row = db.get(pid)
    if not row:
        return JSONResponse({"error": f"id={pid} not found"}, 404)
    db.delete(pid)
    return {"ok": True, "deleted": pid}


# ── 启动 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Prompt DB Server: http://localhost:8000")
    print("  GET  /prompts           - 列出所有记录")
    print("  GET  /prompts?status=done - 按状态过滤")
    print("  GET  /prompts/{id}      - 获取单条记录")
    print("  GET  /stats             - 统计信息")
    print("  GET  /leaderboard       - ELO 排行榜")
    print("  POST /prompts/{id}/score?score=0.8 - 设置评分")
    print("  POST /prompts/{id}/elo?elo=1550   - 设置 ELO")
    print("  POST /prompts/match?winner_id=1&loser_id=2 - ELO 对战")
    print("  DELETE /prompts/{id}    - 删除记录")
    uvicorn.run(app, host="0.0.0.0", port=8000)
