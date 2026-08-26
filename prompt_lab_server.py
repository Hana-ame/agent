"""Prompt 组合探索系统 — FastAPI 后台服务版

启动:  python3 prompt_lab_server.py
端口:  8319
"""

import json
import random
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

from opencode import run as opencode_run

from prompt_lab import PromptDB, find_untried_combinations, build_full_prompt


# ── Config ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DEFAULT_AGENT = "Auto666"
DEFAULT_MODEL = "deepseek-v4-flash-free"
JUDGE_MODEL = "siliconflow-cn/Qwen/Qwen3-8B"

app = FastAPI(title="Prompt Lab", version="1.0")
db = PromptDB()

# 后台运行锁
_run_lock = threading.Lock()
_bg_running = False


# ── 评判 ─────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """你是一个回复质量评判专家。请对以下 AI 回复进行评分。

评分标准（0.0 ~ 1.0）：
- 1.0: 完美，完全解决问题，信息准确完整
- 0.7-0.9: 良好，基本解决问题，有些小瑕疵
- 0.4-0.6: 一般，部分解决问题，有明显不足
- 0.1-0.3: 差，几乎没有解决问题
- 0.0: 完全无用

原始问题：
{original}

AI 回复：
{response}

请只输出一个 0.0~1.0 之间的浮点数，不要任何解释。
评分："""


def judge_response(original_text, response_text, judge_model=JUDGE_MODEL):
    prompt = JUDGE_PROMPT.format(original=original_text, response=response_text)
    try:
        result = opencode_run(prompt, agent="Null", model=judge_model, timeout=60)
        out = result["output"]
        if isinstance(out, dict):
            out = str(out)
        text = str(out).strip()
        m = re.search(r"[\d.]+(?:[eE][+-]?\d+)?", text)
        if m:
            rate = float(m.group())
            return max(0.0, min(1.0, rate))
        return 0.0
    except Exception:
        return 0.0


def call_agent(agent, model, full_prompt, timeout=600):
    try:
        result = opencode_run(full_prompt, agent=agent, model=model, timeout=timeout)
        out = result["output"]
        if isinstance(out, dict):
            text = json.dumps(out, indent=2, ensure_ascii=False)
        else:
            text = str(out)
        return text, result["success"]
    except Exception as e:
        return f"Error: {e}", False


# ── 单次执行 ─────────────────────────────────────────────────────────────

def execute_one():
    """随机选一个未尝试组合，执行+评判。返回结果 dict，没有可尝试时返回 None。"""
    succeed = db.get_succeed_prompts()
    if len(succeed) < 1:
        return None

    succeed_ids = [p["id"] for p in succeed]
    tried = db.get_tried_combinations()
    untried = find_untried_combinations(succeed_ids, tried)

    if not untried:
        return None

    chosen = random.choice(untried)
    chosen_list = sorted(chosen)
    base_text = succeed[-1]["prompt_text"]

    ref_responses = {}
    for pid in chosen_list:
        pr = db.get_prompt_by_id(pid)
        if pr:
            ref_responses[pid] = pr["response"]

    full_prompt = build_full_prompt(base_text, ref_responses)

    pid = db.add_prompt(
        DEFAULT_AGENT, DEFAULT_MODEL,
        chosen_list, base_text, full_prompt,
    )

    response, ok = call_agent(DEFAULT_AGENT, DEFAULT_MODEL, full_prompt)
    status = "succeed" if ok else "failed"
    db.update_result(pid, response, status)

    rate = 0.0
    if ok:
        rate = judge_response(base_text, response)
        db.update_reply_rate(pid, rate)

    return {
        "id": pid,
        "prompt_ids": chosen_list,
        "status": status,
        "reply_rate": rate,
        "response_preview": response[:200],
    }


# ── 后台循环 ─────────────────────────────────────────────────────────────

def bg_loop(interval=30, max_runs=None):
    global _bg_running
    runs = 0
    while _bg_running:
        if max_runs is not None and runs >= max_runs:
            break
        with _run_lock:
            result = execute_one()
        if result:
            runs += 1
            print(f"[BG] #{result['id']} {result['status']} rate={result['reply_rate']:.3f} ids={result['prompt_ids']}")
        else:
            print("[BG] 无未尝试组合，等待中...")
        time.sleep(interval)


# ── API Routes ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """服务状态"""
    succeed = db.get_succeed_prompts()
    tried = db.get_tried_combinations()
    if succeed:
        untried = find_untried_combinations([p["id"] for p in succeed], tried)
    else:
        untried = []
    return {
        "bg_running": _bg_running,
        "total_records": len(db.get_all()),
        "succeed_count": len(succeed),
        "tried_combinations": len(tried),
        "untried_combinations": len(untried),
    }


@app.post("/api/add")
async def api_add(prompt: str = Query(..., description="根问题的 prompt 文本")):
    """添加根 prompt，立即执行+评判"""
    pid = db.add_prompt(DEFAULT_AGENT, DEFAULT_MODEL, [], prompt, prompt)
    response, ok = call_agent(DEFAULT_AGENT, DEFAULT_MODEL, prompt)
    status = "succeed" if ok else "failed"
    db.update_result(pid, response, status)

    rate = 0.0
    if ok:
        rate = judge_response(prompt, response)
        db.update_reply_rate(pid, rate)

    return {
        "id": pid,
        "prompt": prompt[:200],
        "status": status,
        "reply_rate": rate,
        "response_preview": response[:500],
    }


@app.post("/api/run")
async def api_run(count: int = Query(1, description="执行次数（默认1）")):
    """手动触发执行（同步，单次/多次）"""
    results = []
    with _run_lock:
        for i in range(count):
            r = execute_one()
            if r is None:
                results.append({"msg": "无未尝试组合"})
                break
            results.append(r)
    return {"count": len(results), "results": results}


@app.post("/api/bg/start")
async def api_bg_start(
    interval: int = Query(30, description="间隔秒数"),
    max_runs: int = Query(None, description="最大执行次数，不传则无限"),
):
    """启动后台自动循环"""
    global _bg_running
    if _bg_running:
        return {"msg": "已在运行中"}
    _bg_running = True
    t = threading.Thread(target=bg_loop, args=(interval, max_runs), daemon=True)
    t.start()
    return {"msg": "后台循环已启动", "interval": interval, "max_runs": max_runs}


@app.post("/api/bg/stop")
async def api_bg_stop():
    """停止后台循环"""
    global _bg_running
    _bg_running = False
    return {"msg": "后台循环已停止"}


@app.get("/api/list")
async def api_list():
    """列出所有记录"""
    rows = db.get_all()
    return {
        "count": len(rows),
        "records": [
            {
                "id": r["id"],
                "agent": r["agent"],
                "model": r["model"],
                "prompt_ids": json.loads(r["prompt_ids"]) if isinstance(r["prompt_ids"], str) else r["prompt_ids"],
                "prompt_text": r["prompt_text"][:200],
                "status": r["status"],
                "reply_rate": r["reply_rate"],
                "response_preview": (r["response"] or "")[:200],
            }
            for r in rows
        ],
    }


@app.get("/api/detail/{pid}")
async def api_detail(pid: int):
    """查看某条记录完整信息"""
    r = db.get_prompt_by_id(pid)
    if not r:
        return JSONResponse({"error": f"id={pid} 不存在"}, 404)
    return {
        "id": r["id"],
        "agent": r["agent"],
        "model": r["model"],
        "prompt_ids": json.loads(r["prompt_ids"]) if isinstance(r["prompt_ids"], str) else r["prompt_ids"],
        "prompt_text": r["prompt_text"],
        "full_prompt": r["full_prompt"],
        "response": r["response"],
        "status": r["status"],
        "reply_rate": r["reply_rate"],
    }


@app.post("/api/judge/{pid}")
async def api_judge(pid: int):
    """重新评判某条记录"""
    r = db.get_prompt_by_id(pid)
    if not r:
        return JSONResponse({"error": f"id={pid} 不存在"}, 404)
    if not r["response"]:
        return JSONResponse({"error": "无 response"}, 400)
    rate = judge_response(r["prompt_text"], r["response"])
    db.update_reply_rate(pid, rate)
    return {"id": pid, "reply_rate": rate}


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Prompt Lab Server  ")
    print("  http://0.0.0.0:8319")
    print("=" * 50)
    print("API:")
    print("  GET  /api/status          — 状态")
    print("  POST /api/add?prompt=...  — 添加根问题")
    print("  POST /api/run?count=1     — 手动执行")
    print("  POST /api/bg/start        — 启动后台循环")
    print("  POST /api/bg/stop         — 停止后台循环")
    print("  GET  /api/list            — 列表")
    print("  GET  /api/detail/<id>     — 详情")
    print("  POST /api/judge/<id>      — 重新评判")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8319, log_level="info")
