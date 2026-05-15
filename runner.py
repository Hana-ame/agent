"""
Node 轮询引擎 — 独立于 FastAPI 运行，仅通过 state.db 通信。

用法：
    python3 runner.py
"""
import sqlite3
import os
import json
import time
import subprocess
import signal
import sys
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")

_running = True
_last_checks: dict[int, float] = {}


def get_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def call_opencode(prompt: str, model: str = None) -> dict:
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


def try_execute_node(node: dict) -> Optional[dict]:
    tags = [t.strip() for t in node["accept_tags"].split(",") if t.strip()]
    if not tags:
        return None

    db = get_db()
    try:
        selected = {}
        selected_ids = []
        for tag in tags:
            row = db.execute(
                "SELECT id, prompt FROM prompt WHERE tag=? AND processed=0 ORDER BY id ASC LIMIT 1",
                (tag,),
            ).fetchone()
            if not row:
                return None
            selected[tag] = row[1]
            selected_ids.append(str(row[0]))

        filled = node["prompt"]
        for tag, text in selected.items():
            filled = filled.replace(f"{{{tag}}}", text)

        t0 = time.time()
        result = call_opencode(filled, node.get("model") or None)
        elapsed = time.time() - t0

        input_ids = ",".join(selected_ids)
        if result["success"]:
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

        for sid in selected_ids:
            db.execute("UPDATE prompt SET processed=1 WHERE id=?", (int(sid),))

        db.commit()
        status = "success" if result["success"] else "error"
        print(f"[runner] {node['name']} ({','.join(selected_ids)}) → {status}  {elapsed:.1f}s", flush=True)
        return {"node_name": node["name"], "status": status}
    except Exception as e:
        db.rollback()
        print(f"[runner] {node['name']} error: {e}", flush=True)
        return {"node_name": node["name"], "status": "error", "error": str(e)[:500]}
    finally:
        db.close()


def poll_loop():
    global _last_checks
    while _running:
        db = get_db()
        try:
            nodes = db.execute(
                "SELECT id, name, accept_tags, output_tag, model, prompt, interval FROM node"
            ).fetchall()
        finally:
            db.close()

        now = time.time()
        for n in nodes:
            if not _running:
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
            try_execute_node(node_dict)

        time.sleep(1)


def main():
    global _running

    def on_signal(sig, frame):
        global _running
        print("[runner] 收到退出信号，停止中...", flush=True)
        _running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"[runner] 轮询引擎启动 (DB: {DB_PATH})", flush=True)
    poll_loop()
    print("[runner] 已退出", flush=True)


if __name__ == "__main__":
    main()
