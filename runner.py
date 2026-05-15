"""
Node 轮询引擎 — 每个 node 独立线程，各自按 interval 异步轮询。

用法：
    python3 runner.py
    python3 runner.py --log-level debug
"""
import sqlite3
import os
import json
import time
import subprocess
import signal
import sys
import threading
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")

_running = True
_log_level = "info"  # info | debug

# processed 三态: 0=待处理, -1=处理中, 1=已完成
ST_PENDING    = 0
ST_PROCESSING = -1
ST_DONE       = 1

# 节点线程管理
_node_threads: dict[int, threading.Thread] = {}
_node_stops: dict[int, threading.Event] = {}
_lock = threading.Lock()


def log(level: str, msg: str):
    if _log_level == "debug" or level != "debug":
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][{level.upper()}] {msg}", flush=True)


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


def recover_stale_claims():
    db = get_db()
    try:
        cur = db.execute("UPDATE prompt SET processed=? WHERE processed=?", (ST_PENDING, ST_PROCESSING))
        if cur.rowcount > 0:
            log("info", f"启动恢复: 重置 {cur.rowcount} 条僵尸认领 (processing → pending)")
        db.commit()
    finally:
        db.close()


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
                "SELECT id, prompt FROM prompt WHERE tag=? AND processed=? ORDER BY id ASC LIMIT 1",
                (tag, ST_PENDING),
            ).fetchone()
            if not row:
                log("debug", f"  [{node['name']}] 缺少 tag=\"{tag}\"，跳过")
                return None
            selected[tag] = row[1]
            selected_ids.append(int(row[0]))
            log("debug", f"  [{node['name']}] tag=\"{tag}\" → prompt #{row[0]} ({row[1][:50]}...)")

        for sid in selected_ids:
            cur = db.execute(
                "UPDATE prompt SET processed=? WHERE id=? AND processed=?",
                (ST_PROCESSING, sid, ST_PENDING),
            )
            if cur.rowcount == 0:
                db.rollback()
                log("info", f"  [{node['name']}] prompt #{sid} 已被认领，跳过")
                return None
        db.commit()

        log("debug", f"  [{node['name']}] 已认领 {','.join(map(str,selected_ids))}")

        filled = node["prompt"]
        for tag, text in selected.items():
            filled = filled.replace(f"{{{tag}}}", text)

        model_str = node.get("model") or "default"
        log("info", f"▶ [{node['name']}] inputs={','.join(map(str,selected_ids))}  model={model_str}")

        t0 = time.time()
        result = call_opencode(filled, node.get("model") or None)
        elapsed = time.time() - t0

        input_ids = ",".join(map(str, selected_ids))
        if result["success"]:
            cur = db.execute(
                "INSERT INTO prompt (prev_id, tag, prompt) VALUES (?, ?, ?)",
                (selected_ids[0], node["output_tag"], result["output"]),
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
            log("info",
                f"✓ [{node['name']}] inputs={input_ids}  output=#{output_id}({node['output_tag']})  "
                f"tokens={result['usage']['total']}(in:{result['usage']['input']}/out:{result['usage']['output']})  "
                f"elapsed={elapsed:.1f}s")
        else:
            output_id = None
            db.execute(
                "INSERT INTO node_exec (input_ids, output_id, node_name, model, status,"
                " error, elapsed, input_tokens, output_tokens, total_tokens)"
                " VALUES (?, ?, ?, ?, 'error', ?, ?, 0, 0, 0)",
                (input_ids, output_id, node["name"], node.get("model") or "",
                 result["error"][:500], round(elapsed, 3)),
            )
            log("info",
                f"✗ [{node['name']}] inputs={input_ids}  FAILED  "
                f"error={result['error'][:100]}  elapsed={elapsed:.1f}s")

        for sid in selected_ids:
            db.execute("UPDATE prompt SET processed=? WHERE id=?", (ST_DONE, sid))

        db.commit()
        return {"node_name": node["name"], "status": "success" if result["success"] else "error"}
    except Exception as e:
        db.rollback()
        log("info", f"✗ [{node['name']}]  EXCEPTION: {e}")
        return {"node_name": node["name"], "status": "error", "error": str(e)[:500]}
    finally:
        db.close()


def _read_node(node_id: int) -> Optional[dict]:
    db = get_db()
    try:
        row = db.execute(
            "SELECT id, name, accept_tags, output_tag, model, prompt, interval FROM node WHERE id=?",
            (node_id,),
        ).fetchone()
    finally:
        db.close()
    if not row:
        return None
    return {
        "id": row[0], "name": row[1], "accept_tags": row[2],
        "output_tag": row[3], "model": row[4], "prompt": row[5],
        "interval": row[6],
    }


def node_loop(node_id: int):
    """单个 node 的独立轮询线程"""
    stop = _node_stops[node_id]
    log("debug", f"[node-{node_id}] 线程启动")

    while _running and not stop.is_set():
        node = _read_node(node_id)
        if node is None:
            log("info", f"[node-{node_id}] 节点已从 DB 删除，线程退出")
            return

        if node["accept_tags"]:
            try_execute_node(node)

        stop.wait(node["interval"])

    log("debug", f"[node-{node_id}] 线程退出")


def manager_loop():
    """管理线程：定期同步 DB 中的 node 列表，增删工作线程"""
    while _running:
        db = get_db()
        try:
            rows = db.execute("SELECT id FROM node").fetchall()
        finally:
            db.close()

        db_ids = set(r[0] for r in rows)

        with _lock:
            live_ids = set(_node_threads.keys())

            # 新增 node → 启动线程
            for nid in db_ids - live_ids:
                _node_stops[nid] = threading.Event()
                t = threading.Thread(target=node_loop, args=(nid,), daemon=True, name=f"node-{nid}")
                _node_threads[nid] = t
                t.start()
                node = _read_node(nid)
                name = node["name"] if node else str(nid)
                log("info", f"+ [{name}] 新节点加入，线程已启动  interval={node['interval']}s" if node else f"+ [{nid}] 已启动")

            # 删除 node → 通知线程退出
            for nid in live_ids - db_ids:
                _node_stops[nid].set()
                del _node_threads[nid]
                del _node_stops[nid]
                log("info", f"- [node-{nid}] 已标记退出")

        time.sleep(5)


def main():
    global _running, _log_level

    if "--log-level" in sys.argv:
        idx = sys.argv.index("--log-level")
        if idx + 1 < len(sys.argv):
            _log_level = sys.argv[idx + 1]

    def on_signal(sig, frame):
        global _running
        log("info", "收到退出信号，停止中...")
        _running = False
        # 通知所有 node 线程退出
        with _lock:
            for ev in _node_stops.values():
                ev.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log("info", f"轮询引擎启动  DB={DB_PATH}  log_level={_log_level}")
    recover_stale_claims()

    # 启动管理线程
    mgr = threading.Thread(target=manager_loop, daemon=True, name="manager")
    mgr.start()

    # 主线程等待退出信号
    while _running:
        time.sleep(0.5)

    # 等待所有工作线程退出
    with _lock:
        threads = list(_node_threads.values())
    for t in threads:
        t.join(timeout=2)
    log("info", "已退出")


if __name__ == "__main__":
    main()
