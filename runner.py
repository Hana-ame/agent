"""
Node 轮询引擎 — 独立于 FastAPI 运行，仅通过 state.db 通信。

用法：
    python3 runner.py              # 前台运行
    python3 runner.py --log-level debug  # 详细日志
"""
import sqlite3
import os
import json
import time
import subprocess
import signal
import sys
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")

_running = True
_last_checks: dict[int, float] = {}
_log_level = "info"  # info | debug


def log(level: str, msg: str):
    """带时间戳和级别的日志输出"""
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


# processed 三态: 0=待处理, -1=处理中, 1=已完成
ST_PENDING    = 0
ST_PROCESSING = -1
ST_DONE       = 1


def recover_stale_claims():
    """启动时重置崩溃遗留的僵尸认领（processed=-1 → 0）"""
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
        # 1. 查找各 tag 的未处理 prompt
        selected = {}
        selected_ids = []
        for tag in tags:
            row = db.execute(
                "SELECT id, prompt FROM prompt WHERE tag=? AND processed=? ORDER BY id ASC LIMIT 1",
                (tag, ST_PENDING),
            ).fetchone()
            if not row:
                log("debug", f"  {node['name']}: 缺少 tag=\"{tag}\"，跳过")
                return None
            selected[tag] = row[1]
            selected_ids.append(int(row[0]))
            log("debug", f"  {node['name']}: tag=\"{tag}\" → prompt #{row[0]} ({row[1][:50]}...)")

        # 2. 原子认领: UPDATE processed=0 → -1，通过 rowcount 防多实例竞争
        for sid in selected_ids:
            cur = db.execute(
                "UPDATE prompt SET processed=? WHERE id=? AND processed=?",
                (ST_PROCESSING, sid, ST_PENDING),
            )
            if cur.rowcount == 0:
                db.rollback()
                log("info", f"  {node['name']}: prompt #{sid} 已被其他进程认领，放弃本轮")
                return None
        db.commit()

        log("debug", f"  {node['name']}: 已认领 {','.join(map(str,selected_ids))}")

        # 3. 变量替换
        filled = node["prompt"]
        for tag, text in selected.items():
            filled = filled.replace(f"{{{tag}}}", text)

        model_str = node.get("model") or "default"
        log("info", f"▶ {node['name']}  inputs={','.join(map(str,selected_ids))}  model={model_str}")

        # 4. 调用 opencode
        t0 = time.time()
        result = call_opencode(filled, node.get("model") or None)
        elapsed = time.time() - t0

        # 5. 写回结果和日志，同时标记已完成
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
                f"✓ {node['name']}  inputs={input_ids}  output=#{output_id}({node['output_tag']})  "
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
                f"✗ {node['name']}  inputs={input_ids}  FAILED  "
                f"error={result['error'][:100]}  elapsed={elapsed:.1f}s")

        # 标记已完成的输入
        for sid in selected_ids:
            db.execute("UPDATE prompt SET processed=? WHERE id=?", (ST_DONE, sid))

        db.commit()
        return {"node_name": node["name"], "status": "success" if result["success"] else "error"}
    except Exception as e:
        db.rollback()
        log("info", f"✗ {node['name']}  EXCEPTION: {e}")
        return {"node_name": node["name"], "status": "error", "error": str(e)[:500]}
    finally:
        db.close()


def poll_loop():
    global _last_checks
    tick = 0
    while _running:
        tick += 1
        db = get_db()
        try:
            nodes = db.execute(
                "SELECT id, name, accept_tags, output_tag, model, prompt, interval FROM node"
            ).fetchall()
        finally:
            db.close()

        log("debug", f"--- tick #{tick}, {len(nodes)} nodes ---")

        now = time.time()
        executed = 0
        skipped_wait = 0
        skipped_tag = 0
        for n in nodes:
            if not _running:
                break
            node_id = n[0]
            interval = n[6]
            if now - _last_checks.get(node_id, 0) < interval:
                skipped_wait += 1
                continue
            _last_checks[node_id] = now
            node_dict = {
                "id": n[0], "name": n[1], "accept_tags": n[2],
                "output_tag": n[3], "model": n[4], "prompt": n[5],
                "interval": n[6],
            }
            res = try_execute_node(node_dict)
            if res is None:
                skipped_tag += 1
            else:
                executed += 1

        log("debug", f"  tick #{tick} done: {executed} exec, {skipped_tag} no-data, {skipped_wait} waiting")

        time.sleep(1)


def main():
    global _running, _log_level

    if "--log-level" in sys.argv:
        idx = sys.argv.index("--log-level")
        if idx + 1 < len(sys.argv):
            _log_level = sys.argv[idx + 1]

    def on_signal(sig, frame):
        nonlocal on_signal
        global _running
        log("info", "收到退出信号，停止中...")
        _running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log("info", f"轮询引擎启动  DB={DB_PATH}  log_level={_log_level}")
    recover_stale_claims()
    poll_loop()
    log("info", "已退出")


if __name__ == "__main__":
    main()
