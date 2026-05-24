import json
import subprocess
import time
import os
import re
import random
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")

FREE_MODELS = [
    "siliconflow-cn/Qwen/Qwen3-8B",
    "siliconflow-cn/Qwen/Qwen3.5-4B",
    "opencode/big-pickle",
    "opencode/deepseek-v4-flash-free",
    "opencode/minimax-m2.5-free",
    "opencode/nemotron-3-super-free",
    "opencode/qwen3.6-plus-free",
    "google/gemma-4-31b-it",
    "google/gemini-3-flash-preview",
]


def get_db():
    import sqlite3
    return sqlite3.connect(DB_PATH, timeout=10)


def call_opencode(prompt: str, model: str) -> dict:
    cmd = ["opencode", "run", "--format", "json", "-m", model]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            encoding="utf-8", timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Timeout (300s)"}
    except FileNotFoundError:
        return {"success": False, "output": "", "error": "opencode not found"}

    output_text = ""
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "text":
                output_text += obj["part"].get("text", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    return {"success": result.returncode == 0, "output": output_text.strip(), "error": ""}


def substitute_vars(template: str, state: dict) -> str:
    def replacer(match):
        key = match.group(1)
        return str(state.get(key, match.group(0)))
    return re.sub(r'\{(\w+)\}', replacer, template)


def validate_definition(defn: dict) -> tuple[bool, str]:
    if "steps" not in defn or not isinstance(defn["steps"], list):
        return False, "steps must be a list"
    if not defn["steps"]:
        return False, "steps cannot be empty"
    if "entry" not in defn:
        return False, "entry step id is required"

    ids = set()
    for i, step in enumerate(defn["steps"]):
        if "id" not in step:
            return False, f"step[{i}] missing id"
        if step["id"] in ids:
            return False, f"duplicate step id: {step['id']}"
        ids.add(step["id"])
        if "prompt" not in step:
            return False, f"step[{step['id']}] missing prompt"

    if defn["entry"] not in ids:
        return False, f"entry step '{defn['entry']}' not found in steps"

    return True, "ok"


def build_step_map(defn: dict) -> dict:
    return {s["id"]: s for s in defn["steps"]}


def run_pipeline(pipeline_id: int, input_data: dict) -> dict:
    db = get_db()
    try:
        row = db.execute(
            "SELECT id, name, definition FROM pipeline WHERE id=?", (pipeline_id,)
        ).fetchone()
    finally:
        db.close()

    if not row:
        return {"status": "error", "error": "pipeline not found"}

    pipeline_name = row[1]
    try:
        defn = json.loads(row[2])
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid definition JSON"}

    valid, msg = validate_definition(defn)
    if not valid:
        return {"status": "error", "error": msg}

    step_map = build_step_map(defn)
    state = dict(input_data)
    steps_log = []
    start_time = time.time()

    current_id = defn["entry"]
    while current_id:
        step = step_map.get(current_id)
        if not step:
            steps_log.append({
                "step_id": current_id,
                "status": "error",
                "error": f"step '{current_id}' not found in definition"
            })
            break

        step_start = time.time()
        prompt = substitute_vars(step["prompt"], state)
        model = step.get("model") or random.choice(FREE_MODELS)
        output_key = step.get("output_key", current_id)

        step_log = {
            "step_id": current_id,
            "name": step.get("name", current_id),
            "model": model,
            "prompt": prompt,
        }

        result = call_opencode(prompt, model)
        elapsed = time.time() - step_start
        step_log["elapsed"] = round(elapsed, 2)

        if result["success"] and result["output"]:
            state[output_key] = result["output"]
            step_log["status"] = "success"
            step_log["output_preview"] = result["output"][:200]
        else:
            err = result.get("error") or "empty response"
            step_log["status"] = "error"
            step_log["error"] = err
            steps_log.append(step_log)
            break

        steps_log.append(step_log)
        current_id = step.get("next")

    total_elapsed = time.time() - start_time
    is_success = all(s["status"] == "success" for s in steps_log)

    exec_record = {
        "pipeline_id": pipeline_id,
        "input_data": json.dumps(input_data, ensure_ascii=False),
        "output_data": json.dumps(state, ensure_ascii=False),
        "status": "success" if is_success else "error",
        "error": "" if is_success else (steps_log[-1].get("error", "unknown") if steps_log else "unknown"),
        "steps_log": json.dumps(steps_log, ensure_ascii=False),
    }

    db = get_db()
    try:
        db.execute(
            "INSERT INTO pipeline_exec (pipeline_id, input_data, output_data, status, error, steps_log) "
            "VALUES (:pipeline_id, :input_data, :output_data, :status, :error, :steps_log)",
            exec_record,
        )
        db.commit()
    finally:
        db.close()

    return {
        "status": "success" if is_success else "error",
        "error": "" if is_success else exec_record["error"],
        "state": state,
        "steps": steps_log,
        "elapsed": round(total_elapsed, 2),
    }
