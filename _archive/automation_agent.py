import subprocess
import json
import requests
import sys
import os
import re
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    moonchan_script: str = field(
        default_factory=lambda: os.environ.get(
            "MOONCHAN_SCRIPT",
            "/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py"
        )
    )
    log_file: str = field(
        default_factory=lambda: os.environ.get(
            "AUTOMATION_LOG",
            str(Path(__file__).parent / "automation_agent.log")
        )
    )
    history_file: str = field(
        default_factory=lambda: str(Path(__file__).parent / "automation_history.jsonl")
    )
    nickname: str = field(
        default_factory=lambda: os.environ.get("MOONCHAN_NICK", "OpenCode_Agent")
    )
    request_timeout: int = 30
    opencode_timeout: int = 300
    max_retries: int = 2


config = Config()

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

AGENTS = {
    "default": """你是一个只能输出 JSON 的智能体。无论用户给你什么任务，你必须严格遵守以下规则：
1. **整个响应必须是一个单一、合法的 JSON 对象**。
2. 不允许在 JSON 之外输出任何额外的文字、解释、打招呼或 markdown 标记（包括 ```json）。
3. JSON 结构必须包含以下固定字段：
   {
     "status": "success" 或 "error",
     "message": "一句话概括完成的任务或错误原因",
     "data": {}
   }
4. 如果任务成功完成，在 `data` 中放入你生成的结果。
5. 如果任务失败，`status` 必须为 "error"，并在 `message` 中说明具体错误。
6. 不要省略任何字段，哪怕值为空对象或空字符串。""",

    "code": """你是一个代码执行 Agent，必须按顺序执行以下步骤并输出 JSON：
1. 理解用户需求
2. 编写/修改代码
3. 验证代码
4. 输出 JSON 结果
输出格式：
{{
  "status": "success" 或 "error",
  "message": "做了什么",
  "data": {{
    "files_changed": ["file1.py", ...],
    "summary": "具体改动说明"
  }}
}}
不在 JSON 外输出任何额外内容。""",

    "inspect": """你是一个代码审查 Agent，输出格式：
{{
  "status": "success" 或 "error",
  "message": "审查结论",
  "data": {{
    "files": ["file1.py", ...],
    "issues": [{{"file": "...", "line": N, "severity": "warning"/"error", "message": "..."}}],
    "suggestions": ["建议1", ...]
  }}
}}
不在 JSON 外输出任何额外内容。""",
}


def detect_agent_type(instructions: str) -> str:
    keywords = {
        "code": ["写代码", "实现", "编写", "修改", "添加", "开发", "implement", "code", "write", "add", "create", "edit", "编辑"],
        "inspect": ["审查", "review", "检查", "审计", "audit", "inspect", "分析", "analyze"],
    }
    inst_lower = instructions.lower() if instructions else ""
    for agent_type, kws in keywords.items():
        if any(kw in inst_lower for kw in kws):
            return agent_type
    return "default"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(config.log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _extract_url(text: str) -> str:
    urls = re.findall(r'https?://[^\s]+', text)
    return urls[0] if urls else text.strip()


def save_result(post_no: int, instructions: str, result: dict) -> None:
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "post_no": post_no,
        "instructions": instructions,
        "result": result
    }
    with open(config.history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log(f"Result saved to {config.history_file}")


def post_reply(post_no: int, text: str) -> bool:
    log(f"Posting reply to post no.{post_no}...")
    try:
        subprocess.run(
            ["python3", config.moonchan_script, "reply", "666", str(post_no), config.nickname, text],
            check=True, capture_output=True, text=True, timeout=30
        )
        log("Reply posted successfully.")
        return True
    except Exception as e:
        log(f"Failed to post reply: {e}")
        return False


def _parse_opencode_output(output: str) -> dict:
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "text":
                text = obj.get("part", {}).get("text", "")
                txt = text.strip()
                if txt:
                    try:
                        return json.loads(txt)
                    except json.JSONDecodeError:
                        continue
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return {"status": "error", "message": "No valid JSON found", "data": {}}


def get_latest_command() -> tuple[Optional[str], Optional[int], Optional[int]]:

    log("Fetching latest command from Board 666...")
    for attempt in range(config.max_retries):
        try:
            res = subprocess.run(
                ["python3", config.moonchan_script, "list", "666", "--pn", "0"],
                capture_output=True, text=True, check=True, timeout=15
            )
            topics = json.loads(res.stdout)
            if not topics:
                log("No topics found.")
                return None, None, None

            latest = topics[0]
            no = latest.get("no")
            txt = latest.get("txt", "")
            url = _extract_url(txt)

            if not url or not url.startswith("http"):
                log(f"No valid URL found in post no.{no}")
                return None, no, None

            tid = None
            resto = latest.get("resto", 0)
            if resto is not None and resto > 0:
                tid = resto
            else:
                tid = no

            log(f"Latest post no.{no} tid={tid}: {url}")
            resp = requests.get(url, timeout=config.request_timeout)
            resp.raise_for_status()
            return resp.text, no, tid
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log(f"Attempt {attempt + 1} failed (subprocess): {e}")
        except requests.RequestException as e:
            log(f"Attempt {attempt + 1} failed (HTTP): {e}")
        except Exception as e:
            log(f"Attempt {attempt + 1} failed: {e}")
    return None, None, None


def execute_with_json(instructions: str) -> dict:
    agent_type = detect_agent_type(instructions)
    system_prompt = AGENTS.get(agent_type, AGENTS["default"])
    prompt = f"{system_prompt}\n\n现在执行以下任务并输出 JSON：\n{instructions}"
    model = random.choice(FREE_MODELS)
    cmd = ["opencode", "run", "--format", "json", "-m", model]

    log(f"Agent={agent_type} Model={model} Executing via opencode run --format json...")
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            encoding="utf-8", timeout=config.opencode_timeout
        )
        log(f"Return code: {result.returncode}")

        output = result.stdout.strip()
        if not output:
            log("Empty output from opencode")
            return {"status": "error", "message": "Empty output", "data": {}}

        parsed = _parse_opencode_output(output)
        if parsed.get("status") == "error":
            log(f"Execution error: {parsed.get('message')}")
        else:
            log(f"Final result: {json.dumps(parsed, ensure_ascii=False)}")
        return parsed
    except subprocess.TimeoutExpired:
        log("Execution timed out (300s)")
        return {"status": "error", "message": "Timeout", "data": {}}
    except Exception as e:
        log(f"Execution error: {e}")
        return {"status": "error", "message": str(e), "data": {}}


def main() -> dict:
    log("=== Automation Agent v2 (JSON mode) ===")
    instructions, post_no, tid = get_latest_command()
    if not instructions:
        log("No instructions retrieved.")
        return {"status": "error", "message": "No instructions retrieved", "data": {"post_no": post_no}}

    log(f"Instructions length: {len(instructions)} chars")
    result = execute_with_json(instructions)
    
    save_result(post_no, instructions, result)
    
    if result.get("status") == "success":
        reply_text = f"任务已完成！\n\n结果：\n{json.dumps(result.get('data', {}), ensure_ascii=False, indent=2)}"
    else:
        reply_text = f"任务执行失败：{result.get('message', '未知错误')}"
    
    if tid:
        post_reply(tid, reply_text)
    elif post_no:
        post_reply(post_no, reply_text)

    result.setdefault("data", {})["post_no"] = post_no
    log("Done.")
    return result



if __name__ == "__main__":
    output = main()
    print(json.dumps(output, ensure_ascii=False))