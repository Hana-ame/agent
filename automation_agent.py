import subprocess
import json
import requests
import sys
import os
import re
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
    request_timeout: int = 30
    opencode_timeout: int = 120
    max_retries: int = 2


config = Config()

SYSTEM_PROMPT = """你是一个只能输出 JSON 的智能体。无论用户给你什么任务，你必须严格遵守以下规则：

1. **整个响应必须是一个单一、合法的 JSON 对象**。
2. 不允许在 JSON 之外输出任何额外的文字、解释、打招呼或 markdown 标记（包括 ```json）。
3. JSON 结构必须包含以下固定字段：
   {{
     "status": "success" 或 "error",
     "message": "一句话概括完成的任务或错误原因",
     "data": {{}}
   }}
4. 如果任务成功完成，在 `data` 中放入你生成的结果。
5. 如果任务失败，`status` 必须为 "error"，并在 `message` 中说明具体错误。
6. 不要省略任何字段，哪怕值为空对象或空字符串。"""


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(config.log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _extract_url(text: str) -> str:
    urls = re.findall(r'https?://[^\s]+', text)
    return urls[0] if urls else text.strip()


def _parse_opencode_output(output: str) -> dict:
    for line in output.strip().splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "text":
                text = obj.get("part", {}).get("text", "")
                if text.strip():
                    parsed = json.loads(text)
                    return parsed
            else:
                return obj
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return {"status": "error", "message": "No valid JSON found", "data": {}}


def get_latest_command() -> tuple[Optional[str], Optional[int]]:
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
                return None, None

            latest = topics[0]
            no = latest.get("no")
            txt = latest.get("txt", "")
            url = _extract_url(txt)

            if not url or not url.startswith("http"):
                log(f"No valid URL found in post no.{no}")
                return None, no

            log(f"Latest post no.{no}: {url}")
            resp = requests.get(url, timeout=config.request_timeout)
            resp.raise_for_status()
            return resp.text, no
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log(f"Attempt {attempt + 1} failed (subprocess): {e}")
        except requests.RequestException as e:
            log(f"Attempt {attempt + 1} failed (HTTP): {e}")
        except Exception as e:
            log(f"Attempt {attempt + 1} failed: {e}")
    return None, None


def execute_with_json(instructions: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\n现在执行以下任务并输出 JSON：\n{instructions}"
    cmd = ["opencode", "run", "--format", "json"]

    log("Executing via opencode run --format json...")
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
        log("Execution timed out (120s)")
        return {"status": "error", "message": "Timeout", "data": {}}
    except Exception as e:
        log(f"Execution error: {e}")
        return {"status": "error", "message": str(e), "data": {}}


def main() -> dict:
    log("=== Automation Agent v2 (JSON mode) ===")
    instructions, post_no = get_latest_command()
    if not instructions:
        log("No instructions retrieved.")
        return {"status": "error", "message": "No instructions retrieved", "data": {"post_no": post_no}}

    log(f"Instructions length: {len(instructions)} chars")
    result = execute_with_json(instructions)
    result.setdefault("data", {})["post_no"] = post_no
    log("Done.")
    return result


if __name__ == "__main__":
    output = main()
    print(json.dumps(output, ensure_ascii=False))
