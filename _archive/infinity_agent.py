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
from typing import Optional, List, Dict, Any

@dataclass
class Config:
    moonchan_script: str = field(
        default_factory=lambda: os.environ.get(
            "MOONCHAN_SCRIPT",
            "/home/lumin/.claude/skills/moonchan-forum/scripts/moonchan.py"
        )
    )
    log_file: str = field(
        default_factory=lambda: str(Path(__file__).parent / "infinity_agent.log")
    )
    trace_file: str = field(
        default_factory=lambda: str(Path(__file__).parent / "infinity_trace.jsonl")
    )
    request_timeout: int = 30
    opencode_timeout: int = 300
    max_iterations: int = 15
    nickname: str = "InfinityAgent"

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

SYSTEM_PROMPT = """你是一个具有高度自主权的 InfinityAgent。你的目标是彻底完成用户交给你的任务，直到任务完全达成为止。

### 工作协议 (Strict Protocol)
你必须严格以 JSON 格式输出，不允许有任何额外文字。输出结构如下：
{
  "thought": "详细描述你当前的分析、观察到的结果以及下一步计划",
  "action": "发送给 opencode run 的具体指令（包含需要使用的工具或步骤）",
  "status": "in_progress" | "completed",
  "summary": "仅在 status 为 completed 时填写，总结任务最终结果"
}

### 行为准则
1. **观察与调整**：每一步执行后，你会收到 `observation`。请根据观察结果调整你的 `thought` 和 `action`。
2. **不妥协原则**：只要任务没有完全达成（例如：代码写了但没测试，或者测试失败了），必须保持 `status: "in_progress"`。
3. **逐步推进**：不要尝试在一个 action 中完成所有事情。将复杂任务分解为：分析 -> 实现 -> 验证 -> 修正 -> 完成。
4. **自省**：在 `thought` 中诚实地评估当前进度。如果陷入死循环，尝试改变策略。
"""

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(config.log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\\n")

def save_trace(step: int, data: dict) -> None:
    with open(config.trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, **data}, ensure_ascii=False) + "\\n")

def _extract_url(text: str) -> str:
    urls = re.findall(r'https?://[^\s]+', text)
    return urls[0] if urls else text.strip()

def _parse_opencode_output(output: str) -> str:
    # 提取 opencode run 的最终 text 响应
    result_text = ""
    for line in output.strip().splitlines():
        if not line.strip(): continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "text":
                result_text += obj["part"].get("text", "")
        except: continue
    return result_text.strip()

def get_latest_command() -> tuple[Optional[str], Optional[int]]:
    log("Fetching latest target from Board 666...")
    try:
        res = subprocess.run(
            ["python3", config.moonchan_script, "list", "666", "--pn", "0"],
            capture_output=True, text=True, check=True, timeout=15
        )
        topics = json.loads(res.stdout)
        if not topics: return None, None
        latest = topics[0]
        no = latest.get("no")
        url = _extract_url(latest.get("txt", ""))
        if not url or not url.startswith("http"): return None, no
        resp = requests.get(url, timeout=config.request_timeout)
        resp.raise_for_status()
        return resp.text, no
    except Exception as e:
        log(f"Error fetching target: {e}")
        return None, None

def post_reply(post_no: int, text: str) -> None:
    log(f"Posting final report to post no.{post_no}...")
    try:
        subprocess.run(
            ["python3", config.moonchan_script, "reply", "666", str(post_no), config.nickname, text],
            check=True, capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        log(f"Reply failed: {e}")

def execute_step(prompt: str, model: str) -> str:
    cmd = ["opencode", "run", "--format", "json", "-m", model]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            encoding="utf-8", timeout=config.opencode_timeout
        )
        return _parse_opencode_output(result.stdout)
    except Exception as e:
        return f"Error executing step: {str(e)}"

def main():
    log("=== InfinityAgent Autonomous Loop Started ===")
    target, post_no = get_latest_command()
    if not target:
        log("No target found. Exiting.")
        return

    log(f"Target acquired from post #{post_no}: {target[:100]}...")
    
    state = {
        "target": target,
        "history": [],
        "iteration": 0
    }

    while state["iteration"] < config.max_iterations:
        state["iteration"] += 1
        it = state["iteration"]
        log(f"--- Iteration {it} ---")

        # 构造上下文 Prompt
        history_str = "\\n".join([f"Step {i}: {h['thought']} \\nAction: {h['action']} \\nObservation: {h['observation'][:200]}..." 
                                for i, h in enumerate(state["history"], 1)])
        
        full_prompt = f"{SYSTEM_PROMPT}\\n\\nTarget: {target}\\n\\nHistory:\\n{history_str}\\n\\nNow, provide your next thought and action in JSON format."
        
        model = random.choice(FREE_MODELS)
        log(f"Using model {model} for decision...")
        
        raw_response = execute_step(full_prompt, model)
        
        # 尝试解析 JSON
        try:
            # 处理 LLM 可能包含的 markdown 代码块
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
            else:
                decision = json.loads(raw_response)
        except Exception as e:
            log(f"Failed to parse decision JSON: {e}. Raw response: {raw_response[:200]}")
            decision = {
                "thought": "I failed to output valid JSON. I will try to re-evaluate.",
                "action": "Repeat last action with more care",
                "status": "in_progress"
            }

        thought = decision.get("thought", "No thought provided")
        action = decision.get("action", "No action provided")
        status = decision.get("status", "in_progress")
        
        log(f"Thought: {thought}")
        log(f"Action: {action}")

        # 执行 Action
        log(f"Executing action...")
        observation = execute_step(action, model)
        log(f"Observation: {observation[:200]}...")

        # 记录 Trace
        step_data = {
            "thought": thought,
            "action": action,
            "observation": observation,
            "status": status
        }
        state["history"].append(step_data)
        save_trace(it, step_data)

        if status == "completed":
            log("Target reached! Terminating loop.")
            summary = decision.get("summary", "Task completed successfully.")
            post_reply(post_no, f"[InfinityAgent] ✅ 任务完成！\\n\\n总结：{summary}")
            break
    else:
        log("Reached max iterations. Terminating.")
        post_reply(post_no, f"[InfinityAgent] ⚠️ 达到最大迭代次数({config.max_iterations})，任务尚未完全完成。")

if __name__ == "__main__":
    main()
