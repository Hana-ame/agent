"""
OpenCode JSON 输出工具模块

提供两种方法获取结构化 JSON 输出：
1. 命令行参数 (--json / --format json) — 最稳定
2. 系统提示词约束 — 作为补充

用法：
    from opencode_json import run_opencode_json, run_with_agent

    # 直接调用
    result = run_opencode_json("分析这段代码的问题")

    # 使用指定 Agent
    result = run_with_agent("randomAgent", "你的提示词")

    # 双管齐下（命令行 + 系统提示词）
    result = run_opencode_json("提示词", enforce_system_prompt=True)
"""

import subprocess
import json
import os
from typing import Optional

# JSON 强制约束系统提示词
JSON_ENFORCE_PROMPT = """
你是一个只能输出 JSON 的智能体。无论用户给你什么任务，你必须严格遵守以下规则：

1. **整个响应必须是一个单一、合法的 JSON 对象**。
2. 不允许在 JSON 之外输出任何额外的文字、解释、打招呼或 markdown 标记（包括 ```json）。
3. 如果需要思考，必须在内部完成，对外不可见。
4. JSON 结构必须包含以下固定字段，且字段名不可拼写错误：
   {
     "status": "success" 或 "error",
     "message": "一句话概括完成的任务或错误原因",
     "data": {}
   }
5. 如果任务成功完成，在 `data` 中放入你生成的结果。
6. 如果任务失败，`status` 必须为 "error"，并在 `message` 中说明具体错误，`data` 留空对象。
7. 不要省略任何字段，哪怕值为空对象或空字符串。

现在等待用户输入，然后直接输出上述结构的 JSON，不要附加任何其他内容。
"""


def parse_opencode_output(raw_output: str) -> Optional[dict]:
    """
    解析 opencode 的 JSON 格式输出。

    opencode --format json 输出的是 NDJSON（每行一个 JSON 事件），
    需要从中提取最终的文本响应。
    """
    output_text = ""
    usage = {}
    last_json = None

    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            last_json = event
            if event.get("type") == "text":
                output_text += event.get("part", {}).get("text", "")
            elif event.get("type") == "step_finish":
                usage = event.get("part", {}).get("tokens", {})
        except json.JSONDecodeError:
            continue

    # 如果解析到了完整的 JSON 响应，直接返回
    if output_text:
        return {
            "success": True,
            "output": output_text,
            "usage": {
                "input": usage.get("input", 0),
                "output": usage.get("output", 0),
                "total": usage.get("total", 0),
            },
        }

    # 回退：尝试将整个输出作为单个 JSON 解析
    if last_json:
        return {"success": True, "output": json.dumps(last_json), "usage": {}}

    return None


def try_extract_json(raw_output: str) -> Optional[dict]:
    """
    尝试从原始文本中提取 JSON（用于未使用 --json 参数的场景）。
    策略：先尝试解析整个输出，失败则尝试提取最后一行。
    """
    raw = raw_output.strip()

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取最后一行
    lines = raw.splitlines()
    if lines:
        try:
            return json.loads(lines[-1].strip())
        except json.JSONDecodeError:
            pass

    # 尝试查找第一个 { 到最后一个 } 之间的内容
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def run_opencode_json(
    prompt: str,
    model: str = None,
    agent: str = None,
    enforce_system_prompt: bool = False,
    timeout: int = 3600,
) -> dict:
    """
    调用 opencode 并获取结构化 JSON 输出。

    Args:
        prompt: 用户提示词
        model: 指定模型（可选）
        agent: 指定 Agent（可选）
        enforce_system_prompt: 是否在提示词前追加 JSON 约束系统提示
        timeout: 超时时间（秒）

    Returns:
        dict: 包含 success/output/usage 或 success/error 的字典
    """
    cmd = ["opencode", "run", "--format", "json"]

    if model:
        cmd.extend(["-m", model])
    if agent:
        cmd.extend(["--agent", agent])

    # 双管齐下：命令行参数 + 系统提示词约束
    if enforce_system_prompt:
        full_prompt = f"{JSON_ENFORCE_PROMPT}\n\n用户任务：{prompt}"
    else:
        full_prompt = prompt

    cmd.append(full_prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"opencode 调用失败: {e.stderr[:500]}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"opencode 超时 ({timeout}s)"}
    except FileNotFoundError:
        return {"success": False, "error": "找不到 opencode 命令，请确认已安装"}

    # 优先使用 NDJSON 解析
    parsed = parse_opencode_output(result.stdout)
    if parsed:
        return parsed

    # 回退：尝试从原始文本提取 JSON
    extracted = try_extract_json(result.stdout)
    if extracted:
        return {"success": True, "output": json.dumps(extracted), "usage": {}}

    return {
        "success": False,
        "error": "无法从输出中解析 JSON",
        "raw_output": result.stdout[:1000],
    }


def run_with_agent(agent_name: str, prompt: str, model: str = None, timeout: int = 3600) -> dict:
    """
    使用指定 Agent 调用 opencode 并获取 JSON 输出。

    前提：该 Agent 的系统提示词中已包含 JSON 约束指令。
    """
    return run_opencode_json(prompt, model=model, agent=agent_name, timeout=timeout)


def create_json_agent_config(
    agent_name: str,
    description: str = "JSON 输出专用 Agent",
    model: str = "opencode/deepseek-v4-flash-free",
    output_path: str = None,
) -> str:
    """
    生成一个预配置 JSON 约束的 Agent 配置文件内容。

    Args:
        agent_name: Agent 名称（用于文件名）
        description: Agent 描述
        model: 使用的模型
        output_path: 如果提供，直接写入文件

    Returns:
        str: Agent 配置文件的完整内容
    """
    config = f"""---
description: {description}
mode: all
model: {model}
permission:
  bash: allow
  read: allow
  write: allow
  edit: allow
  glob: allow
  grep: allow
  webfetch: allow
  websearch: allow
  task: allow
---

{JSON_ENFORCE_PROMPT.strip()}
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config)

    return config


if __name__ == "__main__":
    # 示例：创建一个 JSON 专用 Agent 配置
    agent_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".opencode", "agents", "JsonAgent.md",
    )
    config = create_json_agent_config(
        agent_name="JsonAgent",
        description="强制输出结构化 JSON 的专用 Agent",
        output_path=agent_path,
    )
    print(f"Agent 配置已写入: {agent_path}")
    print("---")
    print(config)
