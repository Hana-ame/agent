"""
Loop 执行引擎 — 配置驱动 + asyncio 并发执行

配置文件 (loop.json) 包含一个循环配置列表，每个配置项：

    name             循环名称（仅用于日志）
    type             "abstract" 或 "jielong"
    models           模型列表（可选，默认使用内置列表）
    count            每次处理条数，-1 表示全部
    interval_seconds 循环间隔（秒），0 表示只跑一次
    enabled          是否启用（可选，默认 true）

用法：
    python loop.py               # 使用 loop.json
    python loop.py my_conf.json  # 使用自定义配置
"""

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from database import DataBase


# ── 配置 ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("SIMPLEAI_DB", str(BASE_DIR / "simpleai.db"))
DEFAULT_CONFIG = str(BASE_DIR / "loop.json")

ABSTRACT_MODELS = [
    "siliconflow-cn/Qwen/Qwen3.5-4B",
    "siliconflow-cn/Qwen/Qwen3-8B",
    "siliconflow-cn/THUDM/GLM-4-9B-0414",
    "siliconflow-cn/THUDM/GLM-Z1-9B-0414",
]

JIELONG_MODELS = ABSTRACT_MODELS + [
    "opencode/deepseek-v4-flash-free",
    "google/gemma-4-31b-it",
]

MAX_CONTEXT_TOKENS = 8000


# ── Token 估算 ──────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - chinese_chars
    return int(chinese_chars * 2 + other * 1.3)


# ── 配置加载 ────────────────────────────────────────────────────────


def load_config(path: str|None = None) -> list:
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        raise ValueError(f"配置必须是 JSON 数组，得到 {type(configs).__name__}")

    for i, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            raise ValueError(f"配置项 #{i} 必须是对象")
        typ = cfg.get("type")
        if typ not in ("abstract", "jielong"):
            raise ValueError(f"配置项 #{i} type='{typ}' 无效，必需是 abstract 或 jielong")
        cfg.setdefault("count", 1)
        cfg.setdefault("interval_seconds", 60)
        cfg.setdefault("enabled", True)
        cfg.setdefault("models", None)
        cfg.setdefault("name", f"{typ}-{i}")

    return configs


# ── OpenCode 调用（async） ─────────────────────────────────────────


async def call_opencode(prompt: str, model: str|None = None, agent: str|None = None) -> dict:
    cmd = ["opencode", "run", "--format", "json"]
    if agent:
        cmd.extend(["--agent", agent])
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3600)
    except asyncio.TimeoutError:
        return {"success": False, "error": "opencode 超时 (3600s)"}
    except FileNotFoundError:
        return {"success": False, "error": "找不到 opencode 命令，请确认已安装"}

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace")[:500]
        return {"success": False, "error": f"opencode 调用失败: {err}"}

    output_text = ""
    usage = {"input": 0, "output": 0, "total": 0}
    stdout_str = stdout.decode("utf-8", errors="replace")
    for line in stdout_str.strip().split("\n"):
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                output_text += event["part"].get("text", "")
            elif event.get("type") == "step_finish":
                usage = {
                    "input": event["part"].get("tokens", {}).get("input", 0),
                    "output": event["part"].get("tokens", {}).get("output", 0),
                    "total": event["part"].get("tokens", {}).get("total", 0),
                }
        except json.JSONDecodeError:
            continue

    if not output_text and stdout_str.strip():
        output_text = stdout_str.strip()

    return {
        "success": True,
        "output": output_text,
        "usage": usage,
    }


# ── Loop 1: Abstract 生成 ──────────────────────────────────────────


async def loop1_abstract(db: DataBase, count: int = 1, models: list|None = None) -> list:
    results = []

    rows = db.prompts.Read(
        condition="(abstract IS NULL OR abstract = '')",
        order_by="id ASC",
    )

    if not rows:
        print("[Loop1] 没有需要生成 abstract 的 prompt")
        return results

    COL_PROMPT = 2
    COL_RESPONSE = 5
    COL_MODEL = 4

    _models = models if models else ABSTRACT_MODELS
    selected = rows if count == -1 else rows[:count]
    print(f"[Loop1] 找到 {len(rows)} 条，本次处理 {len(selected)} 条")

    for row in selected:
        prompt_id = row[0]
        prompt_text = row[COL_PROMPT] or ""
        response_text = row[COL_RESPONSE] or ""
        model_used = row[COL_MODEL] or random.choice(_models)

        print(f"  [Loop1] prompt_id={prompt_id}, model={model_used}")

        # 构建生成 abstract 的 prompt，包含用户问题和助手回答
        dialogue_content = f"用户: {prompt_text}\n助手: {response_text}"
        abstract_prompt = (
            f"请为以下对话内容生成一个简洁的摘要（abstract），并判断此对话是否应该结束。\n\n"
            f"对话内容：\n{dialogue_content}\n\n"
            f"如果对话中有明确的结束信号（如再见、感谢、结束等），则 should_end=1；否则 should_end=0。\n"
            f"请严格按以下 JSON 格式回复，不要输出额外内容：\n"
            f'{{"abstract": "你的摘要", "should_end": 0 或 1}}'
        )

        start_time = datetime.now().isoformat()
        oc_result = await call_opencode(abstract_prompt, model=model_used)
        end_time = datetime.now().isoformat()

        success = 1 if oc_result.get("success") else 0
        abstract_text = ""
        should_end = 0

        if oc_result.get("success"):
            output = oc_result["output"]
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    abstract_text = parsed.get("abstract", output[:200])
                    should_end = parsed.get("should_end", 0)
                else:
                    abstract_text = output[:200]
            except json.JSONDecodeError:
                if "```json" in output:
                    json_str = output.split("```json")[1].split("```")[0].strip()
                    try:
                        parsed = json.loads(json_str)
                        abstract_text = parsed.get("abstract", output[:200])
                        should_end = parsed.get("should_end", 0)
                    except json.JSONDecodeError:
                        abstract_text = output[:200]
                else:
                    abstract_text = output[:200]

        should_end = 1 if should_end else 0

        db.prompts.Update(
            {"abstract": abstract_text, "should_end": should_end},
            condition=f"id={prompt_id}",
        )

        usage = oc_result.get("usage", {})
        req_id = db.requests.Insert({
            "prompt_id": prompt_id,
            "agent_name": "AbstractAgent",
            "start_time": start_time,
            "end_time": end_time,
            "input_tokens": usage.get("input", 0),
            "output_tokens": usage.get("output", 0),
            "success": success,
            "include_history": 0,
        })

        results.append({
            "prompt_id": prompt_id,
            "abstract": abstract_text,
            "should_end": should_end,
            "request_id": req_id,
            "success": bool(success),
        })
        print(f"    → abstract={'✓' if abstract_text else '✗'}, should_end={should_end}, req_id={req_id}")

    return results


# ── Loop 2: 对话接龙 ──────────────────────────────────────────────


def build_history(db: DataBase, prompt_id: int) -> list:
    history = []
    current_id = prompt_id

    while current_id is not None:
        rows = db.prompts.Read(condition=f"id={current_id}")
        if not rows:
            break
        row = rows[0]
        history.append({
            "id": row[0],
            "previous_id": row[1],
            "prompt": row[2] or "",
            "agent": row[3] or "",
            "model": row[4] or "",
            "response": row[5] or "",
            "abstract": row[6] or "",
            "should_end": row[7] or 0,
        })
        current_id = row[1]

    history.reverse()
    return history


def format_conversation_context(history: list) -> str:
    """将对话历史格式化为上下文文本。"""
    parts = []
    for i, entry in enumerate(history):
        if entry.get("_compressed"):
            parts.append(f"=== 历史上下文总结 ===\n{entry['summary']}\n")
            continue

        parts.append(f"--- 第 {i + 1} 轮 ---")
        parts.append(f"用户: {entry['prompt']}")
        if entry['response']:
            parts.append(f"助手: {entry['response']}")
        if entry['abstract']:
            parts.append(f"摘要: {entry['abstract']}")
        parts.append("")
    return "\n".join(parts)


async def compact_history(db: DataBase, history: list) -> list:
    print(f"  [Compact] 压缩历史: {len(history)} 轮")

    if len(history) <= 2:
        return history

    preserve_count = 2
    to_compress = history[:-preserve_count]
    preserved = history[-preserve_count:]

    # 1. 将所有需要压缩的轮次拼接成一个文本块
    history_text = ""
    for i, entry in enumerate(to_compress):
        history_text += f"轮次 {i+1}:\n用户: {entry['prompt']}\n助手: {entry['response']}\n摘要: {entry['abstract']}\n\n"

    compact_prompt = (
        f"以下是对话的早期历史记录：\n\n{history_text}\n\n"
        f"请将上述所有历史内容压缩成一段精简的摘要，保留关键信息和结论，"
        f"以便在后续对话中作为背景上下文。请直接输出摘要内容，不要包含 '摘要：' 等前缀。"
    )

    # 2. 一次性调用 API 进行全局压缩
    oc_result = await call_opencode(compact_prompt, model="opencode/deepseek-v4-flash-free")
    
    if oc_result.get("success"):
        summary = oc_result["output"].strip()
        # 创建一个标准的压缩总结节点
        compressed_entry = {
            "summary": summary,
            "_compressed": True,
        }
        return [compressed_entry] + preserved
    else:
        # 失败则回退：保留最后 3 轮，其余舍弃
        print("  [Compact] 压缩失败，执行简单截断")
        return history[-3:] if len(history) > 3 else history


async def loop2_jielong(db: DataBase, count: int = 1, models: list|None = None) -> list:
    results = []

    rows = db.prompts.Read(
        condition="abstract IS NOT NULL AND abstract != '' AND (should_end IS NULL OR should_end = 0)",
        order_by="id ASC",
    )

    if not rows:
        print("[Loop2] 没有可接龙的 prompt")
        return results

    COL_PROMPT = 2
    COL_MODEL = 4

    _models = models if models else JIELONG_MODELS
    selected = rows if count == -1 else rows[:count]
    print(f"[Loop2] 找到 {len(rows)} 条可接龙的 prompt，本次处理 {len(selected)} 条")

    for row in selected:
        prompt_id = row[0]
        prompt_text = row[COL_PROMPT] or ""
        model_used = row[COL_MODEL] or random.choice(_models)

        print(f"  [Loop2] prompt_id={prompt_id}, model={model_used}")

        history = build_history(db, prompt_id)
        context_text = format_conversation_context(history)

        estimated = estimate_tokens(context_text)
        compacted = False
        if estimated > MAX_CONTEXT_TOKENS:
            print(f"    Token 超限: {estimated} > {MAX_CONTEXT_TOKENS}，执行压缩")
            history = await compact_history(db, history)
            context_text = format_conversation_context(history)
            compacted = True
            estimated = estimate_tokens(context_text)
            print(f"    压缩后: {estimated} tokens")

        jielong_prompt = (
            f"以下是迄今为止的对话历史：\n\n"
            f"{context_text}\n"
            f"请基于以上对话历史，以用户身份继续提出下一个问题或话题（一句话即可）。\n"
            f"然后用助手身份给出回答。\n\n"
            f"请严格按以下 JSON 格式回复，不要输出额外内容：\n"
            f'{{"next_prompt": "用户的下一个问题", "response": "助手的回答"}}'
        )

        start_time = datetime.now().isoformat()
        oc_result = await call_opencode(jielong_prompt, model=model_used)
        end_time = datetime.now().isoformat()

        success = 1 if oc_result.get("success") else 0
        next_prompt = ""
        response_text = ""

        if oc_result.get("success"):
            output = oc_result["output"]
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    next_prompt = parsed.get("next_prompt", "")
                    response_text = parsed.get("response", "")
            except json.JSONDecodeError:
                if "```json" in output:
                    json_str = output.split("```json")[1].split("```")[0].strip()
                    try:
                        parsed = json.loads(json_str)
                        next_prompt = parsed.get("next_prompt", "")
                        response_text = parsed.get("response", "")
                    except json.JSONDecodeError:
                        pass

        if not next_prompt:
            next_prompt = f"（续）基于之前对话的新问题"
        if not response_text:
            response_text = oc_result.get("output", "")

        new_prompt_id = db.prompts.Insert({
            "previous_id": prompt_id,
            "prompt": next_prompt,
            "agent": "JielongAgent",
            "model": model_used,
            "response": response_text,
            "abstract": "",
            "should_end": 0,
        })

        usage = oc_result.get("usage", {})
        req_id = db.requests.Insert({
            "prompt_id": new_prompt_id,
            "agent_name": "JielongAgent",
            "start_time": start_time,
            "end_time": end_time,
            "input_tokens": usage.get("input", 0),
            "output_tokens": usage.get("output", 0),
            "success": success,
            "include_history": 1,
        })

        results.append({
            "prompt_id": prompt_id,
            "new_prompt_id": new_prompt_id,
            "request_id": req_id,
            "next_prompt": next_prompt,
            "response": response_text,
            "history_count": len(history),
            "context_compacted": compacted,
            "success": bool(success),
        })
        print(f"    → new_prompt_id={new_prompt_id}, req_id={req_id}, compacted={compacted}")

    return results


# ── 循环运行器 ─────────────────────────────────────────────────────


async def run_loop_instance(db: DataBase, config: dict):
    name = config["name"]
    loop_type = config["type"]
    fn = loop1_abstract if loop_type == "abstract" else loop2_jielong

    count = config["count"]
    interval = config["interval_seconds"]
    models = config.get("models")

    print(f"[{name}] 启动 type={loop_type} count={count} interval={interval}s")

    while True:
        try:
            results = await fn(db, count=count, models=models)
            print(f"[{name}] 完成，处理 {len(results)} 条")
        except Exception as e:
            print(f"[{name}] 错误: {e}")

        # 永远运行：如果 interval 为 0，则默认等待 1 秒防止 CPU 满载
        sleep_time = interval if interval > 0 else 1
        await asyncio.sleep(sleep_time)


# ── 主入口 ──────────────────────────────────────────────────────────


async def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    print(f"加载配置: {config_path}")

    try:
        configs = load_config(config_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    print(f"共 {len(configs)} 个循环配置")
    db = DataBase(DB_PATH)

    tasks = []
    for cfg in configs:
        if cfg["enabled"]:
            tasks.append(run_loop_instance(db, cfg))

    if not tasks:
        print("没有启用的循环配置")
        db.close()
        return

    print(f"启动 {len(tasks)} 个并发循环...")
    try:
        await asyncio.gather(*tasks)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
