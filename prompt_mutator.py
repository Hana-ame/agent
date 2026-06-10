"""Context 变异守护进程（LLM 驱动）

使用 LLM 生成新的 context，并验证 JSON 合法性。

用法:  python3 prompt_mutator.py
"""

import sys
import json
import random
import time
import signal

sys.path.insert(0, ".")

from prompt_db import PromptDB
from opencode import run as opencode_run

db = PromptDB()
running = True


def handle_signal(sig, frame):
    global running
    print("\n收到信号，正在退出...")
    running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def parse_context(ctx_str):
    """解析 context 字符串为列表。"""
    try:
        ctx = json.loads(ctx_str)
        if isinstance(ctx, list):
            return ctx
        return [ctx]
    except (json.JSONDecodeError, TypeError):
        if ctx_str:
            return [ctx_str]
        return []


def validate_context(ctx):
    """验证 context 是否为合法的 JSON 列表。"""
    if isinstance(ctx, list):
        return True
    if isinstance(ctx, (str, int)):
        return True
    return False


def call_llm(prompt_text, timeout=120):
    """调用 opencode 生成内容。"""
    try:
        result = opencode_run(prompt_text, timeout=timeout)
        output = result.get("output", "")
        if isinstance(output, dict):
            # JSON 格式的输出
            return output.get("text", str(output))
        return str(output).strip()
    except Exception as e:
        print(f"    LLM 调用失败: {e}")
        return ""


def extract_json_from_response(text):
    """从 LLM 响应中提取 JSON。"""
    import re

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取 [ ... ] 或 { ... }
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                # 尝试将 Python 列表语法转换为 JSON
                try:
                    # 将单引号替换为双引号
                    json_str = match.group().replace("'", '"')
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

    return None


def generate_mutated_context(base_context, all_ids):
    """使用 LLM 生成新的 context。"""
    prompt = f"""请根据下面的 context 生成一个新的 context。

原始 context: {json.dumps(base_context, ensure_ascii=False)}

可用的 ID: {all_ids}

规则：
1. 输出一个 JSON 数组
2. 数组元素可以是整数（ID）或字符串（文本）
3. 只输出 JSON 数组，例如 [1, 2, 3] 或 ["你好", 1]
4. 不要输出任何其他内容

新的 context:"""

    response = call_llm(prompt)
    if not response:
        return None

    result = extract_json_from_response(response)
    if result is None:
        print(f"    JSON 解析失败: {response[:100]}...")
        return None

    if not validate_context(result):
        print(f"    context 不合法: {result}")
        return None

    return result


def process_mutations():
    """使用 LLM 生成新的 context。"""
    rows = db.list_all()
    all_ids = [r["id"] for r in rows]

    if len(rows) < 2:
        return 0

    mutated = 0
    candidates = [r for r in rows if r["context"] and r["context"] != "[]"]
    if not candidates:
        return 0

    # 每次处理 2 条
    sample_size = min(2, len(candidates))
    sampled = random.sample(candidates, sample_size)

    for row in sampled:
        items = parse_context(row["context"])
        if not items:
            continue

        print(f"\n  处理 #{row['id']}: {items}")
        new_items = generate_mutated_context(items, all_ids)

        if new_items is None:
            continue

        if new_items == items:
            print(f"    生成的 context 相同，跳过")
            continue

        pid = db.add(
            new_items,
            agent=row["agent"],
            model=row["model"],
        )
        print(f"    → #{pid}: {new_items}")
        mutated += 1

    return mutated


def main():
    print("=" * 50)
    print("  Prompt Mutator (LLM) 守护进程")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    cycle = 0
    while running:
        cycle += 1
        mutated = process_mutations()

        if mutated > 0:
            print(f"\n  [周期 {cycle}] 生成了 {mutated} 条新记录")

        time.sleep(30)

    print("Mutator 已退出")


if __name__ == "__main__":
    main()
