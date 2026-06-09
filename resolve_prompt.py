"""
递归 Prompt 解析器。

类型定义:
  Prompt = str | int | {"agent": str, "context": str | int | list[Prompt | str | int]}

解析规则:
  1. str → 直接作为 prompt 调用 opencode（无 agent）
  2. int → 查询数据库 prompts 表中 id=int 的记录：
     - 若该行有 response → 直接返回 response
     - 若无 response → 从该行的 context 递归推理（解析 context 中的 id）
     - 若 context 也为空 → 返回 ""
  3. {agent, context}:
     - context 为 str → 直接用该 text 作为 prompt
     - context 为 int → 按规则 2 解析
     - context 为 list → 递归解析每个元素（str/int/Prompt），用 \n\n 拼接
     → 最后调用 opencode --agent <agent> <resolved_context>

  use_cache=True 时，相同结构的 Promise 只执行一次，后续直接复用缓存结果。
"""

import json
from typing import Any

from opencode import run as opencode_run
from prompt_db import PromptDB


def _serialize(prompt: Any) -> str:
    """将 Prompt 结构序列化为唯一标识字符串（用于缓存 key）。"""
    if isinstance(prompt, str):
        return json.dumps({"t": "s", "v": prompt}, ensure_ascii=False)
    if isinstance(prompt, int):
        return json.dumps({"t": "i", "v": prompt})
    context = prompt.get("context", "")
    if isinstance(context, list):
        ctx_key = [_serialize(item) for item in context]
    else:
        ctx_key = context
    return json.dumps({"a": prompt.get("agent", ""), "c": ctx_key}, ensure_ascii=False)


def _resolve_int(pid: int, db: PromptDB, model: str = "", timeout: int = 600) -> str:
    """
    解析 int 引用（SQL 缓存）：
    - 有 response → 直接返回（已缓存）
    - 无 response → 从 context 推理，结果写回 DB 作为缓存
    - 无 context → 返回 ""
    """
    row = db.get(pid)
    if row is None:
        return ""

    if row["response"]:
        return row["response"]

    context = row["context"]
    if not context:
        return ""

    # context 可能是 JSON 数组或纯文本
    try:
        ctx_list = json.loads(context)
    except (json.JSONDecodeError, TypeError):
        # 纯文本，运行 opencode
        result = opencode_run(context, agent=row["agent"], model=model or row["model"], timeout=timeout)
        text = _to_text(result["output"])
        db.done(pid, text, {"source": "opencode_run"})
        return text

    if not isinstance(ctx_list, list):
        return str(ctx_list)

    parts = []
    for item in ctx_list:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, int):
            parts.append(_resolve_int(item, db, model=model, timeout=timeout))
    text = "\n\n".join(parts)

    # 写回 DB，下次引用同一 id 直接命中缓存
    db.done(pid, text, {"source": "context_resolved"})
    return text


def resolve_prompt(
    prompt: Any,
    *,
    db: PromptDB | None = None,
    model: str = "",
    timeout: int = 600,
    use_cache: bool = False,
    _cache: dict | None = None,
) -> str:
    """
    递归解析 Prompt 并调用 opencode，返回 response 文本。

    参数:
      prompt  : str 或 int 或 {"agent": str, "context": ...}
      db      : PromptDB 实例（int 引用必须提供）
      model   : 传递给 opencode 的模型名
      timeout : 超时秒数
      use_cache: 是否启用缓存复用（相同结构只执行一次）
      _cache  : 内部参数，勿手动传入

    返回:
      opencode 的 response 文本字符串
    """
    if _cache is None:
        _cache = {}

    if isinstance(prompt, int):
        if db is None:
            raise ValueError("int 引用需要提供 db 参数")
        key = _serialize(prompt) if use_cache else ""
        if use_cache and key in _cache:
            return _cache[key]
        text = _resolve_int(prompt, db)
        if use_cache:
            _cache[key] = text
        return text

    if isinstance(prompt, str):
        key = _serialize(prompt) if use_cache else ""
        if use_cache and key in _cache:
            return _cache[key]
        result = opencode_run(prompt, model=model, timeout=timeout)
        text = _to_text(result["output"])
        if use_cache:
            _cache[key] = text
        return text

    agent = prompt.get("agent", "")
    context = prompt.get("context", "")

    if isinstance(context, str):
        resolved_text = context
    elif isinstance(context, int):
        if db is None:
            raise ValueError("int 引用需要提供 db 参数")
        resolved_text = _resolve_int(context, db)
    elif isinstance(context, list):
        parts: list[str] = []
        for item in context:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, int):
                if db is None:
                    raise ValueError("int 引用需要提供 db 参数")
                parts.append(_resolve_int(item, db))
            else:
                parts.append(
                    resolve_prompt(item, db=db, model=model, timeout=timeout,
                                   use_cache=use_cache, _cache=_cache)
                )
        resolved_text = "\n\n".join(parts)
    else:
        resolved_text = str(context)

    key = _serialize(prompt) if use_cache else ""
    if use_cache and key in _cache:
        return _cache[key]

    result = opencode_run(resolved_text, agent=agent, model=model, timeout=timeout)
    text = _to_text(result["output"])

    if use_cache:
        _cache[key] = text
    return text


def _to_text(output) -> str:
    """将 opencode 返回值统一转为文本。"""
    if isinstance(output, dict):
        return json.dumps(output, indent=2, ensure_ascii=False)
    return str(output)


# ── 测试 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from prompt_db import PromptDB

    print("=" * 60)
    print("  resolve_prompt 测试（含 int 引用）")
    print("=" * 60)

    db = PromptDB()
    # 清理旧数据
    with db._conn() as conn:
        conn.execute("DELETE FROM prompts")
        conn.commit()

    # ── 原有测试 ──

    print("\n--- 测试 1: 纯文本 ---")
    prompt1 = "1+1等于几？请直接回答数字"
    r1 = resolve_prompt(prompt1, model="opencode/mimo-v2.5-free", timeout=30)
    print(f"  输入: {prompt1}")
    print(f"  输出: {r1}")

    print("\n--- 测试 2: agent + text ---")
    prompt2 = {"agent": "Null", "context": "2+3等于几？"}
    r2 = resolve_prompt(prompt2, model="opencode/mimo-v2.5-free", timeout=30)
    print(f"  输入: {json.dumps(prompt2, ensure_ascii=False)}")
    print(f"  输出: {r2}")

    print("\n--- 测试 3: agent + list ---")
    prompt3 = {
        "agent": "Null",
        "context": ["请说出一种水果", "请说出一种蔬菜"],
    }
    r3 = resolve_prompt(prompt3, model="opencode/mimo-v2.5-free", timeout=60)
    print(f"  输入: {json.dumps(prompt3, ensure_ascii=False)}")
    print(f"  输出: {r3}")

    # ── int 引用测试 ──

    # 预先写入几条有 response 的记录
    pid1 = db.add("你最喜欢什么颜色？", agent="Null", model="mimo-v2.5-free")
    db.done(pid1, "蓝色是我最喜欢的颜色。")
    print(f"\n  [DB] 写入 #{pid1}（有 response）")

    pid2 = db.add("你最喜欢什么食物？", agent="Null", model="mimo-v2.5-free")
    db.done(pid2, "面条是我最喜欢的食物。")
    print(f"  [DB] 写入 #{pid2}（有 response）")

    pid3 = db.add([pid1, pid2], agent="Null", model="mimo-v2.5-free")
    print(f"  [DB] 写入 #{pid3}（无 response，context=[{pid1},{pid2}]）")

    pid4 = db.add("无 context 无 response", agent="Null", model="mimo-v2.5-free")
    print(f"  [DB] 写入 #{pid4}（无 response，无 context）")

    print(f"\n--- 测试 4: int 引用（有 response）---")
    r4 = resolve_prompt(pid1, db=db)
    print(f"  输入: int {pid1}")
    print(f"  输出: {r4}")

    print(f"\n--- 测试 5: int 引用（无 response，从 context 推理）---")
    r5 = resolve_prompt(pid3, db=db)
    print(f"  输入: int {pid3}")
    print(f"  输出: {r5}")

    print(f"\n--- 测试 6: int 引用（无 response 无 context）---")
    r6 = resolve_prompt(pid4, db=db)
    print(f"  输入: int {pid4}")
    print(f"  输出: '{r6}'（空）")

    print(f"\n--- 测试 7: list 中混合 str + int ---")
    prompt7 = {
        "agent": "Null",
        "context": ["你喜欢的水果是", pid1, "你喜欢的食物是", pid2],
    }
    r7 = resolve_prompt(prompt7, db=db, model="opencode/mimo-v2.5-free", timeout=30)
    print(f"  输入: {json.dumps(prompt7, ensure_ascii=False)}")
    print(f"  输出: {r7}")

    print(f"\n--- 测试 8: 嵌套 Prompt 中引用 int ---")
    prompt8 = {
        "agent": "Null",
        "context": [
            "根据以下信息回答：",
            {"agent": "Null", "context": [pid1, pid2]},
        ],
    }
    r8 = resolve_prompt(prompt8, db=db, model="opencode/mimo-v2.5-free", timeout=60)
    print(f"  输入: {json.dumps(prompt8, ensure_ascii=False)}")
    print(f"  输出: {r8}")

    print("\n✅ 全部测试完成")
