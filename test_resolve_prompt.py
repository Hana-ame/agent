"""PromptResolver 测试 — 真实 API，详细输出，DAG 调度器版"""
import json
from pathlib import Path
import tempfile

import resolve_with_db
from resolve_with_db import PromptResolver
from prompt_db import PromptDB

_orig_opencode_run = resolve_with_db.opencode_run

MODEL = "siliconflow-cn/Qwen/Qwen3-8B"
TIMEOUT = 180

_api_calls = []


def _logging_run(*args, **kw):
    result = _orig_opencode_run(*args, **kw)
    _api_calls.append((args, kw, result))
    return result


def _dump_db(db, label):
    print(f"──── DB {label} ────")
    for row in db.list_all():
        print(json.dumps(row, ensure_ascii=False))
    print()


def _call(label, resolver, prompt):
    global _api_calls
    _api_calls = []

    print("=" * 40)
    print(f"调用function：PromptResolver.resolve")
    print(f"signature: def resolve(self, prompt: Any) -> str")
    print()

    # ① 输入
    print("【输入】")
    if isinstance(prompt, dict):
        print(f"  prompt = {json.dumps(prompt, ensure_ascii=False)}  type=dict")
        for k, v in prompt.items():
            if isinstance(v, list):
                print(f"  prompt['{k}'] = {json.dumps(v, ensure_ascii=False)}  type=list")
                for i, item in enumerate(v):
                    print(f"    [{i}] = {json.dumps(item, ensure_ascii=False) if not isinstance(item, int) else str(item)}  type={type(item).__name__}")
                    if isinstance(item, int):
                        row = resolver.db.get(item)
                        if row:
                            print(f"           db.get({item}) = {json.dumps(row, ensure_ascii=False)}")
            else:
                print(f"  prompt['{k}'] = {json.dumps(v, ensure_ascii=False) if not isinstance(v, int) else str(v)}  type={type(v).__name__}")
                if isinstance(v, int):
                    row = resolver.db.get(v)
                    if row:
                        print(f"           db.get({v}) = {json.dumps(row, ensure_ascii=False)}")
    else:
        print(f"  prompt = {repr(prompt)}")

    # 执行
    original_run = resolve_with_db.opencode_run
    resolve_with_db.opencode_run = _logging_run
    try:
        result = resolver.resolve(prompt)
    finally:
        resolve_with_db.opencode_run = original_run

    # ② API 调用
    print()
    print("【API 调用】")
    if not _api_calls:
        print("  (无 API 调用)")
    else:
        for idx, (args, kw, ret) in enumerate(_api_calls):
            if idx > 0:
                print()
            print(f"  #{idx + 1} opencode_run:")
            print(f"    text    = {repr(args[0])}")
            if kw.get("agent"):
                print(f"    agent   = {repr(kw['agent'])}")
            if kw.get("model"):
                print(f"    model   = {repr(kw['model'])}")
            print(f"    ── 返回 ──")
            print(f"    output  = {repr(ret.get('output', ''))}")
            print(f"    success = {ret.get('success')}")
            print(f"    error   = {repr(ret.get('error', ''))}")

    # ③ 最终返回值
    print()
    print("【最终返回值】")
    print(f"  value = {repr(result)}")
    print(f"  type  = {type(result).__name__}")
    print()


def main():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    db = PromptDB(db_path)
    resolver = PromptResolver(db, model=MODEL, timeout=TIMEOUT)

    # 准备 DB 数据
    pid1 = db.add("你最喜欢什么颜色？", agent="Null")
    db.done(pid1, "蓝色是我最喜欢的颜色。")
    pid2 = db.add("你最喜欢什么食物？", agent="Null")
    db.done(pid2, "面条是我最喜欢的食物。")
    pid3 = db.add([pid1, pid2], agent="Null")

    _dump_db(db, "初始")

    # 1. dict + int（引用已有 response）
    _call("dict + int（引用 resp）",
          resolver,
          {"agent": "Null", "context": pid1})

    # 2. list 混合 str + int
    _call("dict + list 混合 str+int",
          resolver,
          {"agent": "Null", "context": ["水果", pid1, "蔬菜", pid2]})

    # 3. 嵌套 dict（pending 自动补跑）
    _call("嵌套 dict（pending）",
          resolver,
          {"agent": "Null", "context": [pid3, "请总结"]})

    # 4. 纯文本 context
    _call("纯文本 context",
          resolver,
          {"agent": "Null", "context": "1+1等于几？请直接回答数字"})

    # 5. 嵌套 dict + int
    _call("嵌套 dict + int",
          resolver,
          {"agent": "Null", "context": ["开头", {"agent": "Null", "context": "嵌套问题：1+1=?"}]})

    # 6. 嵌套 dict 引用 int
    _call("嵌套 dict 引用 int",
          resolver,
          {"agent": "Null", "context": [{"agent": "Null", "context": pid1}]})

    _dump_db(db, "最终")


if __name__ == "__main__":
    main()
