"""
查询数据库中的 pending 记录，随机取一条执行。
配合 resolve_prompt DAG 调度器使用（只接受 dict/JSON str）。

用法:  python3 run_latest_pending.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt


def main():
    db = PromptDB()
    pending = db.list_by_status("pending")
    if not pending:
        print("当前没有 pending 记录。")
        return

    target = random.choice(pending)
    pid = target["id"]
    agent = target["agent"] or "Null"
    model = target["model"] or "siliconflow-cn/Qwen/Qwen3-8B"

    print(f"随机选中 pending 记录: #{pid}")
    print(f"  context: {target['context'][:100]}...")
    print(f"  agent: {agent}")
    print(f"  model: {model}")
    print("正在调用 LLM ...")

    # 构造 DAG 调度器接受的 dict 结构
    prompt = {"agent": agent, "context": target["context"]}

    try:
        result = resolve_prompt(prompt, db=db, model=model, timeout=300)
        print("\n✅ 执行成功，结果已保存到数据库。")
        print("=" * 60)
        print(result[:500])
        if len(result) > 500:
            print("...")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        db.failed(pid, str(e))
        print("已将状态标记为 failed。")


if __name__ == "__main__":
    main()
