"""
查询数据库中最新的 pending 记录，并执行它。
调用 resolve_prompt 完成 LLM 请求并自动更新状态。

用法:  python3 run_latest_pending.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_db import PromptDB
from resolve_prompt import resolve_prompt

def main():
    db = PromptDB()
    pending = db.list_by_status("pending")
    if not pending:
        print("当前没有 pending 记录。")
        return

    latest = pending[-1]
    pid = latest["id"]
    context = latest["context"]
    agent = latest["agent"]
    model = latest["model"]

    print(f"找到最新 pending 记录: #{pid}")
    print(f"  context: {context[:100]}...")
    print(f"  agent: {agent}")
    print(f"  model: {model}")
    print("正在调用 LLM ...")

    try:
        result = resolve_prompt(pid, db=db, timeout=300)
        print(f"\n✅ 执行成功，结果已保存。")
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
