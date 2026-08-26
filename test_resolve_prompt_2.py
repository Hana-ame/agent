"""
测试 resolve_prompt 对特定输入（包含乱码 context）的处理。

输入字符串：
    {"agent": "Qwen3-8B", "context": "1+1ç­‰äºŽå‡ ï¼Ÿè¯·ç›´æŽ¥å›žç­”æ•°å­—"}

说明：context 中的乱码实际上是 UTF-8 编码的 "1+1等于几？请直接回答数字" 
被错误地以 Latin-1/Windows-1252 解码后的结果，但 JSON 解析仍能正常加载为 Unicode 字符串。
该测试直接使用此原始字符串，观察 resolve_prompt 的行为。
"""

import json
from resolve_prompt import resolve_prompt
from prompt_db import PromptDB

# 给定的测试输入（保留原样）
PROMPT_STR = '{"agent": "Qwen3-8B", "context": "1+1ç­‰äºŽå‡ ï¼Ÿè¯·ç›´æŽ¥å›žç­”æ•°å­—"}'

def test_mojibake_context():
    """用包含乱码的 JSON 调用 resolve_prompt。"""
    db = PromptDB()
    # 清空数据库（如需要）
    with db._conn() as conn:
        conn.execute("DELETE FROM prompts")
        conn.commit()

    model = "siliconflow-cn/Qwen/Qwen3-8B"   # 根据实际情况调整
    timeout = 300

    print("输入 JSON 字符串:")
    print(PROMPT_STR)
    print("\n解析后的 context (乱码):")
    data = json.loads(PROMPT_STR)
    print(repr(data["context"]))   # 显示原始 Unicode 码点

    print("\n--- 调用 resolve_prompt ---")
    try:
        result = resolve_prompt(PROMPT_STR, db=db, model=model, timeout=timeout)
        print("输出:")
        print(result)
    except Exception as e:
        print(f"调用失败: {e}")

if __name__ == "__main__":
    test_mojibake_context()