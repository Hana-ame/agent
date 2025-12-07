# 为什么会爆404的,我真的服了.

from simple_ai import SimpleAI

import os
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == "__main__":
    ai = SimpleAI(
        api_key=os.getenv("GROQ_API_KEY") or "",
        endpoint=os.getenv("SILICONFLOW_ENDPOINT") or "",  # 以 DeepSeek 为例
        bug=True,
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        # --- 在这里设置全局参数 ---
        temperature=0.1,  # 设置得很低，让它回答严谨
        max_tokens=2048 * 8,  # 限制回复长度
        frequency_penalty=0.5,  # 也可以传一些不常用的参数
    )

    resp = ai.chat("详细解释量子力学")
    print(resp)

    ai = SimpleAI(
        api_key=os.getenv("SILICONFLOW_API_KEY") or "",
        endpoint=os.getenv("SILICONFLOW_ENDPOINT") or "",  # 以 DeepSeek 为例
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        # --- 在这里设置全局参数 ---
        temperature=0.1,  # 设置得很低，让它回答严谨
        max_tokens=2048 * 8,  # 限制回复长度
        frequency_penalty=0.5,  # 也可以传一些不常用的参数
    )

    resp = ai.chat("详细解释量子力学")
    print(resp)
