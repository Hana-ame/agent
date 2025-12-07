# 为什么会爆404的,我真的服了.

from simple_ai import SimpleAI

import os
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == "__main__":
    ai = SimpleAI(
        os.getenv("GROQ_API_KEY") or """,
        os.getenv("SILICONFLOW_ENDPOINT") or "",
        "THUDM/GLM-4-9B-0414",
        system_prompt= "",
        bug=True,
    )
    
    ai = SimpleAI(
        os.getenv("GROQ_API_KEY") or """,
        os.getenv("SILICONFLOW_ENDPOINT") or "",
        "THUDM/GLM-Z1-9B-0414",
        system_prompt= "",
        bug=True,
    )
    
    
    ai = SimpleAI(
        os.getenv("GROQ_API_KEY") or "",
        os.getenv("SILICONFLOW_ENDPOINT") or "",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        system_prompt= "",
        max_tokens=16*1024,
        bug=True,
    )
    
    
    resp = ai.chat("4141+1414=?")
    print(resp)


