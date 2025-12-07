# 为什么会爆404的,我真的服了.
# 因为超过cloudflare的100秒了。
# ~2000 tokens + ~20 tokens/s 呗。

from simple_ai import SimpleAI

import os
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == "__main__":
    models = [
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
    ]
    for model in models:
        ai = SimpleAI(
            os.getenv("GROQ_API_KEY") or "",
            os.getenv("GROQ_ENDPOINT") or "",
            model=model,
            system_prompt= "",
            bug=True,
        )    
 
        resp = ai.chat("4141+1414=?")
        print(resp)


