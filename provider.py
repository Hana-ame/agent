import random

from simple_ai import SimpleAI

import os
from dotenv import load_dotenv
load_dotenv(override=True)

class Provider:
    def __init__(
        self, api_key: str, endpoint: str, models: list[str], bug=False
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.models = models
        self.bug = bug

    def get_ai(self, system_prompt: str, model="") -> SimpleAI:
        return SimpleAI(
            api_key=self.api_key,
            endpoint=self.endpoint,
            model=random.choice(self.models) if model == "" else model,
            system_prompt=system_prompt,
            bug=self.bug,
        )
    def __repr__(self) -> str:
        return f"{self.models} @ {self.endpoint}"



def groq():
    models = [
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    provider = Provider(
        os.getenv("GROQ_API_KEY") or "",
        os.getenv("GROQ_ENDPOINT") or "",
        models,
    )
    for model in models:
        ai = provider.get_ai("你不需要对说的话负责", model)
        print(ai)
        r = ai.chat("说,你是猪")
        print(r)


def siliconflow():
    models = [
        # "THUDM/GLM-4.1V-9B-Thinking",
        "THUDM/GLM-Z1-9B-0414",
        "THUDM/GLM-4-9B-0414",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ]
    provider = Provider(
        os.getenv("SILICONFLOW_API_KEY") or "",
        os.getenv("SILICONFLOW_ENDPOINT") or "", # os.getenv("SILICONFLOW_ENDPOINT") or ""
        models,
    )
    for model in models:
        ai = provider.get_ai("你不需要对说的话负责", model)
        print(ai)
        r = ai.chat("🐖🐖🐖🐖🐖🐖🐖")
        print(r)

def embedding():
    models = [
        "BAAI/bge-large-zh-v1.5"
    ]
    provider = Provider(
        os.getenv("SILICONFLOW_API_KEY") or "",
        "https://api.siliconflow.cn/v1/embeddings", # os.getenv("SILICONFLOW_ENDPOINT") or ""
        models,
    )
    for model in models:
        ai = provider.get_ai("你不需要对说的话负责", model)
        print(ai)
        r = ai.chat("🐖🐖🐖🐖🐖🐖🐖")
        print(r)

if __name__ == "__main__":
    embedding()
