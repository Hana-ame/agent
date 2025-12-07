# 为什么会爆404的,我真的服了.

from simple_ai import SimpleAI

if __name__ == "__main__":
    models = [
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
    ]
    for model in models:
        ai = SimpleAI(
            "nanaka",
            "https://chat.moonchan.xyz/groq",
            default_model=model,
            system_prompt= "",
            bug=True,
        )    
 
        resp = ai.chat("4141+1414=?")
        print(resp)


