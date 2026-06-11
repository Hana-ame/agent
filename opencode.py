import subprocess
import json


def run(prompt, agent="", model="", timeout=600):
    cmd = ["opencode"]
    if agent:
        cmd.extend(["--agent", agent])
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["run", prompt])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = result.stdout.strip()
    error = result.stderr.strip() if result.returncode != 0 else ""
    try:
        return {"output": json.loads(output), "json": True, "success": result.returncode == 0, "error": error}
    except json.JSONDecodeError:
        return {"output": output, "json":False, "success": result.returncode == 0, "error": error}


SILICONFLOW_MODELS = [
    "siliconflow-cn/Qwen/Qwen3-8B",
    "siliconflow-cn/THUDM/GLM-Z1-9B-0414",
    "siliconflow-cn/THUDM/GLM-4-9B-0414",
]


def models(filter_free=True):
    cmd = ["opencode", "models"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    all_models = [m.strip() for m in result.stdout.strip().split("\n") if m.strip()]
    if filter_free:
        return [m for m in all_models
                if m.startswith("opencode/") or m.startswith("nvidia/") or m in SILICONFLOW_MODELS]
    return all_models


if __name__ == '__main__':
    result = run("自我介绍(about model)：", agent="Null", model="opencode/mimo-v2.5-free", timeout=30)
