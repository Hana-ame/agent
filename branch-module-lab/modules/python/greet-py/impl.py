"""Python 模块：模块级 NAME / describe() / run(payload)。"""

NAME = "greet-py"


def describe() -> str:
    return "Python 模块：生成问候语（v3）"


def run(payload: str) -> str:
    return f"hello, {payload} (python v3)"
