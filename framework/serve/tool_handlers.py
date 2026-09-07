"""Tool handlers for ToolCallEdge demo.

提供可被 ToolCallEdge 调用的工具函数。
"""

import datetime
import json
import os
import subprocess


def get_weather(city: str) -> str:
    """模拟天气查询。返回结构化天气信息。"""
    return json.dumps({
        "city": city,
        "temperature": "25°C",
        "condition": "晴",
        "humidity": "60%",
        "note": "模拟数据",
    }, ensure_ascii=False)


def get_current_time() -> str:
    """返回当前时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_files(directory: str) -> str:
    """列出目录中的文件。"""
    try:
        result = subprocess.run(
            ["ls", "-la", directory],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def calculate(expression: str) -> str:
    """安全计算数学表达式。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return f"Error: 不安全的表达式: {expression}"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
