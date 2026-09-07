"""Tool handlers for ToolCallEdge demo.

Provides tool functions callable by ToolCallEdge.
"""

import datetime
import json
import os
import subprocess


def get_weather(city: str) -> str:
    """Simulate weather query. Returns structured weather information."""
    return json.dumps({
        "city": city,
        "temperature": "25°C",
        "condition": "Sunny",
        "humidity": "60%",
        "note": "Simulated data",
    }, ensure_ascii=False)


def get_current_time() -> str:
    """Return current timestamp."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_files(directory: str) -> str:
    """List files in directory."""
    try:
        result = subprocess.run(
            ["ls", "-la", directory],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return f"Error: Unsafe expression: {expression}"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
