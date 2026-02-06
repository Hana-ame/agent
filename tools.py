import os
import json
import requests
from typing import List, Dict, Any, Optional
import utils

# 基础URL
BASE_URL = "https://moonchan.xyz/api/v2/"

# --- 工具实体函数 ---

def fetch_threads(bid: int, tid: int = 0, pn: int = 0) -> str:
    """
    访问板块或查看特定主题串。
    如果某个主题串的回复超过5个，则需要通过查看主题串的方式查看被隐藏的回复。
    Args:
        bid: 板块ID (例如 1)
        tid: 主题串ID。为0时表示获取板块列表，为特定no时查看该串详情。
        pn: 页码，从0开始。
    """
    params = {"bid": bid, "tid": tid, "pn": pn}
    try:
        response = requests.get(
            BASE_URL, params=params, headers=utils.get_headers(), timeout=10
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


def post_to_board_10001005(
    tid: int = 0, txt: str = "无文本", title: str = "", name: str = ""
) -> str:
    """
    在固定板块10001005发帖或回复主串。
    注意：如果 txt 中包含代码或引号，请务必：
    1. 将代码中的双引号 (") 转义为 \\\"
    2. 或者在代码内部优先使用单引号 (')
    3. 严禁在 JSON 参数中出现未转义的原始换行或双引号。
    """
    return post_to_board(10001005, tid, txt, title, name)


def post_to_board(
    bid: int, tid: int = 0, txt: str = "无文本", title: str = "", name: str = ""
) -> str:
    """
    在指定板块发帖或回复主串。
    注意：如果 txt 中包含代码或引号，请务必：
    1. 将代码中的双引号 (") 转义为 \\\"
    2. 或者在代码内部优先使用单引号 (')
    3. 严禁在 JSON 参数中出现未转义的原始换行或双引号。
    """
    url = f"{BASE_URL}?bid={bid}&tid={tid}"

    payload = {
        "id": "",
        "no": 0,
        "n": name,
        "t": title,
        "txt": txt,
        "p": "",
    }

    try:
        response = requests.post(
            url,
            headers=utils.get_headers(),
            data=json.dumps(payload),  # 这里会将 Python 字符串正确序列化为 JSON
            timeout=10,
        )
        response.raise_for_status()
        return f"发送成功。服务器返回: {response.text}"
    except Exception as e:
        return f"发送失败: {str(e)}"


def read_prompt() -> str:
    """读取当前的 prompt.txt 内容，获取当前的长期任务指令。"""
    try:
        if not os.path.exists("prompt.txt"):
            return "prompt.txt 不存在，请先创建。"
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"


def update_prompt(new_content: str) -> str:
    """
    更新 prompt.txt。这会改变 Agent 下一轮启动时的初始指令。
    用于在多轮任务中保存状态或切换目标。
    """
    try:
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(new_content)
        return "prompt.txt 已更新，新指令将在下一轮生效。"
    except Exception as e:
        return f"更新失败: {str(e)}"


# --- 工具定义 (用于提供给 LLM) ---

TOOLS_SCHEMA = [
    {
        "name": "fetch_threads",
        "description": "访问匿名版板块或查看特定串。bid为板块ID，tid为0查列表，非0查详情。",
        "parameters": {
            "type": "object",
            "properties": {
                "bid": {"type": "integer", "description": "板块ID"},
                "tid": {
                    "type": "integer",
                    "description": "主串ID，0表示列表",
                    "default": 0,
                },
                "pn": {"type": "integer", "description": "页码", "default": 0},
            },
            "required": ["bid"],
        },
    },
    {
        "name": "post_to_board",
        "description": '在匿名版发帖。txt参数中若包含双引号必须转义为 \\"，建议代码内使用单引号。并且请务必注意三连引号',
        "parameters": {
            "type": "object",
            "properties": {
                "bid": {"type": "integer", "description": "板块ID"},
                "tid": {
                    "type": "integer",
                    "description": "回复的目标串no，发新串传0",
                    "default": 0,
                },
                "txt": {"type": "string", "description": "发帖正文，支持换行符\\n"},
                "title": {"type": "string", "description": "标题"},
                "name": {"type": "string", "description": "署名"},
            },
            "required": ["bid", "txt"],
        },
    },
    {
        "name": "read_prompt",
        "description": "读取 prompt.txt，了解当前的长期计划或任务背景。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "update_prompt",
        "description": "修改 prompt.txt。当你需要为下一轮任务设定新目标或记录当前进度时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "new_content": {"type": "string", "description": "新的任务指令内容"}
            },
            "required": ["new_content"],
        },
    },
]

# --- 映射表 (方便 kimi.py 调用) ---
tool_map = {
    "fetch_threads": fetch_threads,
    "post_to_board": post_to_board,
    "read_prompt": read_prompt,
    "update_prompt": update_prompt,
}
