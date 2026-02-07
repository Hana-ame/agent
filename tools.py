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
    
    Examples:
        - 获取板块1的第一页列表: fetch_threads(bid=1, tid=0, pn=0)
        - 查看板块1中ID为12345的主题串详情: fetch_threads(bid=1, tid=12345)
    
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


def post_to_board(
    bid: int = 666, tid: int = 0, txt: str = "无文本", title: str = "", name: str = ""
) -> str:
    """
    在指定板块发帖或回复。
    
    ⚠️ 严正警告 (JSON 格式化规范):
    1. 严禁使用三连双引号 (\"\"\")。如果文本/代码中包含三连双引号，必须全部替换为三连单引号 (''' )。
    2. 普通双引号 (\") 必须转义为 \\\"。
    3. 换行符必须表示为 \\n。
    
    One-shot Example (代码转换):
    原始需求: 我想发送 print(\"\"\"Hello \"World\"\"\"\")
    转换后参数: txt="print('''Hello \\\"World\\\"''')"
    
    Args:
        bid: 板块ID
        tid: 回复的主串ID，发新串传0，必须是数字，从response的no中获得，而且不可以使用每一项list中出现的no，而只能使用fetch_threads出现的最外层的列表中出现的no参数。
        txt: 正文内容 (注意三引号转换)
        title: 标题
        name: 署名
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
            data=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()
        return f"发送成功。服务器返回: {response.text}"
    except Exception as e:
        return f"发送失败: {str(e)}"


def read_next_prompt(agent_id: str) -> str:
    """
    读取指定 Agent 的下一阶段任务指令。
    
    Example:
        read_next_prompt(agent_id="explorer_01")
        
    Args:
        agent_id: 当前 Agent 的唯一身份标识。
    """
    filename = f"prompt_{agent_id}.txt"
    try:
        if not os.path.exists(filename):
            return f"Agent[{agent_id}] 的指令文件不存在。请先使用 update_next_prompt 创建。"
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"


def update_next_prompt(agent_id: str, new_content: str) -> str:
    """
    更新或初始化指定 Agent 的任务指令。用于跨轮次保存进度或切换目标。
    
    Example:
        update_next_prompt(agent_id="explorer_01", new_content="已抓取首页，下一目标是回复 ID 为 999 的串。")
        
    Args:
        agent_id: 当前 Agent 的唯一身份标识。
        new_content: 存入文件的详细指令或状态。
    """
    filename = f"prompt_{agent_id}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Agent[{agent_id}] 指令已更新。"
    except Exception as e:
        return f"更新失败: {str(e)}"


# --- 工具定义 (用于提供给 LLM) ---

TOOLS_SCHEMA = [
    {
        "name": "fetch_threads",
        "description": "访问板块或查串详情。bid为板块ID，tid为0查列表，非0查详情。",
        "parameters": {
            "type": "object",
            "properties": {
                "bid": {"type": "integer", "description": "板块ID"},
                "tid": {"type": "integer", "description": "串ID (0为列表)", "default": 0},
                "pn": {"type": "integer", "description": "页码", "default": 0},
            },
            "required": ["bid"],
        },
    },
    {
        "name": "post_to_board",
        "description": '发帖/回复。⚠️禁止使用三连双引号\"\"\"，请用三连单引号\'\'\'替代。双引号需转义为\\\"。',
        "parameters": {
            "type": "object",
            "properties": {
                "bid": {"type": "integer", "description": "板块ID"},
                "tid": {"type": "integer", "description": "目标串no (发新串传0)", "default": 0},
                "txt": {"type": "string", "description": "内容。若包含代码，请将 \"\"\" 替换为 '''"},
                "title": {"type": "string", "description": "标题"},
                "name": {"type": "string", "description": "署名"},
            },
            "required": ["bid", "txt"],
        },
    },
    {
        "name": "read_next_prompt",
        "description": "通过 agent_id 读取对应的长期任务指令或历史进度。",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent唯一识别码"}
            },
            "required": ["agent_id"]
        },
    },
    {
        "name": "update_next_prompt",
        "description": "为特定 agent_id 写入或更新下一轮执行的指令。",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent唯一识别码"},
                "new_content": {"type": "string", "description": "新的指令内容"}
            },
            "required": ["agent_id", "new_content"],
        },
    },
]

# --- 映射表 ---
tool_map = {
    "fetch_threads": fetch_threads,
    "post_to_board": post_to_board,
    "read_next_prompt": read_next_prompt,
    "update_next_prompt": update_next_prompt,
}