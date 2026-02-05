import os
import json
import requests
from typing import List, Dict, Any, Optional
from utils import *

# 基础URL
BASE_URL = "https://moonchan.xyz/api/v2/"

def fetch_threads(bid: int, tid: int = 0, pn: int = 0) -> str:
    """
    访问板块或查看特定主题串。
    
    Args:
        bid: 板块ID (例如 1)
        tid: 主题串ID。为0时表示获取板块列表，为特定no时查看该串详情。
        pn: 页码，从0开始。
        
    Returns:
        JSON 格式的字符串。包含主串(no)及其回复列表(list)。
        注意：Agent 只能对第一层级中的 "no" 进行回复操作。
    """
    params = {
        "bid": bid,
        "tid": tid,
        "pn": pn
    }
    try:
        response = requests.get(BASE_URL, params=params, headers=get_headers(), timeout=10)
        response.raise_for_status()
        # 返回原始JSON文本供LLM分析
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})

def post_to_board(bid: int, tid: int = 0, txt: str = "无文本", title: str = "", name: str = "") -> str:
    """
    在指定板块发帖或回复主串。
    
    Args:
        bid: 板块ID。
        tid: 回复目标。如果要【发布新串】，传 0；如果要【回复现有主串】，传该主串的 "no"。
             注意：不能回复 list 里面的子回复 no，只能回复第一层级的主串 no。
        txt: 正文内容。
        title: 标题 (可选)。
        name: 署名 (可选)。
        
    Returns:
        服务器响应信息。
    """
    # 构造接口 URL
    url = f"{BASE_URL}?bid={bid}&tid={tid}"
    
    # 构造符合要求的 Payload
    payload = {
        "id": "",     # 留空
        "no": 0,      # 固定传0，系统会自动分配
        "n": name,    # 署名
        "t": title,   # 标题
        "txt": txt,   # 正文
        "p": ""       # 图片路径，留空
    }
    
    try:
        response = requests.post(
            url, 
            headers=get_headers(), 
            data=json.dumps(payload), 
            timeout=10
        )
        response.raise_for_status()
        return f"发送成功。服务器返回: {response.text}"
    except Exception as e:
        return f"发送失败: {str(e)}"
