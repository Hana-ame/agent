import os
from datetime import datetime
import json
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("SILICONFLOW_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}


def completions(payload: dict):
    try:
        # 这里的 timeout 设置为 60 秒，防止网络卡死
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        
        result = response.json()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path_response = f"{timestamp}.json"

        try:
            with open(file_path_response, "wt", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            print(file_path_response)
        except Exception as e:
            print(e)

        return result
    except Exception as e:
        print(e)
        pass

def get_content(result:dict, key="content") -> str:    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"][key]
    return str(result)

def user_prompt(content:str, role="user"):
    return {
        "role": role,
        "content": content,
    }