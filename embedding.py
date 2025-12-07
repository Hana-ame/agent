import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def embedding(input: str):
    try:
        # 这里的 timeout 设置为 60 秒，防止网络卡死 ,
        response = requests.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.getenv("SILICONFLOW_API_KEY")}",
                "Content-Type": "application/json",
            },
            json={
                "model": "BAAI/bge-large-zh-v1.5",
                "input": input,
            },
            timeout=120,
        )

        # 解析结果
        result = response.json()

        response.raise_for_status()

        if "data" in result and len(result["data"]) > 0:
            return result["data"][0]["embedding"]
        else:
            return f"Error: 响应数据为空或格式异常: {result}"

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        # 尝试读取 API 返回的详细错误 JSON
        if e.response is not None:
            try:
                err_json = e.response.json()
                if "error" in err_json:
                    error_msg = f"API Error: {err_json['error']['message']}"
            except:
                pass
        return f"请求失败: {error_msg}"


if __name__ == "__main__":
    r = embedding("123123123123")
    print(r)
