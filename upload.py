import requests
import json
import os
from urllib.parse import quote

def upload_file(path: str) -> str:
    """
    上传文件到指定服务器，并返回可访问的文件URL。

    Args:
        path (str): 本地文件的路径。

    Returns:
        str: 上传后文件的完整URL。

    Raises:
        可能抛出requests异常或文件不存在异常。
    """
    endpoint = "https://upload.moonchan.xyz/api/upload"
    
    # 读取文件内容并以PUT方式发送
    with open(path, 'rb') as f:
        response = requests.put(endpoint, data=f)
    
    # 检查请求是否成功
    response.raise_for_status()
    
    # 解析JSON响应
    data = response.json()
    file_id = data['id']
    filename = os.path.basename(path)
    encoded_filename = quote(filename)
    
    # 构造完整URL
    file_url = f"https://upload.moonchan.xyz/api/{file_id}/{encoded_filename}"
    return file_url

def main():
    # 1. 上传 deepseek-script.js
    print("正在上传 deepseek-script.js...")
    script_url = upload_file("deepseek-script.js")
    print(f"deepseek-script.js 上传成功: {script_url}")

    # 2. 原地修改 config.json
    print("正在修改 config.json...")
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)  # 假设是列表

    if isinstance(config, list) and len(config) > 0 and "url" in config[0]:
        config[0]["url"] = script_url
    else:
        raise ValueError("config.json 格式不正确，应为包含 'url' 字段的对象列表")

    # 写回原文件
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 3. 上传修改后的 config.json
    print("正在上传 config.json...")
    config_url = upload_file("config.json")
    print(f"config.json 上传成功: {config_url}")

    # 4. 打印最终得到的 URL
    print("\n最终得到的 URL:")
    print(config_url)

if __name__ == "__main__":
    main()