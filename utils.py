import os

def get_headers():
    """从环境变量获取Cookie并构建请求头"""
    cookie = os.getenv("MOONCHAN_COOKIE", "auth=DY4X5IHR%7C551e446ae9942a675b7152eb98a1609928c8f5bb4c724a62cdf51eaa8a4c6040;")
    return {
        "Cookie": cookie,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Agent/1.0"
    }
