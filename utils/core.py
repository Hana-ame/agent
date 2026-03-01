
# utils/core.py

import os
from pathlib import Path

class Context:
    """上下文，包含根路径和路径验证"""
    def __init__(self, root_path):
        self.root_path = os.path.abspath(root_path)

    def validate_path(self, user_path):
        """
        将用户提供的相对路径转换为绝对路径，并确保在根目录内。
        返回绝对路径；若非法则抛出 ValueError。
        """
        full_path = os.path.abspath(os.path.join(self.root_path, user_path))
        if not full_path.startswith(self.root_path):
            raise ValueError(f"路径 '{user_path}' 试图访问根目录之外，已阻止。")
        return full_path

def load_root_path():
    """从 .env 文件读取 ROOT_PATH，若不存在则返回当前工作目录"""
    env_path = Path(".env")
    root = os.getcwd()
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "ROOT_PATH":
                        root = value.strip().strip('"\'')
                        break
    return os.path.abspath(root)