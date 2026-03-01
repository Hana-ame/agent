"""
core.py - 工具核心模块，提供上下文和路径验证
"""

import os
import re

class Context:
    def __init__(self, root_path):
        self.root_path = root_path

    def validate_path(self, relative_path):
        """
        验证并返回相对于根目录的绝对路径
        拒绝包含 '..' 的路径和绝对路径（包括 Windows 驱动器字母）
        """
        # 防止路径遍历攻击
        if '..' in relative_path:
            raise ValueError(f"非法路径（包含 '..'）：{relative_path}")
        # 检查 Unix 绝对路径
        if relative_path.startswith('/'):
            raise ValueError(f"非法路径（Unix 绝对路径）：{relative_path}")
        # 检查 Windows 绝对路径（以反斜杠开头或包含驱动器字母）
        if relative_path.startswith('\\'):
            raise ValueError(f"非法路径（Windows 绝对路径）：{relative_path}")
        if re.match(r'^[a-zA-Z]:\\', relative_path):
            raise ValueError(f"非法路径（Windows 驱动器路径）：{relative_path}")
        return os.path.join(self.root_path, relative_path)

def load_root_path():
    """
    从环境变量或当前目录加载根路径
    """
    root = os.getcwd()
    # 尝试从 .env 文件读取 ROOT_PATH
    env_path = os.path.join(root, '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ROOT_PATH'):
                    root = line.strip().split('=', 1)[1].strip()
                    if root.startswith('"') and root.endswith('"'):
                        root = root[1:-1]
                    elif root.startswith("'") and root.endswith("'"):
                        root = root[1:-1]
                    break
    except FileNotFoundError:
        pass
    return os.path.abspath(root)
