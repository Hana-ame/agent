
# [START] UTILS-CORE
# version: 001
# 上下文：被 utils.py 导入，用于创建 Context 对象。
import os
import re

class Context:
    def __init__(self, root_path):
        self.root_path = os.path.abspath(root_path)

    def validate_path(self, relative_path):
        """
        验证并返回绝对路径，防止路径穿越 (../)
        """
        if not relative_path:
            raise ValueError("路径不能为空")
            
        # 简单防穿越
        if '..' in relative_path:
            raise ValueError(f"非法路径（包含 '..'）：{relative_path}")
            
        # 拼接并获取绝对路径
        full_path = os.path.abspath(os.path.join(self.root_path, relative_path))
        
        # 再次确认最终路径是否在 root_path 下
        if not full_path.startswith(self.root_path):
            raise ValueError(f"访问拒绝：路径 {relative_path} 超出工作目录范围")
            
        return full_path

def load_root_path():
    """从环境变量或 .env 获取根路径"""
    root = os.getcwd()
    env_path = os.path.join(root, '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('ROOT_PATH='):
                        val = line.split('=', 1)[1].strip()
                        return val.strip('"').strip("'")
        except:
            pass
    return root
# [END] UTILS-CORE