"""
上海三打一（两副牌斗地主）模拟器 - 服务入口

启动: uvicorn shanghai_sandayi.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "shanghai_sandayi.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
