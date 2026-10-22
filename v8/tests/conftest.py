"""v8 测试配置：注册标记。pi 探测见 tests/_probe.py。"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 真实调用 pi，耗时数十秒")
    config.addinivalue_line("markers", "live: 需要 pi 可用（本地或 WSL）")
