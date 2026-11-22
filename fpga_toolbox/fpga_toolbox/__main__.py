"""
模块目的：支持直接 `python3 -m fpga_toolbox <tool_name> --参数 值` 启动，不再需要写 `.cli` 和 `call`。
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
