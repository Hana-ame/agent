#!/usr/bin/env python3
"""CellAgent - 基于细胞仿生思想的多 Agent 协作系统

快速启动:
  # 1. 初始化数据库
  python -c "from cell_core.db import init_db, seed_default_data; init_db(); seed_default_data()"

  # 2. 启动后台执行器（独立进程）
  python cell_core/scheduler.py &

  # 3. 启动 API 服务（另一个终端）
  uvicorn api.main:app --reload --port 8000

  # 4. 创建任务
  curl -X POST http://localhost:8000/tasks \
    -H "Content-Type: application/json" \
    -d '{"dna_id": 1, "input_json": {"task": "写一个冒泡排序的 Python 代码"}}'
"""
