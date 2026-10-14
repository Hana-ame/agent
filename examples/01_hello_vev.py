"""示例 1：3 行代码快速上手 VEA V5（从 x1 经过 y1 到 z1，从 z1 经过 y2 到 z2）.

运行方式：
    python3 examples/01_hello_vev.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v5 import DAG, Executor


def main() -> None:
    # 1. 创建一张空的 DAG
    dag = DAG("Hello_VEV_Pipeline")

    # 2. 一句一句往里加 "从 V_in 经过 E 到 V_out"（不需要手动设置 start/end 或状态机！）
    dag.add_step("x1", lambda text: [int(n) for n in text.split(",")], "z1")
    dag.add_step("z1", lambda nums: {"count": len(nums), "sum": sum(nums)}, "z2")

    print(dag.visualize_ascii())

    # 3. 传入起点 x1 的数据并直接运行
    report = Executor(dag).run_sync(inputs={"x1": "10, 20, 30, 40"})

    print("\n执行成功:", report.success)
    print("执行的边:", report.executed_edges)
    print("中间节点 z1 的值:", report.all_vertices["z1"])
    print("最终终点 z2 的值:", report.final_output)


if __name__ == "__main__":
    main()
