"""示例 5：VEA V6 极简快速上手 —— 纯标签信息流、预设边拼凑与外部条件终止.

展示 VEA V6 的核心设计：
1. `dag` 用 `{"contents": [], "edges": []}` 表示；
2. 只用加处理边：`(输入的 tag list[str], 处理方式 (str | {模型, 处理prompt}), 输出的 tag list[str])`；
3. 送入信息结构：`{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`；
4. 对一条没有直接输入的 `edge`，自动从现有 `contents` 中找两条信息，通过 `dag.preset_edge` 拼凑出一条信息；
5. `dag.run()` = 跑一次，`dag.context` = 列出所有 context，在外部检测 `tag="result"` 终止。

运行方式：
    python3 examples/05_vea_v6_quickstart.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import DAG, SmartMockLLM


def main() -> None:
    mock_llm = SmartMockLLM(
        rules={
            "提取营收明细": '{"revenue": 5000}',
            "提取成本明细": '{"cost": 3000}',
            "计算净利润并生成结论": lambda p: json.dumps(
                {
                    "net_profit": json.loads(p.split("输入数据：")[-1])["revenue"]
                    - json.loads(p.split("输入数据：")[-1])["cost"],
                    "status": "盈利",
                },
                ensure_ascii=False,
            ),
        }
    )

    # 1. 初始化 V6 DAG（自带 preset_edge 预设边对象）
    dag = DAG(llm=mock_llm)
    print("初始化预设边对象:", dag.preset_edge)

    # 2. 只用加处理边：(input_tags, processor, output_tags)
    dag.add_edge(["raw_report"], "提取营收明细: {content}", ["revenue_part"])
    dag.add_edge(
        ["raw_report"],
        {"模型": "sensenova-v6", "处理prompt": "提取成本明细: {content}"},
        ["cost_part"],
    )
    # 第 3 条边不设单条直接输入（input_tags=[]），运行时自动触发 dag.preset_edge 从现有条目找两条拼凑！
    dag.add_edge(
        [],
        {"模型": "sensenova-v6-reasoner", "处理prompt": "计算净利润并生成结论"},
        ["result"],
    )

    # 3. 送入一条初始信息 (gen=0)
    dag.send(
        {
            "tags": ["raw_report"],
            "content": "Q4季度总营收 5000 万元，总成本 3000 万元",
            "image": "https://example.com/q4_chart.png",
            "gen": 0,
        }
    )

    # 4. 在外部检测是否出现 tag="result" 来终止
    step = 0
    while not any("result" in item["tags"] for item in dag.context):
        step += 1
        dag.run()
        print(f"\n--- 第 {step} 次调用 dag.run() 后的 dag.context (共 {len(dag.context)} 条) ---")
        for idx, item in enumerate(dag.context):
            print(f"  [{idx}] gen={item['gen']} tags={item['tags']} content={item['content']}")

    # 5. 获取 tag="result" 的 content 作为最后回应
    print("\n最终回应 (tag=result):", dag.get_result())


if __name__ == "__main__":
    main()
