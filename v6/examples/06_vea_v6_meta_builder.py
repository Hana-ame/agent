"""示例 6：运行 VEA V6 附赠的“用来构建 DAG 的 DAG” (Meta-DAG Builder).

展示如何使用一个 V6 DAG (`builder_dag`) 来自动构建另一个可运行的业务 V6 DAG：
1. `builder_dag = create_dag_builder_dag()` 自身就是一个标准的 V6 DAG（包含 3 条处理边与预设边）；
2. 向 `builder_dag` 送入一段自然语言步骤描述 (`tags=["input"], gen=0`)；
3. 调用 `builder_dag.run()` 直至 `builder_dag.context` 中出现 `tag="result"` 的条目；
4. 从 `tag="result"` 条目反序列化得到目标业务 V6 DAG，送入业务数据并运行得到最终结果！

运行方式：
    python3 examples/06_vea_v6_meta_builder.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import DEFAULT_PRESET_EDGE, DAG, SmartMockLLM, create_dag_builder_dag


def main() -> None:
    # 1. 创建“用来构建 DAG 的 DAG”
    builder_dag = create_dag_builder_dag(similarity_threshold=0.45)
    print("【步骤 1】用来构建 DAG 的元图 (builder_dag) 边列表:")
    for i, e in enumerate(builder_dag.edges, 1):
        print(f"  Edge {i}: layer={e.layer}, input_tags={e.input_tags}, output_tags={e.output_tags}")

    # 2. 向 builder_dag 送入任务描述信息
    task_description = """
    1. 从 x1 经过 提取商品价格明细 到 商品价格列表
    2. 从 x1 经过 提取会员优惠折扣 到 优惠折扣信息
    3. 从 商品价格清单 经过 汇总商品原价总和 到 原价小计
    4. 从 原价小计与优惠折扣信息 经过 {"模型": "billing-v6", "处理prompt": "核算最终应付金额"} 到 z2
    """
    builder_dag.send({"tags": ["input"], "content": task_description, "gen": 0})

    # 3. 运行 builder_dag，在外部检测 tag="result" 终止
    while not any("result" in item["tags"] for item in builder_dag.context):
        builder_dag.run()

    print("\n【步骤 2】builder_dag 运行完成，全部 context 条目:")
    for idx, item in enumerate(builder_dag.context):
        print(f"  [{idx}] gen={item['gen']} tags={item['tags']} content={item['content'][:90]}...")

    built_dag_json = builder_dag.get_result()
    assert built_dag_json is not None
    built_dag_spec = json.loads(built_dag_json)
    print("\n  -> 对齐日志:", built_dag_spec.get("alignment_logs"))

    # 4. 用构建出的 JSON 规格实例化业务 V6 DAG 并运行
    mock_llm = SmartMockLLM(
        rules={
            "提取商品价格明细": '{"items": [120, 80, 50]}',
            "提取会员优惠折扣": '{"discount": 30}',
            "汇总商品原价总和": lambda p: json.dumps(
                {"subtotal": sum(json.loads(p.split("输入数据：")[-1])["items"])}
            ),
            "核算最终应付金额": lambda p: json.dumps(
                {
                    "final_pay": json.loads(p.split("输入数据：")[-1])["subtotal"]
                    - json.loads(p.split("输入数据：")[-1])["discount"]
                }
            ),
        }
    )

    business_dag = DAG.from_dict(
        built_dag_spec,
        llm=mock_llm,
        preset_edge=DEFAULT_PRESET_EDGE,
        models={
            "billing-v6": {
                "api": "https://api.sensenova.cn/v1",
                "key": "sk-demo",
                "model": "billing-v6",
                "params": {"temperature": 0.0},
            },
        },
    )
    business_dag.send(
        {
            "tags": ["x1"],
            "content": "订单：键盘 120元、鼠标 80元、鼠标垫 50元；使用满减券立减 30元",
            "gen": 0,
        }
    )

    while not any("result" in item["tags"] for item in business_dag.context):
        business_dag.run()

    print("\n【步骤 3】构建出的业务 DAG 运行完毕，最终回应 (tag=result):", business_dag.get_result())


if __name__ == "__main__":
    main()
