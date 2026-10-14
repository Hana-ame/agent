"""示例 2：运行“第一个 VEA DAG（合并器）”——将分立的自然语言句子自动拆成 V-E-V 并用虚拟边连接相似顶点.

场景说明：
用户或 LLM 写出了四句分立的自然语言描述：
  1. "从 x1 经过 提取商品价格明细 到 商品价格列表"
  2. "从 x1 经过 提取会员优惠折扣 到 优惠折扣信息"
  3. "从 商品价格清单 经过 汇总商品原价总和 到 原价小计"
  4. "从 原价小计与优惠折扣信息 经过 核算最终应付金额 到 z2"

注意：
- 第 1 句输出的叫 `商品价格列表`，但第 3 句输入的叫 `商品价格清单`（名字不同，但语义很像！）；
- 第 2 句输出的叫 `优惠折扣信息`、第 3 句输出的叫 `原价小计`，而第 4 句输入的叫 `原价小计与优惠折扣信息`！
合并器（本身是第一个 3 层 VEA DAG）会自动：
- 第 1 层：把每句话拆成独立的 `V - E - V` 三元组；
- 第 2 层：发现这些顶点之间“挺像的”，自动在它们之间架设虚线虚拟边 `VirtualEdge`；
- 第 3 层：合成为一张完整可运行的 DAG 并直接跑出最终账单结果！

运行方式：
    python3 examples/02_sentence_merger_dag.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v5 import Executor, MergeMode, SentenceToDAGMerger, SmartMockLLM


def main() -> None:
    # 分立的一句一句自然语言描述
    raw_sentences = """
    1. 从 x1 经过 提取商品价格明细 到 商品价格列表
    2. 从 x1 经过 提取会员优惠折扣 到 优惠折扣信息
    3. 从 商品价格清单 经过 汇总商品原价总和 到 原价小计
    4. 从 原价小计与优惠折扣信息 经过 核算最终应付金额 到 z2
    """

    print("【步骤 0】看看合并器自身（第一个 VEA DAG）的 3 层结构：")
    merger = SentenceToDAGMerger(
        similarity_threshold=0.45,
        default_merge_mode=MergeMode.JSON_MERGE,
    )
    print(merger.meta_dag.visualize_ascii())

    print("\n【步骤 1】运行合并器 DAG，将分立句子拆成 V-E-V 并用虚拟边连接相似顶点：")
    merge_res = merger.merge(raw_sentences)

    print("\n  -> 第 1 层输出（分立的 V-E-V 三元组）：")
    for t in merge_res.extracted_triplets:
        print(f"     {t}")

    print("\n  -> 第 2 层输出（相似顶点识别与虚拟边架设）：")
    for log in merge_res.similarity_logs:
        print(f"     {log}")

    print("\n  -> 第 3 层输出（合并完成的可执行业务 DAG）：")
    business_dag = merge_res.compiled_dag
    print(business_dag.visualize_ascii())

    # 准备一个模拟大模型来执行合并后的业务 DAG
    mock_llm = SmartMockLLM(
        rules={
            "提取商品价格明细": '{"items": [120, 80, 50]}',
            "提取会员优惠折扣": '{"discount": 30}',
            "汇总商品原价总和": lambda p: json.dumps({"subtotal": sum(json.loads(p.split("输入数据：")[-1])["items"])}),
            "核算最终应付金额": lambda p: json.dumps(
                {
                    "final_pay": json.loads(p.split("输入数据：")[-1])["subtotal"]
                    - json.loads(p.split("输入数据：")[-1])["discount"]
                }
            ),
        }
    )

    print("\n【步骤 2】在 VEA V5 执行器下直接运行合并出来的业务 DAG：")
    report = Executor(business_dag, llm=mock_llm).run_sync(
        inputs={"x1": "订单：键盘 120元、鼠标 80元、鼠标垫 50元；使用满减券立减 30元"}
    )

    print("  运行是否成功:", report.success)
    print("  各顶点数据快照:")
    for v_name, val in report.all_vertices.items():
        print(f"    - {v_name}: {val}")
    print("  最终终点 z2 输出:", report.final_output)


if __name__ == "__main__":
    main()
