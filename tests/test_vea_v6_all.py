"""VEA V6 完整自动化单元测试与元图构建验证套件.

支持以下任意方式运行：
    python3 tests/test_vea_v6_all.py
    python3 -m unittest tests/test_vea_v6_all.py -v
    pytest tests/test_vea_v6_all.py -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import (
    ContentItem,
    DAG,
    Edge,
    MetaDAGBuilder,
    SmartMockLLM,
    create_dag_builder_dag,
    parse_edges_from_description,
)


def test_1_dag_structure_and_edge_formats() -> None:
    """测试 1：验证 V6 DAG 结构 {contents: [], edges: []} 与处理边三元组格式。"""
    mock_llm = SmartMockLLM(
        rules={
            "提取核心数字": "100",
            "按汇率折算": "720",
        }
    )
    dag = DAG(llm=mock_llm)

    # 初始化时即具备 contents, edges 以及预设边 preset_edge
    assert dag.contents == []
    assert dag.edges == []
    assert dag["contents"] == []
    assert dag["edges"] == []
    assert isinstance(dag.preset_edge, Edge)
    assert dag.preset_edge.is_preset is True

    # 只用加处理边：(输入的 tag list[str], 处理方式 (str | {模型, 处理prompt}), 输出的 tag list[str])
    dag.add_edge(["raw_input"], "提取核心数字: {content}", ["num_usd"])
    dag.add_edge(
        ["num_usd"],
        {"模型": "sensenova-v6-pro", "处理prompt": "按汇率折算: {content}"},
        ["result"],
    )

    assert len(dag.edges) == 2
    assert dag.edges[0].input_tags == ["raw_input"]
    assert dag.edges[0].output_tags == ["num_usd"]
    assert dag.edges[1].model == "sensenova-v6-pro"
    assert "按汇率折算" in dag.edges[1].prompt

    # 支持三元组解包与下标访问
    in_tags, proc, out_tags = dag.edges[1]
    assert in_tags == ["num_usd"]
    assert isinstance(proc, dict)
    assert out_tags == ["result"]

    # 送入一条信息：{tags: list[str], content: str, gen: 0}
    dag.send({"tags": ["raw_input"], "content": "订单金额 $100", "gen": 0})
    assert len(dag.context) == 1
    assert dag.context[0]["tags"] == ["raw_input"]
    assert dag.context[0]["gen"] == 0

    # 跑一次并从外部检测 tag="result"
    while not any("result" in item["tags"] for item in dag.context):
        dag.run()

    result_items = [item for item in dag.context if "result" in item["tags"]]
    assert len(result_items) == 1
    assert result_items[-1]["content"] == "720"
    assert result_items[-1]["gen"] == 2
    assert mock_llm.call_history[-1]["model"] == "sensenova-v6-pro"

    # 验证序列化结构为 {"contents": [...], "edges": [...]}
    serialized = dag.to_dict()
    assert set(serialized.keys()) == {"contents", "edges"}
    assert len(serialized["contents"]) == 3
    assert len(serialized["edges"]) == 2


def test_2_message_structure_with_optional_image_and_gen() -> None:
    """测试 2：验证信息条目结构 {tags: list[str], content: str, image?: path|url, gen: 0}。"""
    mock_llm = SmartMockLLM(
        rules={
            "识别商品图片与描述": lambda prompt, model, image: f"已由{model}识别[{image}]: 机械键盘",
        }
    )
    dag = DAG(llm=mock_llm)
    dag.add_edge(
        ["product_post"],
        {"模型": "vision-model-v6", "处理prompt": "识别商品图片与描述: {content}"},
        ["result"],
    )

    # 1) 不带 image 的普通消息，序列化字典中不包含多余的 image 键
    plain_item = ContentItem({"tags": ["note"], "content": "纯文本", "gen": 0})
    assert "image" not in plain_item.to_dict()
    assert plain_item.image is None

    # 2) 送入带 image 字段（path 或 url）的信息条目
    msg = dag.send(
        {
            "tags": ["product_post"],
            "content": "全新客制化套件",
            "image": "https://example.com/keyboard.png",
            "gen": 0,
        }
    )
    assert msg["tags"] == ["product_post"]
    assert msg["content"] == "全新客制化套件"
    assert msg["image"] == "https://example.com/keyboard.png"
    assert msg["gen"] == 0

    dag.run()

    res_item = dag.get_result_item("result")
    assert res_item is not None
    assert res_item["tags"] == ["result"]
    assert res_item["image"] == "https://example.com/keyboard.png"
    assert res_item["gen"] == 1
    assert res_item["content"] == "已由vision-model-v6识别[https://example.com/keyboard.png]: 机械键盘"


def test_3_preset_edge_fallback_when_edge_has_no_input() -> None:
    """测试 3：对一条 edge，如果没有输入，则从现有的所有条目中找到两条，通过初始化预设边拼凑一个信息。"""
    # 场景 A：边定义时直接无输入 (input_tags=[])，从已有的两条信息中通过 preset_edge 拼凑并计算结果
    mock_llm = SmartMockLLM(
        rules={
            "综合核算净利润": lambda prompt: str(
                json.loads(prompt.split("输入数据：")[-1])["revenue"]
                - json.loads(prompt.split("输入数据：")[-1])["cost"]
            ),
        }
    )
    dag_empty_in = DAG(llm=mock_llm)
    dag_empty_in.add_edge([], "综合核算净利润", ["result"])

    dag_empty_in.send({"tags": ["rev_info"], "content": '{"revenue": 5000}', "gen": 0})
    dag_empty_in.send({"tags": ["cost_info"], "content": '{"cost": 1800}', "gen": 0})

    while not dag_empty_in.has_result():
        dag_empty_in.run()

    assert dag_empty_in.get_result() == "3200"
    merged_items = [c for c in dag_empty_in.context if "merged" in c["tags"]]
    assert len(merged_items) == 1
    assert merged_items[0]["gen"] == 1
    assert dag_empty_in.get_result_item()["gen"] == 2

    # 场景 B：两条上游边各自产出单标签信息，下游边需要两个标签 ["price", "discount"]，
    # 现有单条信息均不满足，自动触发 preset_edge 找两条拼凑！
    dag_fan_in = DAG()
    dag_fan_in.add_edge(["order"], lambda s: '{"price": 200}', ["price"])
    dag_fan_in.add_edge(["order"], lambda s: '{"discount": 50}', ["discount"])
    dag_fan_in.add_edge(
        ["price", "discount"],
        lambda s: str(json.loads(s)["price"] - json.loads(s)["discount"]),
        ["result"],
    )

    dag_fan_in.send({"tags": ["order"], "content": "商品200元，优惠50元", "gen": 0})
    while not dag_fan_in.has_result():
        dag_fan_in.run()

    assert dag_fan_in.get_result() == "150"

    # 场景 C：自定义初始化 preset_edge 处理逻辑
    custom_preset = Edge(
        [],
        lambda item1, item2: f"COMBINED({item1['content']} + {item2['content']})",
        ["custom_merged"],
        is_preset=True,
    )
    dag_custom_preset = DAG(preset_edge=custom_preset)
    dag_custom_preset.add_edge([], lambda s: f"FINAL: {s}", ["result"])
    dag_custom_preset.send({"tags": ["part1"], "content": "Alpha", "gen": 0})
    dag_custom_preset.send({"tags": ["part2"], "content": "Beta", "gen": 0})
    dag_custom_preset.run()
    assert dag_custom_preset.get_result() == "FINAL: COMBINED(Alpha + Beta)"


def test_4_external_loop_termination_on_result_tag() -> None:
    """测试 4：多轮迭代任务中每次调用 dag.run() 跑一次，在外部通过检测 tag='result' 终止。"""
    dag = DAG()

    def iterative_doubler(item: ContentItem) -> dict:
        val = int(item["content"]) * 2
        if val >= 50:
            return {"tags": ["result"], "content": str(val), "gen": item["gen"] + 1}
        return {"tags": ["step"], "content": str(val), "gen": item["gen"] + 1}

    dag.add_edge(["step"], iterative_doubler, ["step"])
    dag.send({"tags": ["step"], "content": "10", "gen": 0})

    run_count = 0
    while not any("result" in c["tags"] for c in dag.context):
        dag.run()
        run_count += 1
        assert run_count <= 10, "防止异常死循环"

    assert run_count == 3
    assert dag.get_result() == "80"
    assert dag.get_result_item()["gen"] == 3


def test_5_dict_literal_initialization_and_direct_list_mutation() -> None:
    """测试 5：验证直接用 {"contents": [...], "edges": [...]} 初始化 DAG 及直接向 contents.append 字典。"""
    dag = DAG(
        {
            "contents": [
                {"tags": ["a"], "content": "Hello", "gen": 0},
                {"tags": ["b"], "content": "World", "image": "/tmp/demo.png", "gen": 0},
            ],
            "edges": [
                (["a", "b"], lambda s: s.replace("\n", " ") + "!", ["result"]),
            ],
        }
    )
    dag.run()
    assert dag.get_result() == "Hello World!"
    assert dag.get_result_item()["image"] == "/tmp/demo.png"


def test_6_meta_dag_builder_builds_runnable_v6_dag() -> None:
    """测试 6：验证附赠的‘用来构建 DAG 的 DAG’（Meta-DAG Builder）及其构建出的业务 DAG 完整运行。"""
    raw_builder_dag = create_dag_builder_dag()
    assert isinstance(raw_builder_dag, DAG)
    assert raw_builder_dag.contents == []
    assert len(raw_builder_dag.edges) == 3
    assert isinstance(raw_builder_dag.preset_edge, Edge)

    sentences = """
    1. 从 x1 经过 提取商品价格明细 到 商品价格列表
    2. 从 x1 经过 提取会员优惠折扣 到 优惠折扣信息
    3. 从 商品价格清单 经过 汇总商品原价总和 到 原价小计
    4. 从 原价小计与优惠折扣信息 经过 {"模型": "billing-llm", "处理prompt": "核算最终应付金额"} 到 z2
    """

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

    builder = MetaDAGBuilder(similarity_threshold=0.45, llm=mock_llm)
    build_res = builder.build(sentences)

    assert any("result" in item["tags"] for item in build_res.builder_context)
    assert any("merged" in item["tags"] for item in build_res.builder_context)

    business_dag = build_res.compiled_dag
    assert business_dag.contents == []
    assert len(business_dag.edges) == 4
    assert "result" in business_dag.edges[-1].output_tags
    assert business_dag.edges[-1].model == "billing-llm"

    business_dag.send(
        {
            "tags": ["x1"],
            "content": "订单：键盘 120元、鼠标 80元、鼠标垫 50元；满减优惠 30元",
            "gen": 0,
        }
    )

    while not any("result" in item["tags"] for item in business_dag.context):
        business_dag.run()

    final_resp = business_dag.get_result()
    assert final_resp is not None
    assert json.loads(final_resp) == {"final_pay": 220}


def test_7_parse_edge_sentences_variants() -> None:
    """测试 7：验证多种自然语言句式与无输入边句式解析为 V6 处理边。"""
    text = """
    从 [doc] 经过 提取摘要 到 [summary]
    [doc] --[提取关键词]--> [keywords]
    无输入 经过 {模型: deepseek-v3, 处理prompt: 合并摘要与关键词生成报告} 到 [result]
    """
    edges = parse_edges_from_description(text)
    assert len(edges) == 3
    assert edges[0]["input_tags"] == ["doc"]
    assert edges[0]["output_tags"] == ["summary"]
    assert edges[1]["input_tags"] == ["doc"]
    assert edges[1]["output_tags"] == ["keywords"]
    assert edges[2]["input_tags"] == []
    assert edges[2]["processor"] == {"模型": "deepseek-v3", "处理prompt": "合并摘要与关键词生成报告"}
    assert edges[2]["output_tags"] == ["result"]


class TestVEAV6Suite(unittest.TestCase):
    """兼容 unittest 测试运行器的包装类。"""

    def test_1(self) -> None:
        test_1_dag_structure_and_edge_formats()

    def test_2(self) -> None:
        test_2_message_structure_with_optional_image_and_gen()

    def test_3(self) -> None:
        test_3_preset_edge_fallback_when_edge_has_no_input()

    def test_4(self) -> None:
        test_4_external_loop_termination_on_result_tag()

    def test_5(self) -> None:
        test_5_dict_literal_initialization_and_direct_list_mutation()

    def test_6(self) -> None:
        test_6_meta_dag_builder_builds_runnable_v6_dag()

    def test_7(self) -> None:
        test_7_parse_edge_sentences_variants()


if __name__ == "__main__":
    test_1_dag_structure_and_edge_formats()
    test_2_message_structure_with_optional_image_and_gen()
    test_3_preset_edge_fallback_when_edge_has_no_input()
    test_4_external_loop_termination_on_result_tag()
    test_5_dict_literal_initialization_and_direct_list_mutation()
    test_6_meta_dag_builder_builds_runnable_v6_dag()
    test_7_parse_edge_sentences_variants()
    print("ALL 7 VEA V6 TEST SUITES PASSED SUCCESSFULLY!")
