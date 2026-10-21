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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import (
    DEFAULT_PRESET_EDGE,
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
    """测试 3：某条 edge 在 run 时找不到对应输入 tag 的信息时，用预设边从相近信息合成 输出tag=输入tag 的条目。"""
    # 场景 A：边需要汇聚 ["rev_info", "cost_info"] 两个标签，没有单条信息同时匹配，
    # 自动通过预设边从相近的两条信息合成一条 输出tag=所需输入tag 的条目再执行
    mock_llm = SmartMockLLM(
        rules={
            "综合核算净利润": lambda prompt: str(
                json.loads(prompt.split("输入数据：")[-1])["revenue"]
                - json.loads(prompt.split("输入数据：")[-1])["cost"]
            ),
        }
    )
    dag_empty_in = DAG(llm=mock_llm)
    dag_empty_in.add_edge(["rev_info", "cost_info"], "综合核算净利润", ["result"])

    dag_empty_in.send({"tags": ["rev_info"], "content": '{"revenue": 5000}', "gen": 0})
    dag_empty_in.send({"tags": ["cost_info"], "content": '{"cost": 1800}', "gen": 0})

    while not dag_empty_in.has_result():
        dag_empty_in.run()

    assert dag_empty_in.get_result() == "3200"
    merged_items = [c for c in dag_empty_in.context if c.get("_preset_synthesized")]
    assert len(merged_items) == 1
    assert set(merged_items[0]["tags"]) == {"rev_info", "cost_info"}
    assert merged_items[0]["gen"] == 1
    assert dag_empty_in.get_result_item()["gen"] == 2

    # 场景 B：两条上游边各自产出单标签信息，下游边需要两个标签 ["price", "discount"]，
    # 现有单条信息均不满足，自动触发预设边找相近的两条信息合成！
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
    dag_custom_preset.add_edge(["part1", "part2"], lambda s: f"FINAL: {s}", ["result"])
    dag_custom_preset.send({"tags": ["part1"], "content": "Alpha", "gen": 0})
    dag_custom_preset.send({"tags": ["part2"], "content": "Beta", "gen": 0})
    dag_custom_preset.run()
    assert dag_custom_preset.get_result() == "FINAL: COMBINED(Alpha + Beta)"

    # 场景 D：处理边不允许没有输入 tag / 输出 tag
    for bad_args in (([], "x", ["y"]), (["x"], "x", [])):
        try:
            DAG().add_edge(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError("空输入/输出 tag 的处理边应被拒绝")


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
    assert any(
        item.get("_preset_synthesized")
        and {"parsed_edges", "tag_alignment"} <= set(item["tags"])
        for item in build_res.builder_context
    )

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


def test_8_models_registry_and_layer_ordering() -> None:
    """测试 8：模型注册表（api/key/model/params 按 KEY 查询注入）与 layer 执行顺序。"""
    # 1. 模型注册表：边里 {"模型": "fast"} 按 KEY 查询 dag.models，调用时替换为注册的真实模型名
    mock_llm = SmartMockLLM(rules={"总结": "摘要完成"})
    dag = DAG(
        llm=mock_llm,
        preset_edge=DEFAULT_PRESET_EDGE,
        models={
            "fast": {
                "api": "https://api.demo.cn/v1",
                "key": "sk-x",
                "model": "sensenova-v6-flash",
                "params": {"temperature": 0.1},
            },
        },
    )
    dag.add_edge(["doc"], {"模型": "fast", "处理prompt": "总结: {content}"}, ["result"])
    dag.send({"tags": ["doc"], "content": "长文", "gen": 0})
    dag.run()
    assert dag.get_result() == "摘要完成"
    assert mock_llm.call_history[-1]["model"] == "sensenova-v6-flash"

    # 2. 已设置注册表但查不到对应模型 KEY → KeyError
    dag_bad = DAG(llm=SmartMockLLM(), models={"fast": {"model": "m"}})
    dag_bad.add_edge(["doc"], {"模型": "ghost", "处理prompt": "x"}, ["result"])
    dag_bad.send({"tags": ["doc"], "content": "y", "gen": 0})
    try:
        dag_bad.run()
    except KeyError:
        pass
    else:
        raise AssertionError("未注册的模型 KEY 应抛出 KeyError")

    # 3. layer 执行顺序：总是从 layer 小的边开始跑（与添加顺序无关）
    order_llm = SmartMockLLM(fallback_fn=lambda p, m, i: p.split("\n")[0])
    dag_layer = DAG(llm=order_llm)
    dag_layer.add_edge(["q"], "第二步", ["s2"], layer=2)
    dag_layer.add_edge(["q"], "第一步", ["s1"], layer=1)
    dag_layer.send({"tags": ["q"], "content": "go", "gen": 0})
    dag_layer.run()
    assert order_llm.call_history[0]["prompt"].startswith("第一步")
    assert order_llm.call_history[1]["prompt"].startswith("第二步")
    assert dag_layer.edges[0].layer == 2 and dag_layer.edges[1].layer == 1

    # 4. 找不到对应 tag 时，预设边可从相近的单条信息合成 输出tag=所需输入tag 的条目
    dag_sim = DAG()
    dag_sim.add_edge(["报价单"], lambda s: f"收到[{s}]", ["result"])
    dag_sim.send({"tags": ["报价"], "content": "100元", "gen": 0})
    dag_sim.run()
    fab = [c for c in dag_sim.context if c.get("_preset_synthesized")]
    assert len(fab) == 1 and fab[0]["tags"] == ["报价单"]
    assert dag_sim.get_result() == "收到[100元]"


def test_9_tag_based_fetch_and_reverse() -> None:
    """测试 9：按指定 tag 获取内容、反转处理后保存回 dag.contents。"""
    # 1. 创建 DAG，添加一条以 callable 为处理器、输入 tag=["todo"]、输出 tag=["done"] 的边
    dag = DAG()
    dag.add_edge(["todo"], lambda content: content[::-1], ["done"], layer=1)

    # 2. 送入多条带 "todo" 标签的待办事项 + 一条不同标签的条目
    dag.send({"tags": ["todo"], "content": "买牛奶", "gen": 0})
    dag.send({"tags": ["todo"], "content": "写报告", "gen": 0})
    dag.send({"tags": ["note"], "content": "这是一条备注", "gen": 0})

    # 3. 运行 DAG
    while not dag.has_tag("done"):
        dag.run()

    # 4. 验证反转结果已保存回 dag.contents
    done_items = dag.find_by_tag("done")
    assert len(done_items) == 2, f"应有 2 条 done 条目，实际 {len(done_items)}"
    assert done_items[0]["content"] == "奶牛买", f"反转错误: {done_items[0]['content']}"
    assert done_items[1]["content"] == "告报写", f"反转错误: {done_items[1]['content']}"

    # 5. 验证 "note" 标签的条目未被 "todo" 边处理（tag 过滤生效）
    assert len(dag.find_by_tag("note")) == 1
    # note 条目没有后续处理，仍然只有 1 条
    all_note = [c for c in dag.context if "note" in c["tags"]]
    assert len(all_note) == 1
    assert all_note[0]["content"] == "这是一条备注"

    # 6. 验证 gen 代数递增
    assert done_items[0]["gen"] == 1
    assert done_items[1]["gen"] == 1

    # 7. 验证原始 "todo" 条目仍在 dag.contents 中（未被删除）
    todo_items = [c for c in dag.context if "todo" in c["tags"]]
    assert len(todo_items) == 2
    assert todo_items[0]["content"] == "买牛奶"


def test_10_openai_compatible_api_edge() -> None:
    """测试 10：OpenAI 兼容 API 边处理（model/key/params 四要素按 KEY 注入）。"""
    # 1. 创建 SmartMockLLM 用于捕获调用参数
    mock_llm = SmartMockLLM(
        rules={
            "翻译": lambda p: json.dumps({"translated": "hello"}, ensure_ascii=False),
        }
    )

    # 2. 创建 DAG，注册 OpenAI 兼容模型（api/key/model/params）
    dag = DAG(
        llm=mock_llm,
        models={
            "gpt-4o": {
                "api": "https://api.openai.com/v1",
                "key": "sk-test-key",
                "model": "gpt-4o",
                "params": {"temperature": 0.2, "max_tokens": 256},
            },
        },
    )

    # 3. 添加使用 {"模型": "gpt-4o", "处理prompt": ...} 的边
    dag.add_edge(
        ["text"],
        {"模型": "gpt-4o", "处理prompt": "请翻译: {content}"},
        ["result"],
        layer=1,
    )

    # 4. 送入文本并运行
    dag.send({"tags": ["text"], "content": "你好", "gen": 0})
    dag.run()

    # 5. 验证模型名被正确替换为注册表中的真实模型名
    assert len(mock_llm.call_history) == 1
    call = mock_llm.call_history[0]
    assert call["model"] == "gpt-4o", f"模型名错误: {call['model']}"
    # 验证 prompt 中包含替换后的内容
    assert "你好" in call["prompt"]

    # 6. 验证处理结果
    assert dag.has_result()
    result = json.loads(dag.get_result())
    assert result["translated"] == "hello"

    # 7. 验证序列化结构中 models 注册表完整保存
    serialized = dag.to_dict()
    assert "contents" in serialized
    assert "edges" in serialized
    assert len(serialized["edges"]) == 1
    edge_proc = serialized["edges"][0]["processor"]
    assert isinstance(edge_proc, dict)
    assert edge_proc.get("模型") == "gpt-4o"
    assert "处理prompt" in edge_proc


def test_11_multi_model_registry_switch() -> None:
    """测试 11：多模型注册表切换——不同边使用不同模型 KEY，运行时自动路由。"""
    mock_llm = SmartMockLLM(
        rules={
            "快速摘要": lambda p: "摘要A",
            "深度分析": lambda p: "分析B",
        }
    )

    dag = DAG(
        llm=mock_llm,
        models={
            "fast": {
                "api": "https://api.fast.cn/v1",
                "key": "sk-fast",
                "model": "fast-model-v1",
                "params": {"temperature": 0.1},
            },
            "deep": {
                "api": "https://api.deep.cn/v1",
                "key": "sk-deep",
                "model": "deep-model-v2",
                "params": {"temperature": 0.0},
            },
        },
    )

    # 两条边使用不同模型
    dag.add_edge(["doc"], {"模型": "fast", "处理prompt": "快速摘要: {content}"}, ["quick"], layer=1)
    dag.add_edge(["doc"], {"模型": "deep", "处理prompt": "深度分析: {content}"}, ["deep"], layer=1)

    dag.send({"tags": ["doc"], "content": "测试文档", "gen": 0})
    dag.run()

    # 验证两条边分别使用了正确的模型
    assert len(mock_llm.call_history) == 2
    models_used = [c["model"] for c in mock_llm.call_history]
    assert "fast-model-v1" in models_used
    assert "deep-model-v2" in models_used

    # 验证两个结果都存在
    assert len(dag.find_by_tag("quick")) == 1
    assert len(dag.find_by_tag("deep")) == 1


def test_12_bootstrap_dag_generates_framework_code() -> None:
    """测试 12：自举 DAG —— 输入描述，在 tag="result" 产出完整框架源码。

    这里用**本地确定性 mock**（`SmartMockLLM` 原样回放各文件源码），
    目的是验拓扑与产物形状，不验模型能力。真模型的链路见
    `tools/oneshot_chain.py`（它要求真的发 HTTP）。
    """
    from vea_v6 import create_bootstrap_dag
    from vea_v6.parts.bootstrap import FRAME_FILES, framework_sources
    from vea_v6.parts.llm import SmartMockLLM

    src = framework_sources()
    mock = SmartMockLLM()
    # 按 prompt 里的 "文件：<path>" 回原文；签名请求回全部路径
    for p in FRAME_FILES:
        mock.add_rule(p, src[p])
    mock.add_rule("签名", '{"files": %s}' % __import__("json").dumps(
        [{"path": p, "sig": []} for p in FRAME_FILES], ensure_ascii=False))

    dag = create_bootstrap_dag(llm=mock)
    dag.send({"tags": ["input"], "content": "", "gen": 0})
    ticks = 0
    while not dag.has_result() and ticks < 8:
        dag.run()
        ticks += 1

    code = dag.get_result()
    assert code and len(code) > 20000, "产出过短，不像完整框架"
    # 收敛性：正常一轮 run 就该跑完（实测 ticks=1、新增 15 条、其中合成 2 条，
    # 分属「三路汇聚」与「skeleton_code 汇聚」两条不同的边）。
    # 若哪天某条边开始反复合成，ticks 会先涨——这条比数量断言更早报警。
    assert ticks <= 3, f"DAG 未在 3 轮内收敛（ticks={ticks}），可能有边在重复合成"
    # 产出必须是合法 Python
    compile(code, "<bootstrapped>", "exec")
    # 框架的公共 API 名都在源码里
    for name in ("class DAG", "class ContentItem", "class Edge", "def generate_framework"):
        assert name in code, f"生成物缺少 {name}"

    # 拓扑：三段式（spec / skeleton / flesh）汇聚到 skeleton_code 的那条边缺三路输入，
    # 必须由预设边把三条 layer-1 产物合成一个中间条目，否则整链断在这里。
    #
    # 只锁契约、不锁条数：`_preset_synthesized` 会被多条边复用（三段式汇聚边、
    # skeleton_code 汇聚边各会合成），历史断言 `== 1` 与「只允许三路 tag」都会误报。
    # 真正要保证的是「三路 tag 的那条合成确实存在」——缺了它 skeleton_code 边就断。
    synth = [i for i in dag.context if i.get("_preset_synthesized")]
    assert synth, "预设边没有合成任何中间条目 —— 三段式会在 skeleton_code 边断链"
    want = ["flesh_plan", "skeleton_plan", "spec_plan"]
    three_way = [i for i in synth if sorted(i["tags"]) == want]
    assert three_way, f"没有三路合成条目，实际合成了：{[sorted(i['tags']) for i in synth]}"

    # 产出必须是**整个 vea_v6**：主源自包含，清单与内嵌源码都齐全
    assert "BOOTSTRAP_B64" in code
    assert dag.get_result() == code  # 结果稳定（幂等，不因重复取用而变化）


def _payload_files(main_src: str) -> dict:
    """从任意一份主源文本里，解出它内嵌的其它文件源码（`{路径: 文本}`）。"""
    import json

    from vea_v6.parts.bootstrap import _CODE_RE, _decode, MAIN_SOURCE

    m = _CODE_RE.search(main_src)
    if not m:
        return {}
    mark = "FRAME_CODE = json.loads(r" + chr(39) * 3
    inner = m.group(0)[len(mark):]
    data = json.loads(inner[:inner.rfind(chr(39) * 3)])
    return {i["path"]: _decode(i["code"]).decode("utf-8")
            for i in data.get("files", []) if i["path"] != MAIN_SOURCE}


def _masked_main(main_src: str) -> tuple:
    """主源的**结构层**：把「随压缩器变」的部分掩掉后剩下的全部内容。

    随 zlib 变的只有两处（实测 3.12 vs 3.13 逐字段比对得出）：
      1. `FRAME_CODE` 里各文件的 `code` —— gzip+base64 载荷，deflate 输出随 zlib 构建变；
      2. `FRAME_MANIFEST` 里**主源自己那条**的 `sha256` —— 它是对主源自身字节取
         sha，而主源字节含上面那些载荷，所以自指 sha 也跟着变（6 个非主源条目的
         sha 是解压后源码的 sha，不随 zlib 变，实测全同）。

    除此之外的**任何**差异都是真漂移：BOOTSTRAP_SLOT、清单 bytes/b64_len/sig、
    模块代码、以及载荷解压后的内容。
    """
    import json

    from vea_v6.parts.bootstrap import MAIN_SOURCE

    q3 = chr(39) * 3

    def cut(mark: str):
        idx = main_src.find(mark)
        if idx < 0:
            return None
        inner = main_src[idx + len(mark):]
        end = inner.find(q3 + ")")
        return idx, idx + len(mark) + end + len(q3) + 1, inner[:end]

    man = cut("FRAME_MANIFEST = json.loads(r" + q3)
    cod = cut("FRAME_CODE = json.loads(r" + q3)
    if man is None or cod is None:
        return (main_src, None, None)

    # 两个块之外的正文（模块代码、BOOTSTRAP_SLOT、SELF_FILL）
    spans = sorted([(0, man[0]), (man[1], cod[0]), (cod[1], len(main_src))])
    body = "".join(main_src[a:b] for a, b in spans)

    man_list = json.loads(man[2])
    man_list = man_list.get("files", man_list) if isinstance(man_list, dict) else man_list
    masked_man = []
    for it in man_list:
        d = {k: v for k, v in it.items() if k != "code"}
        if d.get("path") == MAIN_SOURCE:
            d["sha256"] = ""      # 自指 sha：随压缩器变
        masked_man.append(d)

    cod_list = json.loads(cod[2])
    cod_list = cod_list.get("files", cod_list) if isinstance(cod_list, dict) else cod_list
    masked_cod = [{k: v for k, v in it.items() if k != "code"} for it in cod_list]

    return (body, masked_man, masked_cod)


def test_13_bootstrap_fixed_point_is_byte_identical() -> None:
    """测试 13：固定点 —— 同一 SPEC 恒得同一份代码，且生成物能复现自身。

    🔴 字节不等**不等于**漂移（CI 实测得出，见下）：主源内嵌的 payload 走
    `gzip.compress(..., 9, mtime=0)`，而 deflate 的输出**随 zlib 构建变** ——
    同一份输入在 3.12（zlib 1.3.2）与 3.13（zlib 1.3.1）下压缩后长度相同
    （9657）但字节不同。磁盘这份是按某一个 zlib 固化的，换到另一个构建上重解，
    容器字节必然不同（实测 3.12：等长 77618、73 个字节不同）。

    于是旧判据"主源字节不同 ⇒ 需重新固化"是**误导**：在新构建上重固化只会把
    不兼容挪到旧构建上。正确做法是多看一层 —— **内容层**（解出来的其它 6 个
    文件）若逐字节一致，就是"容器不同、内容相同"，不是漂移；内容层不同才是真
    漂移，照样红。

    遗留（未改，属设计层面，需用户定夺）：要让固定点真正**跨解释器可复现**，
    得把内嵌编码换成与 zlib 无关的确定性编码（如 gzip level 0 直接存储、
    或干脆不压缩只 base64），代价是主源体积变大，且需重新固化。
    """
    import zlib

    from vea_v6 import (
        FRAMEWORK_SPEC,
        render_framework_code,
        verify_fixed_point,
        verify_self_reproduction,
    )
    from vea_v6.parts.bootstrap import MAIN_SOURCE, framework_sources

    gen1 = render_framework_code(FRAMEWORK_SPEC)
    gen2 = render_framework_code(FRAMEWORK_SPEC)
    assert gen1 == gen2, "同一 SPEC 两次渲染不一致"

    ok, reason = verify_fixed_point(FRAMEWORK_SPEC)
    if not ok and "与磁盘主源不一致" in reason:
        # 要看**两层**都一致，才能判为"仅压缩载荷字节不同"：
        #   结构层 = FRAME_CODE 块外的全部文本（含 BOOTSTRAP_SLOT、清单 bytes/sha）
        #            + 块内除 `code` 外的字段；
        #   内容层 = 各载荷解压后的源码。
        # 🔴 只看内容层是不够的（变异测试实测）：把 BOOTSTRAP_SLOT 从 48947 改成
        #    999999 后，内容层照样一致，求解器会正确重解出 48947 —— 这是**真漂移**，
        #    只比内容层会把它误放行。
        disk_main = framework_sources().get(MAIN_SOURCE, "")
        content_same = _payload_files(disk_main) == _payload_files(gen1)
        structure_same = _masked_main(disk_main) == _masked_main(gen1)
        if content_same and structure_same:
            print(f"  ⚠️ 固定点：内嵌 payload 的压缩字节因 zlib "
                  f"{zlib.ZLIB_RUNTIME_VERSION} 与固化时不同而不同；"
                  f"结构层 + 内容层（{len(_payload_files(disk_main))} 个文件）均一致 ⇒ 非漂移")
            ok, reason = True, reason + "（结构层与内容层均一致，仅压缩载荷字节不同）"
        else:
            reason += "（" + ("内容层也不同" if not content_same else "结构层不同") + " ⇒ 真漂移）"
    assert ok, reason

    # 更强的一步：把生成物当模块执行，调它自己的 generate_framework()
    ok2, reason2 = verify_self_reproduction(gen1)
    assert ok2, reason2


def test_14_bootstrap_generates_another_framework_from_same_prompt() -> None:
    """测试 14：用同一段提示词换描述，能生成**另一个**同样自稳定的新框架。

    必须显式传 mock：不传 `llm` 时 `bootstrap` 会真的去调模型（这是有意设计，
    为了让"假通过"不可能发生），但单元测试不能依赖网络与 key。
    模型链路由 `tools/oneshot_chain.py` 负责验证。
    """
    import json

    from vea_v6 import FRAME_FILES, bootstrap, verify_self_reproduction
    from vea_v6.parts.bootstrap import framework_sources
    from vea_v6.parts.llm import SmartMockLLM

    src = framework_sources()
    mock = SmartMockLLM()
    for p in FRAME_FILES:
        mock.add_rule(p, src[p])
    mock.add_rule(
        "签名",
        json.dumps({"files": [{"path": p, "sig": []} for p in FRAME_FILES]}, ensure_ascii=False),
    )

    desc = "一个面向多模态检索的 tag 驱动 DAG 框架，边可按 layer 并行，结果合并写回 contents"
    r1 = bootstrap(desc, llm=mock)
    r2 = bootstrap(desc, llm=mock)

    assert r1.stable, r1.reason
    assert r1.framework_code == r2.framework_code, "同一描述两次生成不一致"
    compile(r1.framework_code, "<gen>", "exec")

    ok, reason = verify_self_reproduction(r1.framework_code)
    assert ok, reason

    # 与默认 SPEC 的产物必须不同（说明 SPEC 真的参与了生成）
    from vea_v6 import FRAMEWORK_SPEC, render_framework_code

    # mock 只回放源码本体、但 spec 是我们传的描述，所以：
    # 1) 产物必须能复现自身（自举闭合）；2) SPEC 里必须记着这次描述。
    from vea_v6 import FRAMEWORK_SPEC

    ok2, reason2 = verify_self_reproduction(r1.framework_code)
    assert ok2, reason2
    assert r1.spec["summary"] == desc
    assert r1.spec["summary"] != FRAMEWORK_SPEC["summary"]


def test_15_generated_framework_is_runnable() -> None:
    """测试 15：生成的框架能当普通库用 —— 建 DAG、按 tag 取内容、跑出 result。"""
    import types

    from vea_v6 import render_framework_code

    code = render_framework_code()
    mod = types.ModuleType("minivea_under_test")
    mod.__file__ = "<minivea_under_test>"
    exec(compile(code, "<minivea_under_test>", "exec"), mod.__dict__)

    dag = mod.DAG("t", preset_edge=mod.DEFAULT_PRESET_EDGE)
    dag.add_edge(["input"], lambda s: s[::-1], ["todo"], layer=1)
    dag.add_edge(["todo"], lambda s: s.upper(), ["result"], layer=2)
    dag.send({"tags": ["input"], "content": "abc", "gen": 0})
    dag.run()

    assert dag.has_result()
    assert dag.get_result() == "CBA"
    # 处理结果确实回到了 contents 中
    assert any(i["content"] == "CBA" for i in dag.contents)


def test_16_inline_openai_config_on_edge() -> None:
    """测试 16：内联 OpenAI 配置边（api/key/model/params 直接写在 processor 里）。"""
    from vea_v6 import OpenAICompatibleLLM

    dag = DAG(preset_edge=DEFAULT_PRESET_EDGE, llm=OpenAICompatibleLLM(), models={})
    processor = {
        "api": "https://example.invalid/v1",
        "key": "sk-inline-test",
        "model": "demo-model",
        "params": {"temperature": 0.0, "max_tokens": 32},
        "处理prompt": "echo {content}",
    }
    edge = dag.add_edge(["input"], processor, ["out"], layer=1)

    inline = edge.inline_llm_config
    assert inline is not None
    assert inline["base_url"] == "https://example.invalid/v1"
    assert inline["api_key"] == "sk-inline-test"
    assert inline["model"] == "demo-model"

    cfg = dag._resolve_model_config(None, edge=edge)
    assert cfg["model"] == "demo-model"
    assert cfg["base_url"] == "https://example.invalid/v1"
    assert cfg["api_key"] == "sk-inline-test"
    assert cfg["params"]["max_tokens"] == 32

    # 只写 {"模型": KEY} 不应被判成内联配置（否则会去查一个不存在的端点）
    dag2 = DAG(models={"k": {"api": "https://x.invalid/v1", "key": "s", "model": "m"}})
    edge2 = dag2.add_edge(["input"], {"模型": "k", "处理prompt": "{content}"}, ["out"], layer=1)
    assert edge2.inline_llm_config is None


def test_17_registry_key_not_treated_as_model_name() -> None:
    """测试 17：注册表 KEY + 边内参数覆盖时，不得把 KEY 当成真实模型名。"""
    models = {
        "demo-key": {
            "api": "https://demo.invalid/v1",
            "key": "sk-demo",
            "model": "real-model-name",
            "params": {"temperature": 0.2},
        }
    }
    dag = DAG(models=models)
    edge = dag.add_edge(
        ["input"], {"模型": "demo-key", "参数": {"temperature": 0.9}, "处理prompt": "{content}"},
        ["out"], layer=1,
    )
    cfg = dag._resolve_model_config("demo-key", edge=edge)
    assert cfg["model"] == "real-model-name", "KEY 被误当成模型名（实测会导致 provider 404）"
    assert cfg["params"]["temperature"] == 0.9, "边内参数未覆盖注册表参数"
    assert cfg["base_url"] == "https://demo.invalid/v1"


def test_18_llm_text_extraction_falls_back_to_reasoning() -> None:
    """测试 18：推理模型 content 为空时，回退 reasoning_content / reasoning 字段。"""
    from vea_v6.llm import _extract_message_text

    assert _extract_message_text({"choices": [{"message": {"content": "OK"}}]}) == "OK"
    assert _extract_message_text(
        {"choices": [{"message": {"content": "", "reasoning_content": "思考后作答"}}]}
    ) == "思考后作答"
    assert _extract_message_text(
        {"choices": [{"message": {"reasoning": "only-reasoning"}}]}
    ) == "only-reasoning"
    try:
        _extract_message_text({"choices": [{"message": {"content": "   "}}]})
    except ValueError:
        pass
    else:
        raise AssertionError("空正文应当抛出 ValueError")


def _unittest_only(fn):
    """unittest 兼容包装：pytest 下跳过，避免同一逻辑跑两遍。

    1~23 号在模块级已有同名函数式测试（pytest 收集模块级那份即可）；这个类
    只是让 `python -m unittest` 也能跑。unittest 不认 pytest 的 mark，那边照常执行。
    """
    return pytest.mark.skip(reason="unittest 兼容包装：pytest 下由模块级同名测试覆盖")(fn)


class TestVEAV6Suite(unittest.TestCase):
    """兼容 unittest 测试运行器的包装类。"""

    @_unittest_only
    def test_1(self) -> None:
        test_1_dag_structure_and_edge_formats()

    @_unittest_only
    def test_2(self) -> None:
        test_2_message_structure_with_optional_image_and_gen()

    @_unittest_only
    def test_3(self) -> None:
        test_3_preset_edge_fallback_when_edge_has_no_input()

    @_unittest_only
    def test_4(self) -> None:
        test_4_external_loop_termination_on_result_tag()

    @_unittest_only
    def test_5(self) -> None:
        test_5_dict_literal_initialization_and_direct_list_mutation()

    @_unittest_only
    def test_6(self) -> None:
        test_6_meta_dag_builder_builds_runnable_v6_dag()

    @_unittest_only
    def test_7(self) -> None:
        test_7_parse_edge_sentences_variants()

    @_unittest_only
    def test_8(self) -> None:
        test_8_models_registry_and_layer_ordering()

    @_unittest_only
    def test_9(self) -> None:
        test_9_tag_based_fetch_and_reverse()

    @_unittest_only
    def test_10(self) -> None:
        test_10_openai_compatible_api_edge()

    @_unittest_only
    def test_11(self) -> None:
        test_11_multi_model_registry_switch()

    @_unittest_only
    def test_12(self) -> None:
        test_12_bootstrap_dag_generates_framework_code()

    @_unittest_only
    def test_13(self) -> None:
        test_13_bootstrap_fixed_point_is_byte_identical()

    @_unittest_only
    def test_14(self) -> None:
        test_14_bootstrap_generates_another_framework_from_same_prompt()

    @_unittest_only
    def test_15(self) -> None:
        test_15_generated_framework_is_runnable()

    @_unittest_only
    def test_16(self) -> None:
        test_16_inline_openai_config_on_edge()

    @_unittest_only
    def test_17(self) -> None:
        test_17_registry_key_not_treated_as_model_name()

    @_unittest_only
    def test_18(self) -> None:
        test_18_llm_text_extraction_falls_back_to_reasoning()

    @_unittest_only
    def test_19(self) -> None:
        test_19_every_generation_runs_a_self_chosen_dag()

    @_unittest_only
    def test_20(self) -> None:
        test_20_meta_dag_dispatches_eight_steps_without_network()

    @_unittest_only
    def test_21(self) -> None:
        test_21_provider_failure_is_not_accepted_as_a_generation()

    @_unittest_only
    def test_22(self) -> None:
        test_22_source_gate_rejects_a_generation_with_nothing_new()

    @_unittest_only
    def test_23(self) -> None:
        test_23_head_steps_and_checkout_verification()


def test_19_every_generation_runs_a_self_chosen_dag() -> None:
    """测试 19：**每一代**框架都要能用自己产出一个自选任务的可用 DAG。

    任务与验收都由本测试自选（**不是"再写一个 vea 框架"**）：
    访问日志分析 —— 单输入边切开五个字段，再判成败、去查询串、拼成一行 `k=v`。
    验收用的 DAG **不依赖任何 LLM**，全是有确定性的可调用对象，所以"能不能跑通"
    只取决于这一代的图执行语义，与它的模型调用实现无关。

    判据分三层：
    1. **磁盘这一份必须过** —— 它是后面所有代的参考实现；
    2. **每一次元 DAG 运行的最终代必须过** —— 只要那一轮跑到了第 n 代，
       第 n 代就必须能自己产出可用 DAG。中间那些被闸门回退过的废代
       （`r01/r02/r03`，闸门上线前留下的）不计入，但它们**不能是可用的**这件事
       也不影响结论：判的是"每一轮的最后一代"。
    3. **git 式回档链必须完整** —— 每代的 HEAD.json 父指针不断、合并指纹与
       磁盘文件相符（详见下方注释）。
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    from tools._meta_dag_spec import FRAME_FILES

    root = Path(__file__).resolve().parent.parent
    program = _demo_program()

    def _try(dest: Path, tag: str) -> tuple:
        demo = dest / "demo"
        demo.mkdir(parents=True, exist_ok=True)
        (demo / "vself.py").write_text(program + chr(10), encoding="utf-8", newline="")
        p = subprocess.run(
            [sys.executable, "vself.py"], cwd=str(demo), capture_output=True, text=True, timeout=300
        )
        (demo / "vself.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8", newline="")
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        return p.returncode, (tail[-1] if tail else "")

    # 1) 磁盘这一份：参考实现，必须过
    rc, last = _try(root, "disk")
    assert rc == 0, f"磁盘框架无法产出可用 DAG：{last}"

    # 1b) 磁盘这一份还要能**用自己把自己再生成一遍**（不然造不出下一代）。
    #     用 `SmartMockLLM` 跑，不发网络请求 —— 几秒出结果，可进 CI。
    sys.path.insert(0, str(root))
    import vea_v6

    from tools.meta_dag import DESCRIPTION

    r = vea_v6.bootstrap(DESCRIPTION, llm=vea_v6.SmartMockLLM())
    assert r.stable, f"磁盘框架自举不稳定：{r.reason}"
    assert sorted(r.files) == sorted(FRAME_FILES), sorted(r.files)

    # 2) 每一轮的**最后一代**必须过。只认闸门上线之后的轮次：
    #    闸门上线前的 r01/r02/r03 由旧代码生成，连 import 都过不了，
    #    它们已归入 `bootstrap_runs/meta/abandoned/`（保留不删），不作为基准。
    checked = []
    # 🔴 `_` 前缀 = 证据/暂存，不是基准（boot_rollback 的 find_runs 早就认这个
    # 约定）。曾漏了这条：已知污染的作废轮次 `_s13`（gen2 是 gen1 的逐字节
    # 复印件，provider 回退黑洞的产物）被当良民扫描，套件输出里赫然写着
    # `_s13/gen2(ALL OK)`——废代不当基准，更不该以"通过的代"身份出现。
    # 🔴 目录还可能**根本不存在**：`bootstrap_runs/` 是 gitignore 的本地产物，
    #    干净 clone 里一次都没跑过，此时 `.iterdir()` 直接 FileNotFoundError，
    #    下面"没有可验收的代就不判负"那句根本轮不到执行 —— 实测 CI 从 #123
    #    （2026-10-17）起就一直红在这一行，而本地因为有 bootstrap_runs 一直绿。
    meta_dir = root / "bootstrap_runs" / "meta"
    for run in sorted(d for d in (meta_dir.iterdir() if meta_dir.is_dir() else [])
                      if d.is_dir() and d.name != "abandoned"
                      and not d.name.startswith("_")):
        # 🔴 "有 7 个文件"不等于"是一代"：闸门不过的失败尝试同样把 7 个文件
        # 写满了盘（就是回退出来的上一代），只是没有 HEAD.json。加上
        # "chained（有 gen0）就必须有 HEAD.json"的判据后，一次网络瞬断导致的
        # 失败代会让套件对那一轮永久报错。meta_dag 现在给失败代落 FAIL.json，
        # 这里据此把失败尝试排除在"代"之外。
        gens = sorted(
            (d for d in run.glob("gen*")
             if all((d / f).exists() for f in FRAME_FILES)
             and not (d / "FAIL.json").exists()),
            key=lambda d: int(d.name[3:] or 0),
        )
        if not gens:
            continue
        last_gen = gens[-1]
        rc, out = _try(last_gen, last_gen.relative_to(root).as_posix())
        checked.append((last_gen.relative_to(root).as_posix(), out))
        assert rc == 0, f"{last_gen} 无法用自己产出可用 DAG：{out}"
        # 除 gen0（磁盘快照，链根，不生成任何代）与最后一代外，每一代都得留下
        # "我确实被拿去生成了下一代"的凭据：`ORIGIN.json` 写在**生成者自己**的
        # 目录里，`loaded` 必须等于它自己的 vea_v6 —— 否则说明那一代其实没参与，
        # 下一代是别人（或磁盘上那份）造的。
        for g in gens[1:-1]:
            cred_file = g / "ORIGIN.json"
            assert cred_file.exists(), f"{g.name} 没有留下生成凭据，说明下一代不是它造的"
            cred = json.loads(cred_file.read_text(encoding="utf-8"))
            assert Path(cred["loaded"]).resolve() == (g / "vea_v6").resolve(), (
                f"{g.name} 的凭据对不上：loaded={cred['loaded']} 应为 {(g / 'vea_v6')}")
        # git 式回档链（/goal"每步保存 + git 式回档"的硬验收）：
        # 每一代过闸门后都要留下 HEAD.json（父指针 + 逐文件 sha256），且
        #   - parent 必须等于上一代的 combined_sha256（断链 = 回档会回错代）；
        #   - combined_sha256 必须与磁盘上这一代的文件重算值相同（指纹对不上
        #     说明文件被改过，checkout 出来的不是当时那一代）。
        # 怎么判断"这一轮该有 HEAD.json"：新代码在跑第一代之前就写 gen0 快照，
        # 所以 `gen0/HEAD.json` 存在 ⟺ 这一轮由接入回档链的代码所跑。老轮次
        # （含已被 backfill_head.py 补过链的）有就验、没有不判负。
        from tools.oneshot_chain import combined_sha

        chained = (run / "gen0" / "HEAD.json").exists()
        prev_sha = ""
        for g in gens:
            head_file = g / "HEAD.json"
            if not head_file.exists():
                if chained:
                    assert False, f"{g.name} 没有 HEAD.json，不在 git 式回档链上"
                continue
            head = json.loads(head_file.read_text(encoding="utf-8"))
            if prev_sha:
                assert head.get("parent") == prev_sha, (
                    f"{g.name} 的 parent={head.get('parent')} 应等于上一代的 {prev_sha}")
            srcs = {f: (g / f).read_text(encoding="utf-8") for f in FRAME_FILES}
            assert head["combined_sha256"] == combined_sha(srcs), (
                f"{g.name} 的合并指纹与磁盘文件不符（文件被改过？）")
            prev_sha = head["combined_sha256"]
    # `bootstrap_runs/` 是 gitignore 的本地产物：clone 里可能一次都没跑过。
    # 那种情况下"没有可验收的代"不是框架的错，不判负；但只要跑过，就必须每轮都过。
    if not checked:
        print("  （bootstrap_runs/meta 里没有已跑完的轮次，跳过历史代验收；"
              "跑一次 python tools/meta_dag.py --rounds 3 后这里就会逐轮核）")

    print("  通过的代：disk" + "".join(f" / {c[0]}({c[1]})" for c in checked))
    _ = json


def _demo_program() -> str:
    """自选任务 + 自选验收：返回要写进 `vself.py` 的源码。"""
    from tools.meta_dag import DEMO_TEST

    return DEMO_TEST


def test_20_meta_dag_dispatches_eight_steps_without_network() -> None:
    """测试 20：元 DAG 的 8 条边 / 7 层，按层依次触发 8 个 step（不联网）。"""
    from tools.meta_dag import build_meta_dag, run_round, DESCRIPTION, RecordingTool
    from tools._meta_dag_spec import META_DAG

    dag = build_meta_dag()
    probe = RecordingTool()
    dag.mount_tool("gen", probe)
    out = run_round(dag, DESCRIPTION)
    assert probe.calls == [e[1]["参数"]["step"] for e in META_DAG], probe.calls
    assert len(dag.edges) == 8
    assert len({e.layer for e in dag.edges}) == 7
    assert out["framework"] and out["report"] and out["result"]


def test_21_provider_failure_is_not_accepted_as_a_generation() -> None:
    """测试 21：代际闸门 —— provider 挂了或产出不完整，这一代就不算新造的一代。

    背景（s13 实测的闸门漏洞）：gen2 的 provider 子进程撞上网络瞬断
    （`CHILD.log` 里是 `LLMError: peer closed connection`），7 个文件全部走
    "回退上一代"，而回退出来的副本本身合法 —— `assemble_ok=True`、自举
    stable、demo ALL OK，于是被当成新的一代验收；`ORIGIN.json` 没写、
    gen2 与 gen1 逐字节相同，坐实它是复印件。
    判据：report 必须带 `provider_ok=False` 与原因；provider 完整产出时为 True。
    """
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from vea_v6.parts.bootstrap import framework_sources  # noqa: E402

    from tools._meta_gen_tool import GenTool  # noqa: E402
    from tools.meta_dag import DESCRIPTION  # noqa: E402

    class _StubLLM:
        """expand 步的最小桩：返回不可解析内容，`_step_expand` 会回落静态施工单
        （provider 路径不看施工单，所以桩返回什么无关）。"""

        def _post_sync(self, prompt: str, **kw: object) -> str:
            return "{}"

    cfg = {"model": "stub", "api": "stub"}
    disk = framework_sources()
    tmp = root / "bootstrap_runs" / "_test_provider_gate"
    tmp.mkdir(parents=True, exist_ok=True)

    def _drive(tool: GenTool) -> dict:
        payload = tool(DESCRIPTION, step="expand")
        payload = tool(payload, step="plan_module")
        payload = tool(payload, step="normalize")
        payload = tool(payload, step="audit_contract")
        payload = tool(payload, step="supplement")
        payload = tool(payload, step="assemble")
        return json.loads(tool(payload, step="measure"))

    # 1) provider 一调用就抛（= s13 gen2 那个网络瞬断）→ provider_ok=False；
    #    同时 assemble_ok 仍应为 True —— 这正是漏洞所在：只看 assemble_ok
    #    就会把上一代的复印件当成新的一代放行。
    def boom(_desc: str) -> dict:
        raise RuntimeError("peer closed connection without sending complete message body")

    bad = GenTool(_StubLLM(), cfg, run_dir=tmp / "bad", fallback=disk, source_provider=boom)
    rep = _drive(bad)
    assert rep["provider_ok"] is False, rep
    assert "provider 调用失败" in rep["provider_why"], rep
    assert rep["assemble_ok"] is True, "回退副本仍能装配，这道闸门才不是摆设"

    # 2) provider 只返回一部分文件 → 缺的会被 supplement 补成上一代的，
    #    拼出来是 chimera，同样不算这一代造的。
    def partial(_desc: str) -> dict:
        return dict(list(disk.items())[:3])

    part = GenTool(_StubLLM(), cfg, run_dir=tmp / "partial", fallback=disk, source_provider=partial)
    rep2 = _drive(part)
    assert rep2["provider_ok"] is False, rep2
    assert "产出不完整" in rep2["provider_why"], rep2

    # 3) provider 完整产出 → provider_ok=True（不能把正常路径也拦了）
    good = GenTool(_StubLLM(), cfg, run_dir=tmp / "good",
                   fallback=disk, source_provider=lambda _d: dict(disk))
    rep3 = _drive(good)
    assert rep3["provider_ok"] is True, rep3
    assert rep3["assemble_ok"] is True, rep3
    print("  provider 闸门：失败→拦下 / 不完整→拦下 / 完整→放行")


def test_22_source_gate_rejects_a_generation_with_nothing_new() -> None:
    """测试 22：源码来源闸门 —— 7 个文件全由上一代补，就不算新造的一代。

    这是 provider 闸门的**对称漏洞**：第 1 代没有 provider，源码由引导工具
    直接调 LLM 写；`_write_batch` 的异常被吞掉后返回空，7 个文件全由
    fallback（上一代/磁盘框架）补 —— 拼出来是逐字节副本，而 assemble/自举/
    自选任务全绿（副本当然能用），旧代码照样当新一代收。

    阈值为什么是"全补"而不是"不许补"：**补件是常态**，实测 s12 与 s15 的
    gen1 都是 5 个新造 + `dag.py`/`bootstrap.py` 契约没过回退补；"不许补"
    会否掉每一次真实运行（判据必须能在真实输入上为真，也不能否掉真实输入）。

    判据：`source_ok` = 补件数 < 7；`fallback_files` 照实列出。
    """
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from vea_v6.parts.bootstrap import framework_sources  # noqa: E402

    from tools._meta_dag_spec import FRAME_FILES  # noqa: E402
    from tools._meta_gen_tool import GenTool  # noqa: E402
    from tools.meta_dag import DESCRIPTION  # noqa: E402

    class _StubLLM:
        """返回不可解析内容：expand 回落静态施工单；写码批全部产出为空，
        于是 7 个文件只能由上一代补 —— 正是要拦的"零新造"场景。"""

        def _post_sync(self, prompt: str, **kw: object) -> str:
            return "{}"

    cfg = {"model": "stub", "api": "stub"}
    disk = framework_sources()
    tmp = root / "bootstrap_runs" / "_test_source_gate"
    tmp.mkdir(parents=True, exist_ok=True)

    def _payload(ok_flags: list) -> str:
        return json.dumps({"files": [
            {"path": p, "source": (f"# new {p}\n" if ok else ""),
             "ok": ok, "why": ("" if ok else "空产出")}
            for p, ok in zip(FRAME_FILES, ok_flags)
        ]})

    tool = GenTool(_StubLLM(), cfg, run_dir=tmp / "unit", fallback=disk)

    # 1) 全补 → 这一代没有一点新造的源码
    tool._step_supplement(_payload([False] * 7))
    assert tool.source_ok is False, tool.fallback_files
    assert len(tool.fallback_files) == 7, tool.fallback_files
    assert "没有新造的源码" in tool.source_why

    # 2) 5 新造 + 2 补（s12/s15 gen1 的实测形状）→ 必须放行，不能误杀
    tool._step_supplement(_payload([True, True, True, False, True, False, True]))
    assert tool.source_ok is True, tool.fallback_files
    assert len(tool.fallback_files) == 2, tool.fallback_files
    assert tool.source_why == ""

    # 3) 全新造 → 放行
    tool._step_supplement(_payload([True] * 7))
    assert tool.source_ok is True and tool.fallback_files == []

    # 4) 端到端：走完 8 步（无 provider = 第 1 代路径），码批全空 → 全补 → 拦下。
    #    这正是 gen1 的"静默回退"漏洞被闸门抓住的实证。
    def _drive(t: GenTool) -> dict:
        payload = t(DESCRIPTION, step="expand")
        payload = t(payload, step="plan_module")
        payload = t(payload, step="normalize")
        payload = t(payload, step="audit_contract")
        payload = t(payload, step="supplement")
        payload = t(payload, step="assemble")
        return json.loads(t(payload, step="measure"))

    bad = GenTool(_StubLLM(), cfg, run_dir=tmp / "e2e_bad", fallback=disk)
    rep = _drive(bad)
    assert rep["source_ok"] is False, rep
    assert rep["new_files"] == 0, rep

    # 5) 正向对照：provider 完整产出 → 这一代有新造源码（补件可能少量存在）
    good = GenTool(_StubLLM(), cfg, run_dir=tmp / "e2e_good",
                   fallback=disk, source_provider=lambda _d: dict(disk))
    rep2 = _drive(good)
    assert rep2["source_ok"] is True, rep2
    assert rep2["new_files"] > 0, rep2
    print("  source 闸门：全补→拦下 / 5新+2补→放行 / 全新建→放行")


def test_23_head_steps_and_checkout_verification() -> None:
    """测试 23：链上记录只认最后一次尝试；回档必须当场核对 sha256。

    两处都是"有数据却不用"型的问题：
    1. HEAD.json 的 `steps` 曾把**被闸门否掉那次尝试**的快照（01~08）和被接受的
       （09~16）一起记进去 —— 重试让步骤编号续排（s15 gen2 实测 17 项）。
       回档链上那一代的记录因此把失败尝试当成了自己的产出过程。
    2. `checkout` 曾只 copy 不核对，而 HEAD.json 里明明存着每个文件的 sha256；
       它打印的"复核：fixpoint_v3 --check"验的是不动点，不是"回档落对没有"。
    """
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from tools.boot_rollback import verify_against_head  # noqa: E402
    from tools.meta_dag import last_attempt_steps  # noqa: E402

    names = ["expand", "plan_module", "normalize", "audit_contract",
             "supplement", "assemble", "measure", "passthrough"]
    tmp = Path(tempfile.mkdtemp())

    # 1) 有重试（01~08 失败 + 09~16 被接受）→ 只认 09~16，且不混入 REPORT.json
    steps = tmp / "steps"
    steps.mkdir()
    for i in range(1, 3):
        for j, n in enumerate(names):
            (steps / f"{(i - 1) * 8 + j + 1:02d}_{n}").write_text("x", encoding="utf-8")
    (steps / "REPORT.json").write_text("{}", encoding="utf-8")
    got = last_attempt_steps(steps)
    assert len(got) == 8, got
    assert got[0] == "09_expand" and got[-1] == "16_passthrough", got
    assert "01_expand" not in got and "REPORT.json" not in got, got

    # 2) 无重试 → 原样返回 01~08（不能把正常路径也改坏）
    steps2 = tmp / "steps2"
    steps2.mkdir()
    for j, n in enumerate(names):
        (steps2 / f"{j + 1:02d}_{n}").write_text("x", encoding="utf-8")
    got2 = last_attempt_steps(steps2)
    assert len(got2) == 8 and got2[0] == "01_expand", got2

    # 3) 回档核对：逐字相符 → 空；改坏一个字节 → 准确报出（不连坐其它文件）
    gen = root / "bootstrap_runs" / "meta" / "s15" / "gen3"
    if (gen / "HEAD.json").exists():
        import json
        import shutil

        h = json.loads((gen / "HEAD.json").read_text(encoding="utf-8"))
        tgt = Path(tempfile.mkdtemp())
        for p in h["files"]:
            dst = tgt / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gen / p, dst)
        assert verify_against_head(tgt, h) == []
        victim = tgt / sorted(h["files"])[0]
        victim.write_text(victim.read_text(encoding="utf-8") + "\n# 变异\n", encoding="utf-8")
        bad = verify_against_head(tgt, h)
        assert len(bad) == 1 and bad[0][0] == sorted(h["files"])[0], bad
        assert bad[0][1] != bad[0][2], bad
    print("  链记录取最后一次尝试；回档核对 sha256：相符→通过 / 改坏→报出")


if __name__ == "__main__":
    test_1_dag_structure_and_edge_formats()
    test_2_message_structure_with_optional_image_and_gen()
    test_3_preset_edge_fallback_when_edge_has_no_input()
    test_4_external_loop_termination_on_result_tag()
    test_5_dict_literal_initialization_and_direct_list_mutation()
    test_6_meta_dag_builder_builds_runnable_v6_dag()
    test_7_parse_edge_sentences_variants()
    test_8_models_registry_and_layer_ordering()
    test_9_tag_based_fetch_and_reverse()
    test_10_openai_compatible_api_edge()
    test_11_multi_model_registry_switch()
    test_12_bootstrap_dag_generates_framework_code()
    test_13_bootstrap_fixed_point_is_byte_identical()
    test_14_bootstrap_generates_another_framework_from_same_prompt()
    test_15_generated_framework_is_runnable()
    test_16_inline_openai_config_on_edge()
    test_17_registry_key_not_treated_as_model_name()
    test_18_llm_text_extraction_falls_back_to_reasoning()
    test_19_every_generation_runs_a_self_chosen_dag()
    test_20_meta_dag_dispatches_eight_steps_without_network()
    test_21_provider_failure_is_not_accepted_as_a_generation()
    test_22_source_gate_rejects_a_generation_with_nothing_new()
    test_23_head_steps_and_checkout_verification()
    print("ALL 23 VEA V6 TEST SUITES PASSED SUCCESSFULLY!")
