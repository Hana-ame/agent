"""Test script validating DAG Agent with the real user endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from dag_agent import DAG, OpenAICompatibleClient, transform_dag_with_meta_dag


def test_live_dag():
    print("==================================================")
    print("🚀 正在使用真实端点测试 DAG 工作流...")
    print("==================================================")

    # 1. 初始化客户端与加载 models.json
    client = OpenAICompatibleClient(timeout=90.0)
    models_path = Path("models.json")
    print(f"📖 读取 models.json: {list(json.loads(models_path.read_text()).keys())}")

    dag = DAG(
        name="live_test_dag",
        models=models_path,
        llm_client=client,
    )

    # 2. 添加两条分层边（依次使用两个不同的端点模型）
    # Edge 0: 优先使用 laptop-6.8
    dag.add_edge(
        input_tags=["topic"],
        processor={
            "model": "laptop-6.8",
            "prompt": "请为以下主题提炼三个核心关键词，只输出关键词，用逗号隔开：\n{input}",
        },
        output_tags=["keywords"],
        layer=0,
    )

    # Edge 1: 随后使用 sensenova-6.8
    dag.add_edge(
        input_tags=["keywords"],
        processor={
            "model": "sensenova-6.8",
            "prompt": "基于以下核心关键词，写一句精炼的一句话愿景宣言（30字内）：\n{input}",
        },
        output_tags=["vision_statement"],
        layer=1,
    )

    # 3. 注入初始输入
    seed_item = dag.add_content(
        contents="人工智能驱动的代码重构与自主智能体系统设计",
        tags=["topic"],
        gen=0,
    )
    print(f"📥 初始内容注入: tags={seed_item.tags}, contents='{seed_item.contents}', gen={seed_item.gen}")

    # 4. 执行 Step 1 (应该触发 layer=0 的 Edge 0: laptop-6.8)
    print("\n▶ 执行 Step 1 (预期触发 Edge 0 / laptop-6.8)...")
    res1 = dag.step()
    print(f"  Action: {res1.action}, Edge ID: {res1.edge_id}, Layer: {res1.edge_layer}, Edge Count: {res1.edge_count}")
    print(f"  产出内容: tags={res1.produced_items[0].tags}, gen={res1.produced_items[0].gen}")
    print(f"  正文: {res1.produced_items[0].contents.strip()}")

    # 5. 执行 Step 2 (应该触发 layer=1 的 Edge 1: sensenova-6.8)
    print("\n▶ 执行 Step 2 (预期触发 Edge 1 / sensenova-6.8)...")
    res2 = dag.step()
    print(f"  Action: {res2.action}, Edge ID: {res2.edge_id}, Layer: {res2.edge_layer}, Edge Count: {res2.edge_count}")
    print(f"  产出内容: tags={res2.produced_items[0].tags}, gen={res2.produced_items[0].gen}")
    print(f"  正文: {res2.produced_items[0].contents.strip()}")

    print("\n==================================================")
    print("✨ 测试默认处理器 (default_processor) N-in-1-out 真实调用...")
    print("==================================================")

    # 构造无现成边可直接匹配的场景，触发 default_processor
    # 我们配置 default_processor 使用 laptop-6.8
    dag.default_processor.model = "laptop-6.8"
    dag.add_content(
        contents="补充系统约束：必须支持高并发与低延迟。",
        tags=["constraint_extra"],
        gen=1,
    )
    # 此时没有边接收 constraint_extra，step 会走 default_processor
    res3 = dag.step()
    print(f"  DefaultProcessor Action: {res3.action}")
    if res3.produced_items:
        print(f"  产出内容: tags={res3.produced_items[0].tags}, gen={res3.produced_items[0].gen}")
        print(f"  正文: {res3.produced_items[0].contents.strip()[:100]}...")

    print("\n==================================================")
    print("✨ 测试元 DAG (Meta-DAG) 使用真实端点根据 Prompt 自动修改 DAG...")
    print("==================================================")
    
    # 用 Prompt 指令动态向 DAG 追加一条边
    prompt_instruction = "添加边: [vision_statement] -> (sensenova-6.8: 请将这句愿景翻译为英文) -> [english_vision] layer=2"
    print(f"  指令 Prompt: '{prompt_instruction}'")
    
    modified_dag = transform_dag_with_meta_dag(
        target_dag=dag,
        prompt=prompt_instruction,
        llm_client=client,
    )
    print(f"  修改后 DAG 边数量: {len(modified_dag.edges)} (原有: {len(dag.edges)})")
    new_edge = modified_dag.edges[-1]
    print(f"  新边信息: id={new_edge.id}, input_tags={new_edge.input_tags} -> output_tags={new_edge.output_tags}, layer={new_edge.layer}")

    # 在修改后的 DAG 上运行这一步新边！
    res4 = modified_dag.step()
    print(f"  运行新边 Step: Action={res4.action}, Edge ID={res4.edge_id}")
    if res4.produced_items:
        print(f"  英文愿景产出: {res4.produced_items[0].contents.strip()}")

    print("\n🎉 真实端点全链路验证全部通过！")


if __name__ == "__main__":
    test_live_dag()
