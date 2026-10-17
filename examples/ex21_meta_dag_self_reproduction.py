"""Example 21: Meta-DAG as a Modifier — Modifying a DAG & Creating Another Meta-DAG from 0.

Demonstrates:
1. Using Meta-DAG (meta_dag_1) as a modifier to generate a full business DAG from instructions.
2. Starting from ZERO (a blank canvas DAG with 0 edges/contents) and having Meta-DAG synthesize
   ANOTHER fully functional Meta-DAG (meta_dag_2).
3. Using the newly synthesized Meta-DAG (meta_dag_2) to modify a third DAG (incident_dag).
4. Full recursive self-hosting: Child Meta-DAG reproduces a Grandchild Meta-DAG (meta_dag_3).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import (
    DAG,
    MockLLMClient,
    create_meta_dag,
    transform_dag_with_meta_dag,
)


def stage_1_meta_dag_modifies_business_dag(meta_dag: DAG) -> DAG:
    """Stage 1: Use Meta-DAG as a modifier to generate a multi-stage business DAG."""
    print("\n" + "=" * 70)
    print("📍 [Stage 1] 用元 DAG 作为修改器，修改出一个完整的业务 DAG")
    print("=" * 70)

    # 初始业务 DAG（仅有骨架和初始输入）
    seed_dag = DAG("ecommerce_order_dag", models="models.json", tools="tools.json", llm_client=MockLLMClient())
    seed_dag.add_content("订单 #10086: 购买 2 台高端服务器，单价 $5000", tags=["raw_order"], gen=0)

    # 修改指令：增加两道分层边
    prompt = (
        "添加边: [raw_order] -> (laptop-6.8: 提取商品数量与单价) -> [parsed_order] layer=0; "
        "添加边: [parsed_order] -> (sensenova-6.8: 计算含税总额并生成收据) -> [final_receipt] layer=1"
    )
    print(f"  待注入业务指令: '{prompt}'")

    modified_order_dag = transform_dag_with_meta_dag(
        target_dag=seed_dag,
        prompt=prompt,
        meta_dag=meta_dag,
    )

    print(f"  修改后业务 DAG 边数量: {len(modified_order_dag.edges)}")
    for i, e in enumerate(modified_order_dag.edges):
        print(f"    Edge {i}: {e.input_tags} -> {e.output_tags} (layer={e.layer})")

    # 运行修改后的业务 DAG
    modified_order_dag.run(until_tag="final_receipt", max_steps=5)
    receipt = modified_order_dag.get_result("final_receipt")
    print(f"  业务 DAG 最终产出: {receipt[:80]}...")
    assert receipt is not None
    assert "final_receipt" in modified_order_dag.contents[-1].tags
    print("  ✅ Stage 1 成功！元 DAG 成功修改并实例化业务 DAG。")
    return modified_order_dag


def stage_2_synthesize_meta_dag_from_zero(parent_meta_dag: DAG) -> DAG:
    """Stage 2: Starting from ZERO (blank DAG), synthesize another Meta-DAG."""
    print("\n" + "=" * 70)
    print("📍 [Stage 2] 尝试从 0 (空白图) 修改出另一个元 DAG (Child Meta-DAG)")
    print("=" * 70)

    # 完全从 0 开始：0 条边，0 个初始内容项
    blank_canvas = DAG("meta_dag_v2_from_zero", models="models.json", tools="tools.json", llm_client=MockLLMClient())
    print(f"  初始空白画布: name={blank_canvas.name}, edges={len(blank_canvas.edges)}, contents={len(blank_canvas.contents)}")
    assert len(blank_canvas.edges) == 0

    # 让父元 DAG 根据元图架构指令，将空白图构造为全新元 DAG
    meta_synthesis_prompt = (
        "从0构建元DAG架构: "
        "添加边: [target_dag, prompt] -> (builtin:meta_plan: 解析待修改DAG与Prompt打包为dag_edit_bundle) -> [dag_edit_bundle] layer=0; "
        "添加边: [dag_edit_bundle] -> (builtin:meta_exec: 应用修改指令并验证生成新DAG) -> [modified_dag, result] layer=1"
    )
    print(f"  元图合成指令: '{meta_synthesis_prompt}'")

    child_meta_dag = transform_dag_with_meta_dag(
        target_dag=blank_canvas,
        prompt=meta_synthesis_prompt,
        meta_dag=parent_meta_dag,
    )

    print(f"  新元 DAG 构造完成: name={child_meta_dag.name}, edges={len(child_meta_dag.edges)}")
    for i, e in enumerate(child_meta_dag.edges):
        print(f"    Meta Edge {i}: {e.input_tags} -> {e.output_tags} (processor={e.processor.prompt}, layer={e.layer})")

    assert len(child_meta_dag.edges) == 2
    assert child_meta_dag.edges[0].input_tags == ["target_dag", "prompt"]
    assert child_meta_dag.edges[0].output_tags == ["dag_edit_bundle"]
    assert child_meta_dag.edges[1].input_tags == ["dag_edit_bundle"]
    assert "result" in child_meta_dag.edges[1].output_tags
    print("  ✅ Stage 2 成功！从 0 成功修改并诞生了新一代元 DAG (Child Meta-DAG)！")
    return child_meta_dag


def stage_3_verify_child_meta_dag_in_action(child_meta_dag: DAG) -> DAG:
    """Stage 3: Prove the Child Meta-DAG can independently modify another DAG."""
    print("\n" + "=" * 70)
    print("📍 [Stage 3] 验证新诞生出的元 DAG (Child Meta-DAG) 独立修改第三个业务 DAG")
    print("=" * 70)

    incident_dag = DAG("incident_response_dag", models="models.json", tools="tools.json", llm_client=MockLLMClient())
    incident_dag.add_content("CRITICAL: Redis 集群连接超时数突增", tags=["incident_alert"], gen=0)

    sre_prompt = (
        "添加边: [incident_alert] -> (laptop-6.8: 分析故障等级与受损范围) -> [incident_triage] layer=0; "
        "添加边: [incident_triage] -> (sensenova-6.8: 生成SRE容灾隔离与重启预案) -> [sre_action_plan] layer=1"
    )
    print(f"  由子元 DAG 处理的修改指令: '{sre_prompt}'")

    # 使用新元 DAG 执行修改！
    active_incident_dag = transform_dag_with_meta_dag(
        target_dag=incident_dag,
        prompt=sre_prompt,
        meta_dag=child_meta_dag,
    )

    print(f"  由子元 DAG 产出的业务图边数: {len(active_incident_dag.edges)}")
    active_incident_dag.run(until_tag="sre_action_plan", max_steps=5)
    plan = active_incident_dag.get_result("sre_action_plan")
    print(f"  产出故障处置预案: {plan[:80]}...")
    assert plan is not None
    print("  ✅ Stage 3 成功！新元 DAG 具备完全独立的图修改与构建能力！")
    return active_incident_dag


def stage_4_recursive_self_reproduction(child_meta_dag: DAG) -> DAG:
    """Stage 4: Recursive Quine — Child Meta-DAG reproduces a Grandchild Meta-DAG from 0."""
    print("\n" + "=" * 70)
    print("📍 [Stage 4] 递归自复制：由子元 DAG 从 0 繁衍出孙元 DAG (Grandchild Meta-DAG)")
    print("=" * 70)

    grandchild_blank = DAG("meta_dag_v3_grandchild", models="models.json", tools="tools.json", llm_client=MockLLMClient())
    reproduction_prompt = (
        "从0繁衍第三代元DAG: "
        "添加边: [target_dag, prompt] -> (builtin:meta_plan: 递归元规划) -> [dag_edit_bundle] layer=0; "
        "添加边: [dag_edit_bundle] -> (builtin:meta_exec: 递归元生成) -> [modified_dag, result] layer=1"
    )

    grandchild_meta_dag = transform_dag_with_meta_dag(
        target_dag=grandchild_blank,
        prompt=reproduction_prompt,
        meta_dag=child_meta_dag,
    )

    print(f"  孙元 DAG 繁衍完成: name={grandchild_meta_dag.name}, edges={len(grandchild_meta_dag.edges)}")
    assert len(grandchild_meta_dag.edges) == 2
    assert "result" in grandchild_meta_dag.edges[1].output_tags
    print("  ✅ Stage 4 成功！实现了 Meta-DAG -> Meta-DAG -> Meta-DAG 的完全无损递归自繁衍！")
    return grandchild_meta_dag


def main():
    print("=" * 80)
    print("🚀 开始执行：元 DAG 修改业务图 + 从 0 繁衍新元 DAG 全流程测试")
    print("=" * 80)

    # 1. 创建初代元 DAG (Root Meta-DAG)
    root_meta_dag = create_meta_dag(llm_client=MockLLMClient(), models={"laptop-6.8": {}, "sensenova-6.8": {}})

    # 2. 阶段 1: 用元 DAG 修改业务 DAG
    stage_1_meta_dag_modifies_business_dag(root_meta_dag)

    # 3. 阶段 2: 从 0 繁衍新元 DAG (Child)
    child_meta_dag = stage_2_synthesize_meta_dag_from_zero(root_meta_dag)

    # 4. 阶段 3: 验证子元 DAG 的独立实战能力
    stage_3_verify_child_meta_dag_in_action(child_meta_dag)

    # 5. 阶段 4: 递归繁衍出第三代元 DAG (Grandchild)
    grandchild_meta_dag = stage_4_recursive_self_reproduction(child_meta_dag)

    print("\n" + "=" * 80)
    print("🎉 全部 4 个元图自省与从 0 繁衍测试阶段均 100% 验证通过！完全符合预期！")
    print("=" * 80)


if __name__ == "__main__":
    main()
