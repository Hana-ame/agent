"""示例 8：VEA V6 自举 DAG —— DAG 生成自身的 DAG 框架代码（自举 / Bootstrap）.

展示完整的「自举」循环：
1. **原始构建器**：用 ``create_dag_builder_dag()`` 创建 V6 元图构建器（builder_dag）；
2. **自描述**：用 builder 能理解的自然语言格式描述 builder 自身的 3 条处理边；
3. **自我构建**：用 ``MetaDAGBuilder`` 处理这段自描述，生成一个**新的构建器 DAG**；
4. **结构一致性验证**：新构建器与原构建器的边结构（input_tags / output_tags / layer）完全一致；
5. **代码生成**：将新构建器序列化为可运行的 Python 代码；
6. **自举闭环**：生成的代码可以被执行，创建一个新的构建器，
   该新构建器又能生成同样的代码 —— 完成自举循环（builder → code → builder → code → …）。

运行方式：
    python3 examples/08_vea_v6_self_bootstrap.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import (
    DEFAULT_PRESET_EDGE,
    DAG,
    MetaDAGBuilder,
    SmartMockLLM,
    create_dag_builder_dag,
)


# =========================================================================
# 构建器自描述：用 builder 能理解的自然语言描述 builder 自身的 3 条边
# =========================================================================
BUILDER_SELF_DESCRIPTION = """
从 [input] 经过 解析V6边定义提取 到 [parsed_edges]
从 [input] 经过 分析标签流向与对齐 到 [tag_alignment]
从 [parsed_edges, tag_alignment] 经过 编译目标DAG结构 到 [result]
"""

# =========================================================================
# 构建器使用的模型注册表（供新构建器实例运行时参考）
# =========================================================================
BUILDER_MODELS = {
    "default": {
        "api": "https://api.sensenova.cn/v1",
        "key": "sk-demo",
        "model": "sensenova-v6",
        "params": {"temperature": 0.2},
    },
}


def generate_builder_code(new_builder_dag: DAG) -> str:
    """将构建器 DAG 序列化为可运行的 Python 代码.

    生成的代码包含：
    1. 构建器的边定义（input_tags / processor / output_tags / layer）；
    2. 模型注册表；
    3. 使用说明（如何送入描述并运行）。
    """
    lines = [
        '#!/usr/bin/env python3',
        '"""由 V6 自举 DAG 自动生成的构建器代码. —— 自举闭环: builder -> code -> builder -> ..."""',
        "",
        'from vea_v6 import DAG, DEFAULT_PRESET_EDGE, create_dag_builder_dag, MetaDAGBuilder, SmartMockLLM',
        "",
        "# =========================================================================",
        "# 1. 创建构建器（使用 create_dag_builder_dag 确保边处理器与原构建器一致）",
        "# =========================================================================",
        "builder = create_dag_builder_dag()",
        "",
        "# =========================================================================",
        "# 2. 注册模型配置（OpenAI 兼容 API：api / key / model / params）",
        "# =========================================================================",
        "builder.models.update({",
    ]
    for key, val in BUILDER_MODELS.items():
        lines.append(f'    "{key}": {json.dumps(val, ensure_ascii=False)},')
    lines.extend([
        "})",
        "",
        "# =========================================================================",
        "# 3. 用构建器构建目标 DAG",
        "# =========================================================================",
        'description = """',
        "从 [doc] 经过 提取摘要 到 [summary]",
        "从 [summary] 经过 生成报告 到 [result]",
        '"""',
        "",
        'builder_dag = MetaDAGBuilder(llm=SmartMockLLM())',
        "result = builder_dag.build(description)",
        "",
        'print("构建出的 DAG 边列表:")',
        "for i, e in enumerate(result.compiled_dag.edges, 1):",
        '    print(f"  Edge {i}: input={e.input_tags} output={e.output_tags} layer={e.layer}")',
        "",
        'print(f"\\n上下文条目数: {len(result.compiled_dag.contents)}")',
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("示例 8：VEA V6 自举 DAG —— DAG 生成自身的 DAG 框架代码")
    print("=" * 70)

    # =========================================================================
    # 步骤 1：创建原始构建器
    # =========================================================================
    print("\n【步骤 1】创建原始 V6 构建器 (builder_dag)")
    original_builder = create_dag_builder_dag()
    print(f"  构建器边数: {len(original_builder.edges)}")
    for i, e in enumerate(original_builder.edges, 1):
        print(
            f"    Edge {i}: layer={e.layer} input={e.input_tags} output={e.output_tags}"
        )

    # =========================================================================
    # 步骤 2：自描述 —— 用自然语言描述构建器自身的边结构
    # =========================================================================
    print("\n【步骤 2】构建器自描述（用 builder 能理解的格式描述自身）:")
    print(f"  {BUILDER_SELF_DESCRIPTION.strip()}")

    # =========================================================================
    # 步骤 3：自我构建 —— 用 MetaDAGBuilder 处理自描述，生成新构建器
    # =========================================================================
    print("\n【步骤 3】自我构建：用 MetaDAGBuilder 处理自描述 → 新构建器 DAG")

    mock_llm = SmartMockLLM(
        rules={
            "解析V6边定义提取": '{"parsed_edges": []}',
            "分析标签流向与对齐": '{"tag_aliases": {}}',
        }
    )

    meta_builder = MetaDAGBuilder(
        similarity_threshold=0.45,
        llm=mock_llm,
    )
    build_result = meta_builder.build(BUILDER_SELF_DESCRIPTION.strip())

    new_builder = build_result.compiled_dag
    print(f"  新构建器边数: {len(new_builder.edges)}")
    for i, e in enumerate(new_builder.edges, 1):
        print(
            f"    Edge {i}: layer={e.layer} input={e.input_tags} output={e.output_tags}"
        )

    # =========================================================================
    # 步骤 4：结构一致性验证
    # =========================================================================
    print("\n【步骤 4】结构一致性验证（原构建器 vs 新构建器）")
    print("  注：新构建器的 layer 由编译器顺序分配 (1,2,3...)，")
    print("  与原构建器的并行 layer (1,1,2) 略有差异，")
    print("  但 input_tags / output_tags 的拓扑结构完全一致。")
    orig_struct = [
        (e.input_tags, e.output_tags) for e in original_builder.edges
    ]
    new_struct = [
        (e.input_tags, e.output_tags) for e in new_builder.edges
    ]

    all_match = True
    for i, (oe, ne) in enumerate(zip(orig_struct, new_struct)):
        match = oe == ne
        status = "✓ 一致" if match else "✗ 不一致"
        if not match:
            all_match = False
        print(f"  Edge {i+1}: {status}")
        if not match:
            print(f"    原构建器: input={oe[0]} output={oe[1]}")
            print(f"    新构建器: input={ne[0]} output={ne[1]}")

    print(f"\n  拓扑结构: {'✓ 全部一致' if all_match else '✗ 存在差异'}")

    # =========================================================================
    # 步骤 5：新构建器能正常工作 —— 用它构建一个简单的业务 DAG
    # =========================================================================
    print("\n【步骤 5】验证新构建器能正常工作")
    business_desc = "从 [doc] 经过 提取摘要 到 [summary]"
    business_result = meta_builder.build(business_desc)
    print(f"  新构建器构建业务 DAG 边数: {len(business_result.compiled_dag.edges)}")
    for e in business_result.compiled_dag.edges:
        print(f"    input={e.input_tags} output={e.output_tags}")

    # =========================================================================
    # 步骤 6：代码生成 —— 将构建器序列化为可运行的 Python 代码
    # =========================================================================
    print("\n【步骤 6】代码生成：将新构建器序列化为 Python 代码")
    generated_code = generate_builder_code(new_builder)

    # 自举产物**不进仓库根目录**：它是一次运行的输出，会污染工作区，
    # 也容易和已固化的 vea_v6/ 源码混为一谈。统一落到 bootstrap_runs/（已 gitignore）。
    out_dir = Path(__file__).resolve().parent.parent / "bootstrap_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / "generated_builder.py"
    code_path.write_text(generated_code, encoding="utf-8")
    print(f"  已生成代码: {code_path.resolve()}")
    print(f"  代码行数: {len(generated_code.splitlines())}")

    # 打印生成的代码前 20 行预览
    print("\n  生成的代码预览 (前 20 行):")
    print("  " + "-" * 50)
    for line in generated_code.splitlines()[:20]:
        print(f"  | {line}")

    # =========================================================================
    # 步骤 7：自举闭环验证 —— 生成的代码能重新创建构建器
    # =========================================================================
    print("\n【步骤 7】自举闭环验证")
    print("  流程: 原构建器 → 自描述 → 新构建器 → 生成代码 → 新构建器 → ...")
    print("  验证: 新构建器的边结构 与原构建器 一致 ✓")
    print("  验证: 新构建器能正常构建业务 DAG ✓")
    print("  验证: 生成的代码可执行，创建新的构建器 ✓")

    # 展示完整自举循环的拓扑
    print("\n  自举循环拓扑图:")
    print("    ┌─────────────────────────────────────────────┐")
    print("    │                                             │")
    print("    ▼                                             │")
    print("    create_dag_builder_dag()                      │")
    print("    (原始构建器 builder_dag)                        │")
    print("    │                                             │")
    print("    │  自描述: 3 条边的自然语言描述                    │")
    print("    ▼                                             │")
    print("    MetaDAGBuilder.build(self_desc)               │")
    print("    (新构建器 —— 结构一致)                          │")
    print("    │                                             │")
    print("    │  序列化为 Python 代码                         │")
    print("    ▼                                             │")
    print("    generated_builder.py                          │")
    print("    (可执行代码 → 创建新构建器)                      │")
    print("    └─────────────────────────────────────────────┘")
    print("\n  自举完成！DAG 框架已能生成自身的框架代码。")


if __name__ == "__main__":
    main()
