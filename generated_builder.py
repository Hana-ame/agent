#!/usr/bin/env python3
"""由 V6 自举 DAG 自动生成的构建器代码. —— 自举闭环: builder -> code -> builder -> ..."""

from vea_v6 import DAG, DEFAULT_PRESET_EDGE, create_dag_builder_dag, MetaDAGBuilder, SmartMockLLM

# =========================================================================
# 1. 创建构建器（使用 create_dag_builder_dag 确保边处理器与原构建器一致）
# =========================================================================
builder = create_dag_builder_dag()

# =========================================================================
# 2. 注册模型配置（OpenAI 兼容 API：api / key / model / params）
# =========================================================================
builder.models.update({
    "default": {"api": "https://api.sensenova.cn/v1", "key": "sk-demo", "model": "sensenova-v6", "params": {"temperature": 0.2}},
})

# =========================================================================
# 3. 用构建器构建目标 DAG
# =========================================================================
description = """
从 [doc] 经过 提取摘要 到 [summary]
从 [summary] 经过 生成报告 到 [result]
"""

builder_dag = MetaDAGBuilder(llm=SmartMockLLM())
result = builder_dag.build(description)

print("构建出的 DAG 边列表:")
for i, e in enumerate(result.compiled_dag.edges, 1):
    print(f"  Edge {i}: input={e.input_tags} output={e.output_tags} layer={e.layer}")

print(f"\n上下文条目数: {len(result.compiled_dag.contents)}")
