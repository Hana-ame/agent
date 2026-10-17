"""元 DAG 的固定形态（规格文件，不含执行代码）。

这里只放两样东西：
1. `META_DAG`：元 DAG 的 edges 定义（`(inputs, processor, outputs, layer)` 四元组列表）；
2. `SOURCE_TASKS`：每个文件"该怎么让模型重写"的自然语言指令模板。

为什么要单独一个文件：这个 DAG 会被 `tools/dag_selfgen.py` 实例化，也会被
`tests/test_meta_dag_contract.py` 断言形状（步骤数、层级、透传边），
把形状写在一处才不会两处漂移。
"""

from __future__ import annotations

#: 框架的 7 个文件（与 `vea_v6.parts.bootstrap.FRAME_FILES` 同序）
FRAME_FILES = (
    "vea_v6/parts/__init__.py",
    "vea_v6/parts/primitives.py",
    "vea_v6/parts/llm.py",
    "vea_v6/parts/dag.py",
    "vea_v6/parts/builder.py",
    "vea_v6/parts/bootstrap.py",
    "vea_v6/__init__.py",
)

#: 元 DAG 的边。**9 条边 / 7 个真实步骤**，不是"一条 gen 边 + 一条透传边"。
#:
#: layer 0  input       ──expand──>        briefs      （1 次 LLM：描述 → 逐文件施工单）
#: layer 1  briefs      ──plan_module────>  module      （N 次 LLM，可并发：每文件一份源码）
#: layer 1  module      ──normalize──────>  part        （确定性：剥围栏 / 去过短产出）
#: layer 2  part        ──audit_contract─>  part_ok     （确定性契约闸门：语法/符号/体积）
#: layer 3  part_ok     ──supplement─────>  part_full   （多输入汇聚：缺的用上一代补）
#: layer 4  part_full   ──assemble───────>  framework   （确定性：解固定点，补清单）
#: layer 5  framework   ──measure────────>  report      （确定性：字节数/文件数/闸门结论）
#: layer 6  framework   ──passthrough────>  result      （透传：给图外的人读）
#:
#: 工具名在 `processor` 里，工具实体挂在 `DAG.tools` 槽位上（第 N 代之后
#: `gen` 会被换成"上一代框架自己的生成 DAG"，这就是"框架造框架"）。
META_DAG = [
    # (输入 tag, 处理方式, 输出 tag, layer)
    (["input"], {"工具": "gen", "参数": {"step": "expand"}}, "briefs", 0),
    (["briefs"], {"工具": "gen", "参数": {"step": "plan_module"}}, "module", 1),
    (["module"], {"工具": "gen", "参数": {"step": "normalize"}}, "part", 1),
    (["part"], {"工具": "gen", "参数": {"step": "audit_contract"}}, "part_ok", 2),
    (["part_ok"], {"工具": "gen", "参数": {"step": "supplement"}}, "part_full", 3),
    (["part_full"], {"工具": "gen", "参数": {"step": "assemble"}}, "framework", 4),
    (["framework"], {"工具": "gen", "参数": {"step": "measure"}}, "report", 5),
    (["framework"], {"工具": "gen", "参数": {"step": "passthrough"}}, "result", 6),
]

#: 每个文件的施工单指令（`expand` 步骤只产指令，不产码；码在 `plan_module` 产）
SOURCE_TASKS = {
    "vea_v6/parts/__init__.py": "这是一个内部包入口，只从同目录模块 re-export 公共符号，并给出 __all__。",
    "vea_v6/parts/primitives.py": "这里放数据结构：带标签的内容条目、边（输入 tag / 处理函数 / 输出 tag / 层号）、可选 LLM 协议。",
    "vea_v6/parts/llm.py": "这里放 OpenAI 兼容客户端（流式、可指定 base_url/api key、失败重试）与本地假模型。",
    "vea_v6/parts/dag.py": "这里放 DAG 引擎：加边、送条目、跑一轮、找 result 标签、工具槽位挂载、缺输入时自动补齐。",
    "vea_v6/parts/builder.py": "这里把自然语言描述解析成边定义，并给出一个可运行的建图 DAG。",
    "vea_v6/parts/bootstrap.py": "这里放自举：SPEC、清单、固定点求解、把整份框架重新渲染出来、自复现校验。",
    "vea_v6/__init__.py": "这是包的公开入口（SPEC 文件），只有 __all__ 与 re-export。",
}
