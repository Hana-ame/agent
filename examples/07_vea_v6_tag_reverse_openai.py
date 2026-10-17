"""示例 7：VEA V6 标签内容处理与 OpenAI 兼容 API 边处理.

展示：
1. **按 tag 获取内容**：从 ``dag.contents`` 中取出带有指定 tag（如 ``"todo"``）的条目，
   对内容做确定性处理（**反转**字符串）后，结果自动保存回 ``dag.contents``（输出 tag = ``"done"``）。
2. **OpenAI 兼容 API 边处理**：通过 ``dag.models`` 注册表配置 OpenAI 兼容 API
   （``api`` / ``key`` / ``model`` / ``params`` 四要素），边里的 ``{"模型": <KEY>, "处理prompt": ...}``
   按 KEY 查询注入 api/key/参数，调用时自动路由到对应端点。
3. **混合确定性边与 LLM 边**：展示 callable（确定性反转）与 prompt（LLM 调用）两种处理方式的组合流转。
4. **上下文完整可追溯**：每一步产出都追加到 ``dag.context``，最终通过 ``tag="result"`` 终止。

运行方式：
    python3 examples/07_vea_v6_tag_reverse_openai.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import DEFAULT_PRESET_EDGE, DAG, SmartMockLLM


def reverse_content(content: str) -> str:
    """将内容字符串反转（确定性处理，无需调用 LLM）。"""
    return content[::-1]


def extract_done_count(content: str) -> str:
    """从反转后的内容中提取任务完成数量（模拟下游确定性处理）。"""
    # 约定格式: "完成了3项" → 提取数字
    import re

    m = re.search(r"(\d+)", content)
    if m:
        return str(int(m.group(1)))
    return "0"


def main() -> None:
    # =========================================================================
    # 1. 创建 DAG —— 注册 OpenAI 兼容模型（api/key/model/params）
    # =========================================================================
    # models 注册表以 model 名为 KEY，保存 api/key/model/params 四要素；
    # 边里的 {"模型": <KEY>, "处理prompt": ...} 在调用 LLM 时按 KEY 查询注入。
    # 此处使用 SmartMockLLM 代替真实 API 调用（零外部依赖），
    # 配置结构与 OpenAICompatibleLLM 完全一致，切换只需替换 llm 参数即可。

    mock_llm = SmartMockLLM(
        rules={
            # 模拟 OpenAI 兼容 API 对摘要请求的响应
            "请总结已完成": lambda p: json.dumps(
                {
                    "summary": "所有待办任务已处理完毕",
                    "status": "complete",
                },
                ensure_ascii=False,
            ),
        }
    )

    dag = DAG(
        preset_edge=DEFAULT_PRESET_EDGE,
        llm=mock_llm,
        models={
            # —— OpenAI 兼容 API 模型配置示例 ——
            "gpt-4o": {
                "api": "https://api.openai.com/v1",
                "key": "sk-your-openai-key",
                "model": "gpt-4o",
                "params": {"temperature": 0.2, "max_tokens": 1024},
            },
            # —— 自定义 OpenAI 兼容端点示例（如本地 Ollama、vLLM、SGLang）——
            "local-qwen": {
                "api": "http://localhost:11434/v1",
                "key": "ollama",
                "model": "qwen2.5:7b",
                "params": {"temperature": 0.0, "max_tokens": 512},
            },
        },
    )

    print("=" * 70)
    print("示例 7：标签内容处理 + OpenAI 兼容 API 边处理")
    print("=" * 70)

    # =========================================================================
    # 2. 添加处理边
    # =========================================================================

    # 边 1（确定性 callable）：从 "todo" 标签获取内容，**反转**后输出到 "done"
    #   这就是"从 edge 当中拿取指定 tag 的内容，处理后保存回 dag.contents"的完整流程：
    #   - 输入 tag = ["todo"] → 自动从 dag.contents 中筛选带 "todo" 标签的条目
    #   - 处理器 = reverse_content → 反转字符串
    #   - 输出 tag = ["done"] → 处理结果自动追加到 dag.contents
    dag.add_edge(["todo"], reverse_content, ["done"], layer=1)

    # 边 2（确定性 callable）：从 "done" 提取完成数量
    dag.add_edge(["done"], extract_done_count, ["done_count"], layer=2)

    # 边 3（LLM prompt + OpenAI 兼容模型）：汇总完成数量，调用 gpt-4o 生成摘要
    #   这里演示了 {"模型": "gpt-4o", "处理prompt": ...} 语法：
    #   - "模型" = "gpt-4o" 按 KEY 查询 dag.models，注入 api/key/model/params
    #   - "处理prompt" = 处理提示词模板，{content} 自动替换为当前条目内容
    dag.add_edge(
        ["done_count"],
        {
            "模型": "gpt-4o",
            "处理prompt": "请总结已完成 {content} 项待办任务的情况，输出 JSON 格式。",
        },
        ["result"],
        layer=3,
    )

    # =========================================================================
    # 3. 送入多条带 "todo" 标签的待办事项
    # =========================================================================
    dag.send({"tags": ["todo"], "content": "完成了3项", "gen": 0})
    dag.send({"tags": ["todo"], "content": "完成了5项", "gen": 0})

    # 送入一条 "note" 标签的备注（不会被 "todo" 边处理，展示 tag 过滤效果）
    dag.send({"tags": ["note"], "content": "这是一条备注，不是待办", "gen": 0})

    print("\n【步骤 1】初始化 DAG 后的 contents:")
    for i, item in enumerate(dag.context):
        print(f"  [{i}] gen={item['gen']} tags={item['tags']} content={item['content']}")

    # =========================================================================
    # 4. 运行 DAG，观察从 "todo" → "done" → "done_count" → "result" 的完整流转
    # =========================================================================
    step = 0
    while not dag.has_result():
        step += 1
        new_items = dag.run()
        print(f"\n【步骤 2】第 {step} 次 dag.run() 新增 {len(new_items)} 条:")
        for item in new_items:
            print(f"  + gen={item['gen']} tags={item['tags']} content={item['content']}")

    # =========================================================================
    # 5. 最终结果
    # =========================================================================
    print("\n【步骤 3】最终 dag.context 全量上下文:")
    for i, item in enumerate(dag.context):
        print(f"  [{i}] gen={item['gen']} tags={item['tags']} content={item['content']}")

    print(f"\n【最终结果 tag=result】: {dag.get_result()}")

    # =========================================================================
    # 6. 验证：反转后的内容确实保存回了 dag.contents
    # =========================================================================
    done_items = dag.find_by_tag("done")
    print(f"\n【验证】tag=done 的条目（反转后保存回 dag.contents）:")
    for item in done_items:
        print(f"  反转后: {item['content']}")

    # 验证反转逻辑正确性
    original_todos = [
        c for c in dag.context
        if "todo" in c["tags"] and not c.get("_preset_synthesized")
    ]
    reversed_correct = all(
        c["content"] == orig["content"][::-1]
        for c in done_items
        for orig in original_todos
        if c["content"] == orig["content"][::-1]
    )
    print(f"反转处理验证: {'✓ 通过' if reversed_correct else '✗ 失败'}")


if __name__ == "__main__":
    main()
