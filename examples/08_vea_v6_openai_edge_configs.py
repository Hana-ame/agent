"""示例 8：三种 OpenAI 兼容 API 边的配置方式（注册表 / 内联 / 混合）.

展示 ``vea_v6`` 里「有 model、有 key、有参数」的三种写法，并说明优先级：

| 写法 | 边的 processor | 凭据来源 | 适用场景 |
|---|---|---|---|
| A 注册表 | ``{"模型": "laptop-6.8", "处理prompt": ...}`` | ``models.json`` / ``DAG(models=...)`` | 同一端点被多条边复用 |
| B 内联 | ``{"model": ..., "api": ..., "key": ..., "params": {...}, "处理prompt": ...}`` | 直接写在边里 | 一次性调用、或多端点混用 |
| C 混合 | ``{"模型": "laptop-6.8", "key": "sk-...", "参数": {...}}`` | 注册表打底 + 边内覆盖 | 换 key / 调温度而不复制整份配置 |

优先级：**内联 > 注册表**。注册表项来自 ``vea_v6/models.json``（存在则自动加载），
也可以用 ``DAG(models={...})`` 显式传入；没有任何注册表时纯内联即可工作。

运行方式：
    python3 examples/08_vea_v6_openai_edge_configs.py

    # 真实调用（需要网络 + 有效 key）
    VEA_REAL=1 python3 examples/08_vea_v6_openai_edge_configs.py

默认只做**离线可验**的部分：解析配置、检查解析结果，不发起网络请求。
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v6 import DEFAULT_PRESET_EDGE, DAG, OpenAICompatibleLLM, load_models_file


def show(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    REGISTRY_KEY = "laptop-6.8"

    login = load_models_file()
    show("0. 模型注册表（vea_v6/models.json）")
    if login:
        for k, v in login.items():
            # 只打印端点与模型名，不打印 key
            print(f"  {k}: api={v.get('api')} model={v.get('model')} key=***")
    else:
        print("  未找到 models.json（不影响内联写法）")

    # -------------------------------------------------------------------------
    # A. 注册表写法：边里只写 KEY
    # -------------------------------------------------------------------------
    show("A. 注册表写法 —— 边里只引用 KEY")
    dag_a = DAG(preset_edge=DEFAULT_PRESET_EDGE, llm=OpenAICompatibleLLM(), models=login)
    edge_a = dag_a.add_edge(
        ["input"],
        {"模型": REGISTRY_KEY, "处理prompt": "只回复一个词：{content}"},
        ["out"],
        layer=1,
    )
    print(f"  边 processor = {edge_a.processor}")
    print(f"  inline_llm_config = {edge_a.inline_llm_config}  （None ⇒ 走注册表）")
    if login.get(REGISTRY_KEY):
        cfg = dag_a._resolve_model_config(REGISTRY_KEY, edge=edge_a)
        print(f"  解析结果: model={cfg.get('model')} base_url={cfg.get('base_url')} key=*** "
              f"params={cfg.get('params')}")
        assert cfg.get("model") == login[REGISTRY_KEY]["model"], "注册表 KEY 不应被当成模型名"
        print("  ✓ 注册表解析正确：KEY 查表后注入真实 model 名（不会把 KEY 发给端点）")

    # -------------------------------------------------------------------------
    # B. 内联写法：model / key / params 全写在边里
    # -------------------------------------------------------------------------
    show("B. 内联写法 —— 边的 processor 直接带 api/key/model/params")
    dag_b = DAG(preset_edge=DEFAULT_PRESET_EDGE, llm=OpenAICompatibleLLM(), models={})
    edge_b = dag_b.add_edge(
        ["input"],
        {
            "api": "https://api.openai.com/v1",
            "key": "sk-inline-demo-key",
            "model": "gpt-4o-mini",
            "params": {"temperature": 0.0, "max_tokens": 64},
            "处理prompt": "只回复一个词：{content}",
        },
        ["out"],
        layer=1,
    )
    print(f"  inline_llm_config = {json.dumps(edge_b.inline_llm_config, ensure_ascii=False)}")
    cfg_b = dag_b._resolve_model_config(None, edge=edge_b)
    print(f"  解析结果: base_url={cfg_b.get('base_url')} model={cfg_b.get('model')} "
          f"key={'***' if cfg_b.get('api_key') else None} params={cfg_b.get('params')}")
    assert cfg_b.get("model") == "gpt-4o-mini"
    assert cfg_b.get("base_url", "").startswith("https://api.openai.com")
    assert cfg_b.get("api_key") == "sk-inline-demo-key"
    assert cfg_b.get("params", {}).get("max_tokens") == 64
    print("  ✓ 内联解析正确：无注册表也能独立工作")

    # -------------------------------------------------------------------------
    # C. 混合写法：注册表打底，边内只覆盖 key / 参数
    # -------------------------------------------------------------------------
    show("C. 混合写法 —— 注册表打底 + 边内覆盖（内联优先）")
    dag_c = DAG(preset_edge=DEFAULT_PRESET_EDGE, llm=OpenAICompatibleLLM(), models=login)
    edge_c = dag_c.add_edge(
        ["input"],
        {"模型": REGISTRY_KEY, "参数": {"temperature": 0.9}, "处理prompt": "{content}"},
        ["out"],
        layer=1,
    )
    cfg_c = dag_c._resolve_model_config(REGISTRY_KEY, edge=edge_c)
    base = login.get(REGISTRY_KEY, {})
    print(f"  注册表 temperature = {base.get('params', {}).get('temperature')}")
    print(f"  边内覆盖 temperature = 0.9 → 解析结果 {cfg_c.get('params', {}).get('temperature')}")
    print(f"  model 仍取注册表真实模型名: {cfg_c.get('model')}")
    assert cfg_c.get("params", {}).get("temperature") == 0.9
    assert cfg_c.get("model") == base.get("model")
    print("  ✓ 覆盖生效，且没有把注册表 KEY 误当模型名")

    # -------------------------------------------------------------------------
    # D. 真实调用（可选）
    # -------------------------------------------------------------------------
    show("D. 真实调用（设置 VEA_REAL=1 才执行）")
    if os.environ.get("VEA_REAL") != "1":
        print("  已跳过：export VEA_REAL=1 可发起真实请求")
    else:
        dag = DAG(preset_edge=DEFAULT_PRESET_EDGE, llm=OpenAICompatibleLLM(), models=login)
        dag.add_edge(
            ["input"],
            {"模型": REGISTRY_KEY, "处理prompt": "复述下面这个词，不要加任何其它字符：{content}"},
            ["result"],
            layer=1,
        )
        dag.send({"tags": ["input"], "content": "PONG", "gen": 0})
        while not dag.has_result():
            dag.run()
        print(f"  tag=result: {dag.get_result()!r}")

    show("完成")


if __name__ == "__main__":
    main()
