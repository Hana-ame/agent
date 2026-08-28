"""Vertex 子类：数据校验 + 汇聚成报告。

complex/config.json 的 merge 节点用 ``"script": "../scripts/validator.py"``
指向本文件，框架自动发现 ValidatorVertex 并实例化。

发现背景：原为模块级函数，靠 Vertex.set_pipeline_module 挂载（死代码，
加载即 AttributeError）。随框架统一转纯子类覆盖。on_ready 的 key 也从
旧的 (data_id,(tags,)) 元组改为现模型 str channel。
"""
from framework.vertex import Vertex


class ValidatorVertex(Vertex):
    """on_receive: 拒绝过短字符串；on_ready: 汇聚成 final channel。"""

    def on_receive(self, data, channel, settings):
        min_len = settings.get("min_length", 3)
        if isinstance(data, str) and len(data) < min_len:
            raise ValueError(
                f"Data too short ({len(data)} chars, minimum {min_len})"
            )
        return data

    def on_ready(self, all_data, settings):
        parts = []
        for key in sorted(all_data.keys()):
            parts.append(f"[{key}] {all_data[key]}")
        combined = "\n".join(parts) if parts else ""
        return {"final": combined}
