"""Vertex 子类：大写转换 + 汇聚。

complex/config.json 的 transform 节点用 ``"script": "../scripts/uppercase_handler.py"``
指向本文件，框架自动发现 UpperVertex 并实例化。

发现背景：原为模块级函数 on_receive/on_ready，靠 Vertex.set_pipeline_module
挂载——但 Vertex 从未实现 module 委托，该路径加载即 AttributeError（死代码）。
随框架统一转纯子类覆盖。on_ready 返回值也从旧的 (data_id,(tags,)) 元组 key
改为现模型的 str channel。
"""
from framework.vertex import Vertex


class UpperVertex(Vertex):
    """on_receive: 字符串大写；on_ready: 汇聚所有数据为 result channel。"""

    def on_receive(self, data, channel, settings):
        if isinstance(data, str):
            return data.upper()
        return data

    def on_ready(self, all_data, settings):
        parts = []
        for key in sorted(all_data.keys()):
            val = all_data[key]
            parts.append(val if isinstance(val, str) else str(val))
        combined = " | ".join(parts) if parts else ""
        return {"result": combined}
