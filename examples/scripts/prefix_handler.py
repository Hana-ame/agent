"""Edge script: prefix / suffix handler.

Settings:
    prefix - prepended to the data before the LLM call (default ``[PRE]``)
    suffix - appended to the result after the LLM call (default ``[POST]``)

⚠️ 必须是 Edge 子类。旧的 hook 写法（在模块顶层写 pre_process / post_process
函数、由 Pipeline 层委托调用）已经失效——Pipeline 编排层已并入 Edge，脚本
不再被当模块委托。脚本里没有 Edge 子类时 load_class_from_script 会静默降级
成裸 Edge，本文件的前后缀就悄悄不生效了（complex demo 的 [ANALYZED] 前缀
就是这样丢掉的）。
"""

from framework.edge import Edge


class PrefixEdge(Edge):
    """在 LLM 调用前后给数据加上可配置的前缀 / 后缀。"""

    def pre_process(self, data, settings):
        """发给 PI Agent 之前加前缀。"""
        prefix = settings.get("prefix", "[PRE]")
        if isinstance(data, str):
            return f"{prefix} {data}"
        return data

    def post_process(self, result, settings):
        """拿到 PI Agent 结果之后加后缀。"""
        suffix = settings.get("suffix", "[POST]")
        if isinstance(result, str):
            return f"{result} {suffix}"
        return result
