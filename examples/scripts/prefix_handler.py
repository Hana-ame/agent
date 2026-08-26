"""Edge script: prefix / suffix handler.

边脚本：前缀 / 后缀处理器。
Hooks:  钩子说明：
    pre_process   – prepends ``settings["prefix"]`` to string data
                    在字符串数据前加上 ``settings["prefix"]`` 前缀
    post_process  – appends ``settings["suffix"]``  to string data
                    在字符串数据后追加 ``settings["suffix"]`` 后缀
"""


def pre_process(data, settings):
    """Add a configurable prefix before PI Agent processing.

    在 PI Agent 处理前给数据加上可配置的前缀。
    """
    prefix = settings.get("prefix", "[PRE]")
    if isinstance(data, str):
        return f"{prefix} {data}"
    return data


def post_process(data, settings):
    """Add a configurable suffix after PI Agent processing.

    在 PI Agent 处理后给数据加上可配置的后缀。
    """
    suffix = settings.get("suffix", "[POST]")
    if isinstance(data, str):
        return f"{data} {suffix}"
    return data
