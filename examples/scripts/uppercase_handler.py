"""Vertex script: uppercase handler.

顶点脚本：大写处理。
Hooks:  钩子说明：
    on_receive  – uppercases string data on arrival
                  数据到达时把字符串转为大写
    on_ready    – merges all stored data into a single (result, (analysis,)) key
                  把所有已存数据整合为单一的 (result, (analysis,)) 键
"""


def on_receive(data, data_id, tags, settings):
    """Convert incoming string data to uppercase.

    把接收到的字符串数据转为大写。
    """
    if isinstance(data, str):
        return data.upper()
    return data


def on_ready(all_data, settings):
    """Consolidate all received data into a single output key.

    把所有已接收数据整合为一个输出键。

    Returns a dict of ``{(data_id, (tags,)): value}`` that will be
    merged into the vertex's data store before outgoing edges fire.

    返回形如 ``{(data_id, (tags,)): value}`` 的字典，
    会在触发输出边之前被合并进顶点的数据存储。
    """
    # 把各条数据用 " | " 连接成一个大字符串
    parts = []
    for key in sorted(all_data.keys()):
        val = all_data[key]
        if isinstance(val, str):
            parts.append(val)
        else:
            parts.append(str(val))

    combined = " | ".join(parts) if parts else ""
    return {("result", ("analysis",)): combined}
