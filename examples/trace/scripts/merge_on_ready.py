"""merge 顶点的 on_ready 钩子：把各来源边的数据按【tag】 data 拼接成主数据。"""


def on_ready(all_data, settings):
    """all_data = {(来源边ID, tags): value}，只做【tag】 data 拼接。"""
    parts = []
    for key in sorted(all_data.keys()):
        _edge_id, tags = key
        tag = ",".join(tags) if tags else ""
        parts.append(f"【{tag}】 {all_data[key]}")
    return "\n".join(parts) if parts else ""
