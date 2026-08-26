"""Vertex script: data validator.

顶点脚本：数据校验器。
Hooks:  钩子说明：
    on_receive  – rejects strings shorter than 3 characters
                  拒绝长度小于 3 个字符的字符串(长度可通过配置调整)
    on_ready    – merges all received data into (final, (report,))
                  把所有已接收数据整合为 (final, (report,)) 键
"""


def on_receive(data, data_id, tags, settings):
    """Validate incoming data; reject if too short.

    校验接收到的数据；若过短则拒绝。
    最小长度取自 settings["min_length"]，默认 3。
    """
    min_len = settings.get("min_length", 3)
    if isinstance(data, str) and len(data) < min_len:
        raise ValueError(
            f"Data too short ({len(data)} chars, minimum {min_len})"
        )
    return data


def on_ready(all_data, settings):
    """Merge all inputs into a single report output.

    把所有输入合并为一份报告输出。
    """
    parts = []
    for key in sorted(all_data.keys()):
        # 键转换为 "data_id:tag1,tag2" 的标签形式，便于阅读
        label = f"{key[0]}:{','.join(key[1])}"
        parts.append(f"[{label}] {all_data[key]}")

    combined = "\n".join(parts) if parts else ""
    return {("final", ("report",)): combined}
