"""Vertex script: data validator.

Hooks:
    on_receive  – rejects strings shorter than 3 characters
    on_ready    – merges all received data into (final, (report,))
"""


def on_receive(data, data_id, tags, settings):
    """Validate incoming data; reject if too short."""
    min_len = settings.get("min_length", 3)
    if isinstance(data, str) and len(data) < min_len:
        raise ValueError(
            f"Data too short ({len(data)} chars, minimum {min_len})"
        )
    return data


def on_ready(all_data, settings):
    """Merge all inputs into a single report output."""
    parts = []
    for key in sorted(all_data.keys()):
        label = f"{key[0]}:{','.join(key[1])}"
        parts.append(f"[{label}] {all_data[key]}")

    combined = "\n".join(parts) if parts else ""
    return {("final", ("report",)): combined}
