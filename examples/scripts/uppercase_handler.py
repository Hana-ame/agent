"""Vertex script: uppercase handler.

Hooks:
    on_receive  – uppercases string data on arrival
    on_ready    – merges all stored data into a single (result, (analysis,)) key
"""


def on_receive(data, channel, settings):
    """Convert incoming string data to uppercase."""
    if isinstance(data, str):
        return data.upper()
    return data


def on_ready(all_data, settings):
    """Consolidate all received data into a single output key.

    Returns a dict of ``{(data_id, (tags,)): value}`` that will be
    merged into the vertex's data store before outgoing edges fire.
    """
    parts = []
    for key in sorted(all_data.keys()):
        val = all_data[key]
        if isinstance(val, str):
            parts.append(val)
        else:
            parts.append(str(val))

    combined = " | ".join(parts) if parts else ""
    return {("result", ("analysis",)): combined}
