"""Edge script: prefix / suffix handler.

Hooks:
    pre_process   – prepends ``settings["prefix"]`` to string data
    post_process  – appends ``settings["suffix"]``  to string data
"""


def pre_process(data, settings):
    """Add a configurable prefix before PI Agent processing."""
    prefix = settings.get("prefix", "[PRE]")
    if isinstance(data, str):
        return f"{prefix} {data}"
    return data


def post_process(data, settings):
    """Add a configurable suffix after PI Agent processing."""
    suffix = settings.get("suffix", "[POST]")
    if isinstance(data, str):
        return f"{data} {suffix}"
    return data
