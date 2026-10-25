"""Vertex subclass: Uppercase conversion and aggregation.

Referenced by complex/config.json using ``"script": "../scripts/uppercase_handler.py"``.
The framework discovers UpperVertex and instantiates it.
"""
from framework.vertex import Vertex


class UpperVertex(Vertex):
    """on_receive: uppercase strings; on_ready: combine all data into result channel."""

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
