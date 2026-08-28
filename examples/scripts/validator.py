"""Vertex subclass: Data validation and reporting aggregation.

Referenced by complex/config.json using ``"script": "../scripts/validator.py"``.
The framework discovers ValidatorVertex and instantiates it.
"""
from framework.vertex import Vertex


class ValidatorVertex(Vertex):
    """on_receive: reject short strings; on_ready: combine into final channel."""

    def on_receive(self, data, channel, settings):
        min_len = settings.get("min_length", 3)
        if isinstance(data, str) and len(data) < min_len:
            raise ValueError(
                f"Data too short ({len(data)} chars, minimum {min_len})"
            )
        return data

    def on_ready(self, all_data, settings):
        parts = []
        for key in sorted(all_data.keys()):
            parts.append(f"[{key}] {all_data[key]}")
        combined = "\n".join(parts) if parts else ""
        return {"final": combined}
