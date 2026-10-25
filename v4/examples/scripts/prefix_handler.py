"""Edge script: prefix / suffix handler.

Settings:
    prefix - prepended to the data before the LLM call (default ``[PRE]``)
    suffix - appended to the result after the LLM call (default ``[POST]``)

Must be an Edge subclass. Legacy top-level hook functions (pre_process / post_process)
are deprecated as the orchestration layer has been consolidated into Edge.
If no Edge subclass is defined, load_class_from_script will fall back to base Edge.
"""

from framework.edge import Edge


class PrefixEdge(Edge):
    """Adds configurable prefix/suffix before and after LLM execution."""

    def pre_process(self, data, settings):
        """Prepend prefix before sending to agent."""
        prefix = settings.get("prefix", "[PRE]")
        if isinstance(data, str):
            return f"{prefix} {data}"
        return data

    def post_process(self, result, settings):
        """Append suffix after receiving agent result."""
        suffix = settings.get("suffix", "[POST]")
        if isinstance(result, str):
            return f"{result} {suffix}"
        return result
