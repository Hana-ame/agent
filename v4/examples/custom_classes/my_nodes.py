from framework.vertex import Vertex
from framework.edge import Edge
import logging

logger = logging.getLogger("custom_nodes")

class SafeFilterVertex(Vertex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_strict = self.settings.get("strict_mode", False)
        logger.info(f"[{self.id}] Initialized with strict_mode={self.is_strict}")

    # Signature is unified to (self, data, channel, settings).
    def on_receive(self, data, channel, settings):
        logger.info(f"[{self.id}] Custom on_receive() intercepted data: {data}")

        # Example validation logic
        if self.is_strict and "forbidden" in str(data).lower():
            raise ValueError("Strict mode blocks 'forbidden' keyword!")

        modified_data = f"{data} [VERIFIED]"
        return modified_data


class PrefixEdge(Edge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = self.settings.get("prefix_tag", "[EDGE]")

    def pre_process(self, data, settings):
        """Intercepts data BEFORE sending to the PI Agent (Model)"""
        logger.info(f"[Edge:{self.id}] pre_process adding prefix.")
        return f"{self.prefix} {data}"

    def post_process(self, result, settings):
        """Cleans up data AFTER receiving from the PI Agent (Model)"""
        logger.info(f"[Edge:{self.id}] post_process cleaning result.")
        return str(result).upper()
