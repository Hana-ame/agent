import re
from typing import Tuple
def fix_deepseek(reasoning_buffer: str, content_buffer: str) -> Tuple[str, str]:
    if content_buffer == "":
        content_buffer, reasoning_buffer = reasoning_buffer, content_buffer
    return (
        reasoning_buffer.replace("<｜end▁of▁thinking｜>", "\n"),
        content_buffer.replace("<｜end▁of▁thinking｜>", "\n"),
    )
class DeepSeekParser:
    def __init__(self):
        self.fragment_types = []
        self.reasoning_buffer = []
        self.content_buffer = []
        self.is_done = False
    def on_message(self, data: dict) -> bool:
        if data.get("done") is True:
            self.is_done = True
            return True
        path = data.get("p", "")
        value = data.get("v")
        operation = data.get("o", "")
        if operation == "BATCH" and path == "response":
            items = value if isinstance(value, list) else []
            for item in items:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_done = True
                    return True
        if isinstance(value, dict) and "response" in value:
            fragments = value["response"].get("fragments", [])
            for frag in fragments:
                self._append_frag(frag.get("type"), frag.get("content", ""))
        elif operation == "APPEND" and path == "response/fragments":
            for frag in value:
                self._append_frag(frag.get("type"), frag.get("content", ""))
        else:
            match = re.match(r"response/fragments/(-?\d+)/content", path)
            if match and isinstance(value, str):
                idx = int(match.group(1))
                real_idx = idx if idx >= 0 else len(self.fragment_types) + idx
                if 0 <= real_idx < len(self.fragment_types):
                    self._append_content(self.fragment_types[real_idx], value)
            elif not path and isinstance(value, str) and self.fragment_types:
                self._append_content(self.fragment_types[-1], value)
        return self.is_done
    def get_result(self) -> Tuple[str, str]:
        return fix_deepseek(
            "".join(self.reasoning_buffer), "".join(self.content_buffer)
        )
    def _append_frag(self, raw_type: str, content: str):
        self.fragment_types.append(raw_type)
        self._append_content(raw_type, content)
    def _append_content(self, raw_type: str, content: str):
        if not content:
            return
        print(content, end="", flush=True)
        if raw_type == "THINK":
            self.reasoning_buffer.append(content)
        else:
            self.content_buffer.append(content)
