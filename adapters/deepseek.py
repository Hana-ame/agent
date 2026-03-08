# [START] DEEPSEEK-PKG
# version: 001
# 上下文：模块装载起点。先决调用：无。后续调用：DEEPSEEK-UTIL。
# 输入参数：无
# 输出参数：无
import re
from typing import Tuple
# [END] DEEPSEEK-PKG

# [START] DEEPSEEK-UTIL
# version: 001
# 上下文：最终格式化提取模型输出时调用。先决调用：数据流接收完毕。后续调用：由 DEEPSEEK-GET-RESULT 封装返回。
# 输入参数：reasoning_buffer (str), content_buffer (str)
# 输出参数：reasoning (str), content (str)
def fix_deepseek(reasoning_buffer: str, content_buffer: str) -> Tuple[str, str]:
    if content_buffer == "":
        content_buffer, reasoning_buffer = reasoning_buffer, content_buffer
    return reasoning_buffer.replace("<｜end▁of▁thinking｜>", "\n"), content_buffer.replace("<｜end▁of▁thinking｜>", "\n")
# [END] DEEPSEEK-UTIL

# [START] DEEPSEEK-PARSER
# version: 001
# 上下文：被上层 WS-ADAPTER 实例化以解析单次对话流。先决调用：ADAPTER-WS-NEWPROMPT 发起请求前。后续调用：流式循环中高频触发 DEEPSEEK-ON-MESSAGE。
# 输入参数：无
# 输出参数：无
class DeepSeekParser:
    def __init__(self):
        self.fragment_types =[]
        self.reasoning_buffer = []
        self.content_buffer = []
        self.is_done = False

    #[START] DEEPSEEK-ON-MESSAGE
    # version: 001
    # 上下文：每次接收到 WS 的 SSE Event JSON (数据片段) 时触发。先决调用：建立长连接并开始读取 recv 流。后续调用：外层通过返回值判断是否断开循环。
    # 输入参数：data (dict)
    # 输出参数：is_done (bool)
    def on_message(self, data: dict) -> bool:
        path = data.get("p", "")
        value = data.get("v")
        operation = data.get("o", "")

        if operation == "BATCH" and path == "response":
            items = value if isinstance(value, list) else[]
            for item in items:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_done = True
                    return True

        if isinstance(value, dict) and "response" in value:
            fragments = value["response"].get("fragments",[])
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
    # [END] DEEPSEEK-ON-MESSAGE

    # [START] DEEPSEEK-GET-RESULT
    # version: 001
    # 上下文：DEEPSEEK-ON-MESSAGE 返回 True 中断循环后被调用。先决调用：单次生成周期结束。后续调用：返回到顶层业务。
    # 输入参数：无
    # 输出参数：reasoning (str), content (str)
    def get_result(self) -> Tuple[str, str]:
        return fix_deepseek("".join(self.reasoning_buffer), "".join(self.content_buffer))
    # [END] DEEPSEEK-GET-RESULT

    def _append_frag(self, raw_type: str, content: str):
        self.fragment_types.append(raw_type)
        self._append_content(raw_type, content)

    def _append_content(self, raw_type: str, content: str):
        if not content: return
        print(content, end="",flush=True)
        if raw_type == "THINK":
            self.reasoning_buffer.append(content)
        else:
            self.content_buffer.append(content)
# [END] DEEPSEEK-PARSER