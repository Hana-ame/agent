import asyncio
import json
import websockets
import re


class StreamChunk:
    """统一的流数据块，表示从模型接收到的每一个片段"""

    TYPE_REASONING = "reasoning"  # 思考过程（如 DeepSeek 的 THINK 类型）
    TYPE_CONTENT = "content"  # 正式回答内容
    TYPE_DONE = "done"  # 流结束标记

    def __init__(self, chunk_type, delta):
        self.type = chunk_type  # 块类型
        self.delta = delta  # 增量文本


class DeepSeekBridge:
    """
    将 DeepSeek 的私有协议解析为统一的流式数据块。
    它跟踪片段类型列表，并处理多种更新模式（全量、追加、路径更新、隐式增量）。
    """

    def __init__(self):
        self.fragment_types = []  # 按顺序记录每个片段的类型（'THINK' 或 'TEXT'）
        self.is_finished = False  # 是否收到完成信号

    def parse(self, data: dict):
        """
        解析从 WebSocket 收到的消息字典，生成对应的 StreamChunk 列表。
        支持多种消息格式：
          - 结束信号 (BATCH 模式)
          - 初始全量 fragments
          - 追加新 fragments
          - 更新某个 fragment 的内容
          - 纯增量字符串（追加到最后一个 fragment）
        """
        chunks = []
        path = data.get("p", "")  # 路径，如 "response/fragments"
        value = data.get("v")  # 值，可以是各种类型
        operation = data.get("o", "")  # 操作类型，如 "BATCH", "APPEND"

        # 1. 识别结束信号 (BATCH 模式)
        if operation == "BATCH" and path == "response":
            items = value if isinstance(value, list) else []
            for item in items:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_finished = True
                    chunks.append(StreamChunk(StreamChunk.TYPE_DONE, ""))
                    return chunks

        # 2. 初始全量数据：第一次收到 response 时，包含所有 fragments
        if isinstance(value, dict) and "response" in value:
            fragments = value["response"].get("fragments", [])
            for frag in fragments:
                frag_type = frag.get("type")
                self.fragment_types.append(frag_type)
                content = frag.get("content", "")
                if content:
                    chunks.append(self._create_chunk(frag_type, content))
            return chunks

        # 3. 追加新 fragments (APPEND 模式)
        if operation == "APPEND" and path == "response/fragments":
            for fragment in value:
                frag_type = fragment.get("type")
                self.fragment_types.append(frag_type)
                content = fragment.get("content", "")
                if content:
                    chunks.append(self._create_chunk(frag_type, content))
            return chunks

        # 4. 更新某个 fragment 的内容（路径形如 response/fragments/-1/content）
        match = re.match(r"response/fragments/(-?\d+)/content", path)
        if match and isinstance(value, str):
            idx = int(match.group(1))
            # 将负索引转换为正索引
            real_idx = idx if idx >= 0 else len(self.fragment_types) + idx
            if 0 <= real_idx < len(self.fragment_types):
                chunks.append(self._create_chunk(self.fragment_types[real_idx], value))
            return chunks

        # 5. 隐式更新：路径为空且值为字符串，表示追加到最后一个 fragment
        if not path and isinstance(value, str) and self.fragment_types:
            chunks.append(self._create_chunk(self.fragment_types[-1], value))
            return chunks

        return chunks

    def _create_chunk(self, raw_type, content):
        """
        根据原始类型（'THINK' 或 'TEXT'）创建对应的 StreamChunk。
        THINK 映射为 reasoning 类型，其余为 content 类型。
        """
        chunk_type = (
            StreamChunk.TYPE_REASONING
            if raw_type == "THINK"
            else StreamChunk.TYPE_CONTENT
        )
        return StreamChunk(chunk_type, content)


class LLMClient:
    """与 LLM 服务交互的 WebSocket 客户端，支持发送提示和流式接收回答"""

    def __init__(self, ws_url="ws://localhost:8765/ws/client"):
        self.ws_url = ws_url
        self.ws = None
        self.model = None  # 当前使用的模型名称
        self.parser = None  # 对应模型的协议解析器

    async def connect_and_pair(self, model_name: str) -> bool:
        """
        连接 WebSocket 并与指定模型配对。
        发送配对请求，等待确认结果。
        """
        self.model = model_name
        self.ws = await websockets.connect(self.ws_url)

        # 根据模型选择解析器（目前仅支持 deepseek）
        if model_name == "deepseek":
            self.parser = DeepSeekBridge()
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        await self.ws.send(json.dumps({"type": "pair_request", "model": model_name}))
        response = await self.ws.recv()
        data = json.loads(response)
        return data.get("type") == "pair_result" and data.get("content") is True

    async def new_chat(self):
        """开启一个新的对话（清空上下文）"""
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))

    async def send_prompt(self, text: str):
        """向模型发送提示文本，并重置解析器状态以准备接收新回答"""
        self.parser.__init__()  # 重置解析器（清除片段类型列表和完成标记）
        await self.ws.send(
            json.dumps(
                {
                    "type": "command",
                    "command": "send_prompt",
                    "params": {"prompt": text},
                }
            )
        )

    async def completion(self) -> tuple[str, str]:
        """
        处理 WebSocket 流式响应，实时在控制台显示已接收的思考字符数和回答字符数。
        当收到完成信号时，返回完整的 (思考过程, 回答内容) 元组。
        """
        reasoning_buffer = []  # 收集思考过程的片段
        content_buffer = []  # 收集回答内容的片段
        reasoning_len = 0
        content_len = 0

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)

            # 对方断开连接
            if (
                msg.get("type") == "system"
                and msg.get("content") == "partner_disconnected"
            ):
                print("\n⚠️ 浏览器端已断开连接")
                break

            if msg.get("type") == "token":
                chunks = self.parser.parse(msg.get("content", {}))
                for chunk in chunks:
                    if chunk.type == StreamChunk.TYPE_REASONING:
                        reasoning_buffer.append(chunk.delta)
                        reasoning_len += len(chunk.delta)
                    elif chunk.type == StreamChunk.TYPE_CONTENT:
                        content_buffer.append(chunk.delta)
                        content_len += len(chunk.delta)
                    elif chunk.type == StreamChunk.TYPE_DONE:
                        # 流结束，输出最终统计并返回
                        print(
                            f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                            end="",
                            flush=True,
                        )
                        return "".join(reasoning_buffer), "".join(content_buffer)

                # 每当收到新块，更新进度显示
                if chunks:
                    print(
                        f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                        end="",
                        flush=True,
                    )

        # 如果因断开连接而退出循环，则返回已收集的内容
        return "".join(reasoning_buffer), "".join(content_buffer)

    @property
    def is_finished(self):
        """是否已收到完成信号（由解析器维护）"""
        return self.parser.is_finished if self.parser else False
