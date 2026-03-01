import asyncio
import json
import re
try:
    import websockets
except ImportError:
    print("请安装websockets: pip install websockets")
    raise
try:
    import httpx
except ImportError:
    print("请安装httpx: pip install httpx")
    raise


class StreamChunk:
    """统一的流数据块，表示从模型接收到的每一个片段"""

    TYPE_REASONING = "reasoning"  # 思考过程（如 DeepSeek 的 THINK 类型）
    TYPE_CONTENT = "content"  # 正式回答内容
    TYPE_DONE = "done"  # 流结束标记

    def __init__(self, chunk_type, delta):
        self.type = chunk_type  # 块类型
        self.delta = delta  # 增量文本


# ==============================================================================
# WebSocket 桥接（用于 DeepSeek 私有协议）
# ==============================================================================

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


class WebSocketBridge:
    """WebSocket 桥接，使用 DeepSeek 私有协议"""

    def __init__(self, ws_url: str, model_name: str = "deepseek"):
        self.ws_url = ws_url
        self.model_name = model_name
        self.ws = None
        self.parser = None
        self._is_finished = False

    async def connect(self) -> bool:
        """建立 WebSocket 连接并完成配对"""
        self.ws = await websockets.connect(self.ws_url)

        if self.model_name == "deepseek":
            self.parser = DeepSeekBridge()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        await self.ws.send(json.dumps({"type": "pair_request", "model": self.model_name}))
        response = await self.ws.recv()
        data = json.loads(response)
        return data.get("type") == "pair_result" and data.get("content") is True

    async def send_prompt(self, text: str):
        """发送提示，重置解析器"""
        self.parser.__init__()
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
        """接收流式响应，返回 (reasoning, content)"""
        reasoning_buffer = []
        content_buffer = []
        reasoning_len = 0
        content_len = 0

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)

            if msg.get("type") == "system" and msg.get("content") == "partner_disconnected":
                print("\n?? 浏览器端已断开连接")
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
                        self._is_finished = True
                        print(
                            f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                            end="",
                            flush=True,
                        )
                        return "".join(reasoning_buffer), "".join(content_buffer)

                if chunks:
                    print(
                        f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                        end="",
                        flush=True,
                    )

        return "".join(reasoning_buffer), "".join(content_buffer)

    async def new_chat(self):
        """开启新对话（清空上下文）"""
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))

    async def close(self):
        """关闭 WebSocket 连接"""
        if self.ws:
            await self.ws.close()

    @property
    def is_finished(self) -> bool:
        """是否已收到完成信号"""
        return self._is_finished or (self.parser and self.parser.is_finished)


# ==============================================================================
# HTTP 桥接（用于 OpenAI 兼容 API）
# ==============================================================================

class OpenAIStreamParser:
    """
    解析 OpenAI 兼容 API 的 SSE 流式响应，生成 StreamChunk。
    处理标准格式：data: {...} 和 data: [DONE]
    """

    def __init__(self):
        self.is_finished = False

    def feed_line(self, line: str):
        """
        输入一行 SSE 数据（可能包含多个 data 块，但通常一行一个 data）。
        返回本次解析出的 StreamChunk 列表。
        """
        if not line or not line.startswith("data: "):
            return []

        data_str = line[6:].strip()  # 去除 "data: " 前缀
        if data_str == "[DONE]":
            self.is_finished = True
            return [StreamChunk(StreamChunk.TYPE_DONE, "")]

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            # 忽略解析失败的 data 行（记录日志？）
            return []

        choices = chunk.get("choices", [])
        if not choices:
            return []

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None:
            # 某些实现会在最后一个 chunk 同时携带 finish_reason 和空 delta
            self.is_finished = True
            return [StreamChunk(StreamChunk.TYPE_DONE, "")]

        chunks = []
        reasoning = delta.get("reasoning_content")
        if reasoning:
            chunks.append(StreamChunk(StreamChunk.TYPE_REASONING, reasoning))
        content = delta.get("content")
        if content:
            chunks.append(StreamChunk(StreamChunk.TYPE_CONTENT, content))

        return chunks


class HTTPBridge:
    """HTTP 桥接，使用 OpenAI 兼容 API，从 profiles.json 加载配置"""

    def __init__(self, profile_name: str, profiles_path: str = "agent/profiles.json"):
        self.profile_name = profile_name
        self.profiles_path = profiles_path
        self.client: httpx.AsyncClient = None
        self.endpoint = None
        self.api_key = None
        self.model = None
        self.messages = []  # 对话历史
        self._parser = None
        self._is_finished = False

    async def connect(self) -> bool:
        """加载配置，初始化 HTTP 客户端"""
        import json
        from pathlib import Path

        profiles_path = Path(self.profiles_path)
        if not profiles_path.exists():
            print(f"[错误] 配置文件不存在: {profiles_path}")
            return False

        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception as e:
            print(f"[错误] 读取配置文件失败: {e}")
            return False

        if self.profile_name not in profiles:
            print(f"[错误] 配置文件中不存在 profile: {self.profile_name}")
            return False

        config = profiles[self.profile_name]
        self.endpoint = config.get("endpoint")
        self.api_key = config.get("api_key")
        if not self.endpoint or not self.api_key:
            print(f"[错误] profile '{self.profile_name}' 缺少 endpoint 或 api_key")
            return False

        self.model = config.get("model")
        if not self.model:
            self.model = self._guess_default_model(self.endpoint)
            print(f"[提示] 未指定 model，使用猜测的默认模型: {self.model}")

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self.messages = []
        self._parser = OpenAIStreamParser()
        self._is_finished = False
        return True

    def _guess_default_model(self, endpoint: str) -> str:
        """根据 endpoint URL 猜测常用的模型名称"""
        endpoint_lower = endpoint.lower()
        if "siliconflow" in endpoint_lower:
            return "deepseek-ai/DeepSeek-V2.5"
        elif "nvidia" in endpoint_lower:
            return "meta/llama3-70b-instruct"
        elif "deepseek" in endpoint_lower:
            return "deepseek-chat"
        elif "groq" in endpoint_lower:
            return "mixtral-8x7b-32768"
        else:
            return "gpt-3.5-turbo"

    async def send_prompt(self, text: str):
        """将用户消息添加到历史中，准备发送"""
        self.messages.append({"role": "user", "content": text})
        self._parser = OpenAIStreamParser()
        self._is_finished = False

    async def completion(self) -> tuple[str, str]:
        """发送 HTTP 流式请求，收集思考过程和正式回答"""
        if not self.client or not self.endpoint or not self.api_key:
            raise RuntimeError("客户端未正确初始化，请先调用 connect")

        if not self.messages or self.messages[-1]["role"] != "user":
            return "", ""

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        reasoning_parts = []
        content_parts = []
        reasoning_len = 0
        content_len = 0

        try:
            async with self.client.stream(
                "POST", self.endpoint, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"[错误] HTTP {response.status_code}: {error_text[:200]}")
                    return "", ""

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    for data_line in line.split("\n"):
                        if not data_line:
                            continue
                        chunks = self._parser.feed_line(data_line)
                        for chunk in chunks:
                            if chunk.type == StreamChunk.TYPE_REASONING:
                                reasoning_parts.append(chunk.delta)
                                reasoning_len += len(chunk.delta)
                            elif chunk.type == StreamChunk.TYPE_CONTENT:
                                content_parts.append(chunk.delta)
                                content_len += len(chunk.delta)
                            elif chunk.type == StreamChunk.TYPE_DONE:
                                self._is_finished = True
                                print(
                                    f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                                    end="",
                                    flush=True,
                                )
                                assistant_content = "".join(content_parts)
                                self.messages.append({"role": "assistant", "content": assistant_content})
                                return "".join(reasoning_parts), assistant_content

                        if chunks:
                            print(
                                f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                                end="",
                                flush=True,
                            )

            # 如果循环正常结束（没有遇到 DONE），视为流结束
            self._is_finished = True
            assistant_content = "".join(content_parts)
            self.messages.append({"role": "assistant", "content": assistant_content})
            return "".join(reasoning_parts), assistant_content

        except Exception as e:
            print(f"\n[异常] {type(e).__name__}: {e}")
            self._is_finished = True
            return "", ""

    async def new_chat(self):
        """清空对话历史，开始新会话"""
        self.messages = []
        self._parser = OpenAIStreamParser()
        self._is_finished = False

    async def close(self):
        """关闭 HTTP 客户端"""
        if self.client:
            await self.client.aclose()

    @property
    def is_finished(self) -> bool:
        return self._is_finished


# ==============================================================================
# 统一 LLM 客户端入口
# ==============================================================================

class LLMClient:
    """
    统一的 LLM 客户端，根据 connection_param 自动选择底层桥接：
      - 以 ws:// 或 wss:// 开头 → 使用 WebSocket 桥接（DeepSeek 私有协议）
      - 否则视为 profiles.json 中的 profile 名称 → 使用 HTTP 桥接（OpenAI 兼容 API）
    """

    def __init__(self, connection_param: str, profiles_path: str = "agent/profiles.json"):
        self.connection_param = connection_param
        self.profiles_path = profiles_path
        self._bridge = None

    async def connect(self) -> bool:
        """根据参数初始化并连接对应的桥接"""
        if self.connection_param.startswith(("ws://", "wss://")):
            self._bridge = WebSocketBridge(self.connection_param, model_name="deepseek")
        else:
            self._bridge = HTTPBridge(self.connection_param, self.profiles_path)
        return await self._bridge.connect()

    async def send_prompt(self, text: str):
        await self._bridge.send_prompt(text)

    async def completion(self) -> tuple[str, str]:
        return await self._bridge.completion()

    async def new_chat(self):
        await self._bridge.new_chat()

    async def close(self):
        await self._bridge.close()

    @property
    def is_finished(self) -> bool:
        return self._bridge.is_finished if self._bridge else False