import asyncio
import json
import re
import pathlib
from typing import List, Dict, Optional, Any, Tuple

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

def fix_deepseek(reasoning_buffer: str, content_buffer: str):
    if content_buffer == "":
        content_buffer, reasoning_buffer = reasoning_buffer, content_buffer
    return reasoning_buffer.replace("
", "\n"), content_buffer.replace("
", "\n")

class StreamChunk:
    """统一的流数据块，表示从模型接收到的每一个片段"""

    TYPE_REASONING = "reasoning"  # 思考过程（如 DeepSeek 的 THINK 类型）
    TYPE_THINKING = "thinking"  # 思考过程（如 DeepSeek 的 THINK 类型）
    TYPE_CONTENT = "content"  # 正式回答内容
    TYPE_DONE = "done"  # 流结束标记

    def __init__(self, chunk_type, delta):
        self.type = chunk_type
        self.delta = delta


# ==============================================================================
# WebSocket 桥接（用于 DeepSeek 私有协议）
# ==============================================================================


class DeepSeekBridge:
    """将 DeepSeek 的私有协议解析为统一的流式数据块。"""

    def __init__(self):
        self.fragment_types = []
        self.is_finished = False

    def parse(self, data: dict):
        chunks = []
        path = data.get("p", "")
        value = data.get("v")
        operation = data.get("o", "")

        if operation == "BATCH" and path == "response":
            items = value if isinstance(value, list) else []
            for item in items:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_finished = True
                    chunks.append(StreamChunk(StreamChunk.TYPE_DONE, ""))
                    return chunks

        if isinstance(value, dict) and "response" in value:
            fragments = value["response"].get("fragments", [])
            for frag in fragments:
                frag_type = frag.get("type")
                self.fragment_types.append(frag_type)
                content = frag.get("content", "")
                if content:
                    chunks.append(self._create_chunk(frag_type, content))
            return chunks

        if operation == "APPEND" and path == "response/fragments":
            for fragment in value:
                frag_type = fragment.get("type")
                self.fragment_types.append(frag_type)
                content = fragment.get("content", "")
                if content:
                    chunks.append(self._create_chunk(frag_type, content))
            return chunks

        match = re.match(r"response/fragments/(-?\d+)/content", path)
        if match and isinstance(value, str):
            idx = int(match.group(1))
            real_idx = idx if idx >= 0 else len(self.fragment_types) + idx
            if 0 <= real_idx < len(self.fragment_types):
                chunks.append(self._create_chunk(self.fragment_types[real_idx], value))
            return chunks

        if not path and isinstance(value, str) and self.fragment_types:
            chunks.append(self._create_chunk(self.fragment_types[-1], value))
            return chunks

        return chunks

    def _create_chunk(self, raw_type, content):
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
        self.ws = await websockets.connect(self.ws_url)

        if self.model_name == "deepseek":
            self.parser = DeepSeekBridge()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        await self.ws.send(
            json.dumps({"type": "pair_request", "model": self.model_name})
        )
        response = await self.ws.recv()
        data = json.loads(response)
        return data.get("type") == "pair_result" and data.get("content") is True

    async def send_prompt(self, text: str):
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
        reasoning_buffer = []
        content_buffer = []
        reasoning_len = 0
        content_len = 0

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)

            if (
                msg.get("type") == "system"
                and msg.get("content") == "partner_disconnected"
            ):
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
                        return fix_deepseek("".join(reasoning_buffer), "".join(content_buffer))

                if chunks:
                    print(
                        f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                        end="",
                        flush=True,
                    )

        return fix_deepseek("".join(reasoning_buffer), "".join(content_buffer))

    async def new_chat(self):
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))

    async def close(self):
        if self.ws:
            await self.ws.close()

    @property
    def is_finished(self) -> bool:
        return self._is_finished or (self.parser and self.parser.is_finished)


# ==============================================================================
# HTTP 桥接（用于 OpenAI 兼容 API，支持从 payloads 目录加载配置）
# ==============================================================================


class OpenAIStreamParser:
    """解析 OpenAI 兼容 API 的 SSE 流式响应"""

    def __init__(self):
        self.is_finished = False

    def feed_line(self, line: str):
        if not line or not line.startswith("data: "):
            return []

        data_str = line[6:].strip()
        if data_str == "[DONE]":
            self.is_finished = True
            return [StreamChunk(StreamChunk.TYPE_DONE, "")]

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return []

        choices = chunk.get("choices", [])
        if not choices:
            return []

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None:
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
    """
    HTTP 桥接，使用 OpenAI 兼容 API。
    从 profiles.json 读取 endpoint 和 api_key，从 payloads/ 目录读取请求模板（含 model 等参数）。
    """

    def __init__(
        self,
        profile_name: str,
        payload_name: str,
        profiles_path: pathlib.Path,
        payloads_dir: pathlib.Path,
        root_path: pathlib.Path,
    ):
        self.profile_name = profile_name
        self.payload_name = payload_name
        self.profiles_path = profiles_path
        self.payloads_dir = payloads_dir
        self.root_path = root_path

        self.client: httpx.AsyncClient = None
        self.endpoint = None
        self.api_key = None
        self.payload_base: Dict[str, Any] = (
            {}
        )  # 从 payload 文件加载的基础配置（不含 messages）
        self.messages: List[Dict[str, str]] = []  # 对话历史
        self._parser = None
        self._is_finished = False

    async def connect(self) -> bool:
        """加载配置，初始化 HTTP 客户端"""
        import json

        # 1. 加载 profiles.json
        if not self.profiles_path.exists():
            print(f"[错误] 配置文件不存在: {self.profiles_path}")
            return False

        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
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

        # 2. 确保 payloads 目录存在
        self.payloads_dir.mkdir(parents=True, exist_ok=True)

        # 3. 加载 payload 文件
        payload_file = self.payloads_dir / self.payload_name
        if not payload_file.exists():
            print(f"[错误] Payload 文件不存在: {payload_file}")
            return False

        try:
            with open(payload_file, "r", encoding="utf-8") as f:
                payload_data = json.load(f)
        except Exception as e:
            print(f"[错误] 读取 payload 文件失败: {e}")
            return False

        # 移除下划线开头的控制字段（如 _context），保留其他字段作为基础 payload
        self.payload_base = {
            k: v for k, v in payload_data.items() if not k.startswith("_")
        }
        # 确保 payload 中不包含 messages 字段（会被历史替换）
        self.payload_base.pop("messages", None)

        # 可选：如果 payload 中没有 model，尝试从 profiles 中获取或猜测
        if "model" not in self.payload_base:
            model_from_profile = config.get("model")
            if model_from_profile:
                self.payload_base["model"] = model_from_profile
            else:
                self.payload_base["model"] = self._guess_default_model(self.endpoint)
                print(
                    f"[提示] 未指定 model，使用猜测的默认模型: {self.payload_base['model']}"
                )

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self.messages = []
        self._parser = OpenAIStreamParser()
        self._is_finished = False
        return True

    def _guess_default_model(self, endpoint: str) -> str:
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
        """将用户消息添加到历史中"""
        self.messages.append({"role": "user", "content": text})
        self._parser = OpenAIStreamParser()
        self._is_finished = False

    async def completion(self) -> tuple[str, str]:
        if not self.client or not self.endpoint or not self.api_key:
            raise RuntimeError("客户端未正确初始化，请先调用 connect")

        if not self.messages or self.messages[-1]["role"] != "user":
            return "", ""

        # 构建最终 payload：基础配置 + 当前消息列表
        payload = self.payload_base.copy()
        payload["messages"] = self.messages
        payload["stream"] = True  # 强制流式

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
                            # print(chunk.type, chunk.delta)
                            if (
                                chunk.type == StreamChunk.TYPE_REASONING
                                or chunk.type == StreamChunk.TYPE_THINKING
                            ):
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
                                self.messages.append(
                                    {"role": "assistant", "content": assistant_content}
                                )
                                return "".join(reasoning_parts), assistant_content

                        if chunks:
                            print(
                                f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]",
                                end="",
                                flush=True,
                            )

            # 流正常结束（无 DONE 标记）
            self._is_finished = True
            assistant_content = "".join(content_parts)
            self.messages.append({"role": "assistant", "content": assistant_content})
            return "".join(reasoning_parts), assistant_content

        except Exception as e:
            print(f"\n[异常] {type(e).__name__}: {e}")
            self._is_finished = True
            return "", ""

    async def new_chat(self):
        self.messages = []
        self._parser = OpenAIStreamParser()
        self._is_finished = False

    async def close(self):
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
      - 否则视为 profiles.json 中的 profile 名称 → 使用 HTTP 桥接（OpenAI 兼容 API），
        需要额外指定 payload 名称（默认 "default.json"）
    """

    def __init__(
        self,
        connection_param: str,
        payload_name: str = "default.json",
        profiles_path: pathlib.Path = None,
        root_path: pathlib.Path = None,
    ):
        self.connection_param = connection_param
        self.payload_name = payload_name
        self._bridge = None

        # 确保 root_path 为 pathlib.Path 对象
        if root_path is None:
            root_path = pathlib.Path.cwd()
        self.root_path = pathlib.Path(root_path)

        # 确定 profiles.json 路径：默认放在根目录下
        if profiles_path is None:
            self.profiles_path = self.root_path / "profiles.json"
        else:
            self.profiles_path = pathlib.Path(profiles_path)

        self.payloads_dir = self.root_path / "payloads"

    async def connect(self) -> bool:
        """根据参数初始化并连接对应的桥接"""
        if self.connection_param.startswith(("ws://", "wss://")):
            self._bridge = WebSocketBridge(self.connection_param, model_name="deepseek")
        else:
            self._bridge = HTTPBridge(
                profile_name=self.connection_param,
                payload_name=self.payload_name,
                profiles_path=self.profiles_path,
                payloads_dir=self.payloads_dir,
                root_path=self.root_path,
            )
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