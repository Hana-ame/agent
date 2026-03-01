import asyncio
import json
import websockets
import re

class StreamChunk:
    """通用的流数据结构"""
    TYPE_REASONING = "reasoning"
    TYPE_CONTENT = "content"
    TYPE_DONE = "done"

    def __init__(self, chunk_type, delta):
        self.type = chunk_type
        self.delta = delta

class DeepSeekBridge:
    """将 DeepSeek 的私有协议编译为通用流"""
    def __init__(self):
        self.fragment_types =[]
        self.is_finished = False

    def parse(self, data: dict):
        chunks =[]
        p = data.get("p", "")
        v = data.get("v")
        o = data.get("o", "")

        # 1. 识别结束信号 (BATCH 模式)
        if o == "BATCH" and p == "response":
            val_list = v if isinstance(v, list) else[]
            for item in val_list:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_finished = True
                    chunks.append(StreamChunk(StreamChunk.TYPE_DONE, ""))
                    return chunks

        # 2. 🌟 修复点：处理初始全量数据 (如果不搜索，第一个包会在这里)
        if isinstance(v, dict) and "response" in v:
            fragments = v["response"].get("fragments",[])
            for f in fragments:
                f_type = f.get("type")
                self.fragment_types.append(f_type)
                content = f.get("content", "")
                if content:
                    chunks.append(self._create_chunk(f_type, content))
            return chunks

        # 3. 处理新 Fragment 创建 (APPEND 模式)
        if o == "APPEND" and p == "response/fragments":
            for fragment in v:
                f_type = fragment.get("type")
                self.fragment_types.append(f_type)
                content = fragment.get("content", "")
                if content:
                    chunks.append(self._create_chunk(f_type, content))
            return chunks

        # 4. 处理路径更新模式 (形如 response/fragments/-1/content)
        match = re.match(r"response/fragments/(-?\d+)/content", p)
        if match and isinstance(v, str):
            idx = int(match.group(1))
            # 转换负索引为实际索引
            real_idx = idx if idx >= 0 else len(self.fragment_types) + idx
            if 0 <= real_idx < len(self.fragment_types):
                chunks.append(self._create_chunk(self.fragment_types[real_idx], v))
            return chunks
        
        # 5. 处理隐式更新 (纯增量字符串)
        if not p and isinstance(v, str) and self.fragment_types:
            chunks.append(self._create_chunk(self.fragment_types[-1], v))
            return chunks

        return chunks

    def _create_chunk(self, raw_type, content):
        ctype = StreamChunk.TYPE_REASONING if raw_type == "THINK" else StreamChunk.TYPE_CONTENT
        return StreamChunk(ctype, content)


class LLMClient:
    def __init__(self, ws_url="ws://localhost:8765/ws/client"):
        self.ws_url = ws_url
        self.ws = None
        self.model = None
        self.parser = None

    async def connect_and_pair(self, model_name: str) -> bool:
        self.model = model_name
        self.ws = await websockets.connect(self.ws_url)

        if model_name == "deepseek":
            self.parser = DeepSeekBridge()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        await self.ws.send(json.dumps({"type": "pair_request", "model": model_name}))
        res = await self.ws.recv()
        data = json.loads(res)
        return data.get("type") == "pair_result" and data.get("content") is True

    async def new_chat(self):
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))

    async def send_prompt(self, text: str):
        # 每次提问重置解析器状态
        self.parser.__init__()
        await self.ws.send(
            json.dumps({
                "type": "command",
                "command": "send_prompt",
                "params": {"prompt": text},
            })
        )

    async def completion(self, text: str = "") -> tuple[str, str]:
        """
        处理 WebSocket 流，实时在控制台更新 received length
        完成后返回最终的 (reasoning, content) 元组
        """
        reasoning_buffer = []
        content_buffer =[]
        reasoning_len = 0
        content_len = 0

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)

            # 对方可能断开连接
            if msg.get("type") == "system" and msg.get("content") == "partner_disconnected":
                print("\n🔴 浏览器端已断开连接")
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
                        # 接收结束
                        print(f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]", end="", flush=True)
                        return "".join(reasoning_buffer), "".join(content_buffer)
                
                # 有内容更新时，实时打印控制台进度
                if chunks:
                    print(f"\r[思考字符: {reasoning_len}] | [回答字符: {content_len}]", end="", flush=True)

        return "".join(reasoning_buffer), "".join(content_buffer)

    @property
    def is_finished(self):
        return self.parser.is_finished if self.parser else False