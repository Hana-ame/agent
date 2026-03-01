import asyncio
import json
import websockets
import re

# DeepSeek 专用流解析器
class DeepSeekBridge:
    def __init__(self):
        self.reasoning_buffer = []
        self.content_buffer = []
        self.fragment_types = []  # 核心：记录每一个 fragment 的类型
        self.is_finished = False

    def process(self, data):
        # 1. 提取基本元数据
        p = data.get("p", "")
        v = data.get("v")
        o = data.get("o", "")

        # 2. 识别最终结束信号 (BATCH 模式)
        if o == "BATCH" and p == "response":
            val_list = v if isinstance(v, list) else []
            for item in val_list:
                if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                    self.is_finished = True
                    return self.get_result()

        # 3. 处理新 Fragment 的创建 (APPEND 模式)
        # 此时会明确告知 type: "THINK" 或 "RESPONSE"
        if o == "APPEND" and p == "response/fragments":
            for fragment in v:
                f_type = fragment.get("type")
                self.fragment_types.append(f_type)
                # 如果创建时就自带 content (如日志中的 '嗯' 或 '今天')
                initial_content = fragment.get("content", "")
                if initial_content:
                    self._append_to_buffer(f_type, initial_content)

        # 4. 处理内容更新 (路径更新模式)
        # 匹配格式如: response/fragments/-1/content
        match = re.match(r"response/fragments/(-?\d+)/content", p)
        if match and isinstance(v, str):
            idx = int(match.group(1))
            # 转换负索引为实际索引
            real_idx = idx if idx >= 0 else len(self.fragment_types) + idx
            
            if 0 <= real_idx < len(self.fragment_types):
                f_type = self.fragment_types[real_idx]
                self._append_to_buffer(f_type, v)
        
        # 5. 处理隐式更新 (没有路径 p，只有 v)
        # 这种情况通常延续上一个活跃的 fragment
        elif not p and isinstance(v, str) and self.fragment_types:
            f_type = self.fragment_types[-1]
            self._append_to_buffer(f_type, v)

        # 6. 处理初始全量数据 (如果 API 第一次返回了完整结构)
        elif isinstance(v, dict) and "response" in v:
            fragments = v["response"].get("fragments", [])
            for f in fragments:
                f_type = f.get("type")
                self.fragment_types.append(f_type)
                self._append_to_buffer(f_type, f.get("content", ""))

        return self.get_result()

    def _append_to_buffer(self, f_type, text):
        if not text or not isinstance(text, str):
            return
        if f_type == "THINK":
            self.reasoning_buffer.append(text)
        elif f_type == "RESPONSE":
            self.content_buffer.append(text)

    def get_result(self):
        """统一返回当前状态"""
        return "".join(self.reasoning_buffer), "".join(self.content_buffer)


# 统一的客户端控制封装类
class LLMClient:
    def __init__(self, ws_url="ws://localhost:8765/ws/client"):
        self.ws_url = ws_url
        self.ws = None
        self.model = None
        self.parser = None

    async def connect_and_pair(self, model_name: str) -> bool:
        """连接服务端并请求配对指定的模型浏览器"""
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
        # 重置解析器，确保新对话不携带旧状态
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

    async def send_command(self, command: str):
        await self.ws.send(json.dumps({"type": "command", "command": command}))

    def completion(self, msg: dict) -> tuple:
        if not isinstance(msg, dict) or msg.get("type") != "token":
            return self.parser.get_result()
        return self.parser.process(msg.get("content", {}))

    @property
    def is_finished(self):
        return self.parser.is_finished