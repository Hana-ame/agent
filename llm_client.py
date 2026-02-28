import asyncio
import json
import websockets


# DeepSeek 专用流解析器
class DeepSeekBridge:
    def __init__(self):
        self.reasoning_buffer = []
        self.content_buffer = []
        self.current_type = "THINK"
        self.is_finished = False

    def process(self, data):
        # 1. 精确识别整个对话的【最终结束信号】
        # 特征：o='BATCH', p='response', 且 v 列表中包含 quasi_status = FINISHED
        if data.get("o") == "BATCH" and data.get("p") == "response":
            val_list = data.get("v", [])
            if isinstance(val_list, list):
                for item in val_list:
                    if item.get("p") == "quasi_status" and item.get("v") == "FINISHED":
                        self.is_finished = True
                        # 结束信号不包含内容，直接返回
                        return "".join(self.reasoning_buffer), "".join(
                            self.content_buffer
                        )

        # 2. 处理常规内容提取
        v = data.get("v")
        p = data.get("p", "")
        o = data.get("o", "")

        # 处理初次返回的完整 fragments 结构
        if isinstance(v, dict):
            for f in v.get("response", {}).get("fragments", []):
                self.current_type = f.get("type", self.current_type)
                self._append(self.current_type, f.get("content", ""))

        # 处理后续追加的内容 (APPEND 模式)
        elif o == "APPEND" and isinstance(v, list):
            for item in v:
                # 过滤掉非内容的更新（比如 status: FINISHED 的更新）
                if "content" in str(item):
                    self.current_type = item.get("type", self.current_type)
                    self._append(self.current_type, item.get("content", ""))

        # 处理路径式更新 (例如：p='response/fragments/0/content')
        elif "content" in p and isinstance(v, str):
            # 如果路径里包含 fragments/0，通常是思考过程
            f_type = "THINK" if "fragments/0" in p else "CONTENT"
            self._append(f_type, v)

        elif isinstance(v, str) and p == "":  # 兜底逻辑
            self._append(self.current_type, v)

        return "".join(self.reasoning_buffer), "".join(self.content_buffer)

    def _append(self, f_type, text):
        if not text or not isinstance(text, str):
            return
        if f_type == "THINK":
            self.reasoning_buffer.append(text)
        elif f_type == "RESPONSE":
            self.content_buffer.append(text)


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

        # 1. 动态挂载对应的解析器
        if model_name == "deepseek":
            self.parser = DeepSeekBridge()
        elif model_name == "qwen":
            # 未来实现 self.parser = QwenBridge()
            raise ValueError(f"Unsupported model: {model_name}")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 2. 发送配对请求
        await self.ws.send(json.dumps({"type": "pair_request", "model": model_name}))

        # 3. 等待配对结果
        res = await self.ws.recv()
        data = json.loads(res)
        if data.get("type") == "pair_result" and data.get("content") is True:
            return True
        return False

    async def new_chat(self):
        """新建对话：触发按钮点击"""
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))

    async def send_prompt(self, text: str):
        """发送信息：最新的一条"""
        # 发送前重置解析器状态，准备迎接新的数据流
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
        """
        核心数据处理方法。
        传入 ws 收到的 msg，直接返回 (reasoning, content) 完整字符串。
        """
        if not isinstance(msg, dict) or msg.get("type") != "token":
            # 如果不是 token 消息，返回目前累积的状态
            return "".join(self.parser.reasoning_buffer), "".join(
                self.parser.content_buffer
            )

        return self.parser.process(msg.get("content", {}))

    @property
    def is_finished(self):
        return self.parser.is_finished
