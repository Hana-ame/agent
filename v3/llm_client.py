import json
from pathlib import Path
from typing import Tuple
from adapter import WSAdapter, OpenAIAdapter
class LLMClient:
    def __init__(
        self,
        connection_param: str,
        payload_name: str = "default.json",
        profiles_path: Path = None,
        root_path: Path = None,
    ):
        self.connection_param = connection_param
        self.payload_name = payload_name
        self.profiles_path = profiles_path
        self.root_path = root_path
        self.adapter = None
        self._current_prompt = ""
        self._connected = False  # 新增：表示连接是否活跃
    async def connect(self) -> bool:
        try:
            if self.connection_param.startswith(("ws://", "wss://")):
                self.adapter = WSAdapter(ws_url=self.connection_param)
                success = await self.adapter.connect()
                if success:
                    self._connected = True
                return success
            else:
                if not self.profiles_path or not self.profiles_path.exists():
                    print(f"[错误] 找不到配置文件: {self.profiles_path}")
                    return False
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                if self.connection_param not in profiles:
                    print(f"[错误] Profile '{self.connection_param}' 未在 {self.profiles_path} 中定义")
                    return False
                config = profiles[self.connection_param]
                endpoint = config.get("endpoint")
                api_key = config.get("api_key")
                model = config.get("model", "gpt-3.5-turbo")
                if not endpoint or not api_key:
                    print(f"[错误] Profile '{self.connection_param}' 缺少 endpoint 或 api_key")
                    return False
                self.adapter = OpenAIAdapter(endpoint=endpoint, api_key=api_key, model=model)
                self._connected = True
                return True
        except Exception as e:
            print(f"[异常] 连接初始化失败: {e}")
            return False
    async def send_prompt(self, text: str):
        self._current_prompt = text
    async def completion(self) -> Tuple[str, str]:
        if not self.adapter:
            raise RuntimeError("尚未连接到任何底层 Adapter")
        reasoning, content = await self.adapter.new_prompt(self._current_prompt)
        return reasoning, content
    async def send_prompt_and_completion(self, text: str) -> Tuple[str, str]:
        if not self.adapter:
            raise RuntimeError("尚未连接到任何底层 Adapter")
        reasoning, content = await self.adapter.new_prompt(text)
        return reasoning, content
    async def new_chat(self):
        if self.adapter:
            await self.adapter.new_chat()
    async def close(self):
        if self.adapter:
            await self.adapter.close()
            self._connected = False
