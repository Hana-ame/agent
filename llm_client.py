# [START] LLMCLIENT-PKG
# version: 001
# 上下文：LLM 客户端统一管理模块装载起点。先决调用：无。后续调用：被 agent.py 导入实例化。
# 输入参数：无
# 输出参数：无
import json
from pathlib import Path
from typing import Tuple

# 引入您提供的 adapter.py 中的底层适配器
from adapter import WSAdapter, OpenAIAdapter
# [END] LLMCLIENT-PKG

# [START] LLMCLIENT-CORE
# version: 001
# 上下文：统一的大语言模型客户端外观模式封装。先决调用：无。后续调用：agent.py 生命周期初始化。
# 输入参数：connection_param (str), payload_name (str), profiles_path (Path), root_path (Path)
# 输出参数：无
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
        self.is_finished = False

    # [START] LLMCLIENT-CONNECT
    # version: 001
    # 上下文：创建客户端并尝试握手或加载环境配置。先决调用：LLMClient 初始化之后。后续调用：若成功则进入主循环，否则触发外层重连。
    # 输入参数：无
    # 输出参数：连接是否成功 (bool)
    async def connect(self) -> bool:
        try:
            if self.connection_param.startswith(("ws://", "wss://")):
                self.adapter = WSAdapter(ws_url=self.connection_param)
                self.is_finished = True
                return await self.adapter.connect()
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
                model = config.get("model", "gpt-3.5-turbo") # 提供兜底默认值
                
                if not endpoint or not api_key:
                    print(f"[错误] Profile '{self.connection_param}' 缺少 endpoint 或 api_key")
                    return False
                    
                self.adapter = OpenAIAdapter(endpoint=endpoint, api_key=api_key, model=model)
                self.is_finished = True
                return True
        except Exception as e:
            print(f"[异常] 连接初始化失败: {e}")
            return False
    # [END] LLMCLIENT-CONNECT

    # [START] LLMCLIENT-SEND-PROMPT
    # last modify: 2026-03-08 
    # version: 001
    # 上下文：Agent 需要发送新一轮对话指令，适配器在此暂存状态。先决调用：LLMCLIENT-CONNECT 成功并处于空闲态。后续调用：触发 LLMCLIENT-COMPLETION 进行阻塞执行。
    # 输入参数：text (str)
    # 输出参数：无
    async def send_prompt(self, text: str):
        self._current_prompt = text
        self.is_finished = False
    # [END] LLMCLIENT-SEND-PROMPT

    # [START] LLMCLIENT-COMPLETION
    # last modify: 2026-03-08 
    # version: 001
    # 上下文：阻塞等待底层 LLM 接口返回完整的双轨结果。先决调用：LLMCLIENT-SEND-PROMPT 已暂存用户输入。后续调用：结果转交 RuleProcessor 模式引擎处理。
    # 输入参数：无
    # 输出参数：reasoning (str), content (str)
    async def completion(self) -> Tuple[str, str]:
        if not self.adapter:
            raise RuntimeError("尚未连接到任何底层 Adapter")
        
        reasoning, content = await self.adapter.new_prompt(self._current_prompt)
        self.is_finished = True
        return reasoning, content
    # [END] LLMCLIENT-COMPLETION

    # [START] LLMCLIENT-SEND-PROMPT-AND-COMPLETION
    # last modify: 2026-03-08 
    # version: 001
    # 上下文：Agent 需要发送新一轮对话指令并立即等待完整结果，组合了发送和完成两个步骤。先决调用：LLMCLIENT-CONNECT 成功并处于空闲态。后续调用：结果转交 RuleProcessor 模式引擎处理。
    # 输入参数：text (str)
    # 输出参数：reasoning (str), content (str)
    async def send_prompt_and_completion(self, text: str) -> Tuple[str, str]:
        if not self.adapter:
            raise RuntimeError("尚未连接到任何底层 Adapter")
        self._current_prompt = text
        self.is_finished = False
        reasoning, content = await self.adapter.new_prompt(text)
        self.is_finished = True
        return reasoning, content
    # [END] LLMCLIENT-SEND-PROMPT-AND-COMPLETION

    # [START] LLMCLIENT-NEW-CHAT
    # version: 001
    # 上下文：Agent 被要求强制重置本地或远端会话记忆。先决调用：底层连接已建立。后续调用：展开全新周期的对话。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        if self.adapter:
            await self.adapter.new_chat()
    # [END] LLMCLIENT-NEW-CHAT

    # [START] LLMCLIENT-CLOSE
    # version: 001
    # 上下文：Agent 主动退出或网络异常重连前的资源清理。先决调用：系统正在运行。后续调用：退出应用或重新触发 LLMCLIENT-CONNECT。
    # 输入参数：无
    # 输出参数：无
    async def close(self):
        if self.adapter:
            await self.adapter.close()
    # [END] LLMCLIENT-CLOSE
#[END] LLMCLIENT-CORE