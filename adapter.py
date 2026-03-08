#[START] ADAPTER-MAIN-PKG
# version: 001
# 上下文：统一 LLM 调用桥接装载起点。先决调用：无。后续调用：ADAPTER-BASE。
# 输入参数：无
# 输出参数：无
import json
import asyncio
import httpx
import websockets
from typing import Tuple, List, Dict

# from adapter.deepseek import DeepSeekParser
from adapters.deepseek import DeepSeekParser
# [END] ADAPTER-MAIN-PKG

# [START] ADAPTER-BASE
# version: 001
# 上下文：作为系统抽象接口规范存在。先决调用：无。后续调用：被 ADAPTER-WS 与 ADAPTER-OPENAI 继承实现。
# 输入参数：无
# 输出参数：无
class BaseAdapter:
    
    # [START] ADAPTER-BASE-NEWCHAT
    # version: 001
    # 上下文：业务层要求开启全新会话或切断记忆时调用。先决调用：无。后续调用：发起新的 ADAPTER-BASE-NEWPROMPT。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        raise NotImplementedError
    # [END] ADAPTER-BASE-NEWCHAT

    # [START] ADAPTER-BASE-NEWPROMPT
    # version: 001
    # 上下文：业务层抛出用户 Prompt 指令并需要获取双轨返回值时调用。先决调用：实例已连接目标终点。后续调用：渲染 UI 或写入下游数据结构。
    # 输入参数：prompt (str)
    # 输出参数：reasoning (str), content (str)
    async def new_prompt(self, prompt: str) -> Tuple[str, str]:
        raise NotImplementedError
    # [END] ADAPTER-BASE-NEWPROMPT

# [END] ADAPTER-BASE

# [START] ADAPTER-WS
# version: 001
# 上下文：基于私有 WS 协议桥接（DeepSeek 浏览器端）被选择时装载。先决调用：ADAPTER-BASE 接口规范。后续调用：WS 生命周期管理。
# 输入参数：ws_url (str), model_name (str)
# 输出参数：无
class WSAdapter(BaseAdapter):
    def __init__(self, ws_url: str, model_name: str = "deepseek"):
        self.ws_url = ws_url
        self.model_name = model_name
        self.ws = None
        self.parser = None

    async def connect(self) -> bool:
        self.ws = await websockets.connect(self.ws_url)
        await self.ws.send(json.dumps({"type": "pair_request", "model": self.model_name}))
        resp = json.loads(await self.ws.recv())
        return resp.get("type") == "pair_result" and resp.get("content") is True

    # [START] ADAPTER-WS-NEWCHAT
    # version: 001
    # 上下文：通知对侧浏览器刷新聊天窗口。先决调用：WS 处于连接状态。后续调用：无。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        await self.ws.send(json.dumps({"type": "command", "command": "new_chat"}))
    # [END] ADAPTER-WS-NEWCHAT

    # [START] ADAPTER-WS-NEWPROMPT
    # version: 001
    # 上下文：将新 Prompt 透传给浏览器并阻塞收集 SSE Event。先决调用：WS 联通。后续调用：DEEPSEEK-ON-MESSAGE 以及 DEEPSEEK-GET-RESULT。
    # 输入参数：prompt (str)
    # 输出参数：reasoning (str), content (str)
    async def new_prompt(self, prompt: str) -> Tuple[str, str]:
        self.parser = DeepSeekParser()
        await self.ws.send(json.dumps({
            "type": "command",
            "command": "send_prompt",
            "params": {"prompt": prompt}
        }))

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)
            
            if msg.get("type") == "system" and msg.get("content") == "partner_disconnected":
                break
                
            if msg.get("type") == "token":
                is_done = self.parser.on_message(msg.get("content", {}))
                if is_done:
                    print("----")
                    break

        return self.parser.get_result()
    # [END] ADAPTER-WS-NEWPROMPT

    async def close(self):
        if self.ws:
            await self.ws.close()
#[END] ADAPTER-WS

# [START] ADAPTER-OPENAI
# version: 001
# 上下文：基于标准 OpenAI 兼容 HTTP API 调用被选择时装载。先决调用：ADAPTER-BASE 接口规范。后续调用：HTTP 客户端生命周期管理。
# 输入参数：endpoint (str), api_key (str), model (str)
# 输出参数：无
class OpenAIAdapter(BaseAdapter):
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.messages: List[Dict[str, str]] =[]
        self.client = httpx.AsyncClient(timeout=60.0)

    # [START] ADAPTER-OPENAI-NEWCHAT
    # version: 001
    # 上下文：清空本地历史记忆栈。先决调用：无。后续调用：无。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        self.messages.clear()
    # [END] ADAPTER-OPENAI-NEWCHAT

    # [START] ADAPTER-OPENAI-NEWPROMPT
    # version: 001
    # 上下文：封装上下文记忆，发起 HTTP 流式请求并收集标准 SSE 响应。先决调用：HTTP 客户端就绪。后续调用：请求结束，返回给顶层业务提取结果。
    # 输入参数：prompt (str)
    # 输出参数：reasoning (str), content (str)
    async def new_prompt(self, prompt: str) -> Tuple[str, str]:
        self.messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True
        }

        reasoning_parts =[]
        content_parts =[]

        async with self.client.stream("POST", self.endpoint, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                return "", ""

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    
                    if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                        reasoning_parts.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"] is not None:
                        content_parts.append(delta["content"])
                except Exception:
                    continue

        final_reasoning = "".join(reasoning_parts)
        final_content = "".join(content_parts)
        
        self.messages.append({"role": "assistant", "content": final_content})
        return final_reasoning, final_content
    # [END] ADAPTER-OPENAI-NEWPROMPT

    async def close(self):
        await self.client.aclose()
# [END] ADAPTER-OPENAI