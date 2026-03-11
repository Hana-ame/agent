#[START] ADAPTER-MAIN-PKG
# version: 0.0.1
# 上下文：统一 LLM 调用桥接装载起点。先决调用：无。后续调用：ADAPTER-BASE。
# 输入参数：无
# 输出参数：无
import json
import asyncio
import httpx
import websockets
from typing import Tuple, List, Dict

from adapters.deepseek import DeepSeekParser
# [END] ADAPTER-OPENAI-PKG

# [START] ADAPTER-BASE
# version: 0.0.2
# 上下文：作为系统抽象接口规范存在。先决调用：无。后续调用：被 ADAPTER-WS 等继承实现。
# 输入参数：无
# 输出参数：无
class BaseAdapter:
    
    # [START] ADAPTER-BASE-NEWCHAT
    # version: 0.0.2
    # 上下文：业务层要求开启全新会话或切断记忆时调用。先决调用：无。后续调用：发起新的 ADAPTER-BASE-NEWPROMPT。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        raise NotImplementedError
    # [END] ADAPTER-BASE-NEWCHAT

    # [START] ADAPTER-BASE-NEWPROMPT
    # version: 0.0.2
    # 上下文：业务层抛出用户 Prompt 指令并需要获取双轨返回值时调用。先决调用：实例已连接目标终点。后续调用：渲染 UI 或写入下游数据结构。
    # 输入参数：prompt (str)
    # 输出参数：reasoning (str), content (str)
    async def new_prompt(self, prompt: str) -> Tuple[str, str]:
        raise NotImplementedError
    # [END] ADAPTER-BASE-NEWPROMPT

# [END] ADAPTER-BASE


# [START] ADAPTER-WS
# version: 0.0.2
# 上下文：基于私有 WS 协议 V2 桥接被选择时装载。先决调用：ADAPTER-BASE 接口规范。后续调用：WS 生命周期管理及基于 Channel 的数据分发。
# 输入参数：ws_url (str), model_name (str)
# 输出参数：无
class WSAdapter(BaseAdapter):
    def __init__(self, ws_url: str, model_name: str = "deepseek"):
        self.ws_url = ws_url  # V2 对应地址应为: ws://127.26.3.1:8080/ws/client
        self.model_name = model_name
        self.ws = None
        self.parser = None

    # [START] ADAPTER-WS-CONNECT
    # version: 0.0.2
    # 上下文：初始化连接时调用。先决调用：WS Server 启动并存有 Browser 节点。后续调用：通过 list 查询可用节点，再通过组合键执行 pair 动作。
    # 输入参数：无
    # 输出参数：是否成功连接并配对 (bool)
    async def connect(self) -> bool:
        self.ws = await websockets.connect(self.ws_url)
        
        # 1. 向系统频道请求查询当前所有注册的浏览器节点
        await self.ws.send(json.dumps({
            "channel": "system",
            "payload": {"action": "list"}
        }))
        
        target_browser = None
        
        # 阻塞等待返回列表结果
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("channel") == "system" and resp.get("payload", {}).get("action") == "list_result":
                browsers = resp["payload"].get("content",[])
                # 寻找匹配指定 model_name (type) 的首个可用 Browser
                target_browser = next((b for b in browsers if b.get("type") == self.model_name), None)
                break
                
        if not target_browser:
            print(f"❌ 系统频道报告: 未找到类型为 [{self.model_name}] 的可用 Browser 节点")
            return False

        # 2. 向系统频道发送锁定配对请求 (利用复合标识)
        await self.ws.send(json.dumps({
            "channel": "system",
            "payload": {
                "action": "pair",
                "type": target_browser.get("type"),
                "title": target_browser.get("title"),
                "created_at": target_browser.get("created_at")
            }
        }))
        
        # 阻塞等待配对结果
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("channel") == "system" and resp.get("payload", {}).get("action") == "pair_result":
                is_success = resp["payload"].get("content") == "success"
                if is_success:
                    print(f"🔗 成功与 Browser [{target_browser['title']}] 建立配对关系")
                return is_success
    # [END] ADAPTER-WS-CONNECT

    # [START] ADAPTER-WS-NEWCHAT
    # version: 0.0.2
    # 上下文：通知对侧浏览器刷新聊天窗口。先决调用：WS 处于配对状态。后续调用：通过 client 频道透传给 Browser 执行 DOM 操作。
    # 输入参数：无
    # 输出参数：无
    async def new_chat(self):
        await self.ws.send(json.dumps({
            "channel": "client",
            "payload": {
                "command": "new_chat"
            }
        }))
    # [END] ADAPTER-WS-NEWCHAT

    # [START] ADAPTER-WS-NEWPROMPT
    # version: 0.0.2
    # 上下文：将新 Prompt 透传给浏览器并阻塞收集流式回复。先决调用：WS 联通并配对完成。后续调用：解析 browser 频道转发过来的 token 事件。
    # 输入参数：prompt (str)
    # 输出参数：reasoning (str), content (str)
    async def new_prompt(self, prompt: str) -> Tuple[str, str]:
        self.parser = DeepSeekParser()
        
        # 将指令通过 client 频道发送给系统，系统会自动透传到配对的 Browser
        await self.ws.send(json.dumps({
            "channel": "client",
            "payload": {
                "command": "send_prompt",
                "params": {"prompt": prompt}
            }
        }))

        while True:
            raw_msg = await self.ws.recv()
            msg = json.loads(raw_msg)
            
            channel = msg.get("channel")
            payload = msg.get("payload", {})
            
            # 监听系统频道的异常断开事件
            if channel == "system" and payload.get("action") == "unpaired":
                print(f"\n⚠️ 会话被中断, 原因: {payload.get('content')}")
                break
                
            # 监听对端发回的业务数据事件
            if channel == "browser" and payload.get("type") == "token":
                # 解析 XHR 劫持送过来的数据
                is_done = self.parser.on_message(payload.get("data", {}))
                if is_done:
                    print("\n----")
                    break

        return self.parser.get_result()
    # [END] ADAPTER-WS-NEWPROMPT

    # [START] ADAPTER-WS-CLOSE
    # version: 0.0.2
    # 上下文：主动断开资源或对象被销毁时调用。先决调用：WS 存在且开启。后续调用：网络关闭，远端触发解绑。
    # 输入参数：无
    # 输出参数：无
    async def close(self):
        if self.ws:
            # 优雅退出：先主动向系统频道请求解绑
            await self.ws.send(json.dumps({
                "channel": "system",
                "payload": {"action": "unpair"}
            }))
            await self.ws.close()
# [END] ADAPTER-WS

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