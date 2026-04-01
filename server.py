#!/usr/bin/env python3
import asyncio
import json
import hashlib
import time
import traceback
from typing import Dict, Optional, Tuple, List, Any
import websockets

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import uvicorn

# ==============================================================================
# 你提供的核心适配器代码 (DeepSeekBridge & OpenAIAdapter)
# ==============================================================================

class DeepSeekBridge:
    """管理 WebSocket 连接，解析 DeepSeek 响应"""
    def __init__(self, ws):
        self.ws = ws
        self.available_browsers = []
        self.paired = False
        self._is_think = False
        self._is_response = False
        self._is_finished = True
        self._think = []
        self._response = []
        self._inbox = asyncio.Queue()
        self.debug = True

    def log(self, msg):
        if self.debug:
            print(f"[Bridge] {msg}")

    async def send(self, channel: str, payload: dict):
        msg = json.dumps({"channel": channel, "payload": payload}, ensure_ascii=False)
        self.log(f"发送: {msg}")
        await self.ws.send(msg)

    async def _set_status(self, status: str):
        prev_finished = self._is_finished
        self._is_finished = status == "FINISHED"
        self._is_think = status == "THINK"
        self._is_response = status == "RESPONSE"
        self.log(f"状态变更: {status} (finished:{self._is_finished})")

        if self._is_finished and not prev_finished:
            think_text = "".join(self._think)
            response_text = "".join(self._response)
            self.log(f"完成！think长度:{len(think_text)}, response长度:{len(response_text)}")
            await self._inbox.put((think_text, response_text))
            self._think, self._response = [], []

    async def _append_v(self, v):
        if not isinstance(v, str):
            return
        if v == "FINISHED":
            await self._set_status(v)
        elif self._is_think:
            self._think.append(v)
        elif self._is_response:
            self._response.append(v)

    async def _deepseek_append_parser(self, obj: dict):
        typ = obj.get("type")
        content = obj.get("content")
        self.log(f"fragment: type={typ}, content={content[:50] if content else ''}...")
        await self._set_status(typ)
        await self._append_v(content)

    async def _deepseek_object_parser(self, obj: dict):
        o = obj.get("o")
        p = obj.get("p")
        v = obj.get("v")

        self.log(f"object: o={o}, p={p}, v类型={type(v)}")

        if (o is None and p is None) or p in (
            "response/fragments/-1/content",
            "quasi_status",
            "response/status",
        ):
            if isinstance(v, dict):
                response = v.get("response")
                if response:
                    fragments = response.get("fragments")
                    if isinstance(fragments, list):
                        for frag in fragments:
                            await self._deepseek_append_parser(frag)
            await self._append_v(v)
        elif o == "APPEND":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for frag in v_list:
                    await self._deepseek_append_parser(frag)
            else:
                self.log(f"APPEND 但 v 不是列表: {v_list}")
        elif o == "BATCH":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for sub_obj in v_list:
                    await self._deepseek_object_parser(sub_obj)
            else:
                self.log(f"BATCH 但 v 不是列表: {v_list}")

    async def listen(self):
        try:
            async for raw in self.ws:
                self.log(f"原始消息: {raw[:200]}")
                data = json.loads(raw)
                channel = data.get("channel")
                payload = data.get("payload", {})

                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass

                if channel == "system":
                    command = payload.get("command")
                    if command == "list_result":
                        self.available_browsers = payload.get("message", [])
                        self.log(f"获取浏览器列表: {self.available_browsers}")
                    elif command == "pair_result":
                        if payload.get("message") == "success":
                            self.paired = True
                            self.log("✅ 配对成功")
                        else:
                            self.log(f"❌ 配对失败: {payload.get('message')}")
                else:
                    await self._deepseek_object_parser(payload)
        except websockets.exceptions.ConnectionClosed:
            self.log("🛑 连接断开")
        except Exception as e:
            traceback.print_exc()
            self.log(f"监听异常: {e}")

    async def pop_response(self, timeout: float = 60.0) -> Tuple[str, str]:
        try:
            think, response = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            if think == "" and response == "":
                return await self.pop_response(timeout)
            return think, response
        except asyncio.TimeoutError:
            raise Exception(f"等待响应超时 {timeout} 秒")

    async def call_match(self, keyword: str):
        await self.send("client", {"command": "match", "message": keyword})

    async def call_new_chat(self):
        await self.send("client", {"command": "new_chat"})

    async def call_send_prompt(self, text: str, image_b64: Optional[str] = None):
        payload = {"command": "send_prompt", "message": text}
        if image_b64:
            payload["image"] = image_b64
        await self.send("client", payload)

class OpenAIAdapter:
    def __init__(self, ws_url: str = "wss://moonchan.publicvm.com/ws/client"):
        self.ws_url = ws_url

    async def _send_and_receive(self, text: str, new_chat: bool = False) -> str:
        print(f"\n=== 开始处理消息: {text[:50]}... ===")
        async with websockets.connect(self.ws_url) as ws:
            client = DeepSeekBridge(ws)
            listen_task = asyncio.create_task(client.listen())

            await client.send("system", {"command": "list"})
            await asyncio.sleep(1.0)

            if not client.available_browsers:
                print("等待浏览器列表...")
                await asyncio.sleep(1.0)
                if not client.available_browsers:
                    raise Exception("没有可用的浏览器")

            target = client.available_browsers[0]
            print(f"目标浏览器: {target}")
            await client.send(
                "system",
                {
                    "command": "pair",
                    "title": target.get("title"),
                    "type": target.get("type"),
                    "timestamp": target.get("timestamp"),
                },
            )
            await asyncio.sleep(1.0)
            if not client.paired:
                raise Exception("配对失败")

            await client.call_match("chat.deepseek.com")
            await asyncio.sleep(1.0)

            if new_chat:
                await client.call_new_chat()
                await asyncio.sleep(1.0)

            await client.call_send_prompt(text=text)

            think, response = await client.pop_response(timeout=120.0)

            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

            print(f"=== 收到响应，长度: {len(response)} ===")
            
            # 如果想在普通的客户端中看到思考过程，可以将 think 和 response 拼接
            # 目前按你的逻辑是：如果有 response 则返回，否则返回 think
            return response if response else think

    async def handle_request(self, request_data: Dict) -> Dict:
        try:
            messages = request_data.get("messages", [])
            model = request_data.get("model", "deepseek-chat")

            user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if not user_msg:
                raise ValueError("未找到用户消息")

            new_chat = any(
                msg.get("role") == "system"
                and msg.get("content", "").strip().upper() == "[NEW_CHAT]"
                for msg in messages
            )

            answer = await self._send_and_receive(user_msg, new_chat)

            return {
                "id": f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_msg) // 4,
                    "completion_tokens": len(answer) // 4,
                    "total_tokens": (len(user_msg) + len(answer)) // 4,
                },
            }
        except Exception as e:
            traceback.print_exc()
            raise


# ==============================================================================
# FastAPI HTTP Server 代码
# ==============================================================================

app = FastAPI(title="DeepSeek WebSocket to OpenAI API Bridge")

# 添加跨域支持，允许所有客户端连接（如 Web 版的 ChatGPT-Next-Web）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 实例化适配器，由于你的每次请求都是独立新建 WebSocket，共享这一个实例是线程/协程安全的
adapter = OpenAIAdapter()

# 虚拟的安全验证（接受任何 Bearer Token，防止部分客户端强制要求 Token 报错）
security = HTTPBearer(auto_error=False)

# 定义客户端请求的数据模型，使用 extra='allow' 兼容不规范的请求
class Message(BaseModel):
    role: str
    content: Any # 兼容部分客户端发送复杂 content 结构

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[Message]
    stream: bool = False
    
    model_config = {
        "extra": "allow"
    }

@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepSeek API Bridge is running!"}

@app.get("/v1/models")
async def list_models():
    """伪装 models 接口，防止某些客户端在发送请求前先校验模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1686935002,
                "owned_by": "deepseek"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, token: HTTPBearer = Depends(security)):
    """处理聊天补全请求"""
    try:
        # 直接获取原始 JSON 数据，保证最大兼容性
        request_data = await request.json()
        
        # 如果客户端请求的是流式输出 (stream=True)
        # 你的当前代码暂不支持流式返回SSE事件，这里给出提示但依旧返回完整内容（大多数客户端兼容此降级行为）
        if request_data.get("stream"):
            print("[Warning] 客户端请求了流式响应 (stream=True)，但当前适配器仅支持非流式，降级处理中...")
            
        # 调用适配器获取标准的 OpenAI 格式返回值
        response_data = await adapter.handle_request(request_data)
        return response_data

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

if __name__ == "__main__":
    # 启动服务器，默认监听 8000 端口
    print("🚀 启动 DeepSeek API Bridge，端口: 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)