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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# ==============================================================================
# 核心适配器代码
# ==============================================================================

class DeepSeekBridge:
    """管理 WebSocket 连接，解析 DeepSeek 响应并支持流式和非流式"""
    def __init__(self, ws):
        self.ws = ws
        self.available_browsers = []
        self.paired = False
        
        self._is_think = False
        self._is_response = False
        self._is_finished = True
        
        self._think = []
        self._response = []
        
        # 针对非流式的队列
        self._inbox = asyncio.Queue()
        # 针对流式的队列（存储每一个 fragment）
        self._stream_queue = asyncio.Queue()
        
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

        # 状态切换到完成时，发送结束信号
        if self._is_finished and not prev_finished:
            # 流式结束信号
            await self._stream_queue.put({"type": "done"})
            
            # 非流式完成信号
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
            # 实时推入流式队列
            await self._stream_queue.put({"type": "think", "content": v})
        elif self._is_response:
            self._response.append(v)
            # 实时推入流式队列
            await self._stream_queue.put({"type": "response", "content": v})

    async def _deepseek_append_parser(self, obj: dict):
        typ = obj.get("type")
        content = obj.get("content")
        # self.log(f"fragment: type={typ}, content={content[:50] if content else ''}...")
        await self._set_status(typ)
        await self._append_v(content)

    async def _deepseek_object_parser(self, obj: dict):
        o = obj.get("o")
        p = obj.get("p")
        v = obj.get("v")

        # self.log(f"object: o={o}, p={p}, v类型={type(v)}")

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
        elif o == "BATCH":
            v_list = obj.get("v")
            if isinstance(v_list, list):
                for sub_obj in v_list:
                    await self._deepseek_object_parser(sub_obj)

    async def listen(self):
        try:
            async for raw in self.ws:
                # self.log(f"原始消息: {raw[:200]}")
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
        """等待完整非流式响应"""
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

    async def _setup_and_send(self, client: DeepSeekBridge, text: str, image_b64: Optional[str], new_chat: bool):
        """抽取出的公共前置连接、配对、发送过程"""
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

        # 这里传入解析出来的 text 和 image_b64
        await client.call_send_prompt(text=text, image_b64=image_b64)

    async def handle_stream_request(self, request_data: Dict):
        """处理流式 (Stream) 请求的生成器，返回 SSE 格式数据"""
        messages = request_data.get("messages", [])
        model = request_data.get("model", "deepseek-chat")

        # 解包文本和图片
        user_msg, image_b64 = self._extract_user_msg(messages)
        new_chat = self._is_new_chat(messages)

        chat_id = f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:24]}"
        created_time = int(time.time())

        print(f"\n=== 开始流式处理消息: {user_msg[:50]}... (含图片: {bool(image_b64)}) ===")

        initial_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(initial_chunk, ensure_ascii=False)}\n\n"

        async with websockets.connect(self.ws_url) as ws:
            client = DeepSeekBridge(ws)
            listen_task = asyncio.create_task(client.listen())

            try:
                # 传入 user_msg 和 image_b64
                await self._setup_and_send(client, user_msg, image_b64, new_chat)

                while True:
                    try:
                        item = await asyncio.wait_for(client._stream_queue.get(), timeout=60.0)
                    except asyncio.TimeoutError:
                        break

                    if item["type"] == "done":
                        finish_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    
                    delta = {}
                    if item["type"] == "think":
                        delta["reasoning_content"] = item["content"]
                    elif item["type"] == "response":
                        delta["content"] = item["content"]

                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            except Exception as e:
                traceback.print_exc()
                print(f"[流式错误] {e}")
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                print("=== 流式响应结束 ===")

    async def handle_request(self, request_data: Dict) -> Dict:
        """处理非流式 (Non-stream) 请求"""
        messages = request_data.get("messages", [])
        model = request_data.get("model", "deepseek-chat")

        # 解包文本和图片
        user_msg, image_b64 = self._extract_user_msg(messages)
        new_chat = self._is_new_chat(messages)

        print(f"\n=== 开始非流式处理消息: {user_msg[:50]}... (含图片: {bool(image_b64)}) ===")
        
        async with websockets.connect(self.ws_url) as ws:
            client = DeepSeekBridge(ws)
            listen_task = asyncio.create_task(client.listen())

            try:
                # 传入 user_msg 和 image_b64
                await self._setup_and_send(client, user_msg, image_b64, new_chat)
                think, response = await client.pop_response(timeout=120.0)
                
                answer = response if response else think
                
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
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass

    def _extract_user_msg(self, messages: list) -> Tuple[str, Optional[str]]:
        """
        提取用户消息，兼容纯字符串格式和多模态数组格式。
        返回 (文本内容, base64图片字符串)
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                
                # 1. 如果 content 是普通字符串，直接返回
                if isinstance(content, str):
                    return content, None
                
                # 2. 如果 content 是数组 (多模态格式)
                elif isinstance(content, list):
                    text_parts = []
                    image_b64 = None
                    
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                            
                        # 提取文本
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                            
                        # 提取图片 (如果客户端发了图片)
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            # 取出 base64 数据部分 (data:image/jpeg;base64,xxxxxx...)
                            if url.startswith("data:image"):
                                try:
                                    image_b64 = url.split(",")[1]
                                except IndexError:
                                    pass
                                    
                    return "\n".join(text_parts), image_b64
                    
        raise ValueError("未找到用户消息")

    def _is_new_chat(self, messages: list) -> bool:
        """检查是否有新建对话的系统指令，同样兼容数组格式的 content"""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join([i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text"])
                
                if text.strip().upper() == "[NEW_CHAT]":
                    return True
        return False
# ==============================================================================
# FastAPI HTTP Server 代码
# ==============================================================================

app = FastAPI(title="DeepSeek WebSocket to Streaming API Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter = OpenAIAdapter()
security = HTTPBearer(auto_error=False)

@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepSeek Streaming API Bridge is running!"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "deepseek-chat", "object": "model", "created": 1686935002, "owned_by": "deepseek"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, token: HTTPBearer = Depends(security)):
    try:
        request_data = await request.json()
        
        # 核心判断：如果客户端请求了 Stream，则返回 StreamingResponse
        if request_data.get("stream", False):
            return StreamingResponse(
                adapter.handle_stream_request(request_data),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # 否则降级回原来的等待完整内容的 JSON 响应
            response_data = await adapter.handle_request(request_data)
            return response_data

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

if __name__ == "__main__":
    print("🚀 启动 DeepSeek 流式(Streaming) API Bridge，端口: 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)