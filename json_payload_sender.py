import os
import json
from json_api_requester import JsonApiRequester

from typing import List, Optional, Any, Dict

class JsonPayloadSender:
    """
    [使用上下文]
    逻辑构造类。通过传入 JsonApiRequester 实例实现发送。
    
    [模式说明]
    1. Context 模式 (use_context=True)：
       消息 = [_context 外部文件消息] + [传入 dynamic_messages]
    2. Payload 模式 (use_context=False)：
       消息 = [payload.json 中的消息] + [传入 dynamic_messages]
    """
    def __init__(self, requester: JsonApiRequester, payload_file_path: str = "payload.json"):
        # [组合模式] 持有 requester 实例
        self.requester = requester
        self.payload_path = payload_file_path
        
        if not os.path.exists(self.payload_path):
            self._save_to_file({"messages": [], "model": "gpt-3.5-turbo", "_context": ""})
        
        print(f"Sender 初始化完成：逻辑绑定至 {self.payload_path}")

    def _read_from_file(self, path: str) -> Any:
        if not os.path.exists(path): return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_to_file(self, data: Dict):
        with open(self.payload_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def update_payload_field(self, key: str, value: Any):
        """实时更新字段并持久化到本地 payload.json"""
        data = self._read_from_file(self.payload_path) or {}
        data[key] = value
        self._save_to_file(data)
        print(f"字段持久化: {key} -> {value}")

    def send_request_with_messages(self, dynamic_messages: List[Dict], use_context: bool = False):
        """
        核心业务逻辑：构造最终 Payload 并调用持有的 requester 发送。
        """
        main_payload = self._read_from_file(self.payload_path)
        base_messages = []

        if use_context:
            context_path = main_payload.get('_context')
            if context_path and os.path.exists(context_path):
                content = self._read_from_file(context_path)
                base_messages = content if isinstance(content, list) else []
            print(f"[Context 模式] 加载外部消息: {len(base_messages)}条")
        else:
            base_messages = main_payload.get("messages", [])
            print(f"[Payload 模式] 加载预设消息: {len(base_messages)}条")

        # 组装最终 Payload
        final_payload = main_payload.copy()
        final_payload["messages"] = base_messages + dynamic_messages
        
        # 清理内部私有字段 (以 _ 开头的)
        clean_payload = {k: v for k, v in final_payload.items() if not k.startswith('_')}
        
        # 调用 requester 实例的方法
        return self.requester.send_request(clean_payload)