import json
import requests
from typing import Callable, Optional, Any, Dict

class JsonApiRequester:
    """
    [使用上下文]
    底层请求类。在初始化时绑定全局 Hook（请求钩子与 SSE 钩子）。
    负责维护最后一次请求的状态记录。
    """
    def __init__(self, 
                 json_file_path: str, 
                 sse_hook: Optional[Callable] = None, 
                 request_hook: Optional[Callable] = None):
        with open(json_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.endpoint = config.get("endpoint")
        self.auth_token = config.get("api_key")
        
        # 在实例化时绑定 Hook
        self.sse_hook = sse_hook
        self.request_hook = request_hook
        
        # 状态记录
        self.last_request = None
        self.last_response = None

        print(f"Requester 初始化成功: Endpoint -> {self.endpoint}")

    def send_request(self, request_body: Dict):
        # 自动处理流模式逻辑
        if self.sse_hook and not request_body.get("stream"):
            request_body["stream"] = True
        
        is_stream = request_body.get("stream", False)
        self.last_request = request_body

        # 触发请求钩子
        if self.request_hook:
            self.request_hook(request_body)

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.endpoint,
                data=json.dumps(request_body),
                headers=headers,
                stream=is_stream
            )
            self.last_response = response

            if is_stream and response.status_code == 200:
                return self._handle_sse(response)
            
            return response
        except Exception as e:
            print(f"请求异常: {e}")
            return None

    def _handle_sse(self, response):
        if not self.sse_hook:
            return response

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]
                    if content == "[DONE]": break
                    try:
                        self.sse_hook(json.loads(content))
                    except: continue
        return response