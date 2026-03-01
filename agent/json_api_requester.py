"""
JsonApiRequester - 流式/非流式 JSON API 请求封装器

【设计目标】
- 为 LLM API（OpenAI/Claude/DeepSeek 等）提供统一的请求层
- 支持 Server-Sent Events (SSE) 流式响应的实时处理
- 通过 Hook 机制实现请求/响应的可观测性，便于日志、监控、重试等扩展
- 关键设计：不设置请求超时，避免长连接流式响应因网络抖动导致 token 浪费

【典型调用链】
    JsonApiRequester ──→ send_request() ──→ 触发 request_hook
                              │
                    ┌────────┴────────┐
                    ↓                 ↓
                 stream=False      stream=True
                    │                 │
               直接返回响应      _handle_sse()
                                      │
                              逐 chunk 触发 sse_hook
                              直到 [DONE] 或连接中断

【配置示例】config.json
    {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    }

【Hook 签名规范】
    request_hook(request_body: Dict) -> None
        【调用时机】HTTP 请求发送前，stream 参数自动注入完成后
        【调用效果】可修改 request_body（副作用可见），或用于日志/审计
        【调用上下文】仍在同一线程，阻塞 send_request 的执行
    
    sse_hook(chunk: Dict) -> None  
        【调用时机】SSE 流中每个有效 data 行解析后，[DONE] 标记前
        【调用效果】实时处理流式内容，如：逐字渲染、token 计数、 early stopping
        【调用上下文】在迭代 response.iter_lines() 的循环中，单线程顺序执行
        【注意】若 sse_hook 抛出异常，将终止流式读取，异常向上传播
"""

import json
import requests
from typing import Callable, Optional, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    """
    【用途】请求全生命周期上下文，用于跨 Hook 传递状态
    
    【字段说明】
        request_body: 最终发送的请求体（含自动注入的 stream 参数）
        start_time: 请求开始时间（ISO 格式，用于耗时计算）
        attempt_count: 当前重试次数（预留，供重试策略使用）
        metadata: 用户自定义透传数据（如 trace_id、user_id 等）
    """
    request_body: Dict[str, Any]
    start_time: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    attempt_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ResponseSummary:
    """
    【用途】响应摘要，替代直接保存整个 Response 对象
    
    【设计原因】原始 Response 包含文件句柄，长期持有导致连接泄漏
    """
    status_code: int
    headers: Dict[str, str]
    content_type: Optional[str]
    total_chunks: int = 0  # 流式响应的 chunk 计数
    first_chunk_time_ms: Optional[float] = None  # 首包延迟（TTFT）
    total_stream_time_ms: Optional[float] = None  # 完整流式耗时


class JsonApiRequester:
    """
    【使用上下文】
    底层请求类，专注于 LLM API 的流式/非流式统一封装。
    
    【核心设计决策】
    1. 无默认超时：LLM 流式响应可能持续数分钟，超时中断会导致已生成 token 费用
       但暴露 timeout 参数供调用方显式控制非流式场景
    
    2. Hook 实例化绑定：确保同一 Requester 的所有请求共享相同的处理逻辑
       如需不同逻辑，应创建不同 Requester 实例
    
    3. 请求体深拷贝：避免自动注入 stream 参数污染调用方原始对象
    
    4. 状态轻量化：不保存原始 Response（防泄漏），仅保存摘要和上下文
    """
    
    def __init__(
        self,
        json_file_path: str,
        sse_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        response_summary_hook: Optional[Callable[[ResponseSummary], None]] = None,
    ):
        """
        【调用效果】
        加载配置文件，初始化所有 Hook，打印初始化信息
        
        【调用上下文】
        通常在应用启动时执行一次，创建全局或作用域内的 Requester 实例
        
        【参数说明】
            json_file_path: API 配置文件路径
            sse_hook: 流式响应回调，为 None 时禁用流式处理（即使请求声明 stream=True）
            request_hook: 请求预处理回调
            response_summary_hook: 响应完成后的摘要回调，用于 metrics 上报
        
        【其他内容】
        配置文件中必须包含 "endpoint" 和 "api_key" 字段
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self._endpoint: str = config["endpoint"]  # 必须存在，缺失则 KeyError
        self._auth_token: str = config["api_key"]  # 必须存在
        
        # Hook 绑定：实例生命周期内固定，确保行为一致性
        self._sse_hook: Optional[Callable[[Dict[str, Any]], None]] = sse_hook
        self._request_hook: Optional[Callable[[Dict[str, Any]], None]] = request_hook
        self._response_summary_hook: Optional[Callable[[ResponseSummary], None]] = response_summary_hook
        
        # 状态追踪：轻量级，仅保存最近一次的上下文和摘要
        self._last_context: Optional[RequestContext] = None
        self._last_summary: Optional[ResponseSummary] = None
        
        print(f"[JsonApiRequester] 初始化成功 | Endpoint: {self._endpoint[:50]}... | "
              f"SSE: {'enabled' if sse_hook else 'disabled'}")

    @property
    def last_context(self) -> Optional[RequestContext]:
        """【调用效果】获取最近一次请求的上下文快照（只读）"""
        return self._last_context
    
    @property  
    def last_summary(self) -> Optional[ResponseSummary]:
        """【调用效果】获取最近一次响应的摘要（只读）"""
        return self._last_summary

    def send_request(
        self,
        request_body: Dict[str, Any],
        *,
        timeout: Optional[Union[float, tuple]] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[requests.Response]:
        """
        【调用效果】
        发送 HTTP POST 请求，根据配置自动处理流式响应，返回原始 Response 对象
        
        【调用上下文】
        业务代码的直接调用点，如聊天接口、批量推理等
        
        【参数说明】
            request_body: API 请求体，将被深拷贝后修改（原对象不受影响）
            timeout: 显式超时设置，默认 None（无限等待）
                - 非流式场景建议设置 (connect_timeout, read_timeout)
                - 流式场景建议 None，或仅设置 connect_timeout
            context_metadata: 透传至 RequestContext 的自定义数据，跨 Hook 可用
        
        【返回说明】
            - 成功：requests.Response 对象（调用方需负责 .close() 或上下文管理）
            - 失败：None（网络层异常，已打印错误日志）
            - 注意：流式场景下 Response 已迭代耗尽，如需原始流需禁用 sse_hook
        
        【其他内容】
        自动流式逻辑：若 sse_hook 存在且 request_body 未显式设置 stream=False，
        则强制启用 stream=True 并注入请求体。
        
        首包延迟(TTFT)计算：流式场景下记录从请求发送到首个 chunk 的时间。
        """
        import copy
        import time
        from datetime import datetime
        
        # 深拷贝：保护调用方原始对象不被修改
        body = copy.deepcopy(request_body)
        
        # 自动流式决策：sse_hook 存在 → 默认开启流式（除非显式关闭）
        auto_stream = self._sse_hook is not None and body.get("stream") is not False
        if auto_stream:
            body["stream"] = True
        
        is_stream: bool = body.get("stream", False)
        
        # 构建上下文
        ctx = RequestContext(
            request_body=body,
            metadata=context_metadata or {}
        )
        self._last_context = ctx
        
        # 触发请求钩子（同步阻塞，异常将中断请求）
        if self._request_hook:
            try:
                self._request_hook(body)
            except Exception as e:
                print(f"[JsonApiRequester] request_hook 异常: {e}")
                raise  # 钩子异常视为致命，不上送请求
        
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }
        
        request_start = time.perf_counter()
        
        try:
            # 关键：无默认 timeout，避免流式响应被中断
            # 调用方可显式传入 timeout=(3.0, 27.0) 控制连接/读取超时
            response = requests.post(
                self._endpoint,
                data=json.dumps(body, ensure_ascii=False),
                headers=headers,
                stream=is_stream,
                timeout=timeout,  # None = 无限等待，符合 LLM 场景需求
            )
            
            # 摘要初始化
            summary = ResponseSummary(
                status_code=response.status_code,
                headers=dict(response.headers),
                content_type=response.headers.get('Content-Type'),
            )
            
            # 流式处理分支
            if is_stream and response.status_code == 200:
                summary = self._handle_sse_with_metrics(
                    response, 
                    summary, 
                    request_start
                )
            
            self._last_summary = summary
            
            # 触发摘要钩子
            if self._response_summary_hook:
                self._response_summary_hook(summary)
            
            return response
            
        except requests.exceptions.ConnectionError as e:
            # 连接层异常：DNS、拒绝连接、网络不可达
            print(f"[JsonApiRequester] 连接失败: {e}")
            return None
            
        except requests.exceptions.Timeout as e:
            # 仅当调用方显式设置 timeout 时可能触发
            print(f"[JsonApiRequester] 请求超时: {e}")
            return None
            
        except Exception as e:
            # 其他异常：SSL 错误、重定向过多等
            print(f"[JsonApiRequester] 未预期异常: {type(e).__name__}: {e}")
            return None

    def _handle_sse_with_metrics(
        self,
        response: requests.Response,
        summary: ResponseSummary,
        request_start: float,
    ) -> ResponseSummary:
        """
        【调用效果】
        迭代 SSE 流，逐 chunk 触发 sse_hook，更新并返回带指标的摘要
        
        【调用上下文】
        send_request 的内部方法，仅当流式响应成功时调用
        
        【指标计算】
        - first_chunk_time_ms: 请求发送到首个有效 chunk 的耗时（TTFT）
        - total_chunks: 有效 data 事件计数（不含 [DONE]）
        - total_stream_time_ms: 完整流式传输耗时（含 [DONE] 等待）
        
        【其他内容】
        解析异常处理：单条 data 行解析失败时记录警告但继续，避免单条损坏
        导致整个流中断。可通过日志聚合监控解析错误率。
        """
        if self._sse_hook is None:
            # 无 hook 时不消费流，返回原始响应让调用方自行处理
            return summary
        
        import time
        
        first_chunk_received = False
        chunk_count = 0
        
        try:
            for line in response.iter_lines():
                if not line:
                    continue  # 跳过空行（SSE 心跳）
                
                decoded = line.decode('utf-8', errors='replace').strip()
                
                if not decoded.startswith("data: "):
                    continue  # 非 data 字段（如 event: / id:），按 SSE 规范忽略
                
                content = decoded[6:]  # 去除 "data: " 前缀
                
                if content == "[DONE]":
                    # OpenAI 标准结束标记
                    break
                
                # TTFT 计算
                if not first_chunk_received:
                    summary.first_chunk_time_ms = (time.perf_counter() - request_start) * 1000
                    first_chunk_received = True
                
                # 解析并触发 Hook
                try:
                    chunk = json.loads(content)
                    chunk_count += 1
                    self._sse_hook(chunk)
                    
                except json.JSONDecodeError as e:
                    # 单条解析失败，记录但继续
                    print(f"[JsonApiRequester] SSE 解析警告: {e} | content: {content[:100]}...")
                    continue
                    
                except Exception as e:
                    # sse_hook 业务异常，终止流式处理
                    print(f"[JsonApiRequester] sse_hook 异常: {e}")
                    raise  # 向上传播，调用方决定是否重试
        
        finally:
            # 确保指标更新（即使异常退出）
            summary.total_chunks = chunk_count
            summary.total_stream_time_ms = (time.perf_counter() - request_start) * 1000
            
            # 主动关闭连接，避免资源泄漏
            response.close()
        
        return summary

    def close(self):
        """
        【调用效果】
        清理资源，重置状态
        
        【调用上下文】
        应用关闭前或 Requester 实例废弃时调用
        """
        self._last_context = None
        self._last_summary = None
        print("[JsonApiRequester] 已清理")

    def __enter__(self):
        """支持上下文管理器：with JsonApiRequester(...) as r:"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理"""
        self.close()
        return False  # 不吞异常


# ═══════════════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════════════

def example_usage():
    """
    【完整调用示例】包含非流式、流式、带指标监控的多种场景
    """
    import sys
    
    # ── 配置创建（实际项目中为静态文件）────────────────────────────────────────
    config = {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_key": "sk-xxx"
    }
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    # ── Hook 定义 ─────────────────────────────────────────────────────────────
    
    def log_request(body: Dict):
        """【调用效果】记录所有 outgoing 请求，用于审计"""
        print(f"[Hook:Request] model={body.get('model')}, "
              f"stream={body.get('stream')}, "
              f"messages_count={len(body.get('messages', []))}")
    
    def render_stream(chunk: Dict):
        """【调用效果】实时渲染流式内容，模拟打字机效果"""
        # 适配 OpenAI 格式
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        content = delta.get('content', '')
        if content:
            print(content, end='', flush=True)
    
    def report_metrics(summary: ResponseSummary):
        """【调用效果】上报 Prometheus/日志，用于 SLO 监控"""
        print(f"\n[Hook:Metrics] status={summary.status_code}, "
              f"chunks={summary.total_chunks}, "
              f"TTFT={summary.first_chunk_time_ms:.1f}ms, "
              f"total_time={summary.total_stream_time_ms:.1f}ms")
    
    # ── 场景 1: 流式对话（推荐）────────────────────────────────────────────────
    print("\n" + "="*50 + " 场景 1: 流式对话 " + "="*50)
    
    with JsonApiRequester(
        config_path,
        sse_hook=render_stream,
        request_hook=log_request,
        response_summary_hook=report_metrics,
    ) as requester:
        
        response = requester.send_request({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "讲个短笑话"}],
            # stream 未指定，自动启用为 True
        })
        
        # 流式场景下，response 已被迭代耗尽，不可再次读取
        # 如需原始内容，需通过 sse_hook 积累或禁用 sse_hook
        
        print(f"\n最后请求模型: {requester.last_context.request_body['model']}")
    
    # ── 场景 2: 强制非流式（覆盖自动逻辑）──────────────────────────────────────
    print("\n" + "="*50 + " 场景 2: 强制非流式 " + "="*50)
    
    with JsonApiRequester(
        config_path,
        sse_hook=render_stream,  # 存在，但会被 stream=False 覆盖
        request_hook=log_request,
    ) as requester:
        
        response = requester.send_request({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,  # 显式关闭，sse_hook 不会触发
        }, timeout=(3.0, 10.0))  # 非流式可安全设置超时
        
        if response:
            result = response.json()
            print(f"非流式结果: {result['choices'][0]['message']['content'][:50]}...")
            response.close()  # 显式关闭（非流式 response 需手动管理）
    
    # ── 场景 3: 无 Hook 原始模式（调用方完全控制）──────────────────────────────
    print("\n" + "="*50 + " 场景 3: 原始模式 " + "="*50)
    
    with JsonApiRequester(config_path) as requester:  # 无 Hook
        
        response = requester.send_request({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "测试"}],
            "stream": True,  # 声明流式，但无 sse_hook
        })
        
        # 原始流式响应，调用方自行迭代
        if response:
            print("原始 SSE 行：")
            for line in response.iter_lines():
                if line:
                    print(f"  {line.decode()[:80]}...")
            response.close()
    
    # 清理
    os.unlink(config_path)


if __name__ == "__main__":
    example_usage()