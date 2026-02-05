#!/usr/bin/env python3
"""
Advanced Typewriter Effect HTTP Client for OpenAI-compatible APIs (Strict Version).
Requires explicit configuration via profiles.json (auth) and config.json (request body).
No default values are applied.
"""

import argparse
import base64
import io
import json
import mimetypes
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import requests

# ============ Constants & Utilities ============

# Markdown Code Block Delimiter (Constant as requested)
CODE_BLOCK = "`" * 3


class Colors:
    """ANSI Color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def info(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.END}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.END}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.END}"

    @classmethod
    def thinking(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.END}"


def configure_stdout_utf8():
    if hasattr(sys.stdout, "buffer"):
        # Recreate stdout with UTF-8 encoding
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


configure_stdout_utf8()

# ============ Configuration Management ============


class ConfigManager:
    """管理配置加载与合并逻辑 - 严格模式"""

    @staticmethod
    def load_auth(profile_name: str) -> Tuple[str, str]:
        """从 profiles.json 加载 endpoint 和 api_key"""
        # 查找 profiles.json
        profile_paths = [
            Path("profiles.json"),
            Path("../profiles.json"),
            Path.home() / ".ai_chat_profiles.json",
        ]

        profile_path = None
        for p in profile_paths:
            if p.exists():
                profile_path = p
                break

        if not profile_path:
            print(f"{Colors.error('❌ 错误: 未找到 profiles.json')}", file=sys.stderr)
            print(
                f"请确保在以下位置之一创建 profiles.json: {[str(p) for p in profile_paths]}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception as e:
            print(
                f"{Colors.error(f'❌ 读取 profiles.json 失败: {e}')}", file=sys.stderr
            )
            sys.exit(1)

        if profile_name not in profiles:
            print(
                f"{Colors.error(f'❌ 错误: profiles.json 中未找到预设 \"{profile_name}\"')}",
                file=sys.stderr,
            )
            sys.exit(1)

        profile_data = profiles[profile_name]

        if "endpoint" not in profile_data or "api_key" not in profile_data:
            print(
                f"{Colors.error(f'❌ 错误: 预设 \"{profile_name}\" 缺少 endpoint 或 api_key')}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"{Colors.info('✓')} 已加载认证预设: {profile_name}", file=sys.stderr)
        return profile_data["endpoint"], profile_data["api_key"]

    @staticmethod
    def load_request_body(config_path: str) -> Dict[str, Any]:
        """从 config.json 加载请求体，不做任何修改或添加默认值"""
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            print(
                f"{Colors.error(f'❌ 错误: 配置文件不存在: {config_path}')}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"{Colors.info('✓')} 已加载请求配置: {path}", file=sys.stderr)
            return data
        except Exception as e:
            print(
                f"{Colors.error(f'❌ 读取配置文件失败 ({config_path}): {e}')}",
                file=sys.stderr,
            )
            sys.exit(1)

    @staticmethod
    def build_final_payload(args) -> Tuple[str, str, Dict[str, Any]]:
        """构建最终的 Endpoint, API Key 和 Request Payload"""

        # 1. 加载认证信息
        endpoint, api_key = ConfigManager.load_auth(args.profile)

        # 2. 加载配置文件 (请求体)
        payload = ConfigManager.load_request_body(args.config)

        # 3. 构建消息列表
        # 优先级: --context (历史) > config.json 中的 messages > 命令行 prompt
        messages = []

        # A. 加载历史记录
        if args.context:
            try:
                with open(args.context, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if isinstance(history, list):
                        messages.extend(history)
                    elif isinstance(history, dict) and "messages" in history:
                        messages.extend(history["messages"])
                print(
                    f"{Colors.info('✓')} 已加载历史对话: {args.context}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"{Colors.error(f'❌ 加载历史对话失败: {e}')}", file=sys.stderr)
                sys.exit(1)

        # B. 加载配置文件中的 messages
        if "messages" in payload:
            if isinstance(payload["messages"], list):
                messages.extend(payload["messages"])
            # 从 payload 中移除 messages，稍后统一设置
            del payload["messages"]

        # C. 追加当前用户输入
        if args.prompt:
            user_messages, _ = FileContextBuilder.build_user_messages(args)
            messages.extend(user_messages)

        if not messages:
            print(
                f"{Colors.error('❌ 错误: 没有可发送的消息。请提供 prompt 或在 config.json 中包含 messages。')}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 4. 更新 payload 中的 messages
        payload["messages"] = messages

        return endpoint, api_key, payload


# ============ File & Context Handling ============


class FileContextBuilder:
    """处理文件上传与上下文构建"""

    @staticmethod
    def load_file_content(filepath: str) -> Any:
        """加载文件内容，支持文本和图片"""
        try:
            if filepath.startswith("@"):
                filepath = filepath[1:]

            path = Path(filepath).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {filepath}")

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = "text/plain"

            # 图片处理
            if mime_type.startswith("image/"):
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }

            # 文本处理
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # 简单的代码块标记逻辑
            if path.suffix in [".md", ".txt", ".rst"]:
                return f"{content}\n\n[文件: {path.name}]"
            else:
                # 代码或配置文件
                # 使用常量 CODE_BLOCK 替换 ``` 以避免格式冲突
                suffix = path.suffix[1:] if path.suffix else "text"
                return f"{CODE_BLOCK}{suffix}\n{content}\n{CODE_BLOCK}\n[文件: {path.name}]"

        except Exception as e:
            print(f"{Colors.warn('⚠')} 加载文件失败 {filepath}: {e}", file=sys.stderr)
            return f"[无法加载文件: {filepath}]"

    @staticmethod
    def build_user_messages(args) -> Tuple[List[Dict], str]:
        """构建用户输入的 messages"""
        if not args.prompt:
            return [], "chat"

        content_parts = []
        has_image = False

        for part in args.prompt:
            if part.startswith("@"):
                file_content = FileContextBuilder.load_file_content(part)
                if isinstance(file_content, dict):  # 图片
                    content_parts.append(file_content)
                    has_image = True
                else:
                    content_parts.append(file_content)
            else:
                content_parts.append(part)

        # 构建消息体
        if has_image:
            multimodal_content = []
            current_text = []
            for part in content_parts:
                if isinstance(part, dict):  # 图片
                    if current_text:
                        multimodal_content.append(
                            {"type": "text", "text": "\n".join(current_text)}
                        )
                        current_text = []
                    multimodal_content.append(part)
                else:
                    current_text.append(part)

            if current_text:
                multimodal_content.append(
                    {"type": "text", "text": "\n\n".join(current_text)}
                )

            return [{"role": "user", "content": multimodal_content}], args.prompt[0]
        else:
            user_content = "\n\n".join(content_parts)
            return [{"role": "user", "content": user_content}], args.prompt[0]


# ============ Typewriter Effect & Output ============


class TypewriterPrinter:
    """处理打字机效果和文件输出"""

    def __init__(self, output_file: Optional[io.TextIOWrapper] = None):
        self.output_file = output_file
        self.reasoning_buffer = ""
        self.content_buffer = ""
        self.reasoning_printed_len = 0
        self.content_printed_len = 0
        self.in_reasoning = True

        # 统计
        self.start_time = time.time()
        self.first_token_time = None
        self.tokens_count = 0

    def write(self, text: str):
        """写入文件"""
        if self.output_file:
            try:
                self.output_file.write(text)
                self.output_file.flush()
            except Exception as e:
                print(f"\n{Colors.error('[文件写入错误]')} {e}", file=sys.stderr)

    def update_reasoning(self, delta: str):
        """更新并打印推理过程"""
        if not delta:
            return

        if self.first_token_time is None:
            self.first_token_time = time.time()

        self.reasoning_buffer += delta
        new_text = self.reasoning_buffer[self.reasoning_printed_len :]

        # 打印到终端
        sys.stdout.write(new_text)
        sys.stdout.flush()

        # 写入文件
        self.write(new_text)

        self.reasoning_printed_len = len(self.reasoning_buffer)

    def switch_to_content(self):
        """切换到正文输出模式"""
        if self.in_reasoning:
            self.write("\n\n---\n\n")  # 文件分隔符
            sys.stdout.write(f"\n{Colors.BOLD}{'='*50}{Colors.END}\n")
            sys.stdout.write(f"{Colors.CYAN}✨ 正式回复：{Colors.END}\n")
            sys.stdout.write(f"{Colors.BOLD}{'='*50}{Colors.END}\n")
            sys.stdout.flush()
            self.in_reasoning = False

    def update_content(self, delta: str):
        """更新并打印正文内容"""
        if not delta:
            return

        if self.in_reasoning:
            self.switch_to_content()

        if self.first_token_time is None:
            self.first_token_time = time.time()

        self.content_buffer += delta
        new_text = self.content_buffer[self.content_printed_len :]

        sys.stdout.write(new_text)
        sys.stdout.flush()
        self.write(new_text)

        self.content_printed_len = len(self.content_buffer)
        self.tokens_count += 1  # 估算

    def finalize(self):
        """完成输出"""
        if self.output_file:
            self.output_file.flush()


# ============ HTTP Client with Retry ============


class APIClient:
    """HTTP 客户端，支持重试和流式解析"""

    def __init__(self, endpoint: str, api_key: str, max_retries: int = 3):
        self.endpoint = endpoint
        self.api_key = api_key
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def request(
        self, payload: Dict[str, Any], printer: TypewriterPrinter
    ) -> Dict[str, Any]:
        """执行请求，处理流式和非流式，支持重试"""
        retry_count = 0
        last_error = None

        # 从 payload 中提取 metadata 用于显示
        metadata = {
            "model": payload.get("model", "unknown"),
            "start_time": datetime.now().isoformat(),
            "finish_reason": None,
            "usage": {},
        }

        is_stream = payload.get("stream", False)
        print(f"\n{Colors.CYAN}🚀 正在请求 {metadata['model']}...{Colors.END}")

        while retry_count <= self.max_retries:
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    stream=is_stream,
                    timeout=300,
                )
                response.raise_for_status()

                if not is_stream:
                    return self._handle_non_stream(response, printer, metadata)
                else:
                    return self._handle_stream(response, printer, metadata)

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                last_error = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    wait_time = 2**retry_count
                    print(
                        f"{Colors.warn(f'⚠️ 请求失败 ({e})，{wait_time}秒后重试... ({retry_count}/{self.max_retries}')}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                else:
                    raise

        raise Exception(f"请求失败，已达到最大重试次数: {last_error}")

    def _handle_non_stream(
        self, response, printer: TypewriterPrinter, metadata: Dict
    ) -> Dict:
        """处理非流式响应"""
        data = response.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})

        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")

        if reasoning:
            print(f"\n{Colors.BLUE}💭 Thinking:{Colors.END}")
            printer.update_reasoning(reasoning)

        printer.switch_to_content()
        printer.update_content(content)

        metadata["finish_reason"] = choice.get("finish_reason")
        metadata["usage"] = data.get("usage", {})
        metadata["full_content"] = content
        metadata["full_reasoning"] = reasoning

        return metadata

    def _handle_stream(
        self, response, printer: TypewriterPrinter, metadata: Dict
    ) -> Dict:
        """处理流式响应"""
        buffer = ""
        full_reasoning = ""
        full_content = ""

        # 打印 Thinking 标题（如果启用）
        if printer.in_reasoning:
            print(f"\n{Colors.BLUE}💭 Thinking:{Colors.END}")

        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            buffer += chunk.decode("utf-8", errors="replace")

            # 按行分割处理 SSE
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    if finish_reason:
                        metadata["finish_reason"] = finish_reason

                    # 处理 Usage (通常在最后)
                    if "usage" in data:
                        metadata["usage"] = data["usage"]

                    # 处理内容
                    reasoning_delta = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    content_delta = delta.get("content", "")

                    if reasoning_delta:
                        printer.update_reasoning(reasoning_delta)
                        full_reasoning += reasoning_delta

                    if content_delta:
                        printer.update_content(content_delta)
                        full_content += content_delta

                except json.JSONDecodeError:
                    # 忽略不完整的 JSON 块
                    continue

        metadata["full_content"] = full_content
        metadata["full_reasoning"] = full_reasoning
        return metadata


# ============ Argument Parser ============


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Chat CLI (Strict) - 需要配置文件",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
配置说明:
  1. profiles.json: 包含 endpoint 和 api_key。
     格式: { "default": { "endpoint": "...", "api_key": "..." } }
  2. config.json: 包含请求体 (model, temperature 等)。
     格式: { "model": "...", "temperature": 0.7, ... }

示例:
  # 使用默认配置 (profiles.json 中的 default, config.json)
  python chat.py "你好"
  
  # 使用指定配置
  python chat.py -p deepseek -f deepseek_config.json "解释 Python"
  
  # 结合文件
  python chat.py @code.py "解释这段代码"
  
  # 继续历史
  python chat.py -c history.json "接着刚才的说"
        """,
    )

    # 基础配置
    parser.add_argument(
        "--profile",
        "-p",
        default="default",
        help="profiles.json 中的预设名称 (默认: default)",
    )
    parser.add_argument(
        "--config",
        "-f",
        default="config.json",
        help="请求体配置文件 (默认: config.json)",
    )

    # 输入
    parser.add_argument("prompt", nargs="*", help="提示词 (支持 @文件名)")
    parser.add_argument("--context", "-c", help="历史对话JSON文件路径")

    # 输出
    parser.add_argument("--output", "-o", help="输出文件路径 (默认自动生成)")

    return parser.parse_args()


# ============ Main & Save Logic ============


def save_result(
    output_path: Path, payload: Dict, metadata: Dict, printer: TypewriterPrinter
):
    """保存结果到 Markdown 和 JSON"""

    # 1. Markdown
    # 移除 messages 字段以避免日志过长，保留参数
    display_payload = payload.copy()
    if "messages" in display_payload:
        display_payload["messages"] = (
            f"[{len(display_payload['messages'])} messages hidden]"
        )

    # 使用常量 CODE_BLOCK 替换 ``` 以避免格式冲突
    md_content = f"""# AI 对话记录
**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**模型**: {metadata['model']}  

## 配置
{CODE_BLOCK}json
{json.dumps(display_payload, indent=2, ensure_ascii=False)}
{CODE_BLOCK}

---
"""
    # 添加回复
    md_content += "\n## 回复\n"
    if metadata.get("full_reasoning"):
        md_content += f"<details><summary>💭 Thinking Process</summary>\n\n{metadata['full_reasoning']}\n\n</details>\n\n"

    md_content += f"{metadata['full_content']}\n\n---\n## 统计\n"
    usage = metadata.get("usage", {})
    if usage:
        md_content += f"- Tokens: {usage.get('total_tokens', 'N/A')} (Prompt: {usage.get('prompt_tokens')}, Completion: {usage.get('completion_tokens')})\n"
    md_content += f"- Finish Reason: {metadata.get('finish_reason')}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. JSON (用于继续对话)
    json_path = output_path.with_suffix(".json")
    new_messages = payload.get("messages", []).copy()
    new_messages.append(
        {
            "role": "assistant",
            "content": metadata.get("full_content", ""),
            "reasoning_content": metadata.get("full_reasoning", ""),
        }
    )

    # 保存完整上下文
    save_context = {"messages": new_messages}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_context, f, ensure_ascii=False, indent=2)

    print(f"\n{Colors.info('💾 已保存:')} {output_path} (Markdown)")
    print(f"{Colors.info('💾 已保存:')} {json_path} (JSON)")


def main():
    args = parse_arguments()

    try:
        # 1. 构建配置
        endpoint, api_key, payload = ConfigManager.build_final_payload(args)

        # 2. 准备输出文件
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 尝试从 prompt 或 model 生成文件名
            name_hint = (
                args.prompt[0][:20] if args.prompt else payload.get("model", "chat")
            )
            safe_name = re.sub(r"[^\w\s-]", "", name_hint).strip().replace(" ", "_")
            output_path = Path(f"chat_{safe_name}_{timestamp}.md")

        tee_file = open(output_path, "w", encoding="utf-8")

        try:
            # 3. 初始化打印器和客户端
            printer = TypewriterPrinter(tee_file)
            client = APIClient(endpoint, api_key)

            # 4. 发起请求
            metadata = client.request(payload, printer)
            printer.finalize()

            # 5. 打印统计信息
            duration = time.time() - printer.start_time
            print(f"\n{Colors.GREEN}✅ 完成{Colors.END} (耗时: {duration:.2f}s)")

            usage = metadata.get("usage", {})
            if usage:
                print(
                    f"📊 Tokens: {usage.get('total_tokens')} | Prompt: {usage.get('prompt_tokens')} | Completion: {usage.get('completion_tokens')}"
                )

            # 6. 保存文件
            save_result(output_path, payload, metadata, printer)

        except KeyboardInterrupt:
            print(f"\n{Colors.warn('⚠️ 用户中断')}", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\n{Colors.error(f'❌ 错误: {e}')}", file=sys.stderr)
            raise
        finally:
            tee_file.close()

    except Exception as e:
        # 配置阶段错误已在 ConfigManager 中处理并退出，这里捕获意外错误
        print(f"{Colors.error(str(e))}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
