#!/usr/bin/env python3
"""
打字机效果的HTTP客户端 - 支持OpenAI格式JSON配置、文件上下文、历史对话和Tee输出
"""

import json
import time
import sys
import io
import os
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests
from dotenv import load_dotenv

THREE_DOTS = "`" * 3

# ============ UTF-8 编码设置 ============


def setup_utf8():
    """设置UTF-8编码环境"""
    if sys.platform == "win32":
        import ctypes

        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except:
            pass

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )


setup_utf8()

# ============ 配置加载 ============


def load_config(args):
    """从.env和profile加载配置"""
    # 1. 加载基础 .env 配置
    env_paths = [".env", "../.env", os.path.expanduser("~/.ai_chat.env")]
    base_config = {
        "endpoint": "",
        "api_key": "",
        "model": "Pro/moonshotai/Kimi-K2.5",
    }
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✓ 已加载环境配置: {env_path}", file=sys.stderr)
            break
    
    base_config.update({
        "endpoint": os.getenv("ENDPOINT", ""),
        "api_key": os.getenv("API_KEY", ""),
        "model": os.getenv("MODEL", base_config["model"]),
    })

    # 2. 加载 Profile 配置（从 ~/.ai_chat_profiles.json）
    profile_config = load_profile(args.profile)
    
    # 3. 合并：Profile > .env > 默认
    final_config = {**base_config, **profile_config}
    
    return final_config


def load_profile(profile_name: str) -> Dict:
    """加载指定 profile 的配置"""
    profile_paths = [
        Path("profiles.json"),
        Path("../profiles.json"),
        Path.home() / ".ai_chat_profiles.json",
    ]
    
    for path in profile_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                
                if profile_name in profiles:
                    print(f"✓ 已加载预设: {profile_name} ({path})", file=sys.stderr)
                    return profiles[profile_name]
                elif profile_name != "default":
                    print(f"⚠️  未找到预设 '{profile_name}'，使用默认配置", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  加载 profile 失败: {e}", file=sys.stderr)
    
    return {}


# ============ OpenAI格式配置构建 ============


def build_request_body(args, config: Dict, messages: List[Dict]) -> Dict[str, Any]:
    """
    构建OpenAI格式的请求体
    优先级: 命令行参数 > JSON配置文件 > .env > 默认值
    """
    # 1. 从JSON文件加载基础配置（如果提供）
    request_body = {}
    if args.config:
        try:
            config_path = Path(args.config).expanduser().resolve()
            with open(config_path, "r", encoding="utf-8") as f:
                request_body = json.load(f)
            print(f"✓ 已加载JSON配置: {config_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  加载JSON配置失败: {e}", file=sys.stderr)
            raise

    # 2. 处理messages（合并历史+用户输入）
    final_messages = request_body.get("messages", []).copy()

    # 加载历史对话（--context参数，追加到messages）
    if args.context:
        try:
            with open(args.context, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                if isinstance(history_data, list):
                    final_messages.extend(history_data)
                elif isinstance(history_data, dict) and "messages" in history_data:
                    final_messages.extend(history_data["messages"])
            print(f"✓ 已加载历史对话: {args.context}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  加载历史对话失败: {e}", file=sys.stderr)

    # 添加当前用户输入
    if messages:
        final_messages.extend(messages)

    if final_messages:
        request_body["messages"] = final_messages

    # 3. 命令行参数覆盖JSON配置（OpenAI标准参数）
    if args.model:
        request_body["model"] = args.model
    elif "model" not in request_body:
        request_body["model"] = config["model"]

    if args.temperature is not None:
        request_body["temperature"] = args.temperature
    elif "temperature" not in request_body:
        request_body["temperature"] = 0.7

    if args.max_tokens is not None:
        request_body["max_tokens"] = args.max_tokens
    elif "max_tokens" not in request_body:
        request_body["max_tokens"] = 8192

    # 流式输出设置
    if args.no_stream:
        request_body["stream"] = False
    elif "stream" not in request_body:
        request_body["stream"] = True

    # 特定API扩展参数（如enable_thinking）
    if hasattr(args, "enable_thinking") and args.enable_thinking is not None:
        request_body["enable_thinking"] = args.enable_thinking
    elif "enable_thinking" not in request_body:
        # request_body["enable_thinking"] = True
        pass # deepseek does not support this flag?

    # 其他OpenAI标准参数（如果JSON中有，保留）
    # top_p, presence_penalty, frequency_penalty, stop, seed 等

    return request_body


# ============ 参数解析 ============


def parse_arguments():
    """解析参数，支持@文件名语法和OpenAI格式JSON配置"""
    parser = argparse.ArgumentParser(
        description="AI Chat CLI - 支持OpenAI格式JSON配置、文件上下文和Tee输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用JSON配置（OpenAI格式）
  python chat.py --config request.json
  
  # JSON配置 + 命令行覆盖
  python chat.py --config request.json --model "gpt-4" --temperature 0.5
  
  # 传统用法：直接输入提示
  python chat.py "你好"
  python chat.py @document.txt "总结这个文件"
  
  # 完整示例
  python chat.py --config base.json --history chat.json @code.py "解释代码" -o result.md
  
OpenAI JSON格式示例:
  {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stream": true,
    "enable_thinking": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"}
    ]
  }
        """,
    )
    
    parser.add_argument("--profile", "-p", default="default", 
                   help="使用预设配置 (如: kimi, gpt4, local)")
    
    # OpenAI格式JSON配置
    parser.add_argument(
        "--config", "-f", default="config.json", help="OpenAI格式的JSON配置文件路径"
    )

    # 输入提示（支持@文件名）
    parser.add_argument("prompt", nargs="*", help="输入提示（支持@文件名加载文件内容）")

    # 历史对话
    parser.add_argument(
        "--context", "-c", help="加载历史对话JSON文件（追加到messages）"
    )

    # 输出设置
    parser.add_argument("--output", "-o", help="输出文件路径（默认自动生成）")

    # API配置（覆盖JSON和.env）
    parser.add_argument("--endpoint", "-e", help="API端点（覆盖.env配置）")
    parser.add_argument("--api-key", "-k", help="API密钥（覆盖.env配置）")
    parser.add_argument("--model", "-m", help="模型名称（覆盖JSON和.env配置）")

    # OpenAI标准参数（覆盖JSON）
    parser.add_argument("--temperature", "-t", type=float, help="温度参数(0-2)")
    parser.add_argument("--max-tokens", type=int, help="最大token数")
    parser.add_argument("--top-p", type=float, help="核采样概率")
    parser.add_argument("--presence-penalty", type=float, help="存在惩罚")
    parser.add_argument("--frequency-penalty", type=float, help="频率惩罚")
    parser.add_argument("--seed", type=int, help="随机种子")

    # 流式与思考选项
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=None,
        help="启用思考过程（特定API支持）",
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="禁用思考过程",
    )

    return parser.parse_args()


# ============ 文件上下文处理 ============


def load_file_content(filepath: str):
    """加载文件内容，支持相对路径和绝对路径"""
    try:
        if filepath.startswith("@"):
            filepath = filepath[1:]

        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        suffix = path.suffix.lower()

        # 图片文件处理
        if suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            import base64

            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            mime_type = f"image/{suffix[1:]}" if suffix != ".jpg" else "image/jpeg"
            # OpenAI多模态格式
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }

        # 文本文件
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 根据文件类型添加代码块标记
        if suffix in [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".rb",
            ".php",
        ]:
            return f"{THREE_DOTS}{suffix[1:]}\n{content}\n{THREE_DOTS}\n[文件: {path.name}]"
        elif suffix in [".md", ".txt", ".rst"]:
            return f"{content}\n\n[文件: {path.name}]"
        elif suffix in [".json", ".yaml", ".yml", ".xml"]:
            return f"{THREE_DOTS}yaml\n{content}\n{THREE_DOTS}\n[文件: {path.name}]"
        else:
            return f"{THREE_DOTS}\n{content}\n{THREE_DOTS}\n[文件: {path.name}]"

    except Exception as e:
        print(f"⚠️  加载文件失败 {filepath}: {e}", file=sys.stderr)
        return f"[无法加载文件: {filepath}]"


def build_user_messages(args) -> Tuple[List[Dict], str]:
    """构建用户输入的messages（支持@文件语法）"""
    if not args.prompt:
        return [], "chat"

    content_parts = []
    has_image = False

    for part in args.prompt:
        if part.startswith("@"):
            file_content = load_file_content(part)
            if isinstance(file_content, dict):  # 图片（OpenAI多模态格式）
                content_parts.append(file_content)
                has_image = True
            else:
                content_parts.append(file_content)
        else:
            content_parts.append(part)

    # 如果有图片，使用OpenAI多模态content格式（数组）
    if has_image:
        # 将纯文本部分合并，保持顺序
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
        # 纯文本，传统格式
        user_content = "\n\n".join(content_parts)
        return [{"role": "user", "content": user_content}], args.prompt[0]


# ============ 核心：安全的打字机打印 ============


class SafePrinter:
    """安全的打字机打印器，正确处理中文，同时支持Tee输出"""

    def __init__(self, tee_file: Optional[io.TextIOWrapper] = None):
        self.tee_file = tee_file
        self.reasoning_printed_chars = 0
        self.content_printed_chars = 0
        self.in_reasoning_phase = True
        self.full_reasoning = ""
        self.full_content = ""

    def write_to_file(self, text: str, is_reasoning: bool = False):
        """写入到文件（不打印）"""
        if self.tee_file:
            try:
                self.tee_file.write(text)
                self.tee_file.flush()
            except Exception as e:
                print(f"\n[文件写入错误: {e}]", file=sys.stderr)

    def print_reasoning(self, full_reasoning: str, delay: float = 0.01) -> None:
        """打印thinking过程，只打印新增部分"""
        if not full_reasoning:
            return

        if isinstance(full_reasoning, bytes):
            full_reasoning = full_reasoning.decode("utf-8", errors="replace")

        self.full_reasoning = full_reasoning
        new_part = full_reasoning[self.reasoning_printed_chars :]

        for char in new_part:
            print(char, end="", flush=True)
            if self.tee_file:
                self.write_to_file(char, is_reasoning=True)
            time.sleep(delay)

        self.reasoning_printed_chars = len(full_reasoning)

    def print_content(self, full_content: str, delay: float = 0.03) -> None:
        """打印正式回复，只打印新增部分"""
        if not full_content:
            return

        if isinstance(full_content, bytes):
            full_content = full_content.decode("utf-8", errors="replace")

        self.full_content = full_content
        new_part = full_content[self.content_printed_chars :]

        for char in new_part:
            print(char, end="", flush=True)
            if self.tee_file:
                self.write_to_file(char, is_reasoning=False)
            time.sleep(delay)

        self.content_printed_chars = len(full_content)

    def switch_to_content(self) -> None:
        """从thinking切换到content阶段"""
        if self.in_reasoning_phase:
            print()
            print("\n" + "-" * 50)
            print("✨ 正式回复：")
            print("-" * 50)
            if self.tee_file:
                self.write_to_file("\n\n---\n✨ 正式回复：\n---\n\n")
            self.in_reasoning_phase = False

    def finalize(self):
        """完成输出，确保文件写入"""
        if self.tee_file:
            self.tee_file.flush()


# ============ HTTP客户端 ============


class TypewriterHTTPClient:
    """打字机效果的HTTP客户端"""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def stream_request(
        self, request_body: Dict[str, Any], printer: SafePrinter
    ) -> Tuple[str, str, Dict]:
        """
        流式请求，打字机效果显示thinking和content
        request_body: OpenAI格式的完整请求体
        返回: (reasoning, content, metadata)
        """
        print("=" * 70)
        print("正在发送请求...")
        print(f"Endpoint: {self.endpoint}")
        print(f"Model: {request_body.get('model', 'unknown')}")
        print(f"Stream: {request_body.get('stream', True)}")
        print(f"Messages: {len(request_body.get('messages', []))} 轮对话")

        # 显示其他OpenAI参数
        params = {
            k: v
            for k, v in request_body.items()
            if k not in ["messages", "model", "stream"] and v is not None
        }
        if params:
            print(f"Parameters: {json.dumps(params, ensure_ascii=False)}")

        full_reasoning = ""
        full_content = ""
        metadata = {
            "start_time": datetime.now().isoformat(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": None,
            "request_body": request_body,  # 记录完整请求
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=request_body,
                stream=request_body.get("stream", True),
                timeout=120000,
            )
            response.raise_for_status()

            print(f"\n连接成功 (Status: {response.status_code})")
            print("-" * 70)

            if not request_body.get("stream", True):
                # 非流式处理
                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                full_content = message.get("content", "")
                full_reasoning = message.get("reasoning_content", "")

                # 打印结果
                if full_reasoning:
                    print("\n💭 Thinking 过程：")
                    print("-" * 50)
                    print(full_reasoning)
                    printer.write_to_file(full_reasoning)

                print("\n✨ 正式回复：")
                print("-" * 50)
                print(full_content)
                printer.write_to_file(full_content)

                # 更新metadata
                if "usage" in data:
                    metadata.update(
                        {
                            "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                            "completion_tokens": data["usage"].get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": data["usage"].get("total_tokens", 0),
                        }
                    )
                metadata["finish_reason"] = choice.get("finish_reason")

            else:
                # 流式处理
                header_printed = False
                buffer_bytes = b""

                for chunk in response.iter_content(chunk_size=128):
                    if not chunk:
                        continue

                    buffer_bytes += chunk

                    while b"\n" in buffer_bytes:
                        line_bytes, buffer_bytes = buffer_bytes.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()

                        if line.startswith("data: "):
                            data_str = line[6:]

                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)

                                # 更新usage信息（通常在最后一条）
                                if "usage" in data and data["usage"]:
                                    metadata["prompt_tokens"] = data["usage"].get(
                                        "prompt_tokens", 0
                                    )
                                    metadata["completion_tokens"] = data["usage"].get(
                                        "completion_tokens", 0
                                    )
                                    metadata["total_tokens"] = data["usage"].get(
                                        "total_tokens", 0
                                    )

                                choices = data.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason:
                                    metadata["finish_reason"] = finish_reason

                                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                                content_delta = delta.get("content")

                                # 打印header（第一次收到数据）
                                if not header_printed and (
                                    reasoning_delta or content_delta
                                ):
                                    header_printed = True
                                    if reasoning_delta or request_body.get(
                                        "enable_thinking"
                                    ):
                                        print("\n💭 Thinking 过程：")
                                        print("-" * 50)
                                        if printer.tee_file:
                                            printer.write_to_file(
                                                "\n💭 Thinking 过程：\n"
                                                + "-" * 50
                                                + "\n"
                                            )

                                # 累积并打印reasoning
                                if reasoning_delta and isinstance(reasoning_delta, str):
                                    full_reasoning += reasoning_delta
                                    printer.print_reasoning(full_reasoning)

                                # 切换到content阶段
                                if content_delta and printer.in_reasoning_phase:
                                    printer.switch_to_content()

                                # 累积并打印content
                                if content_delta and isinstance(content_delta, str):
                                    full_content += content_delta
                                    printer.print_content(full_content)

                            except json.JSONDecodeError:
                                pass

                # 处理剩余buffer
                if buffer_bytes:
                    try:
                        line = buffer_bytes.decode("utf-8", errors="replace").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str and data_str != "[DONE]":
                                data = json.loads(data_str)
                                # 处理最后的数据...
                    except:
                        pass

            print("\n" + "=" * 70)
            print("✅ 请求完成")

            if full_reasoning:
                print(f"\n📊 Thinking: {len(full_reasoning)} 字符")
            print(f"📊 Content: {len(full_content)} 字符")
            if metadata["total_tokens"]:
                print(
                    f"📊 Tokens: {metadata['total_tokens']} (Prompt: {metadata['prompt_tokens']}, Completion: {metadata['completion_tokens']})"
                )
            if metadata["finish_reason"]:
                print(f"📊 Finish reason: {metadata['finish_reason']}")

            metadata["end_time"] = datetime.now().isoformat()
            metadata["reasoning_chars"] = len(full_reasoning)
            metadata["content_chars"] = len(full_content)

            return full_reasoning, full_content, metadata

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback

            traceback.print_exc()
            raise


# ============ 文件保存 ============


def save_conversation(
    output_path: Path,
    reasoning: str,
    content: str,
    metadata: Dict,
    messages: List[Dict],
):
    """保存对话到文件（Markdown格式）"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    request_body = metadata.get("request_body", {})

    md_content = f"""# AI 对话记录

**时间**: {timestamp}  
**模型**: {request_body.get('model', metadata.get('model', 'unknown'))}  
**Token 消耗**: {metadata.get('total_tokens', 'N/A')} (Prompt: {metadata.get('prompt_tokens', 'N/A')}, Completion: {metadata.get('completion_tokens', 'N/A')})  
**结束原因**: {metadata.get('finish_reason', 'N/A')}

## 请求参数

{THREE_DOTS}json
{json.dumps({k: v for k, v in request_body.items() if k != 'messages'}, indent=2, ensure_ascii=False)}
{THREE_DOTS}

---

## 对话历史

"""

    # 添加对话历史
    for msg in messages:
        role = msg.get("role", "unknown")
        content_text = msg.get("content", "")

        # 处理多模态content（数组格式）
        if isinstance(content_text, list):
            texts = []
            for item in content_text:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    texts.append("[图片]")
            content_text = "\n".join(texts)

        md_content += f"### {role.upper()}\n\n{content_text}\n\n"

    # 添加当前回复
    md_content += f"""---

## 当前回复

"""

    if reasoning:
        md_content += f"""<details>
<summary>💭 Thinking 过程 ({metadata.get('reasoning_chars', len(reasoning))} 字符)</summary>

{reasoning}
</details>

"""

    md_content += f"""### ✨ 正式回复

{content}

---

## 元数据

{THREE_DOTS}json
{json.dumps(metadata, indent=2, ensure_ascii=False, default=str)}
{THREE_DOTS}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return output_path


def save_json_history(
    output_path: Path,
    messages: List[Dict],
    reasoning: str,
    content: str,
    metadata: Dict,
):
    """保存为JSON格式（OpenAI兼容，便于后续加载继续对话）"""
    # 添加助手回复到消息列表
    assistant_message = {"role": "assistant", "content": content}
    if reasoning:
        assistant_message["reasoning_content"] = reasoning

    full_messages = messages.copy()
    full_messages.append(assistant_message)

    data = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "total_tokens": metadata.get("total_tokens"),
            "model": metadata.get("request_body", {}).get("model"),
        },
        "messages": full_messages,
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path


# ============ 主程序 ============


def main():
    # 加载环境配置
    # 原来是：env_config = load_config()
    # 改为：
    args = parse_arguments()
    env_config = load_config(args)

    # 确定API端点和密钥
    endpoint = args.endpoint or env_config["endpoint"]
    api_key = args.api_key or env_config["api_key"]

    if not endpoint:
        print(
            "❌ 错误: 未设置API端点。请使用--endpoint参数、-f JSON配置或设置.env文件",
            file=sys.stderr,
        )
        sys.exit(1)

    if not api_key:
        print(
            "❌ 错误: 未设置API密钥。请使用--api-key参数、-f JSON配置或设置.env文件",
            file=sys.stderr,
        )
        sys.exit(1)

    # 构建用户输入的messages（处理@文件）
    user_messages, prompt_hint = build_user_messages(args)

    # 构建完整的OpenAI格式请求体
    try:
        request_body = build_request_body(args, env_config, user_messages)
    except Exception as e:
        print(f"❌ 构建请求失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 检查是否有messages
    if not request_body.get("messages"):
        print(
            "❌ 错误: 没有输入内容。请提供提示文本、使用@文件加载，或在JSON配置中提供messages。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 确定输出文件
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_hint = re.sub(r"[^\w\s-]", "", prompt_hint)[:20].strip() or "chat"
        output_path = Path(f"chat_{safe_hint}_{timestamp}.md")

    # 打开输出文件（Tee模式）
    tee_file = open(output_path, "w", encoding="utf-8")

    try:
        # 初始化打印器（带Tee）
        printer = SafePrinter(tee_file=tee_file)

        # 写入文件头
        tee_file.write(
            f"# AI对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        tee_file.write("## 请求配置\n\n")
        tee_file.write(
            f"{THREE_DOTS}json\n{json.dumps({k: v for k, v in request_body.items() if k != 'messages'}, indent=2, ensure_ascii=False)}\n{THREE_DOTS}\n\n"
        )
        tee_file.write("## 对话内容\n\n")

        # 创建客户端并发送请求
        client = TypewriterHTTPClient(endpoint=endpoint, api_key=api_key)

        reasoning, content, metadata = client.stream_request(request_body, printer)

        # 完成文件写入
        printer.finalize()

        # 保存完整对话记录（Markdown）
        save_conversation(
            output_path, reasoning, content, metadata, request_body["messages"]
        )

        # 同时保存JSON历史（便于继续对话）
        json_path = save_json_history(
            output_path, request_body["messages"], reasoning, content, metadata
        )

        print(f"\n💾 对话已保存:")
        print(f"   Markdown: {output_path.absolute()}")
        print(f"   JSON历史: {json_path.absolute()}")
        print(f"\n💡 提示: 使用 --context {json_path.name} 继续此对话")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 程序错误: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if tee_file:
            tee_file.close()


if __name__ == "__main__":
    main()
