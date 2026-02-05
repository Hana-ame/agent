#!/usr/bin/env python3
"""
Advanced Typewriter Effect HTTP Client for OpenAI-compatible APIs (Enhanced Version).
Zero-default policy with graceful fallback - avoids crashes when encountering unsupported parameters.
"""

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union

import requests
from dotenv import load_dotenv

# ============ Setup & Constants ============

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

CODE_BLOCK = "`" * 3


class Constants:
    """Application constants"""
    CHUNK_SIZE = 1024
    STREAM_DELAY_REASONING = 0.01
    STREAM_DELAY_CONTENT = 0.03
    TIMEOUT = 60*60  # 保持 1 小时超时


class Colors:
    """ANSI Color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
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
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.END}"


# ============ Exceptions ============

class ChatError(Exception):
    """Base exception for chat errors"""
    pass


class ConfigurationError(ChatError):
    """Missing required configuration"""
    pass


class APIError(ChatError):
    """API communication error"""
    pass


class FileProcessingError(ChatError):
    """File I/O error"""
    pass


# ============ UTF-8 Setup ============

def setup_utf8():
    """Force UTF-8 encoding for Windows and Unix"""
    if sys.platform == "win32":
        import ctypes
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True
        )


# ============ Data Models ============

@dataclass
class APIConfig:
    """API connection configuration - only endpoint and api_key"""

    endpoint: str = ""
    api_key: str = ""
    timeout: int = Constants.TIMEOUT  # HTTP timeout, not API parameter

    def validate(self) -> None:
        if not self.endpoint:
            raise ConfigurationError("API endpoint is required")
        if not self.api_key:
            raise ConfigurationError("API key is required")


@dataclass
class RequestMetadata:
    """Metadata for request tracking and reporting"""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    duration: float = 0.0

    def finalize(self) -> None:
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()

    def get_token_rate(self) -> Optional[float]:
        """Calculate tokens per second"""
        if self.duration > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.duration
        return None


# ============ Configuration Management ============

class ConfigManager:
    """Manages configuration loading with graceful fallback"""

    ENV_PATHS = [".env", "../.env", Path.home() / ".ai_chat.env"]
    PROFILE_PATHS = [
        Path("profiles.json"),
        Path("../profiles.json"),
        Path.home() / ".ai_chat_profiles.json",
    ]

    @staticmethod
    def load_connection_config(args) -> APIConfig:
        """Load endpoint and api_key with priority: CLI > Profile > Env > Default"""
        config = APIConfig()

        # 1. Load .env files first
        for env_path in ConfigManager.ENV_PATHS:
            if Path(env_path).exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment: {env_path}")
                break

        # 2. From environment variables
        config.endpoint = os.getenv("ENDPOINT", "")
        config.api_key = os.getenv("API_KEY", "")

        # 3. Try to load from profile
        profile_config = ConfigManager._load_profile(args.profile)
        if profile_config:
            if "endpoint" in profile_config:
                config.endpoint = profile_config["endpoint"]
            if "api_key" in profile_config:
                config.api_key = profile_config["api_key"]
            if "apiKey" in profile_config:
                config.api_key = profile_config["apiKey"]

        # 4. CLI arguments override everything
        if hasattr(args, "endpoint") and args.endpoint:
            config.endpoint = args.endpoint
        if hasattr(args, "api_key") and args.api_key:
            config.api_key = args.api_key

        return config

    @staticmethod
    def _load_profile(profile_name: str) -> Optional[Dict[str, Any]]:
        """Load profile from various locations"""
        for path in ConfigManager.PROFILE_PATHS:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        profiles = json.load(f)

                    if profile_name in profiles:
                        print(
                            f"{Colors.info('✓')} Loaded profile: {profile_name} from {path}",
                            file=sys.stderr,
                        )
                        return profiles[profile_name]
                except Exception as e:
                    logger.error(f"Failed to load profile {path}: {e}")

        if profile_name != "default":
            print(
                f"{Colors.warn('⚠')} Profile '{profile_name}' not found, using default",
                file=sys.stderr,
            )
        return None

    @staticmethod
    def load_request_body(config_path: str, args) -> Dict[str, Any]:
        """Load request body from config.json with safe parameter handling"""
        path = Path(config_path).expanduser().resolve()

        if not path.exists():
            if (
                config_path != "config.json"
            ):  # User explicitly specified non-existent file
                raise FileNotFoundError(f"Config file not found: {config_path}")
            # Use empty config if default config.json doesn't exist
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"{Colors.info('✓')} Loaded request config: {path}", file=sys.stderr)

            # Apply CLI overrides for safety-critical parameters
            ConfigManager._apply_cli_overrides(data, args)

            return data
        except Exception as e:
            raise ValueError(f"Failed to load config file ({config_path}): {e}")

    @staticmethod
    def _apply_cli_overrides(data: Dict[str, Any], args) -> None:
        """Apply CLI arguments to request body, removing unsupported parameters"""
        # Safety: Remove problematic parameters if CLI provides alternatives
        if hasattr(args, "model") and args.model:
            data["model"] = args.model
            # Remove any model-specific parameters that might conflict
            data.pop("temperature", None)  # Some models don't support temperature
            data.pop("top_p", None)
            data.pop("frequency_penalty", None)
            data.pop("presence_penalty", None)

        # OpenAI standard parameters override
        if hasattr(args, "temperature") and args.temperature is not None:
            data["temperature"] = args.temperature
        if hasattr(args, "max_tokens") and args.max_tokens is not None:
            data["max_tokens"] = args.max_tokens
        if hasattr(args, "top_p") and args.top_p is not None:
            data["top_p"] = args.top_p
        if hasattr(args, "presence_penalty") and args.presence_penalty is not None:
            data["presence_penalty"] = args.presence_penalty
        if hasattr(args, "frequency_penalty") and args.frequency_penalty is not None:
            data["frequency_penalty"] = args.frequency_penalty
        if hasattr(args, "seed") and args.seed is not None:
            data["seed"] = args.seed

        # Stream handling
        if hasattr(args, "no_stream") and args.no_stream:
            data["stream"] = False
        elif "stream" not in data:
            data["stream"] = True  # Default to stream for better UX

        # Thinking enable/disable
        if hasattr(args, "enable_thinking"):
            if args.enable_thinking is not None:
                data["enable_thinking"] = args.enable_thinking
                # Some APIs don't support reasoning_content, remove if disabled
                if not args.enable_thinking:
                    data.pop("reasoning_content", None)


# ============ File & Message Building ============

class MessageBuilder:
    """Builds messages with file context support"""

    IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    @staticmethod
    def load_file(filepath: str) -> Union[str, Dict[str, Any]]:
        """Load file content with MIME type detection"""
        if filepath.startswith("@"):
            filepath = filepath[1:]

        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "application/octet-stream"

        # Image handling
        if mime_type.startswith("image/") or path.suffix.lower() in MessageBuilder.IMAGE_TYPES:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }

        # Text file handling with encoding detection
        encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]
        content = None
        
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, errors="strict") as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        # Format based on file type
        suffix = path.suffix[1:] if path.suffix else "text"
        if path.suffix in [".md", ".txt", ".rst"]:
            return f"{content}\n\n[File: {path.name}]"
        else:
            return f"{CODE_BLOCK}{suffix}\n{content}\n{CODE_BLOCK}\n[File: {path.name}]"

    @staticmethod
    def build_messages(
        args, config_messages: List[Dict] | None = None
    ) -> List[Dict[str, Any]]:
        """Build final message list with priority: context > config > prompt"""
        messages = []

        # 1. Load context if provided
        if hasattr(args, "context") and args.context:
            try:
                context_messages = MessageBuilder._load_context(args.context)
                messages.extend(context_messages)
                print(
                    f"{Colors.info('✓')} Loaded conversation context: {args.context}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"{Colors.warn('⚠')} Failed to load context: {e}", file=sys.stderr
                )

        # 2. Add messages from config
        if config_messages:
            messages.extend(config_messages)

        # 3. Add current user input
        if hasattr(args, "prompt") and args.prompt:
            user_messages = MessageBuilder._build_user_messages(args.prompt)
            messages.extend(user_messages)

        if not messages:
            raise ConfigurationError(
                "No messages to send. Please provide prompt or messages in config."
            )

        return messages

    @staticmethod
    def _load_context(context_path: str) -> List[Dict]:
        """Load conversation history from JSON file"""
        path = Path(context_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "messages" in data:
            return data["messages"]
        else:
            raise ValueError(
                "Invalid context format: expected list or {messages: [...]}"
            )

    @staticmethod
    def _build_user_messages(prompt_parts: List[str]) -> List[Dict[str, Any]]:
        """Build user messages from prompt parts"""
        if not prompt_parts:
            return []

        content_parts = []
        has_image = False

        for part in prompt_parts:
            if part.startswith("@"):
                try:
                    file_content = MessageBuilder.load_file(part)
                    if isinstance(file_content, dict):
                        has_image = True
                        print(f"{Colors.info('✓')} Attached image: {part[1:]}", file=sys.stderr)
                    else:
                        print(f"{Colors.info('✓')} Attached file: {part[1:]}", file=sys.stderr)
                    content_parts.append(file_content)
                except Exception as e:
                    print(
                        f"{Colors.warn('⚠')} Failed to load file {part}: {e}",
                        file=sys.stderr,
                    )
                    content_parts.append(f"[Failed to load file: {part}]")
            else:
                content_parts.append(part)

        # Build message content
        if has_image:
            multimodal_content = []
            current_text = []

            for part in content_parts:
                if isinstance(part, dict):
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

            return [{"role": "user", "content": multimodal_content}]
        else:
            user_content = "\n\n".join(content_parts)
            return [{"role": "user", "content": user_content}]


# ============ Typewriter Output ============

class TypewriterPrinter:
    """Typewriter effect printer with file output"""

    def __init__(self, output_file: Optional[io.TextIOWrapper] = None):
        self.output_file = output_file
        self.reasoning_buffer = ""
        self.content_buffer = ""
        self.reasoning_printed = 0
        self.content_printed = 0
        self.in_reasoning = True
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0

        # Performance tracking
        self.reasoning_start = None
        self.content_start = None

    def write(self, text: str, is_reasoning: bool = False):
        """Write text to file"""
        if self.output_file:
            try:
                self.output_file.write(text)
                if not is_reasoning:
                    self.output_file.flush()
            except Exception as e:
                print(f"\n{Colors.error('[File write error]')} {e}", file=sys.stderr)

    def update_reasoning(self, delta: str):
        """Update and print reasoning content"""
        if not delta:
            return

        if self.first_token_time is None:
            self.first_token_time = time.time()

        if self.reasoning_start is None:
            self.reasoning_start = time.time()
            print(f"\n{Colors.BLUE}💭 Thinking:{Colors.END}")
            if self.output_file:
                self.write("\n💭 Thinking:\n", is_reasoning=True)

        self.reasoning_buffer += delta
        new_text = self.reasoning_buffer[self.reasoning_printed :]

        # Print to console with typewriter effect
        for char in new_text:
            print(char, end="", flush=True)
            time.sleep(Constants.STREAM_DELAY_REASONING)

        # Write to file
        self.write(new_text, is_reasoning=True)

        self.reasoning_printed = len(self.reasoning_buffer)
        self.token_count += len(delta.split())  # Rough estimate

    def switch_to_content(self):
        """Switch from reasoning to content output"""
        if self.in_reasoning:
            self.in_reasoning = False
            self.content_start = time.time()

            # Console output
            print(f"\n{Colors.bold('='*50)}")
            print(f"{Colors.CYAN}✨ Response:{Colors.END}")
            print(f"{Colors.bold('='*50)}")

            # File output
            if self.output_file:
                self.write(f"\n\n{'='*50}\n✨ Response:\n{'='*50}\n\n")

    def update_content(self, delta: str):
        """Update and print response content"""
        if not delta:
            return

        if self.in_reasoning:
            self.switch_to_content()

        if self.first_token_time is None:
            self.first_token_time = time.time()

        self.content_buffer += delta
        new_text = self.content_buffer[self.content_printed :]

        # Typewriter effect
        for char in new_text:
            print(char, end="", flush=True)
            time.sleep(Constants.STREAM_DELAY_CONTENT)
        
        self.write(new_text)

        self.content_printed = len(self.content_buffer)
        self.token_count += len(delta.split())  # Rough estimate

    def finalize(self):
        """Finalize output"""
        if self.output_file:
            try:
                self.output_file.flush()
            except Exception as e:
                print(f"{Colors.warn('⚠')} Failed to flush file: {e}", file=sys.stderr)


# ============ HTTP Client ============

class APIClient:
    """HTTP client with retry and error handling using Session"""

    def __init__(self, config: APIConfig, max_retries: int = 3):
        self.config = config
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def _estimate_tokens(self, messages: List[Dict]) -> Tuple[int, int]:
        """估算 Prompt 的 Token 数（保守估计：每 3 个字符 1 个 Token）"""
        try:
            # 将 messages 转为字符串计算
            text = json.dumps(messages, ensure_ascii=False)
            char_count = len(text)
            # 保守估计：英文约 4 字符/token，中文约 1-2 字符/token，混合按 3 计算
            estimated_tokens = char_count // 3
            return char_count, estimated_tokens
        except:
            return 0, 0

    def request(
        self, payload: Dict[str, Any], printer: TypewriterPrinter
    ) -> RequestMetadata:
        """Execute API request with retry logic"""
        metadata = RequestMetadata(model=payload.get("model"))
        is_stream = payload.get("stream", True)

        # 计算 Prompt 信息
        messages = payload.get("messages", [])
        char_count, estimated_tokens = self._estimate_tokens(messages)
        max_tokens = payload.get("max_tokens", "Not set")
        
        # 显示请求信息，包括 Token 预估
        print(
            f"\n{Colors.CYAN}🚀 Requesting {metadata.model or 'unknown model'}...{Colors.END}"
        )
        print(f"   Endpoint: {self.config.endpoint}")
        print(f"   Stream: {'Yes' if is_stream else 'No'}")
        print(f"   Prompt Size: {char_count} chars (Est. ~{estimated_tokens} tokens)")
        
        if max_tokens != "Not set":
            print(f"   Max Tokens: {max_tokens}")
            if isinstance(max_tokens, int) and estimated_tokens > 0:
                total_est = estimated_tokens + max_tokens
                print(f"   Total Est.: ~{total_est} tokens")
                # 警告：如果预估超过常见模型的 32k 或 128k 限制
                # if total_est > 32000:
                #     print(f"{Colors.warn('⚠️  Warning:')} High token usage! Risk of context overflow.")
        else:
            print(f"   Max Tokens: {Colors.warn('Not set')} (model default)")

        # Safety: Remove any empty or None parameters that might cause API errors
        clean_payload = self._clean_payload(payload)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.config.endpoint,
                    json=clean_payload,
                    stream=is_stream,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

                if is_stream:
                    return self._handle_stream(response, printer, metadata)
                else:
                    return self._handle_non_stream(response, printer, metadata)

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    print(
                        f"{Colors.warn(f'⚠️  Request failed ({e}), retrying in {wait_time}s...')}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                else:
                    raise APIError(
                        f"Request failed after {self.max_retries} retries: {e}"
                    )

        raise APIError("Max retries exceeded")

    def _clean_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Remove potentially problematic parameters"""
        clean = payload.copy()

        # Remove None values
        for key in list(clean.keys()):
            if clean[key] is None:
                del clean[key]

        # Remove empty strings
        for key in list(clean.keys()):
            if isinstance(clean[key], str) and not clean[key].strip():
                del clean[key]

        return clean

    def _handle_stream(
        self, response, printer: TypewriterPrinter, metadata: RequestMetadata
    ) -> RequestMetadata:
        """Handle streaming response"""
        buffer = ""

        for chunk in response.iter_content(chunk_size=Constants.CHUNK_SIZE):
            if not chunk:
                continue

            buffer += chunk.decode("utf-8", errors="replace")

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
                    self._process_stream_data(data, printer, metadata)
                except json.JSONDecodeError:
                    continue

        metadata.finalize()
        return metadata

    def _process_stream_data(
        self, data: Dict, printer: TypewriterPrinter, metadata: RequestMetadata
    ):
        """Process individual stream data chunk"""
        # Update usage
        if "usage" in data and data["usage"]:
            metadata.prompt_tokens = data["usage"].get("prompt_tokens", 0)
            metadata.completion_tokens = data["usage"].get("completion_tokens", 0)
            metadata.total_tokens = data["usage"].get("total_tokens", 0)

        choices = data.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})

        # Update finish reason
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            metadata.finish_reason = finish_reason

        # Process content
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
        content_delta = delta.get("content", "")

        if reasoning_delta:
            printer.update_reasoning(reasoning_delta)

        if content_delta:
            printer.update_content(content_delta)

    def _handle_non_stream(
        self, response, printer: TypewriterPrinter, metadata: RequestMetadata
    ) -> RequestMetadata:
        """Handle non-streaming response"""
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content", "")
        content = message.get("content", "")

        if reasoning:
            printer.update_reasoning(reasoning)

        printer.switch_to_content()
        printer.update_content(content)

        # Update metadata
        if "usage" in data:
            metadata.prompt_tokens = data["usage"].get("prompt_tokens", 0)
            metadata.completion_tokens = data["usage"].get("completion_tokens", 0)
            metadata.total_tokens = data["usage"].get("total_tokens", 0)

        metadata.finish_reason = choice.get("finish_reason")
        metadata.finalize()

        return metadata


# ============ Result Saving ============

class ResultSaver:
    """Saves conversation results to files"""

    @staticmethod
    def save(
        base_path: Path,
        payload: Dict[str, Any],
        metadata: RequestMetadata,
        printer: TypewriterPrinter,
    ) -> Tuple[Path, Path]:
        """Save results as Markdown and JSON"""
        # Create directory if needed
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save Markdown
        md_path = ResultSaver._save_markdown(base_path, payload, metadata, printer)

        # 2. Save JSON for continuation
        json_path = ResultSaver._save_json(base_path, payload, metadata, printer)

        return md_path, json_path

    @staticmethod
    def _save_markdown(
        base_path: Path,
        payload: Dict[str, Any],
        metadata: RequestMetadata,
        printer: TypewriterPrinter,
    ) -> Path:
        """Save conversation as Markdown with full history"""
        md_path = base_path.with_suffix(".md")

        # Prepare payload for display (hide messages to save space)
        display_payload = payload.copy()
        messages = display_payload.pop("messages", [])
        if messages:
            display_payload["messages"] = f"[{len(messages)} messages hidden]"

        # Build Markdown content
        md_content = f"""# AI Conversation Log
**Time**: {metadata.start_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Model**: {metadata.model or 'Unknown'}  
**Duration**: {metadata.duration:.2f}s  

## Request Configuration
{CODE_BLOCK}json
{json.dumps(display_payload, indent=2, ensure_ascii=False)}
{CODE_BLOCK}

## Conversation History

"""

        # Add all messages
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            # Handle multimodal content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            texts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            texts.append("[Image]")
                content = "\n".join(texts)
            
            md_content += f"### {role}\n\n{content}\n\n"

        # Add reasoning if present
        if printer.reasoning_buffer:
            md_content += f"""
---

## 💭 Thinking Process
<details>
<summary>Click to expand ({len(printer.reasoning_buffer)} chars)</summary>

{printer.reasoning_buffer}
</details>

"""

        # Add response
        md_content += f"""
---

## ✨ Response
{printer.content_buffer}

---
## 📊 Statistics

- **Finish Reason**: {metadata.finish_reason or 'N/A'}
- **Tokens**: {metadata.total_tokens or 'N/A'} (Prompt: {metadata.prompt_tokens or 'N/A'}, Completion: {metadata.completion_tokens or 'N/A'})
- **Token Rate**: {f'{metadata.get_token_rate():.1f} tokens/s' if metadata.get_token_rate() else 'N/A'}
- **First Token Latency**: {f'{(printer.first_token_time - printer.start_time):.2f}s' if printer.first_token_time else 'N/A'}
- **Estimated Total Tokens**: {printer.token_count}
"""

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_path

    @staticmethod
    def _save_json(
        base_path: Path,
        payload: Dict[str, Any],
        metadata: RequestMetadata,
        printer: TypewriterPrinter,
    ) -> Path:
        """Save conversation as JSON for continuation"""
        json_path = base_path.with_suffix(".json")

        # Build conversation for continuation
        continuation = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "model": metadata.model,
                "total_tokens": metadata.total_tokens,
                "duration": metadata.duration,
            },
            "messages": payload.get("messages", []).copy(),
        }

        # Add assistant response
        assistant_msg = {
            "role": "assistant",
            "content": printer.content_buffer,
        }
        if printer.reasoning_buffer:
            assistant_msg["reasoning_content"] = printer.reasoning_buffer

        continuation["messages"].append(assistant_msg)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(continuation, f, ensure_ascii=False, indent=2)

        return json_path


# ============ Argument Parser ============

def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Chat CLI - Enhanced with graceful parameter handling",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Configuration Strategy:
  1. Unified settings via config.json to avoid model-specific parameter issues
  2. CLI arguments safely override config.json (removes conflicting params)
  3. profiles.json or .env provides endpoint and api_key

Image Support:
  Use @filename.png to attach images (png, jpg, jpeg, gif, webp, bmp supported)
  Example: python chat.py @photo.jpg "Describe this image"

Token Estimation:
  Before sending, the CLI will display estimated prompt tokens and max_tokens
  to help you avoid context overflow.

Examples:
  # Basic usage with default config
  python chat.py "Hello, how are you?"
  
  # With file attachment (code or image)
  python chat.py @code.py "Explain this code"
  python chat.py @image.png "What's in this image?"
  
  # Custom profile and config
  python chat.py -p deepseek -f deepseek_config.json "Explain quantum computing"
  
  # Continue conversation
  python chat.py --context conversation.json "Continue from here"
  
  # Disable streaming
  python chat.py --no-stream "Generate a long response"
  
  # Override parameters
  python chat.py --temperature 0.5 --max-tokens 2000 "Creative writing"
        """,
    )

    # Connection
    parser.add_argument(
        "--profile", "-p", default="default", help="Profile name in profiles.json"
    )
    parser.add_argument("--endpoint", "-e", help="Override API endpoint")
    parser.add_argument("--api-key", "-k", help="Override API key")

    # Configuration
    parser.add_argument(
        "--config", "-c", default="config.json", help="Request body configuration file"
    )

    # Input
    parser.add_argument("prompt", nargs="*", help="Prompt text (supports @filename for files/images)")
    parser.add_argument("--context", help="Conversation context JSON file for continuation")

    # OpenAI Parameters (override config.json)
    parser.add_argument("--model", "-m", help="Model name (overrides config)")
    parser.add_argument("--temperature", "-t", type=float, help="Sampling temperature (0-2)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling parameter")
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Behavior (safely override config)
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming output"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=None,
        help="Enable thinking process",
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking process",
    )

    # Output
    parser.add_argument("--output", "-o", help="Output file base name")

    return parser


# ============ Main Function ============

def handle_interrupt(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n{Colors.warn('⚠️  Interrupted by user')}")
    sys.exit(130)


def main():
    """Main entry point"""
    # Setup
    setup_utf8()
    signal.signal(signal.SIGINT, handle_interrupt)

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # 1. Load connection configuration
        api_config = ConfigManager.load_connection_config(args)
        api_config.validate()

        # 2. Load request body from config.json
        request_body = ConfigManager.load_request_body(args.config, args)

        # 3. Build messages
        config_messages = request_body.pop(
            "messages", []
        )  # Remove from body temporarily
        messages = MessageBuilder.build_messages(args, config_messages)
        request_body["messages"] = messages

        # 4. Determine output path
        if args.output:
            output_base = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hint = (
                args.prompt[0][:20]
                if args.prompt
                else request_body.get("model", "chat")
            )
            safe_name = re.sub(r"[^\w\s-]", "", prompt_hint).strip().replace(" ", "_")
            output_base = Path(f"chat_{safe_name}_{timestamp}")

        # 5. Execute request
        with open(output_base.with_suffix(".md"), "w", encoding="utf-8") as out_file:
            printer = TypewriterPrinter(out_file)
            
            with APIClient(api_config) as client:
                metadata = client.request(request_body, printer)
            
            printer.finalize()

            # 6. Print statistics
            print(f"\n{Colors.info('✅ Complete')}")
            print(f"   Duration: {metadata.duration:.2f}s")

            if metadata.total_tokens:
                print(
                    f"   Tokens: {metadata.total_tokens} (Prompt: {metadata.prompt_tokens}, Completion: {metadata.completion_tokens})"
                )
                if token_rate := metadata.get_token_rate():
                    print(f"   Token Rate: {token_rate:.1f} tokens/s")

            if metadata.finish_reason:
                print(f"   Finish Reason: {metadata.finish_reason}")

            if printer.first_token_time:
                print(
                    f"   First Token: {(printer.first_token_time - printer.start_time):.2f}s"
                )

            # 7. Save results
            md_path, json_path = ResultSaver.save(
                output_base, request_body, metadata, printer
            )

            print(f"\n{Colors.info('💾 Saved:')}")
            print(f"   {md_path}")
            print(f"   {json_path}")
            print(
                f"\n{Colors.info('💡 Tip:')} Continue with: --context {json_path.name}"
            )

    except KeyboardInterrupt:
        print(f"\n{Colors.warn('⚠️  Interrupted')}", file=sys.stderr)
        sys.exit(130)
    except (ConfigurationError, FileNotFoundError) as e:
        print(f"{Colors.error('❌ Configuration Error:')} {e}", file=sys.stderr)
        sys.exit(1)
    except APIError as e:
        print(f"{Colors.error('❌ API Error:')} {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"{Colors.error('❌ Unexpected error:')} {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()