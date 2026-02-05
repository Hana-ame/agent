#!/usr/bin/env python3
"""
AI Chat CLI - Typewriter-effect HTTP client with OpenAI-compatible JSON config.
Zero-default policy: No default values for API parameters; fail if not provided.
Profiles provide only endpoint/api_key; config.json provides the raw request body.
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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# ============ Constants ============
THREE_DOTS = "`" * 3

class Constants:
    """Application constants (non-configurable defaults)"""
    CHUNK_SIZE = 1024
    STREAM_DELAY_REASONING = 0.01
    STREAM_DELAY_CONTENT = 0.03
    
    # ANSI Colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ============ Exceptions ============
class ChatError(Exception):
    """Base exception"""
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


# ============ Data Models ============
@dataclass
class APIConfig:
    """Connection configuration - only endpoint and api_key, no defaults"""
    endpoint: str = ""
    api_key: str = ""
    timeout: int = 120  # HTTP timeout is client-side, not API parameter

    def validate(self) -> None:
        """Validate that required fields are present"""
        if not self.endpoint:
            raise ConfigurationError(
                "API endpoint is required. Provide via --endpoint, profile, or ENDPOINT env var."
            )
        if not self.api_key:
            raise ConfigurationError(
                "API key is required. Provide via --api-key, profile, or API_KEY env var."
            )


@dataclass
class ConversationMetadata:
    """Response metadata tracking"""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    model: Optional[str] = None

    def finalize(self, finish_reason: Optional[str] = None):
        self.end_time = datetime.now().isoformat()
        self.finish_reason = finish_reason


# ============ UTF-8 Setup ============
def setup_utf8() -> None:
    """Force UTF-8 encoding"""
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


# ============ Configuration Management ============
class ConfigManager:
    """Loads endpoint/api_key from env/profiles. No defaults, no model params."""

    ENV_PATHS = [".env", "../.env", os.path.expanduser("~/.ai_chat.env")]
    PROFILE_PATHS = [
        Path("profiles.json"),
        Path("../profiles.json"),
        Path.home() / ".ai_chat_profiles.json",
    ]

    @classmethod
    def load(cls, profile_name: str = "default") -> APIConfig:
        """Load connection config (endpoint/api_key only)"""
        config = APIConfig()

        # Load .env files
        for env_path in cls.ENV_PATHS:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info(f"Loaded environment: {env_path}")
                break

        # From environment (no defaults)
        config.endpoint = os.getenv("ENDPOINT", "")
        config.api_key = os.getenv("API_KEY", "")

        # Load profile if specified (only endpoint/api_key extracted)
        if profile_name != "default":
            profile_data = cls._load_profile(profile_name)
            if profile_data:
                # Only extract connection info, ignore other fields
                if "endpoint" in profile_data:
                    config.endpoint = profile_data["endpoint"]
                if "api_key" in profile_data:
                    config.api_key = profile_data["api_key"]
                if "apiKey" in profile_data:  # Alternative key name
                    config.api_key = profile_data["apiKey"]

        return config

    @classmethod
    def _load_profile(cls, profile_name: str) -> Optional[Dict[str, Any]]:
        """Load profile JSON"""
        for path in cls.PROFILE_PATHS:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                if profile_name in profiles:
                    logger.info(f"Loaded profile '{profile_name}' from {path}")
                    return profiles[profile_name]
            except Exception as e:
                logger.error(f"Failed to load profile {path}: {e}")
        return None


# ============ File Processing ============
class FileProcessor:
    """Handles @file syntax with multimodal support"""

    IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    @classmethod
    def load(cls, filepath: str) -> Union[str, Dict[str, Any]]:
        """Load file content or image data"""
        clean_path = filepath[1:] if filepath.startswith("@") else filepath
        path = Path(clean_path).expanduser().resolve()

        if not path.exists():
            raise FileProcessingError(f"File not found: {filepath}")

        suffix = path.suffix.lower()

        if suffix in cls.IMAGE_TYPES:
            return cls._load_image(path)
        return cls._load_text(path)

    @classmethod
    def _load_image(cls, path: Path) -> Dict[str, Any]:
        """Base64 encode image for multimodal"""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        mime = mimetypes.guess_type(path)[0] or f"image/{path.suffix[1:]}"
        if path.suffix.lower() == ".jpg":
            mime = "image/jpeg"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"}
        }

    @classmethod
    def _load_text(cls, path: Path) -> str:
        """Load text with encoding detection"""
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

        # Format based on extension
        ext = path.suffix[1:] if path.suffix else ""
        if ext in ["py", "js", "ts", "java", "cpp", "c", "go", "rs"]:
            return f"{THREE_DOTS}{ext}\n{content}\n{THREE_DOTS}\n[File: {path.name}]"
        return f"{content}\n\n[File: {path.name}]"


# ============ Message Building ============
class MessageBuilder:
    """Builds message list from prompt arguments"""

    @staticmethod
    def build(args_prompt: List[str]) -> Tuple[List[Dict[str, Any]], str]:
        """Process prompt list with @file support"""
        if not args_prompt:
            return [], "chat"

        parts = []
        has_image = False
        hint = re.sub(r"[^\w\s-]", "", args_prompt[0])[:20].strip() or "chat"

        for part in args_prompt:
            if part.startswith("@"):
                try:
                    content = FileProcessor.load(part)
                    if isinstance(content, dict):
                        has_image = True
                    parts.append(content)
                except FileProcessingError as e:
                    logger.error(str(e))
                    parts.append(f"[Error: {part}]")
            else:
                parts.append(part)

        if has_image:
            return MessageBuilder._build_multimodal(parts), hint
        return [{"role": "user", "content": "\n\n".join(parts)}], hint

    @staticmethod
    def _build_multimodal(parts: List[Union[str, Dict]]) -> List[Dict[str, Any]]:
        """Construct multimodal content array"""
        content = []
        text_buffer = []

        for part in parts:
            if isinstance(part, dict):  # Image
                if text_buffer:
                    content.append({"type": "text", "text": "\n".join(text_buffer)})
                    text_buffer = []
                content.append(part)
            else:
                text_buffer.append(part)

        if text_buffer:
            content.append({"type": "text", "text": "\n\n".join(text_buffer)})

        return [{"role": "user", "content": content}]


# ============ Request Construction ============
class RequestBuilder:
    """
    Builds request body from config.json (raw body) + messages.
    No defaults injected. Command line args override config.json values.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build(self, user_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build request body:
        1. Start with config.json content (if exists)
        2. Add/merge messages
        3. Override with command line args (if provided)
        """
        # 1. Load base config (raw body parameters)
        body = self._load_json_config()

        # 2. Handle messages: config.json messages + context + user_messages
        existing_messages = body.get("messages", [])
        
        # Load context history if specified
        if self.args.context:
            history = self._load_context(self.args.context)
            existing_messages.extend(history)
        
        # Append current user messages
        existing_messages.extend(user_messages)
        body["messages"] = existing_messages

        # 3. Command line overrides for behavior flags only
        # Stream handling: CLI flag overrides config
        if self.args.no_stream:
            body["stream"] = False
        elif "stream" not in body:
            # If not specified anywhere, default to True for UX
            # This is a client behavior default, not an API parameter default
            body["stream"] = True

        # Enable thinking if specified
        if self.args.enable_thinking is not None:
            body["enable_thinking"] = self.args.enable_thinking

        return body

    def _load_json_config(self) -> Dict[str, Any]:
        """Load config.json as raw body template"""
        config_path = Path(self.args.config).expanduser().resolve()
        
        if not config_path.exists():
            if self.args.config != "config.json":  # User explicitly specified non-existent file
                raise ConfigurationError(f"Config file not found: {self.args.config}")
            # Default config.json not found is OK, start empty
            return {}
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded request config: {config_path}")
            return data
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {self.args.config}: {e}")

    def _load_context(self, context_path: str) -> List[Dict]:
        """Load conversation history"""
        path = Path(context_path).expanduser().resolve()
        if not path.exists():
            raise ConfigurationError(f"Context file not found: {context_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "messages" in data:
            return data["messages"]
        else:
            raise ConfigurationError("Invalid context format: expected list or {messages: [...]}")


# ============ Typewriter Output ============
class TypewriterPrinter:
    """Typewriter effect with optional file tee"""

    def __init__(self, tee_file: Optional[io.TextIOWrapper] = None):
        self.tee = tee_file
        self.reasoning_acc = ""
        self.content_acc = ""
        self.reasoning_len = 0
        self.content_len = 0
        self.in_reasoning = True

    def print_reasoning(self, text: str):
        """Print thinking content"""
        if not text:
            return
        new_part = text[self.reasoning_len:]
        for char in new_part:
            print(char, end="", flush=True)
            if self.tee:
                self.tee.write(char)
                self.tee.flush()
            time.sleep(Constants.STREAM_DELAY_REASONING)
        self.reasoning_len = len(text)
        self.reasoning_acc = text

    def switch_to_content(self):
        """Transition to response"""
        if self.in_reasoning:
            print(f"\n\n{Constants.CYAN}{'='*50}{Constants.RESET}")
            print(f"{Constants.BOLD}Response:{Constants.RESET}\n")
            if self.tee:
                self.tee.write(f"\n\n{'='*50}\nResponse:\n{'='*50}\n\n")
            self.in_reasoning = False

    def print_content(self, text: str):
        """Print response content"""
        if not text:
            return
        if self.in_reasoning:
            self.switch_to_content()
        new_part = text[self.content_len:]
        for char in new_part:
            print(char, end="", flush=True)
            if self.tee:
                self.tee.write(char)
                self.tee.flush()
            time.sleep(Constants.STREAM_DELAY_CONTENT)
        self.content_len = len(text)
        self.content_acc = text

    def finalize(self):
        if self.tee:
            self.tee.flush()

    def get_content(self) -> Tuple[str, str]:
        return self.reasoning_acc, self.content_acc


# ============ HTTP Client ============
class AIChatClient:
    """Streaming HTTP client"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def chat(
        self,
        body: Dict[str, Any],
        printer: TypewriterPrinter
    ) -> Tuple[str, str, ConversationMetadata]:
        """Execute request with typewriter output"""
        metadata = ConversationMetadata()
        
        # Display request info
        print(f"\n{Constants.GREEN}Request:{Constants.RESET}")
        print(f"  Endpoint: {self.config.endpoint}")
        print(f"  Model: {body.get('model', 'Not specified')}")
        print(f"  Messages: {len(body.get('messages', []))}")
        print(f"  Stream: {body.get('stream', True)}")
        print()

        is_stream = body.get("stream", True)
        
        try:
            resp = self.session.post(
                self.config.endpoint,
                json=body,
                stream=is_stream,
                timeout=self.config.timeout
            )
            resp.raise_for_status()

            if is_stream:
                return self._handle_stream(resp, printer, metadata)
            else:
                return self._handle_non_stream(resp, printer, metadata)

        except requests.exceptions.Timeout:
            raise APIError(f"Timeout after {self.config.timeout}s")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}")

    def _handle_stream(
        self,
        resp: requests.Response,
        printer: TypewriterPrinter,
        meta: ConversationMetadata
    ) -> Tuple[str, str, ConversationMetadata]:
        """Process SSE stream"""
        reasoning = ""
        content = ""
        header_printed = False

        data = {}
        for line in self._iter_sse(resp):
            if not line.startswith("data: "):
                continue
            
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            
            try:
                data = json.loads(data_str) or {}
                
                # Update usage if present
                if "usage" in data:
                    meta.prompt_tokens = data["usage"].get("prompt_tokens", 0)
                    meta.completion_tokens = data["usage"].get("completion_tokens", 0)
                    meta.total_tokens = data["usage"].get("total_tokens", 0)

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")
                if finish:
                    meta.finish_reason = finish

                r_delta = delta.get("reasoning_content") or delta.get("reasoning", "")
                c_delta = delta.get("content", "")

                if not header_printed and (r_delta or c_delta):
                    header_printed = True
                    if r_delta:
                        print(f"{Constants.YELLOW}Thinking:{Constants.RESET}")

                if r_delta:
                    reasoning += r_delta
                    printer.print_reasoning(reasoning)
                
                if c_delta:
                    content += c_delta
                    printer.print_content(content)

            except json.JSONDecodeError:
                continue

        meta.finalize(meta.finish_reason)
        meta.model = data.get("model") if 'data' in locals() else None
        return reasoning, content, meta

    def _handle_non_stream(
        self,
        resp: requests.Response,
        printer: TypewriterPrinter,
        meta: ConversationMetadata
    ) -> Tuple[str, str, ConversationMetadata]:
        """Process complete response"""
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")

        if reasoning:
            print(f"{Constants.YELLOW}Thinking:{Constants.RESET}")
            printer.print_reasoning(reasoning)
        
        printer.print_content(content)

        if "usage" in data:
            meta.prompt_tokens = data["usage"].get("prompt_tokens", 0)
            meta.completion_tokens = data["usage"].get("completion_tokens", 0)
            meta.total_tokens = data["usage"].get("total_tokens", 0)
        
        meta.finalize(choice.get("finish_reason"))
        meta.model = data.get("model")
        return reasoning, content, meta

    def _iter_sse(self, resp: requests.Response) -> Iterator[str]:
        """Iterate SSE lines"""
        buffer = ""
        for chunk in resp.iter_content(chunk_size=Constants.CHUNK_SIZE, decode_unicode=True):
            if chunk:
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield line.strip()
        if buffer:
            yield buffer.strip()


# ============ Persistence ============
class ConversationSaver:
    """Save conversations to disk"""

    @staticmethod
    def save(
        base_path: Path,
        reasoning: str,
        content: str,
        metadata: ConversationMetadata,
        messages: List[Dict],
        request_body: Dict
    ) -> Tuple[Path, Path]:
        """Save as Markdown and JSON"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Markdown
        md = f"""# AI Conversation Log

**Time**: {timestamp}  
**Model**: {metadata.model or 'unknown'}  
**Tokens**: {metadata.total_tokens} (Prompt: {metadata.prompt_tokens}, Completion: {metadata.completion_tokens})  
**Finish Reason**: {metadata.finish_reason}

## Request Body

{THREE_DOTS}json
{json.dumps(request_body, indent=2, ensure_ascii=False)}
{THREE_DOTS}

## Messages

"""
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content_data = msg.get("content", "")
            if isinstance(content_data, list):
                texts = [item.get("text", "[Image]") if isinstance(item, dict) else str(item) 
                        for item in content_data]
                content_data = "\n".join(texts)
            md += f"### {role}\n{content_data}\n\n"

        md += "## Response\n\n"
        if reasoning:
            md += f"<details>\n<summary>Thinking Process</summary>\n\n{reasoning}\n\n</details>\n\n"
        md += f"{content}\n"

        md_path = base_path.with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")

        # JSON (for context continuation)
        json_data = {
            "metadata": {
                "timestamp": timestamp,
                "model": metadata.model,
                "total_tokens": metadata.total_tokens
            },
            "messages": messages + [{
                "role": "assistant",
                "content": content,
                **({"reasoning_content": reasoning} if reasoning else {})
            }]
        }
        json_path = base_path.with_suffix(".json")
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return md_path, json_path


# ============ CLI ============
def create_parser() -> argparse.ArgumentParser:
    """Argument parser - no defaults for API params"""
    p = argparse.ArgumentParser(
        description="AI Chat CLI - Zero-default typewriter client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config.json contains all API parameters (model, temperature, etc.)
  python chat.py --config api_params.json "Hello"

  # Use profile for endpoint/key, config.json for body params
  python chat.py --profile production --config request.json @file.txt "Analyze this"
  
  # Continue conversation
  python chat.py --config request.json --context chat_20231225_143022.json "Follow up question"
        """
    )

    # Connection (profiles/env provide these, but can override)
    p.add_argument("--profile", "-p", default="default", help="Profile name (default: default)")
    p.add_argument("--endpoint", "-e", help="API endpoint URL")
    p.add_argument("--api-key", "-k", help="API key")

    # Request body config (raw JSON)
    p.add_argument("--config", "-f", default="config.json", 
                  help="JSON file containing request body parameters (default: config.json)")

    # Input
    p.add_argument("prompt", nargs="*", help="Prompt text (supports @file)")
    p.add_argument("--context", "-c", help="Previous conversation JSON to continue")

    # Behavior
    p.add_argument("--no-stream", action="store_true", help="Disable streaming")
    p.add_argument("--enable-thinking", action="store_true", default=None)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--output", "-o", help="Output file base name (auto if omitted)")

    return p


def handle_sigint(signum, frame):
    print(f"\n{Constants.YELLOW}Interrupted{Constants.RESET}")
    sys.exit(130)


def main():
    setup_utf8()
    signal.signal(signal.SIGINT, handle_sigint)

    parser = create_parser()
    args = parser.parse_args()

    try:
        # 1. Load connection config (endpoint/api_key only)
        config = ConfigManager.load(args.profile)
        
        # Command line overrides for connection
        if args.endpoint:
            config.endpoint = args.endpoint
        if args.api_key:
            config.api_key = args.api_key
        
        config.validate()

        # 2. Build request body from config.json + messages
        builder = RequestBuilder(args)
        user_msgs, hint = MessageBuilder.build(args.prompt)
        request_body = builder.build(user_msgs)

        if not request_body.get("messages"):
            raise ConfigurationError("No messages to send")

        # 3. Determine output path
        if args.output:
            out_path = Path(args.output)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^\w\-]", "", hint)[:20] or "chat"
            out_path = Path(f"chat_{safe}_{ts}")

        # 4. Execute
        with open(out_path.with_suffix(".md"), "w", encoding="utf-8") as tee, \
             AIChatClient(config) as client:
            
            printer = TypewriterPrinter(tee_file=tee)
            
            # Write header to file
            tee.write(f"# Chat Log - {datetime.now().isoformat()}\n\n")
            tee.write(f"## Request\n{THREE_DOTS}json\n{json.dumps(request_body, indent=2)}\n{THREE_DOTS}\n\n## Response\n\n")
            
            reasoning, content, metadata = client.chat(request_body, printer)
            printer.finalize()

            # 5. Save
            md_path, json_path = ConversationSaver.save(
                out_path, reasoning, content, metadata,
                request_body["messages"], request_body
            )

            # Summary
            print(f"\n\n{Constants.GREEN}{'='*50}{Constants.RESET}")
            print(f"{Constants.BOLD}Complete{Constants.RESET}")
            if reasoning:
                print(f"  Reasoning: {len(reasoning)} chars")
            print(f"  Content: {len(content)} chars")
            if metadata.total_tokens:
                print(f"  Tokens: {metadata.total_tokens}")
            print(f"\n{Constants.CYAN}Saved:{Constants.RESET}")
            print(f"  MD:  {md_path}")
            print(f"  JSON: {json_path}")
            print(f"\nContinue with: --context {json_path.name}")

    except ConfigurationError as e:
        print(f"{Constants.RED}Config Error: {e}{Constants.RESET}", file=sys.stderr)
        sys.exit(1)
    except APIError as e:
        print(f"{Constants.RED}API Error: {e}{Constants.RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"{Constants.RED}Error: {e}{Constants.RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()