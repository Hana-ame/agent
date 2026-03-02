#!/usr/bin/env python3
"""
ChatCLI - DeepSeek CLI 对话工具

【设计目标】
- 提供交互式与单次调用双模式，支持文档引用与多会话管理
- 实时流式渲染：思考链(reasoning_content)与正式回答分离展示
- 状态持久化：对话历史自动保存，支持 context 文件热切换

【典型调用模式】

1. 交互模式（默认）：
    $ python chat.py                    # 使用默认 history.json
    $ python chat.py -c project_a.json  # 指定会话文件

2. 单次查询：
    $ python chat.py "解释量子计算"
    $ python chat.py -d paper.pdf "总结这篇论文" --new

3. 文档分析：
    $ python chat.py -d code.py "优化这段代码" -c code_review.json

【快捷键】
    Ctrl+C / Ctrl+D : 安全退出交互模式
"""

import os
import json
import sys
import argparse
import signal
from pathlib import Path
from typing import List, Dict, Optional, Any

# 依赖组件
from json_api_requester import JsonApiRequester
from json_payload_sender import JsonPayloadSender


# ═══════════════════════════════════════════════════════════════════════════════
# 终端渲染配置
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI 颜色码，支持自动检测禁用（管道/非 TTY）"""
    def __init__(self, force_color: bool = False):
        self.enabled = force_color or (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
    
    def __getattr__(self, name: str) -> str:
        codes = {
            'blue': '\033[94m', 'green': '\033[92m', 
            'yellow': '\033[93m', 'red': '\033[91m',
            'cyan': '\033[96m', 'gray': '\033[90m',
            'bold': '\033[1m', 'reset': '\033[0m'
        }
        return codes.get(name, '') if self.enabled else ''


# ═══════════════════════════════════════════════════════════════════════════════
# 核心对话类
# ═══════════════════════════════════════════════════════════════════════════════

class ChatCLI:
    """
    【使用上下文】
    终端交互层，整合 Requester/Sender，管理会话状态与渲染。
    
    【状态管理】
    - current_reasoning: 当前流式思考链累积
    - current_content: 当前正式回答累积  
    - history_file: 当前绑定的持久化文件路径
    - _reasoning_printed: 思考链是否已开始输出（控制换行）
    - _content_printed: 正式回答是否已开始输出
    """

    # 默认配置文件路径（类级别，便于覆盖）
    DEFAULT_ENDPOINT = "endpoint.json"
    DEFAULT_PAYLOAD = "payload.json"

    def __init__(
        self,
        history_file: str = "history.json",
        endpoint_file: Optional[str] = None,
        payload_file: Optional[str] = None,
        *,
        no_color: bool = False,
    ):
        """
        【调用效果】
        初始化完整对话环境：网络层、消息组装层、历史持久化
        
        【调用上下文】
        main() 入口或测试夹具中，每个进程一个实例
        
        【参数说明】
            history_file: 对话历史 JSON 文件路径，自动创建/复用
            endpoint_file: API 配置覆盖（默认 endpoint.json）
            payload_file: Payload 模板覆盖（默认 payload.json）
            no_color: 强制禁用颜色输出
        
        【初始化流程】
        1. 加载/创建历史文件
        2. 初始化 Requester（绑定 SSE 渲染钩子）
        3. 初始化 Sender（关联 Requester，同步 _context 指向）
        """
        self.colors = Colors(force_color=not no_color)
        self.history_file: Path = Path(history_file).resolve()
        
        # 状态重置
        self._reset_stream_state()
        
        # 文件路径配置
        self._endpoint_file = endpoint_file or self.DEFAULT_ENDPOINT
        self._payload_file = payload_file or self.DEFAULT_PAYLOAD
        
        # 初始化网络与消息层
        self._init_components()
        
        # 历史文件绑定
        self._bind_history_file()

    def _reset_stream_state(self) -> None:
        """【调用效果】重置流式渲染状态，每次新查询前执行"""
        self.current_reasoning: str = ""
        self.current_content: str = ""
        self._reasoning_started: bool = False
        self._content_started: bool = False

    def _init_components(self) -> None:
        """
        【调用效果】
        创建 Requester 与 Sender，建立 SSE 渲染链路
        
        【异常处理】
        配置文件缺失时抛出 FileNotFoundError，向上传播至 main 处理
        """
        self.requester = JsonApiRequester(
            json_file_path=self._endpoint_file,
            sse_hook=self._handle_sse_chunk
        )
        self.sender = JsonPayloadSender(
            requester=self.requester,
            payload_file_path=self._payload_file
        )

    def _bind_history_file(self) -> None:
        """
        【调用效果】
        确保历史文件存在，并同步更新 Sender 的 _context 指向
        
        【设计说明】
        _context 指向历史文件，使 Sender 的 Context 模式能加载完整对话历史
        """
        if not self.history_file.exists():
            self._write_history([])
            print(f"{self.colors.gray}[初始化] 创建新会话: {self.history_file.name}{self.colors.reset}")
        else:
            self._sync_context_path()
            # 静默同步，不打扰交互

    def _sync_context_path(self) -> None:
        """【调用效果】将 Sender 的 _context 字段同步为当前 history_file 路径"""
        try:
            self.sender.update_payload_field("_context", str(self.history_file), persist=True)
        except Exception as e:
            print(f"{self.colors.red}[警告] 同步 context 路径失败: {e}{self.colors.reset}")

    def _handle_sse_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        【调用效果】
        SSE 流式响应实时渲染处理器，区分思考链与正式回答
        
        【调用上下文】
        被 JsonApiRequester 的迭代循环回调，单线程顺序执行
        
        【渲染逻辑】
        reasoning_content: 黄色前缀，首次输出时打印"思考中..."提示
        content: 绿色前缀，首次输出时换行并打印"回答："提示
        
        【状态副作用】
        累积内容至 self.current_*, 控制 _started 标志管理换行
        """
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        c = self.colors
        
        # 处理思考链（DeepSeek-R1 等模型的特色）
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self._reasoning_started:
                print(f"\n{c.yellow}━ 思考中 ━{c.reset}", end="", flush=True)
                self._reasoning_started = True
            print(f"{c.yellow}{reasoning}{c.reset}", end="", flush=True)
            self.current_reasoning += reasoning
            return
        
        # 处理正式回答
        content = delta.get("content")
        if content:
            if not self._content_started:
                # 思考链与回答的分隔
                prefix = "\n\n" if self._reasoning_started else "\n"
                print(f"{prefix}{c.green}━ 回答 ━{c.reset}\n", end="", flush=True)
                self._content_started = True
            print(f"{c.green}{content}{c.reset}", end="", flush=True)
            self.current_content += content

    def _write_history(self, data: List[Dict[str, Any]]) -> None:
        """
        【调用效果】
        原子化写入历史记录，并同步更新 Sender 的 context 指向
        
        【调用上下文】
        对话结束保存时、重置历史时
        """
        try:
            # 原子写入：临时文件 + 重命名
            temp_path = self.history_file.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(self.history_file)
            
            # 同步 context 路径（文件可能已被移动/重命名）
            self._sync_context_path()
            
        except Exception as e:
            print(f"{self.colors.red}[错误] 保存历史失败: {e}{self.colors.reset}")
            raise

    def _read_history(self) -> List[Dict[str, Any]]:
        """【调用效果】安全读取历史，损坏时返回空列表"""
        try:
            if not self.history_file.exists():
                return []
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"{self.colors.yellow}[警告] 历史文件损坏，重置为空")
            return []
        except Exception as e:
            print(f"{self.colors.red}[错误] 读取历史失败: {e}")
            return []

    def clear_history(self) -> None:
        """【调用效果】清空当前绑定历史文件，开始全新会话"""
        self._write_history([])
        self._reset_stream_state()
        print(f"{self.colors.blue}[系统] 已重置: {self.history_file.name}{self.colors.reset}")

    def save_to_history(self, user_text: str, is_new: bool = False) -> None:
        """
        【调用效果】
        将当前轮对话追加保存，支持 reasoning_content 扩展字段
        
        【参数说明】
            is_new: True 时不加载旧历史（已清空），直接追加新记录
        """
        history: List[Dict[str, Any]] = [] if is_new else self._read_history()
        
        # 用户消息
        history.append({"role": "user", "content": user_text})
        
        # 助手消息（含可选思考链）
        assistant_msg: Dict[str, Any] = {
            "role": "assistant", 
            "content": self.current_content
        }
        if self.current_reasoning:
            assistant_msg["reasoning_content"] = self.current_reasoning
        history.append(assistant_msg)
        
        self._write_history(history)

    def _build_query(self, message: str, file_path: Optional[str] = None) -> str:
        """
        【调用效果】
        构造最终查询文本，支持文档引用包裹格式
        
        【文档格式】
        【参考文档：{filename}】
        {content}
        【参考文档：{filename}结束】
        """
        parts: List[str] = []
        
        if file_path:
            fp = Path(file_path)
            if fp.exists() and fp.is_file():
                try:
                    content = fp.read_text(encoding='utf-8')
                    # 限制过大文件（>100KB 截断提示）
                    max_size = 100 * 1024
                    if len(content) > max_size:
                        content = content[:max_size] + f"\n... [截断，原文件 {len(content)} 字符]"
                    
                    parts.append(f"【参考文档：{fp.name}】\n{content}\n【参考文档：{fp.name}结束】")
                except Exception as e:
                    parts.append(f"[文档读取失败: {fp.name} - {e}]")
            else:
                parts.append(f"[文档不存在: {file_path}]")
        
        parts.append(message)
        return "\n\n".join(parts)

    def execute_query(
        self,
        message: str,
        file_path: Optional[str] = None,
        is_new: bool = False,
    ) -> bool:
        """
        【调用效果】
        执行单轮查询：构造消息、发送请求、渲染响应、持久化历史
        
        【调用上下文】
        交互模式每次输入、命令行单次调用
        
        【参数说明】
            message: 用户核心提问
            file_path: 可选的参考文档路径
            is_new: 是否忽略历史，开启全新对话
        
        【返回】
            True: 请求成功（HTTP 层），False: 失败
        
        【渲染流程】
        1. 重置流式状态
        2. 构造完整 query（含文档）
        3. Sender 发送（use_context=not is_new）
        4. SSE 实时渲染（通过回调）
        5. 保存历史
        6. 打印分隔线
        """
        # 重置状态
        self._reset_stream_state()
        
        # 可选：清空历史
        if is_new:
            self.clear_history()
        
        # 构造查询
        full_query = self._build_query(message, file_path)
        
        # 发送请求
        c = self.colors
        print(f"\n{c.cyan}━ 发送中 ━{c.reset}", end="", flush=True)
        
        try:
            result = self.sender.send_request_with_messages(
                dynamic_messages=[{"role": "user", "content": full_query}],
                use_context=(not is_new),
                request_metadata={
                    "history_file": str(self.history_file),
                    "has_document": file_path is not None,
                    "is_new_session": is_new,
                }
            )
            
            # 确保换行（流式可能未结束于换行）
            if self._reasoning_started or self._content_started:
                print()  # 结束当前行
            
            # 检查发送结果
            if not result.success:
                print(f"\n{c.red}[请求失败] {result.error_info or 'Unknown'}{c.reset}")
                return False
            
            # 保存历史
            self.save_to_history(full_query, is_new=is_new)
            
            # 结束标记
            print(f"{c.gray}\n{'─' * 40}{c.reset}")
            return True
            
        except Exception as e:
            print(f"\n{c.red}[异常] {type(e).__name__}: {e}{c.reset}")
            return False

    def interactive_loop(self) -> None:
        """
        【调用效果】
        启动交互式 REPL，支持优雅退出与信号处理
        
        【快捷键】
            Ctrl+D (EOF): 退出
            Ctrl+C: 中断当前输入/退出
        """
        c = self.colors
        print(f"\n{c.bold}{c.blue}╔{'═' * 46}╗")
        print(f"║{'ChatCLI':^46}║")
        print(f"╠{'═' * 46}╣")
        print(f"║ 会话文件: {str(self.history_file):<33}║")
        print(f"║ 命令: exit, quit, /reset, /clear, /new        ║")
        print(f"╚{'═' * 46}╝{c.reset}\n")
        
        # 信号处理：优雅退出
        def signal_handler(sig, frame):
            print(f"\n{c.gray}[退出]{c.reset}")
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            try:
                # 提示符
                prompt_symbol = f"{c.blue}❯{c.reset} "
                user_input = input(prompt_symbol).strip()
                
                # 空输入
                if not user_input:
                    continue
                
                # 内置命令
                cmd = user_input.lower()
                if cmd in ('exit', 'quit'):
                    break
                if cmd in ('/reset', '/clear'):
                    self.clear_history()
                    continue
                if cmd == '/new':
                    self.execute_query("你好，开始新对话。", is_new=True)
                    continue
                
                # 普通查询
                self.execute_query(user_input, is_new=False)
                
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except Exception as e:
                print(f"{c.red}[错误] {e}{c.reset}")


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def create_parser() -> argparse.ArgumentParser:
    """【调用效果】配置 argparse，支持 --help 自动生成"""
    parser = argparse.ArgumentParser(
        prog='chat.py',
        description='DeepSeek CLI 对话工具 - 支持流式思考链与多会话管理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s                           # 交互模式，默认历史文件
  %(prog)s -c project.json           # 指定会话文件
  %(prog)s "你好"                    # 单次查询
  %(prog)s -d doc.txt "分析" --new   # 新会话+文档分析
        '''
    )
    
    parser.add_argument(
        '-d', '--document',
        metavar='PATH',
        help='附加参考文档路径（将包裹在标记中发送）'
    )
    
    parser.add_argument(
        '-c', '--context',
        metavar='FILE',
        default='history.json',
        help='指定历史记录 JSON 文件（默认: history.json）'
    )
    
    parser.add_argument(
        '--endpoint',
        metavar='FILE',
        help=f'覆盖 API 配置文件（默认: {ChatCLI.DEFAULT_ENDPOINT}）'
    )
    
    parser.add_argument(
        '--payload',
        metavar='FILE', 
        help=f'覆盖 Payload 模板文件（默认: {ChatCLI.DEFAULT_PAYLOAD}）'
    )
    
    parser.add_argument(
        '--new', action='store_true',
        help='开启全新对话，清空指定的 context 文件'
    )
    
    parser.add_argument(
        '--no-color', action='store_true',
        help='禁用 ANSI 颜色输出'
    )
    
    parser.add_argument(
        'message', nargs='?',
        help='直接发送的消息内容（不提供则进入交互模式）'
    )
    
    return parser


def main() -> int:
    """
    【调用效果】
    命令行入口：解析参数，初始化 CLI，分发执行模式
    
    【返回】
        0: 成功，1: 错误
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # 参数验证
    if args.document and not Path(args.document).exists():
        print(f"[错误] 文档不存在: {args.document}", file=sys.stderr)
        return 1
    
    try:
        # 初始化 CLI
        cli = ChatCLI(
            history_file=args.context,
            endpoint_file=args.endpoint,
            payload_file=args.payload,
            no_color=args.no_color,
        )
        
        # 模式分发
        if args.message or args.document:
            # 单次模式
            msg = args.message or "请分析以上文档。"
            success = cli.execute_query(msg, args.document, is_new=args.new)
            return 0 if success else 1
        else:
            # 交互模式
            if args.new:
                cli.clear_history()
            cli.interactive_loop()
            return 0
            
    except FileNotFoundError as e:
        print(f"[错误] 配置文件缺失: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[致命错误] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    print("你妈的又没")
    sys.exit(main())

