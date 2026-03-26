#!/usr/bin/env python3
"""
综合处理脚本：
1. 从 LAST_RESPONSE.txt 读取响应，解析代码块。
2. 执行 bash 代码块，保存其它代码块（支持 [PATH] 注释）。
3. 分析 agent_v3.py 的依赖，复制必要文件到 v3 目录，并简单优化（删除注释和空行）。
4. 生成当前目录的探索报告 README.txt。
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# ========== 配置 ==========
RESPONSE_FILE = "LAST_RESPONSE.txt"          # 存放 AI 响应的文件
BASH_TIMEOUT = 30                             # bash 执行超时（秒）
UNTITLED_DIR = "untitled"                     # 无路径时保存的目录
V3_DIR = "v3"                                 # 优化后代码输出目录
REPORT_FILE = "README.txt"                    # 探索报告文件名

# 语言 -> 注释符号映射（用于识别 [PATH] 行）
COMMENT_MAP = {
    "python": "#",
    "javascript": "//",
    "js": "//",
    "typescript": "//",
    "ts": "//",
    "java": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "//",
    "ruby": "#",
    "go": "//",
    "rust": "//",
    "php": "//",
    "html": "<!--",
    "xml": "<!--",
    "css": "/*",
    "scss": "//",
    "json": None,
    "yaml": "#",
    "markdown": None,
}

# 语言 -> 默认扩展名（路径无扩展名时补全）
EXT_MAP = {
    "python": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "csharp": ".cs",
    "ruby": ".rb",
    "go": ".go",
    "rust": ".rs",
    "php": ".php",
    "html": ".html",
    "xml": ".xml",
    "css": ".css",
    "scss": ".scss",
    "yaml": ".yaml",
}

# ========== 辅助函数 ==========
def extract_path_from_content(content: str, lang: str) -> Optional[str]:
    """从代码块内容中提取 [PATH] 注释指定的路径。返回路径或 None。"""
    comment = COMMENT_MAP.get(lang)
    if comment is None:
        return None
    # 构建正则匹配注释符号后跟 [PATH]
    if comment in ("<!--", "/*"):
        pattern = re.escape(comment) + r"\s*\[PATH\]\s*(.*)"
    else:
        pattern = re.escape(comment) + r"\s*\[PATH\]\s*(.*)"
    for line in content.splitlines():
        m = re.search(pattern, line)
        if m:
            return m.group(1).strip()
    return None

def save_file(path: str, content: str, remove_path_line: bool = True) -> bool:
    """保存文件，若 remove_path_line 为 True 则移除包含 [PATH] 的行。"""
    if remove_path_line:
        lines = content.splitlines()
        lines = [line for line in lines if "[PATH]" not in line]
        content = "\n".join(lines).strip()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"已保存: {path}")
        return True
    except Exception as e:
        print(f"保存失败 {path}: {e}")
        return False

def execute_bash(code: str) -> Tuple[str, str]:
    """执行 bash 代码，返回 (stdout, stderr)。"""
    try:
        # 尝试使用 Git Bash 的 bash，或系统默认 bash
        bash_path = r"C:\Program Files\Git\usr\bin\bash.exe" if sys.platform == "win32" else "/bin/bash"
        if not os.path.exists(bash_path):
            bash_path = "bash"  # 回退到 PATH 中的 bash
        proc = subprocess.run(
            [bash_path, "-c", code],
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            env=os.environ.copy(),
        )
        return proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"
    except Exception as e:
        return "", str(e)

def parse_response(response: str) -> None:
    """解析响应，处理所有代码块。"""
    pattern = r"