# [START] CODEBLOCK-RULE
# version: 0.0.1
# 上下文：拦截并提取 Markdown 代码块，支持 write_multiple 格式自动写入和 snippet 格式更新。

import re
import os
import hashlib
from .base import BaseRule

class CodeBlockRule(BaseRule):
    def __init__(self, root_path: str, agent_dir: str):
        self.root_path = root_path
        self.agent_dir = agent_dir

    def _parse_write_multiple_blocks(self, content: str):
        """解析符合 write_multiple 格式的代码块内容，返回 (路径, 内容) 列表"""
        files = []
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            start_match = re.match(r'^=== (.+) ===$', line)
            if start_match:
                rel_path = start_match.group(1).strip()
                i += 1
                content_lines = []
                while i < len(lines):
                    current_line = lines[i].rstrip('\n')
                    end_match = re.match(r'^=== end of (.+) ===$', current_line)
                    if end_match and end_match.group(1).strip() == rel_path:
                        break
                    content_lines.append(lines[i])
                    i += 1
                if i < len(lines) and re.match(r'^=== end of .+ ===$', lines[i].rstrip('\n')):
                    i += 1
                file_content = ''.join(content_lines)
                # 去除可能的首尾 ``` 标记
                fc_lines = file_content.splitlines()
                if fc_lines and fc_lines[0].startswith("```"):
                    fc_lines.pop(0)
                if fc_lines and fc_lines[-1].startswith("```"):
                    fc_lines.pop()
                file_content = "\n".join(fc_lines)
                files.append((rel_path, file_content))
            else:
                i += 1
        return files

    def _extract_snippet(self, content: str):
        """从内容中提取第一个代码段，返回 (name, snippet_content) 或 None"""
        pattern = re.compile(
            r'^[ \t]*#[ \t]*\[START\][ \t]+([\w-]+)[ \t]*$\n?(.*?)^[ \t]*#[ \t]*\[END\][ \t]+\1[ \t]*$',
            re.DOTALL | re.MULTILINE
        )
        match = pattern.search(content)
        if match:
            return match.group(1), match.group(2)
        return None, None

    # [START] CODEBLOCK-RULE-MATCH
    # version: 0.0.1
    # 上下文：正则扫描全文中带有 ``` 包裹的代码段。
    # 输入参数：text (str)
    # 输出参数：文件保存位置信息的通知文本 (str)
    async def match_and_execute(self, text: str) -> str:
        results = []
        pattern = re.compile(r'```([a-zA-Z0-9_\-+]+)?\n(.*?)```', re.DOTALL)
        
        for match in pattern.finditer(text):
            ext = match.group(1) or 'txt'
            content = match.group(2)
            
            # 尝试解析为 write_multiple 格式
            files = self._parse_write_multiple_blocks(content)
            if files:
                for rel_path, file_content in files:
                    full_path = os.path.join(self.root_path, rel_path)
                    name, snippet_content = self._extract_snippet(file_content)
                    if name is not None:
                        # 处理为代码段更新
                        try:
                            if os.path.exists(full_path):
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    existing = f.read()
                            else:
                                existing = ""
                            from utils.snippet import replace_snippet
                            new_content = replace_snippet(existing, name, snippet_content)
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            results.append(f"通过代码块更新代码段 '{name}' 到文件: {rel_path}")
                        except Exception as e:
                            results.append(f"更新代码段 {rel_path} 失败: {e}")
                    else:
                        # 普通文件写入
                        try:
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(file_content)
                            results.append(f"通过代码块写入文件: {rel_path}")
                        except Exception as e:
                            results.append(f"写入文件 {rel_path} 失败: {e}")
            else:
                # 回退到原始行为：保存到 .agent/ 目录
                hash_val = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
                filename = f"{hash_val}.{ext}"
                filepath = os.path.join(self.agent_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                results.append(f"检测到代码块，已提取并保存至: .agent/{filename}")
            
        return "\n".join(results)
    # [END] CODEBLOCK-RULE-MATCH
# [END] CODEBLOCK-RULE