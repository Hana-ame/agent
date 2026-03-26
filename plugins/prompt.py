
import re
import os
from .base import Plugin

class DefaultPrompt(Plugin):
    def __init__(self, default_prompt:str):
        self.default_prompt = default_prompt
        
    def before_prompt(self, args, req):
        if str(req.get("prompt", ""))  == "":
            req["prompt"] = self.default_prompt
        return False            

    def after_prompt(self, args, req, resp):
        return False

class SaveCodePlugin(Plugin):
    """自动保存响应中带有 [PATH] 注释的代码块到文件；若无路径指示则保存到 untitled/snippet_xx.txt"""
    
    # 语言 -> 注释符号（单行），若无法确定则 None
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

    # 语言 -> 默认扩展名（当路径无扩展名时使用）
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

    # 类级别计数器，用于生成默认文件名
    _snippet_counter = 1

    def __init__(self):
        # 记录本次会话中保存的文件路径，供下一次 before_prompt 使用
        self.saved_files = []

    def before_prompt(self, args, req):
        # 将之前保存的文件列表传给请求
        if self.saved_files:
            req["code"] = self.saved_files
        else:
            # 也可设置为空列表，但无必要
            pass
        # 清空列表，准备记录下一轮保存的文件
        self.saved_files = []
        return False

    def after_prompt(self, args, req, resp):
        response = resp.get("response", "")
        if not response:
            return False

        code_block_pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        for match in code_block_pattern.finditer(response):
            lang = match.group(1).strip().lower()
            content = match.group(2)

            # 尝试提取路径
            target_path = self._extract_path(content, lang)
            if target_path:
                # 若有扩展名则保留，否则根据语言补充
                if not os.path.splitext(target_path)[1]:
                    ext = self.EXT_MAP.get(lang, "")
                    if ext:
                        target_path += ext
                # 保存到指定路径
                if self._save_file(target_path, content, remove_path_line=True):
                    self.saved_files.append(target_path)
            else:
                # 回退：保存到 untitled/snippet_xx.txt
                default_filename = f"snippet_{self._snippet_counter:02d}.txt"
                default_path = os.path.join("untitled", default_filename)
                if self._save_file(default_path, content, remove_path_line=False):
                    self.saved_files.append(default_path)
                    self.__class__._snippet_counter += 1

        return False

    def _extract_path(self, content, lang):
        """根据语言注释符号提取 [PATH] 行中的路径，未找到返回 None"""
        comment_prefix = self.COMMENT_MAP.get(lang)
        if comment_prefix is None:
            return None

        if comment_prefix in ("<!--", "/*"):
            pattern = re.escape(comment_prefix) + r"\s*\[PATH\]\s*(.*)"
        else:
            pattern = re.escape(comment_prefix) + r"\s*\[PATH\]\s*(.*)"

        path_re = re.compile(pattern)
        for line in content.splitlines():
            m = path_re.search(line)
            if m:
                return m.group(1).strip()
        return None

    def _save_file(self, filepath, content, remove_path_line):
        """保存文件，返回是否成功"""
        if remove_path_line:
            lines = content.splitlines()
            path_line_idx = None
            for i, line in enumerate(lines):
                if "[PATH]" in line:
                    path_line_idx = i
                    break
            if path_line_idx is not None:
                del lines[path_line_idx]
            content = "\n".join(lines).strip()

        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[SaveCodePlugin] Saved to {filepath}")
            return True
        except Exception as e:
            print(f"[SaveCodePlugin] Error saving {filepath}: {e}")
            return False