import re
import os
import subprocess
from .base import Plugin
class DefaultPrompt(Plugin):
    def __init__(self, default_prompt: str, system_prompt: str):
        """
        :param default_prompt: Either a string containing the default prompt,
                               or a file path (if it ends with .txt or .md)
        """
        if os.path.isfile(default_prompt):
            with open(default_prompt, "r", encoding="utf-8") as f:
                self.default_prompt = f.read()
        else:
            self.default_prompt = default_prompt
        if os.path.isfile(system_prompt):
            with open(system_prompt, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt
        self.first_prompt = True
    def before_prompt(self, args, req):
        code = req.get("code", "")
        output = req.get("output", "")
        user_prompt = req.get("prompt", "")
        if not user_prompt:
            if code or output:
                req["prompt"] = f"CodeBlocks:\n{code}\n\nOutput:\n{output}"
            else:
                if self.system_prompt:
                    req["prompt"] = self.system_prompt
                    self.system_prompt = ""
                else:
                    req["prompt"] = self.default_prompt
        else:
            if code or output:
                req["prompt"] = f"Code:\n{code}\n\nOutput:\n{output}\n\n{user_prompt}"
        return False
    def after_prompt(self, args, req, resp):
        return False
class SaveCodePlugin(Plugin):
    """自动保存响应中带有 [PATH] 注释的代码块到文件；若无路径指示则保存到 untitled/snippet_xx.txt"""
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
    _snippet_counter = 1
    def __init__(self):
        self.saved_files = []
    def before_prompt(self, args, req):
        if self.saved_files:
            req["code"] = self.saved_files
        else:
            pass
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
            target_path = self._extract_path(content, lang)
            if target_path:
                if not os.path.splitext(target_path)[1]:
                    ext = self.EXT_MAP.get(lang, "")
                    if ext:
                        target_path += ext
                if self._save_file(target_path, content, remove_path_line=True):
                    self.saved_files.append(target_path)
            else:
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
class RunBashCodeBlock(Plugin):
    def __init__(self, bash_path: str = r"C:\Program Files\Git\usr\bin\bash.exe"):
        """
        :param bash_path: Path to the bash executable (Git Bash on Windows)
        """
        self.bash_path = bash_path
        self.results = []
        self.output = ""
        self.bin_dir = os.path.dirname(bash_path)  # e.g., C:\Program Files\Git\usr\bin
    def before_prompt(self, args, req):
        req["output"] = self.output
        self.output = ""
        return False
    def after_prompt(self, args, req, resp):
        response = resp.get("response", "")
        if not response:
            return False
        pattern = r"```bash\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            return False
        output_parts = []
        self.results = []
        env = os.environ.copy()
        current_path = env.get("PATH", "")
        if self.bin_dir not in current_path.split(os.pathsep):
            env["PATH"] = self.bin_dir + os.pathsep + current_path
        for idx, code in enumerate(matches, start=1):
            try:
                result = subprocess.run(
                    [self.bash_path, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=False,
                    env=env,  # use enhanced environment
                )
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                block_result = {
                    "index": idx,
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout": False,
                    "error": None,
                }
                self.results.append(block_result)
                if stdout:
                    output_parts.append(f"## Output of block {idx}:\n{stdout}")
                if stderr:
                    output_parts.append(f"## Error of block {idx}:\n{stderr}")
            except subprocess.TimeoutExpired:
                output_parts.append(f"## Block {idx}: Execution timed out (30s)")
                self.results.append(
                    {
                        "index": idx,
                        "code": code,
                        "stdout": "",
                        "stderr": "Execution timed out (30s)",
                        "timeout": True,
                        "error": None,
                    }
                )
            except Exception as e:
                output_parts.append(f"## Block {idx}: Execution failed - {e}")
                self.results.append(
                    {
                        "index": idx,
                        "code": code,
                        "stdout": "",
                        "stderr": str(e),
                        "timeout": False,
                        "error": e,
                    }
                )
        if output_parts:
            self.output = "[Execution results]\n" + "\n\n".join(output_parts)
            print(self.output)
        return False
