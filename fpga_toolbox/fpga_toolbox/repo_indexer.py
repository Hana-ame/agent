"""Tools 1 & 2: Syntax-aware RTL repo indexing and BFS transitive dependency resolution."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fpga_toolbox"

from .core.sv_parser import SVModuleAST, parse_sv_file, sv_tokenize
from .core.types import err_result, ok_result

RTL_EXTS = {".v", ".sv", ".vh", ".svh"}
_INC_RE = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)
_NON_INST = {
    "module", "endmodule", "if", "else", "for", "while", "case", "begin", "end",
    "always", "always_ff", "always_comb", "assign", "function", "task", "generate",
    "assert", "property", "input", "output", "inout", "wire", "reg", "logic",
}


def _find_instantiations(src: str, known: Set[str], self_name: str) -> List[str]:
    toks, found, i = sv_tokenize(src), [], 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.kind == "IDENT" and t.value in known and t.value != self_name and t.value not in _NON_INST:
            j = i + 1
            if j < n and toks[j].value == "#":
                j += 1
                if j < n and toks[j].value == "(":
                    depth = 0
                    while j < n:
                        depth += 1 if toks[j].value == "(" else (-1 if toks[j].value == ")" else 0)
                        j += 1
                        if depth == 0:
                            break
            if j + 1 < n and toks[j].kind == "IDENT" and toks[j + 1].value == "(":
                if t.value not in found:
                    found.append(t.value)
        i += 1
    return found


def _module_body(name: str, src: str) -> str:
    """提取指定 module 自身的定义体源码（从 `module <name>` 到配对的 `endmodule`）。

    背景（真实踩坑）：有些仓库用 `include` 把多个子模块拼进一个翻译单元
    （如 Single_Cycle_Top.v include 了 PC/ALU/寄存器堆等全部子模块）。此时
    若把整个文件源交给例化扫描，文件里每一个模块都会继承该文件的全部例化
    （叶子模块 ALU/Register_File 被误标成例化了 PC/存储器——那些其实是
    顶层 Single_Cycle_Top 的例化）。

    用 token 的行号定位 module 声明与配对 endmodule，只返回该模块自身
    定义体的源码。找不到时回退到整个文件源（保持旧行为）。
    """
    toks = sv_tokenize(src)
    start = None
    depth = 0
    end_line = None
    for k, t in enumerate(toks):
        if start is None:
            if t.kind == "IDENT" and t.value == "module" and k + 1 < len(toks) \
                    and toks[k + 1].kind == "IDENT" and toks[k + 1].value == name:
                start = t.line
        else:
            if t.kind == "IDENT" and t.value == "endmodule":
                end_line = t.line
                break
    if start is None:
        # 该 module 在本文件的原始文本里没有 module 声明 —— 它是通过
        # `include 从别的文件展开进来的（slang 展开后误归到本文件）。
        # 返回空串（不归属任何例化），而不是回退到整个文件源。
        return ""
    # endmodule 找不到（罕见）时取到文件末尾
    lines = src.splitlines()
    end_idx = end_line if end_line is not None else len(lines)
    return "\n".join(lines[start - 1:end_idx])


def tool_index_repo(project_root: str) -> Dict[str, Any]:
    root = Path(project_root)
    if not root.is_dir():
        return err_result("index_repo", f"Directory not found: {project_root}")

    rtl_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in RTL_EXTS)
    sources: Dict[Path, str] = {}
    parsed: List[Tuple[Path, SVModuleAST]] = []
    for fpath in rtl_files:
        sources[fpath] = fpath.read_text(encoding="utf-8", errors="ignore")
        if fpath.suffix.lower() in {".v", ".sv"}:
            parsed.extend((fpath, m) for m in parse_sv_file(fpath))

    known = {m.name for _, m in parsed}
    headers = {p.name: str(p.relative_to(root)) for p in rtl_files if p.suffix.lower() in {".vh", ".svh"}}

    # 归属修正：`include 会把子模块展开进顶层文件，导致同一模块出现在多个文件
    # 的解析结果里（如 datapath 同时出现在 datapath.v 和 include 它的
    # top_proc.v）。modules 字典按名去重时「排序靠后的文件覆盖前面的」，会把
    # datapath 错标成 top_proc.v。正确归属应是**原始文本里真正声明
    # `module <name>`** 的文件（_module_body 在该文件能定位到自身定义体）。
    owner: Dict[str, Path] = {}
    for fpath in rtl_files:
        if fpath.suffix.lower() not in {".v", ".sv"}:
            continue
        for mname in {m.name for _, m in parsed}:
            if mname in owner:
                continue
            if _module_body(mname, sources[fpath]):
                owner[mname] = fpath

    modules = {
        m.name: {
            "name": m.name,
            "file_path": str(owner.get(m.name, fp).relative_to(root)),
            "port_count": len(m.ports),
            "parameters": m.parameters,
            "instantiated_modules": _find_instantiations(
                _module_body(m.name, sources[fp]), known, m.name),
            "includes": list(dict.fromkeys(_INC_RE.findall(sources[fp]))),
        }
        for fp, m in parsed
    }
    return ok_result("index_repo", {
        "project_root": str(root.resolve()),
        "total_rtl_files": len(rtl_files),
        "headers": headers,
        "modules": modules,
    })


def tool_resolve_deps(
    target_module: str,
    repo_index: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    if repo_index is None:
        if not project_root:
            return err_result("resolve_deps", "Provide 'repo_index' or 'project_root'.")
        idx = tool_index_repo(project_root)
        if not idx["ok"]:
            return err_result("resolve_deps", idx["error"] or "Indexing failed.")
        repo_index = idx["data"]

    mods, hdrs = repo_index.get("modules", {}), repo_index.get("headers", {})
    if target_module not in mods:
        return err_result("resolve_deps", f"Module '{target_module}' not found. Available: {sorted(mods)}")

    closure, seen, edges = [], set(), []
    rtl_set, hdr_set, missing_inc = set(), set(), []
    q = deque([target_module])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        closure.append(cur)
        info = mods[cur]
        rtl_set.add(info["file_path"])
        for inc in info.get("includes", []):
            (hdr_set.add(hdrs[inc]) if inc in hdrs else missing_inc.append(inc))
        for child in info.get("instantiated_modules", []):
            edges.append({"from": cur, "to": child})
            if child not in seen and child in mods:
                q.append(child)

    diags = [{"severity": "WARNING", "code": "W101", "path": inc, "message": f"Header '{inc}' not found."} for inc in sorted(set(missing_inc))]
    return ok_result("resolve_deps", {
        "target_module": target_module,
        "target_file": mods[target_module]["file_path"],
        "dependency_closure": closure,
        "dependency_edges": edges,
        "rtl_files": sorted(rtl_set),
        "header_files": sorted(hdr_set),
        "all_files": sorted(rtl_set | hdr_set),
    }, diagnostics=diags)


if __name__ == "__main__":
    import sys

    from .cli import run_single_tool

    # 传了 --target-module 走依赖闭包，否则只建索引
    pick = "resolve_deps" if any(a.startswith("--target-module") or a.startswith("--target_module") for a in sys.argv[1:]) else "index_repo"
    sys.exit(run_single_tool(pick))
