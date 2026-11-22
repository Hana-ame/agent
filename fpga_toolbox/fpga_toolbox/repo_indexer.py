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
    modules = {
        m.name: {
            "name": m.name,
            "file_path": str(fp.relative_to(root)),
            "port_count": len(m.ports),
            "parameters": m.parameters,
            "instantiated_modules": _find_instantiations(sources[fp], known, m.name),
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
