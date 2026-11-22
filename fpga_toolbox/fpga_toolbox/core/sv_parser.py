"""Verilog-2001 / SystemVerilog lexer, recursive-descent parser & optional slang AST backend."""

from __future__ import annotations

import ast
import json
import operator
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SVToken:
    kind: str
    value: str
    line: int


@dataclass(frozen=True)
class SVPortAST:
    name: str
    direction: str
    width_expr: str
    width: int
    line: int
    is_interface: bool = False
    interface_type: str = ""
    modport: str = ""


@dataclass(frozen=True)
class SVModuleAST:
    name: str
    file_path: str
    parameters: Dict[str, int]
    ports: Tuple[SVPortAST, ...]
    parser_backend: str = "builtin"


_TOKEN_RE = re.compile(
    r"""
    (?P<WS>[ \t\r]+)
  | (?P<NL>\n)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<NUMBER>[0-9]+'[sS]?[bBdDhHoO][0-9a-fA-FxXzZ_]+|[0-9][0-9_]*)
  | (?P<SYM>[#()\[\],;=:\+\-\*/<>\{\}\.\?@!~&|\^])
    """,
    re.VERBOSE,
)

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"(?m)^[ \t]*`[^\n]*", "", src)
    return src


def sv_tokenize(src: str) -> List[SVToken]:
    cleaned = strip_comments(src)
    tokens: List[SVToken] = []
    line, pos, n = 1, 0, len(cleaned)
    while pos < n:
        m = _TOKEN_RE.match(cleaned, pos)
        if not m:
            pos += 1
            continue
        kind, val, pos = m.lastgroup, m.group(0), m.end()
        if kind == "NL":
            line += 1
        elif kind != "WS":
            tokens.append(SVToken(kind=str(kind), value=val, line=line))
    return tokens


def _parse_sv_literal(val: str) -> Optional[int]:
    val = val.replace("_", "")
    if "'" in val:
        radix_part = val.split("'", 1)[1].lstrip("sS")
        base_map = {"b": 2, "d": 10, "h": 16, "o": 8}
        if radix_part and radix_part[0].lower() in base_map and not re.search(r"[xXzZ?]", radix_part[1:]):
            return int(radix_part[1:], base_map[radix_part[0].lower()])
        return None
    return int(val, 10) if val.isdigit() else None


def eval_int_expr(expr: str, params: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    expr = re.sub(
        r"\$clog2\s*\(([^()]+)\)",
        lambda m: str(max(1, ((eval_int_expr(m.group(1), params) or 1) - 1).bit_length())),
        expr,
    )
    expr = re.sub(
        r"[0-9]+'[sS]?[bBdDhHoO][0-9a-fA-FxXzZ_]+",
        lambda m: str(_parse_sv_literal(m.group(0)) or 0),
        expr,
    )
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def _ev(nd: ast.AST) -> int:
        if isinstance(nd, ast.Constant) and isinstance(nd.value, int):
            return int(nd.value)
        if isinstance(nd, ast.Name) and nd.id in params:
            return params[nd.id]
        if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, ast.USub):
            return -_ev(nd.operand)
        if isinstance(nd, ast.BinOp) and type(nd.op) in _SAFE_BINOPS:
            return _SAFE_BINOPS[type(nd.op)](_ev(nd.left), _ev(nd.right))
        raise ValueError("unsupported")

    try:
        return _ev(node)
    except Exception:
        return None


class SVModuleParser:
    _TYPES = {"wire", "reg", "logic", "bit", "signed", "unsigned", "int", "integer", "var", "tri"}
    _DIRS = {"input", "output", "inout"}

    def __init__(self, tokens: List[SVToken], file_path: str):
        self.tokens, self.file_path, self.i, self.n = tokens, file_path, 0, len(tokens)

    def _peek(self, off: int = 0) -> Optional[SVToken]:
        idx = self.i + off
        return self.tokens[idx] if 0 <= idx < self.n else None

    def _adv(self) -> SVToken:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _match(self, val: str) -> bool:
        if self._peek() and self._peek().value == val:
            self.i += 1
            return True
        return False

    def _read_balanced(self, open_s: str, close_s: str) -> List[SVToken]:
        out, depth = [], 0
        while self.i < self.n:
            tok = self._adv()
            if tok.value == open_s:
                depth += 1
                if depth > 1:
                    out.append(tok)
            elif tok.value == close_s:
                depth -= 1
                if depth == 0:
                    break
                out.append(tok)
            else:
                out.append(tok)
        return out

    def _split_top(self, toks: List[SVToken], sep: str = ",") -> List[List[SVToken]]:
        groups, cur, depth = [], [], 0
        for t in toks:
            if t.value in ("(", "[", "{"):
                depth += 1
            elif t.value in (")", "]", "}"):
                depth = max(0, depth - 1)
            if t.value == sep and depth == 0:
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            groups.append(cur)
        return groups

    def _parse_params(self, toks: List[SVToken]) -> Dict[str, int]:
        params: Dict[str, int] = {}
        for item in self._split_top(toks):
            eq = next((i for i, t in enumerate(item) if t.value == "="), None)
            if eq and eq > 0:
                val = eval_int_expr(" ".join(t.value for t in item[eq + 1:]), params)
                if val is not None:
                    params[item[eq - 1].value] = val
        return params

    def _parse_range(self, toks: List[SVToken], start: int, params: Dict[str, int]) -> Tuple[str, int, int]:
        depth, end, inner = 0, start, []
        while end < len(toks):
            t = toks[end]
            if t.value == "[":
                depth += 1
                if depth > 1:
                    inner.append(t)
            elif t.value == "]":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
                inner.append(t)
            else:
                inner.append(t)
            end += 1
        colon = next((i for i, t in enumerate(inner) if t.value == ":"), None)
        expr = "[" + "".join(t.value for t in inner) + "]"
        if colon is not None:
            msb = eval_int_expr(" ".join(t.value for t in inner[:colon]), params)
            lsb = eval_int_expr(" ".join(t.value for t in inner[colon + 1:]), params)
            w = abs(msb - lsb) + 1 if (msb is not None and lsb is not None) else 1
        else:
            w = 1
        return expr, w, end

    def _try_parse_interface_port(self, item: List[SVToken]) -> Optional[SVPortAST]:
        """Recognize SV interface/modport port declarations like `axi_lite_if.slave axi` or `interface.mp name`."""
        if not item:
            return None
        if any(t.value in self._DIRS for t in item):
            return None
        # Form 1: <iface_type> . <modport> <port_name>
        if len(item) >= 4 and item[0].kind == "IDENT" and item[1].value == "." and item[2].kind == "IDENT" and item[3].kind == "IDENT":
            return SVPortAST(
                name=item[3].value,
                direction="interface",
                width_expr="interface",
                width=0,
                line=item[3].line,
                is_interface=True,
                interface_type=item[0].value,
                modport=item[2].value,
            )
        # Form 2: <iface_type> <port_name> where iface_type looks like *_if or generic interface
        if len(item) == 2 and item[0].kind == "IDENT" and item[1].kind == "IDENT":
            tname = item[0].value
            if tname == "interface" or tname.lower().endswith("_if"):
                return SVPortAST(
                    name=item[1].value,
                    direction="interface",
                    width_expr="interface",
                    width=0,
                    line=item[1].line,
                    is_interface=True,
                    interface_type=tname,
                    modport="",
                )
        return None

    def _parse_ansi(self, toks: List[SVToken], params: Dict[str, int]) -> List[SVPortAST]:
        ports, cur_dir, cur_wexpr, cur_w = [], "input", "1", 1
        for item in self._split_top(toks):
            iface_port = self._try_parse_interface_port(item)
            if iface_port is not None:
                ports.append(iface_port)
                continue
            idx = 0
            while idx < len(item):
                v = item[idx].value
                if v in self._DIRS:
                    cur_dir, cur_wexpr, cur_w = v, "1", 1
                    idx += 1
                elif v in self._TYPES:
                    idx += 1
                elif v == "[":
                    cur_wexpr, cur_w, idx = self._parse_range(item, idx, params)
                else:
                    break
            name_tok = None
            for t in item[idx:]:
                if t.value in ("=", "["):
                    break
                if t.kind == "IDENT":
                    name_tok = t
            if name_tok:
                ports.append(SVPortAST(name_tok.value, cur_dir, cur_wexpr, cur_w, name_tok.line))
        return ports

    def _parse_non_ansi(self, names: List[str], params: Dict[str, int]) -> List[SVPortAST]:
        declared: Dict[str, SVPortAST] = {}
        while self.i < self.n and self._peek().value != "endmodule":
            val = self._peek().value
            if val in ("parameter", "localparam"):
                self._adv()
                stmt = []
                while self.i < self.n and self._peek().value != ";":
                    stmt.append(self._adv())
                self._match(";")
                params.update(self._parse_params(stmt))
            elif val in self._DIRS:
                direction = self._adv().value
                stmt = []
                while self.i < self.n and self._peek().value != ";":
                    stmt.append(self._adv())
                self._match(";")
                idx, wexpr, w = 0, "1", 1
                while idx < len(stmt):
                    if stmt[idx].value in self._TYPES:
                        idx += 1
                    elif stmt[idx].value == "[":
                        wexpr, w, idx = self._parse_range(stmt, idx, params)
                    else:
                        break
                for grp in self._split_top(stmt[idx:]):
                    ident = next((t for t in grp if t.kind == "IDENT"), None)
                    if ident:
                        declared[ident.value] = SVPortAST(ident.value, direction, wexpr, w, ident.line)
            else:
                self._adv()
        return [declared[n] for n in names if n in declared]

    def parse_all(self) -> List[SVModuleAST]:
        mods: List[SVModuleAST] = []
        while self.i < self.n:
            if self._adv().value != "module":
                continue
            nt = self._peek()
            if not nt or nt.kind != "IDENT":
                continue
            mod_name = self._adv().value
            params: Dict[str, int] = {}
            if self._match("#") and self._peek() and self._peek().value == "(":
                params = self._parse_params(self._read_balanced("(", ")"))
            ports: List[SVPortAST] = []
            if self._peek() and self._peek().value == "(":
                ptoks = self._read_balanced("(", ")")
                self._match(";")
                has_dir = any(t.value in self._DIRS for t in ptoks)
                has_iface = any(self._try_parse_interface_port(g) is not None for g in self._split_top(ptoks))
                if has_dir or has_iface:
                    ports = self._parse_ansi(ptoks, params)
                else:
                    names = [g[0].value for g in self._split_top(ptoks) if g and g[0].kind == "IDENT"]
                    ports = self._parse_non_ansi(names, params)
            else:
                self._match(";")
            while self.i < self.n and self._peek().value != "endmodule":
                self._adv()
            self._match("endmodule")
            mods.append(SVModuleAST(mod_name, self.file_path, params, tuple(ports), parser_backend="builtin"))
        return mods


def _find_slang_context(rtl_path: Path) -> Tuple[List[str], List[str]]:
    """为 slang 推导同批编译文件与 include 目录。

    真实 RTL 普遍 `import xxx_pkg::*`。只把单个文件喂给 slang 时，它无法解析
    这些 import，会报 `unknown package` 并以非 0 退出（但仍可能输出合法 AST
    JSON）。为了让 slang 后端对真实项目生效，这里：
      1) 收集目标文件所在目录及其父目录下所有 `*_pkg.sv` 作为同批编译文件；
      2) 把这些目录也作为 `-I` include 目录传给 slang。
    """
    extra_files: List[str] = []
    inc_dirs: List[str] = []
    seen_dirs = set()
    # 用绝对路径：slang 子进程的 cwd 与调用方可能不同，相对路径会解析失败
    rtl_abs = rtl_path.resolve()
    for d in (rtl_abs.parent, *rtl_abs.parent.parents):
        if not d.is_dir() or d in seen_dirs:
            continue
        seen_dirs.add(d)
        # 收集同目录的 package 定义（*_pkg.sv）和 SV interface 定义（*_if.sv）
        # ——真实 RTL 里模块会 `import xxx_pkg::*`、引用 SV interface（如
        # axi_lite_if.slave）。不把它们的定义文件一起喂给 slang，会报
        # unknown package / unknown interface。timescale 报错属于源文件自身
        # 缺声明，slang 会提示但不妨碍输出可用 AST。
        for pat in ("*_pkg.sv", "*_if.sv"):
            for pf in sorted(d.glob(pat)):
                if pf != rtl_abs:
                    extra_files.append(str(pf))
        # 即使本目录没有收集到定义文件，也作为 include 目录（可能被间接引用）
        inc_dirs.append(str(d))
        # 只向上走两层，避免在大仓库里扫描过多目录拖慢 slang
        if len(seen_dirs) >= 3:
            break
    # 去重保序
    seen_f = set()
    extra_files = [f for f in extra_files if not (f in seen_f or seen_f.add(f))]
    return extra_files, inc_dirs


def _run_slang(
    slang_bin: str,
    rtl_path: Path,
    extra_files: List[str],
    inc_dirs: List[str],
) -> Tuple[Optional[Any], List[str]]:
    """跑 slang --ast-json。返回 (AST tree, diagnostics)。

    判定策略（关键）：slang 在遇到未知 package 时会 **exit 非 0 但仍输出完整
    AST JSON**。旧实现因 `returncode != 0` 整段丢弃，导致 slang 后端对几乎
    所有真实 RTL 静默失效、回退到 builtin。这里改为「stdout 有合法 JSON 就用」，
    同时把 stderr 诊断透传给调用方，绝不静默丢弃。
    """
    diags: List[str] = []
    # 用绝对路径：slang 子进程 cwd 可能与调用方不同
    rtl_arg = str(rtl_path.resolve())
    # 先带依赖包编译（最可能干净通过），失败再退回只编译目标文件
    inc_args: List[str] = []
    for d in inc_dirs:
        inc_args += ["-I", d]
    # 大量 RTL 不写 `timescale`，slang 会为此报 error（噪声，不影响 AST）。
    # 提供默认 timescale 消除这类报错，让 stderr 里只剩真正的问题。
    timescale_args = ["--timescale", "1ns/1ps"]

    attempts: List[Tuple[List[str], bool]] = []  # (cmd, is_single_file_attempt)
    # 顺序关键：**先只编译目标文件**，失败才带同目录依赖文件重试。
    # 旧实现反过来（先带 *_if.sv/*_pkg.sv），会把依赖文件里的模块也解析进
    # 结果——实测 /tmp/ghsv 下 gen_mux.v + arr_if.sv 同目录时，extract 把
    # arr_if.sv 的 arr_user/axi_bus 也算成 gen_mux.v 的模块（多出 2 个不
    # 属于目标文件的模块）。先单文件可保证「只返回目标文件的模块」，只有
    # 目标文件真有跨文件依赖（import/interface）解析失败时才带上依赖重试。
    attempts.append(([slang_bin, "--ast-json", "-", *inc_args, *timescale_args, rtl_arg], True))
    if extra_files:
        attempts.append((
            [slang_bin, "--ast-json", "-", *inc_args, *timescale_args, *extra_files, rtl_arg],
            False,
        ))

    for cmd, is_single_file in attempts:
        attempt_diags: List[str] = []
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
        except Exception as exc:
            attempt_diags.append(f"slang invocation error: {exc}")
            diags.extend(attempt_diags)
            continue
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines():
                if "error" in line.lower() or "warning" in line.lower():
                    attempt_diags.append(line.strip()[:200])
        out = proc.stdout.strip()
        if not out:
            diags.extend(attempt_diags)
            continue
        # slang --ast-json - 会把人类可读 banner（"Top level design units:"
        # / "Build succeeded: ...") 和 JSON 混在同一个 stdout，JSON 从第一个
        # "{" 开始。直接 json.loads 整段必然失败，必须先定位 JSON 起点。
        brace = out.find("{")
        if brace < 0:
            attempt_diags.append("slang stdout contained no JSON object")
            diags.extend(attempt_diags)
            continue
        try:
            tree = json.loads(out[brace:])
        except Exception as exc:
            attempt_diags.append(f"slang AST JSON parse error: {exc}")
            diags.extend(attempt_diags)
            continue
        # 该 attempt 拿到合法 JSON。若是「只编译单文件」的 attempt 1 成功，
        # 它的 stderr 里会有 unknown interface/package 之类**预期噪声**（单文件
        # 必然缺跨文件定义），不应作为 W013 报给调用方。只有 attempt 2（带依赖）
        # 的 stderr 才是真实诊断。
        if is_single_file:
            attempt_diags = []
        return tree, attempt_diags
    return None, diags


def _parse_with_slang(
    rtl_path: Path, slang_bin: str
) -> Tuple[Optional[List[SVModuleAST]], List[str]]:
    """Optional slang AST JSON backend when slang is installed on PATH."""
    extra_files, inc_dirs = _find_slang_context(rtl_path)
    tree, diags = _run_slang(slang_bin, rtl_path, extra_files, inc_dirs)
    if tree is None:
        return None, diags or ["slang produced no usable AST JSON"]

    mods: List[SVModuleAST] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("kind") == "Instance" and isinstance(node.get("body"), dict):
            body = node["body"]
            mname = str(body.get("name") or node.get("name") or "")
            params: Dict[str, int] = {}
            ports: List[SVPortAST] = []
            for mem in body.get("members", []):
                mk = mem.get("kind")
                if mk == "Parameter" and isinstance(mem.get("value"), (int, str)):
                    val = eval_int_expr(str(mem["value"]), params)
                    if val is not None and mem.get("name"):
                        params[str(mem["name"])] = val
                elif mk == "Port":
                    dmap = {"In": "input", "Out": "output", "InOut": "inout"}
                    d = dmap.get(str(mem.get("direction", "")), "input")
                    tstr = str(mem.get("type", "logic"))
                    m_rng = re.search(r"\[(\d+):(\d+)\]", tstr)
                    w = abs(int(m_rng.group(1)) - int(m_rng.group(2))) + 1 if m_rng else 1
                    ports.append(SVPortAST(str(mem.get("name", "")), d, tstr, w, 0))
                elif mk == "InterfacePort":
                    ports.append(
                        SVPortAST(
                            name=str(mem.get("name", "")),
                            direction="interface",
                            width_expr="interface",
                            width=0,
                            line=0,
                            is_interface=True,
                            interface_type=str(mem.get("interfaceDef", "")),
                            modport=str(mem.get("modport", "")),
                        )
                    )
            if mname:
                mods.append(SVModuleAST(mname, str(rtl_path), params, tuple(ports), parser_backend="slang"))
        for v in node.values():
            if isinstance(v, list):
                for item in v:
                    _walk(item)
            elif isinstance(v, dict):
                _walk(v)

    _walk(tree)
    # 返回 list（而非 tuple）以便调用方能原地补齐 interface 信息
    return (list(mods) or None), diags


def parse_sv_file(
    rtl_path: Path,
    backend: str = "auto",
    diagnostics: Optional[List[str]] = None,
) -> List[SVModuleAST]:
    """Parse an RTL file using slang AST when available (backend='auto'/'slang'), else builtin parser.

    当 slang 不可用或解析失败而回退到 builtin 时，把原因追加到 `diagnostics`
    （若调用方提供该 list）。这样「slang 后端没生效」对调用方是可见信号，
    而不是静默回退——否则「可选 slang AST 后端」就是虚报。
    """
    if backend in ("auto", "slang"):
        slang_bin = shutil.which("slang")
        if slang_bin:
            slang_mods, slang_diags = _parse_with_slang(rtl_path, slang_bin)
            if slang_mods:
                # 已知限制（实测）：slang 12.x 的 ast-json 不导出 InterfacePort
                # 的 interfaceDef/modport，只给 {name,kind,addr,isGeneric}。为
                # 不丢这些信息，用 builtin 解析的 interface 类型/modport 补齐
                # slang 结果里为空的 interface 端口。
                # _parse_with_slang 返回 tuple，转成 list 以便原地补齐
                slang_list = list(slang_mods)
                _enrich_from_builtin(rtl_path, slang_list)
                if diagnostics is not None:
                    diagnostics.extend(slang_diags)
                return slang_list
            if diagnostics is not None:
                diagnostics.extend(slang_diags)
                diagnostics.append(
                    f"slang backend produced no modules for {rtl_path.name}; "
                    "falling back to builtin parser"
                )
        elif diagnostics is not None:
            diagnostics.append("slang not found on PATH; using builtin parser")
        if backend == "slang":
            raise RuntimeError("Requested parser backend 'slang' is unavailable or failed to parse.")
    return SVModuleParser(sv_tokenize(rtl_path.read_text(encoding="utf-8")), str(rtl_path)).parse_all()


def _enrich_from_builtin(rtl_path: Path, slang_mods: List[SVModuleAST]) -> None:
    """用 builtin 解析结果补齐 slang 缺失的 interface 类型/modport。

    slang mods 是 frozen dataclass，因此直接改写其 ports tuple 里匹配端口的
    三个字段（新建 SVPortAST）。builtin 拿不到对应模块/端口时保持原值不动。
    """
    try:
        builtin_mods = SVModuleParser(
            sv_tokenize(rtl_path.read_text(encoding="utf-8")), str(rtl_path)
        ).parse_all()
    except Exception:
        return
    builtin_ports = {}
    for bm in builtin_mods:
        for bp in bm.ports:
            if bp.is_interface and bp.interface_type:
                builtin_ports.setdefault(bp.name, (bp.interface_type, bp.modport))

    enriched = []
    changed = False
    for sm in slang_mods:
        new_ports = list(sm.ports)
        for i, sp_ in enumerate(new_ports):
            if sp_.is_interface and not sp_.interface_type and sp_.name in builtin_ports:
                itype, modp = builtin_ports[sp_.name]
                new_ports[i] = SVPortAST(
                    name=sp_.name, direction=sp_.direction, width_expr=sp_.width_expr,
                    width=sp_.width, line=sp_.line, is_interface=True,
                    interface_type=itype, modport=modp,
                )
                changed = True
        enriched.append(SVModuleAST(sm.name, sm.file_path, sm.parameters, tuple(new_ports), sm.parser_backend))
    if changed:
        # 原地替换（slang_mods 是调用方持有的 list，直接改写其元素）
        slang_mods[:] = enriched
