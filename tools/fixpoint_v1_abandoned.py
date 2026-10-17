"""固定点求解器：给一份 bootstrap.py 主源 + 其余文件，解出自指 payload 的精确长度。

原理：主源里 `BOOTSTRAP_B64` 的值是**定长填充行**（N 个 'F'），它的长度 L 恰好等于
自身 gzip 后 base64 文本的长度。因为 payload 只由 'F' 组成，gzip 体积对 L 近乎线性，
于是 `L = 4⌈(z(L)+1)/3⌉` 有唯一不动点，用「估 L → 实渲染 → 量 z → 重算 L」迭代 4~10 轮即稳定。

与 v1 的 L = C + 2 + 4⌈L/3⌉ 不同：那里 L 出现在**内容**里（长度随自身增长），系数 4/3 > 1 必发散；
这里 L 只决定**填充行长度**，内容本身不依赖 L，所以是「解方程」而不是「解递归」。
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
import sys
from typing import Dict, List, Tuple

SELF_FILL = "SELFB64FILL"
ZERO = bytes(1)  # 单个零字节：用来量 base64 长度
SELF_MARK = "BOOTSTRAP_B64"
MAIN = "vea_v6/parts/bootstrap.py"
SPEC_FILES = ("vea_v6/__init__.py",)
FRAME_FILES = (
    "vea_v6/parts/__init__.py",
    "vea_v6/parts/primitives.py",
    "vea_v6/parts/llm.py",
    "vea_v6/parts/dag.py",
    "vea_v6/parts/builder.py",
    MAIN,
    "vea_v6/__init__.py",
)

MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)
SELF_LINE_RE = re.compile(r"^" + SELF_MARK + r" = \"(.+)\"$", re.M)


def signature(path: str, source: str) -> List[str]:
    if path in SPEC_FILES:
        m = re.search(r"^__all__\s*=\s*\[(.*?)\]", source, re.S | re.M)
        if not m:
            raise ValueError(f"{path}: SPEC 文件必须有 __all__")
        return re.findall(r'"([^"]+)"', m.group(1))
    tree = ast.parse(source)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                out.append(f"def {node.name}({', '.join(a.arg for a in node.args.args)})")
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                out.append(f"class {node.name}({', '.join(ast.unparse(b) for b in node.bases)})")
    return out


def b64of(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")


def gz(raw: bytes) -> bytes:
    return gzip.compress(raw, 9, mtime=0)


def render(source: str, manifest_text: str, code_text: str, pad: int = 0) -> str:
    out, n1 = MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    if n1 != 1:
        raise ValueError("主源缺少 FRAME_MANIFEST 块")
    out, n2 = CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("主源缺少 FRAME_CODE 块")
    return out.replace(SELF_FILL, "F" * pad)


def probe(src: str) -> Dict:
    ns: Dict = {"__name__": "_vea_fixpoint_probe"}
    exec(compile(src, "<fixed_point>", "exec"), ns)  # noqa: S102
    return ns


def solve(main_src: str, manifest_of, code_text: str):
    """解 pad 使 len(gz(渲染结果)) == pad。

    两段：先按 z(pad) 逼近（单调，几轮进到 z 的抖动带内），再在邻域 ±64 内线性扫，
    取第一个精确不动点。单一 z(pad) 迭代会在 z 的 1 字节抖动附近来回跳，扫邻域是必要的。
    """
    pad, sha, src = 1, "", ""
    for _ in range(20):
        src = render(main_src, manifest_of(pad, sha), code_text, pad=pad)
        actual = len(gz(src.encode("utf-8")))
        if actual == pad:
            return pad, src
        pad = actual
        sha = hashlib.sha256(render(main_src, manifest_of(pad, ""), code_text, pad=pad).encode()).hexdigest()
    base = pad
    for cand in range(max(1, base - 64), base + 65):
        cand_sha = hashlib.sha256(render(main_src, manifest_of(cand, ""), code_text, pad=cand).encode()).hexdigest()
        trial = render(main_src, manifest_of(cand, cand_sha), code_text, pad=cand)
        if len(gz(trial.encode("utf-8"))) == cand:
            return cand, trial
    raise RuntimeError(f"固定点未收敛（pad≈{base}）")


def build(files: Dict[str, str], spec: Dict) -> Dict[str, str]:
    """把 `{路径: 源码}` 变成自洽的框架源码集（主源已填入 manifest/code/payload）。"""
    for p in FRAME_FILES:
        if p not in files:
            raise ValueError(f"缺文件 {p}")
        ast.parse(files[p])

    others = [p for p in FRAME_FILES if p != MAIN]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: b64of(raw[p]) for p in others}

    def manifest_obj(pad_len: int, self_sha: str = "") -> Dict:
        docs = []
        for p in FRAME_FILES:
            if p == MAIN:
                docs.append({"path": p, "bytes": 0, "b64_len": pad_len, "sha256": self_sha or "PENDING",
                             "sig": signature(p, files[p])})
            else:
                docs.append({"path": p, "bytes": len(raw[p]), "b64_len": len(payloads[p]),
                             "sha256": hashlib.sha256(raw[p]).hexdigest(), "sig": signature(p, files[p])})
        return {"spec": spec, "files": docs}

    def code_obj(pad_len: int) -> Dict:
        items = [{"path": p, "key": f"{p.replace('/', '_').replace('.', '_').upper()}_B64", "code": payloads[p]}
                 for p in others]
        items.append({"path": MAIN, "key": SELF_MARK, "code": "SELF"})
        return {"files": items}

    code_text = json.dumps(code_obj(0), ensure_ascii=False, indent=2)
    pad, final = solve(files[MAIN], lambda pad, sha: json.dumps(manifest_obj(pad, sha), ensure_ascii=False,
                                                                 indent=2), code_text)
    # 终校验：主源跑起来能逐字节复现自己
    ns = probe(final)
    gen = ns["generate_framework"]()
    for p in FRAME_FILES:
        want = files[p] if p != MAIN else final
        if gen.get(p) != want:
            raise RuntimeError(f"复现失败: {p}")
    out = {p: files[p] for p in FRAME_FILES if p != MAIN}
    out[MAIN] = final
    return out


def current() -> Dict[str, str]:
    return {p: open(p, encoding="utf-8").read() for p in FRAME_FILES}


if __name__ == "__main__":
    from pathlib import Path
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    files = {p: (root / p).read_text(encoding="utf-8") for p in FRAME_FILES}
    spec = json.loads(re.search(r"FRAMEWORK_SPEC: Dict\[str, Any\] = (\{.*?\n\})\n", files[MAIN], re.S).group(1).replace("True", "true").replace("False", "false")) if False else None
    # SPEC 直接从前代主源里 import 出来，避免手抄
    spec_mod = {}
    exec(re.search(r"^FRAMEWORK_SPEC: Dict\[str, Any\] = \{.*?^\}", files[MAIN], re.S | re.M).group(0), spec_mod)
    out = build(files, spec_mod["FRAMEWORK_SPEC"])
    (root / MAIN).write_text(out[MAIN], encoding="utf-8")
    print("fixed point, main bytes =", len(out[MAIN].encode()))
