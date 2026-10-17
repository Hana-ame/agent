"""固定点求解器（独立实现）：给一份 bootstrap.py 主源 + 其余文件，解出自指段的精确长度。

主源里 `BOOTSTRAP_B64 = "FFF…F"` 是一行**定长填充**，长度 pad 必须等于整份主源
gzip 压缩后的字节数——因为 `generate_framework()` 就是"用这条 payload 重建源码"。

朴素迭代 `pad ← z(pad)` 有 ±1 抖动（z 对 pad 不是严格单调的），会在两个值间打转。
所以分两段：先逼近，再在邻域 ±32 内线性扫，取第一个严格不动点。

用法：
    python tools/fixpoint.py            # 读仓库文件 → 固化主源
    python tools/fixpoint.py --check    # 只检查当前主源是否是固定点
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
import sys
from typing import Callable, Dict, List, Tuple

SELF_FILL = "SELFB64FILL"
SLOT_RE = re.compile(r"^BOOTSTRAP_SLOT = \d+", re.M)
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


def render_slot(source: str, slot: int) -> str:
    """把 `BOOTSTRAP_SLOT = N` 写成给定值（槽宽本身就是字段，写进去不改变字节数）。"""
    out, n = SLOT_RE.subn(f"BOOTSTRAP_SLOT = {slot}", source, count=1)
    if n != 1:
        raise ValueError("主源缺少 BOOTSTRAP_SLOT 行")
    return out


def render(source: str, manifest_text: str, code_text: str, pad: int = 0) -> str:
    out, n1 = MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    if n1 != 1:
        raise ValueError("主源缺少 FRAME_MANIFEST 块")
    out, n2 = CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("主源缺少 FRAME_CODE 块")
    return render_slot(out.replace(SELF_FILL, "F" * pad), pad)


def code_text_of(payloads: Dict[str, str]) -> str:
    files = [
        {"path": p, "key": p.replace("/", "_").replace(".", "_").upper() + "_B64", "code": payloads[p]}
        for p in payloads
        if p != MAIN
    ]
    files.append({"path": MAIN, "key": SELF_MARK, "code": SELF_MARK})
    return json.dumps(
        {
            "_self": "除本文件外的源码都 gzip+base64 内嵌在 code 字段；"
            "本文件的源码就是本文件自身，用 " + SELF_MARK + " 标记占位",
            "files": files,
        },
        ensure_ascii=False,
        indent=2,
    )


def solve(main_src: str, manifest_of: Callable[[int, str], str], code_text: str) -> Tuple[int, str]:
    """求 pad 使 `len(gzip(渲染结果)) == pad` 且清单里的 sha 等于自身 sha（严格不动点）。

    `z(pad)` 不是常函数（pad 变化会改变清单里 6 位数字的文本，压缩率随之抖动），
    但**大体单调**：z 随 pad 缓慢上升，`z(pad) - pad` 随 pad 严格下降。
    朴素迭代 `pad ← 4⌈z/3⌉` 会放大这个量级（实测收敛到 42840，离真解 32127 很远）——**弃用**。
    改成二分定位变号点，再在邻域逐点判定严格不动点。取证见 tools/probe_window.py。
    """
    z_guess = 4 * ((len(gz(main_src.encode("utf-8"))) + 2) // 3)
    seen: Dict[int, int] = {}  # pad -> z(pad)：z 与 pad 无关的部分不该重算

    def trial_at(cand: int) -> Tuple[int, str, str]:
        """返回 (实际 gzip 体积, 最终源码, 清单里的 sha)。"""
        sha = hashlib.sha256(render(main_src, manifest_of(cand, ""), code_text, pad=cand).encode()).hexdigest()
        text = render(main_src, manifest_of(cand, sha), code_text, pad=cand)
        return len(gz(text.encode("utf-8"))), text, sha

    def d(x: int) -> int:
        if x not in seen:
            seen[x] = trial_at(x)[0] - x
        return seen[x]

    # 先用二分把区间压到 `z(pad) - pad` 变号处，再在该点邻域取第一个严格不动点。
    # 单调性并不严格（压缩率抖动会让 z 上下 1~2），所以二分只是"快速定位"，判据仍是逐点验证。
    lo, hi = 1, max(4096, z_guess)
    while d(hi) > 0:  # 上界要落在 d<0 一侧
        lo, hi = hi, hi * 2
    span = 1
    for _ in range(40):
        mid = (lo + hi) // 2
        if mid in (lo, hi):
            span = hi - lo + 1
            break
        if d(mid) > 0:
            lo = mid
        else:
            hi = mid
    else:
        span = hi - lo + 1
    for radius in (span, 64, 512, 4096, 65536):
        for cand in range(max(1, lo - radius), hi + radius + 1):
            size, text, sha = trial_at(cand)
            if size == cand and hashlib.sha256(text.encode("utf-8")).hexdigest() == sha:
                return cand, text
    raise RuntimeError(f"固定点未收敛（区间 [{lo}, {hi}]）")


def probe(src: str) -> Dict:
    """在独立模块对象里执行源码（dataclass 需要模块在 sys.modules 里注册）。"""
    import types

    from pathlib import Path

    mod = types.ModuleType("_vea_fixpoint_probe")
    mod.__file__ = str(Path(__file__).resolve().parent.parent / MAIN)
    sys.modules["_vea_fixpoint_probe"] = mod
    try:
        exec(compile(src, "<fixed_point>", "exec"), mod.__dict__)  # noqa: S102
    finally:
        pass
    return mod.__dict__


def build(files: Dict[str, str], spec: Dict) -> Dict[str, str]:
    """把 `{路径: 源码}` 变成自洽的框架源码集（主源已填入 manifest/code/payload）。"""
    for p in FRAME_FILES:
        if p not in files:
            raise ValueError(f"缺文件 {p}")
        ast.parse(files[p])

    others = [p for p in FRAME_FILES if p != MAIN]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: b64of(raw[p]) for p in others}
    code_text = code_text_of(payloads)

    def manifest_of(pad: int, sha: str) -> str:
        # 主源三项同源：pad = gzip(最终源码) 的字节数 = 内嵌 payload 的 base64 长度；
        # 而 base64 长度 == 源码字节数（每 3 字节进 4 字符）。三者必须一致，
        # 否则 bootstrap.py 的 manifest() 与本工具会写出不同的清单（曾因此 FAIL）。
        docs = []
        for p in FRAME_FILES:
            if p == MAIN:
                docs.append({"path": p, "bytes": pad, "b64_len": pad,
                             "sha256": sha or "PENDING", "sig": signature(p, files[p])})
            else:
                docs.append({"path": p, "bytes": len(raw[p]), "b64_len": len(payloads[p]),
                             "sha256": hashlib.sha256(raw[p]).hexdigest(), "sig": signature(p, files[p])})
        return json.dumps({"spec": spec, "files": docs}, ensure_ascii=False, indent=2)

    pad, final = solve(files[MAIN], manifest_of, code_text)

    ns = probe(final)
    gen = ns["generate_framework"]()
    for p in FRAME_FILES:
        want = files[p] if p != MAIN else final
        if gen.get(p) != want:
            raise RuntimeError(f"复现失败: {p}")
    out = {p: files[p] for p in FRAME_FILES if p != MAIN}
    out[MAIN] = final
    return out


def main(argv: List[str]) -> int:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    files = {p: (root / p).read_text(encoding="utf-8") for p in FRAME_FILES}
    main_src = files[MAIN]
    if "--check" in argv:
        ns = probe(main_src)
        ok, why = ns["verify_self_reproduction"]()
        print(("OK  " if ok else "FAIL ") + why)
        return 0 if ok else 1
    ns: Dict = {"Any": object, "Dict": dict, "List": list}
    exec(re.search(r"^FRAMEWORK_SPEC: Dict\[str, Any\] = \{.*?^\}", main_src, re.S | re.M).group(0), ns)
    out = build(files, ns["FRAMEWORK_SPEC"])
    (root / MAIN).write_text(out[MAIN], encoding="utf-8")
    pad = len(re.search(r'^BOOTSTRAP_B64 = "(F+)"$', out[MAIN], re.M).group(1))
    print(f"固化完成：主源 {len(out[MAIN].encode())} 字节，pad={pad}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
