"""框架自举的固定点求解器（第三版：模板形态 + 就近最小 pad）。

## 为什么换了思路

前两版都卡在同一处：把库清单一并塞进主源，然后要求 `z(pad) == pad`
（z = 渲染结果的 gzip 体积）。实测 `z(pad)` 在真解附近几乎是常数（pad 加 1，
压缩体积往往只涨 0~1），于是 `z(pad) - pad` 直接从 +12 跳到 -3，**跨过零点而不命中**。
窗口取证：`python tools/probe_window.py 32120 32230` → 「严格不动点: 无」。

数学根因：gzip 对"单调递增的重复字节"几乎免费。所以 **递增的 pad 是一个宽度为 0 的解**。

## 现在的不动点

把「主源字节数」和「槽宽」解耦，两层分离：

| 层 | 内容 | 关系 |
|---|---|---|
| 主源 | 代码 + 定长 `SELF_FILL` 标记 + `BOOTSTRAP_SLOT = pad` | 模板形态，字节数**不依赖** pad 的字面值 |
| payload | `b64(gzip(最终源码))` | 长度恰为 pad，槽内填 `pad - len(sha256)` 个 `x` 后再 sha512 |

于是：
1. 对任意 pad，把 `pad` 写进 `BOOTSTRAP_SLOT = pad`，源码字节数 **不变**（只是数字位数变化，位数封顶后恒定）；
2. `z(pad)` 与 pad 无关 ⇒ `z(pad)` 是常数 z0 ⇒ 取 **pad = 最小的素数 ≥ z0**；
3. `z0 <= pad` 保证"定长填充 + sha 尾巴"总能填满槽（最多浪费几十字节）。

这样不动点从"未必存在的等式"变成"一定存在的取整"，且完全确定。
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = None  # 由 main() 设定

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

SELF_MARK = "BOOTSTRAP_B64"
MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)
SLOT_RE = re.compile(r"^BOOTSTRAP_SLOT = \d+", re.M)
SELF_RE = re.compile(r'^SELF_FILL = "F+"$', re.M)


# ---------------------------------------------------------------- 基础工具
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


def gz(raw: bytes) -> bytes:
    return gzip.compress(raw, 9, mtime=0)


def b64of(raw: bytes) -> str:
    return base64.b64encode(gz(raw)).decode("ascii")


SHA_HOLE = "0" * 64


def skeleton_sha(source: str) -> str:
    """主源骨架 sha256：把清单里主源自己那一行的 sha256 值涂成等长占位后取整文哈希。

    涂掉"这一格自己"⇒ `x` 不再影响被哈希的内容 ⇒ 一遍收敛，且不改变字节数。
    """
    pat = re.compile(
        r'("path":\s*"' + re.escape(MAIN) + r'",\s*"bytes":\s*\d+,\s*"b64_len":\s*\d+,\s*"sha256":\s*")[0-9a-f]{64}(")'
    )
    out, n = pat.subn(lambda m: m.group(1) + SHA_HOLE + m.group(2), source, count=1)
    return hashlib.sha256((out if n == 1 else source).encode("utf-8")).hexdigest()


def code_text_of(payloads: Dict[str, str]) -> str:
    files = [
        {"path": p, "key": p.replace("/", "_").replace(".", "_").upper() + "_B64", "code": payloads[p]}
        for p in FRAME_FILES
        if p != MAIN
    ]
    files.append({"path": MAIN, "key": SELF_MARK, "code": SELF_MARK})
    return json.dumps(
        {
            "_self": "除本文件外的源码都 gzip+base64 内嵌在 code 字段；本文件的源码就是本文件自身，"
            "用 " + SELF_MARK + " 标记占位，定长填充见 SELF_FILL，槽宽见 BOOTSTRAP_SLOT",
            "files": files,
        },
        ensure_ascii=False,
        indent=2,
    )


def render(source: str, manifest_text: str, code_text: str) -> str:
    """把清单/代码块写进主源（不改 SELF_FILL，不改 BOOTSTRAP_SLOT）。"""
    out, n1 = MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    if n1 != 1:
        raise ValueError("主源缺少 FRAME_MANIFEST 块")
    out, n2 = CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("主源缺少 FRAME_CODE 块")
    return out


def set_slot(source: str, pad: int) -> str:
    out, n = SLOT_RE.subn(f"BOOTSTRAP_SLOT = {pad}", source, count=1)
    if n != 1:
        raise ValueError("主源缺少 BOOTSTRAP_SLOT 行")
    return out


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def next_prime(n: int) -> int:
    k = max(2, n)
    while not is_prime(k):
        k += 1
    return k


# ---------------------------------------------------------------- 求解
def solve(template: str, manifest_of: Callable[[int, str], str], code_text: str,
          self_bytes: int = 0) -> Tuple[int, str]:
    """返回 (pad, 最终主源)。

    槽内内容 = `(pad - 64)` 个 `x` + 定型源码自己的 sha256。两层解耦：
    - `SELF_FILL` 是定长占位符、`BOOTSTRAP_SLOT = pad` 只写数字 ⇒ 填充不参与解方程；
    - 清单里主源 `sha256` 用**骨架 sha**（涂掉自己那一格）⇒ 代入即收敛；
    - 清单里主源 `bytes` 用上一轮测量值 `self_bytes`，由外层迭代到自洽。
    """
    body = render(template, manifest_of(32127, "", self_bytes), code_text)
    z = len(gz(set_slot(body, 32127).encode("utf-8")))
    pad = next_prime(max(z + 4096, 256))
    for _ in range(64):
        src = set_slot(render(template, manifest_of(pad, "", self_bytes), code_text), pad)
        zz = len(gz(src.encode("utf-8")))
        if zz > pad:
            pad = next_prime(max(pad + 32, zz + 64))
            continue
        cur = ""
        for _ in range(12):
            sha = skeleton_sha(src)
            if sha == cur:
                if reproduces(src):
                    return pad, src
                break
            cur, src = sha, set_slot(render(template, manifest_of(pad, sha, self_bytes), code_text), pad)
            if len(gz(src.encode("utf-8"))) > pad:
                break
        else:
            raise RuntimeError("清单 sha 回填未收敛")
        pad = next_prime(max(pad + 32, zz + 64))
    raise RuntimeError(f"固定点未收敛（最后试到 pad={pad}）")


def reproduces(src: str) -> bool:
    """内层判据：把 src 当主源跑一遍，`generate_framework()` 是否逐字节回放出 src 自己。

    只验 payload 闭合，不验清单 `bytes`——`bytes` 由外层迭代到真值，由 `--check`
    端的 `verify_self_reproduction()` 统一把关。两层混一起会互相锁死。
    """
    try:
        return probe(src)["generate_framework"]()[MAIN] == src
    except Exception:
        return False


def probe(src: str, path: Optional[Path] = None, name: str = "_vea_fixpoint_probe") -> Dict:
    """在独立模块对象里执行源码（dataclass 需要模块在 sys.modules 里注册）。"""
    import types

    mod = types.ModuleType(name)
    mod.__file__ = str(path or (ROOT / MAIN))
    sys.modules[name] = mod
    exec(compile(src, "<fixed_point>", "exec"), mod.__dict__)  # noqa: S102 - 只执行自家源码
    return mod.__dict__


def build(files: Dict[str, str], spec: Dict) -> str:
    for p in FRAME_FILES:
        if p not in files:
            raise ValueError(f"缺文件 {p}")
        ast.parse(files[p])

    others = [p for p in FRAME_FILES if p != MAIN]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: b64of(raw[p]) for p in others}
    code_text = code_text_of(payloads)

    def manifest_of(pad: int, sha: str, self_bytes: int = 0) -> str:
        # 主源的 bytes 用真实字节数（与 pad 无关），b64_len 就是槽宽。
        # 两者若混为一谈，模块侧 manifest() 与本工具会写出不同清单（曾因此恒 FAIL）。
        docs = []
        for p in FRAME_FILES:
            if p == MAIN:
                docs.append({"path": p, "bytes": self_bytes or len(str(pad)), "b64_len": pad,
                             "sha256": sha or "PENDING", "sig": signature(p, files[p])})
            else:
                docs.append({"path": p, "bytes": len(raw[p]), "b64_len": 4 * ((len(gz(raw[p])) + 2) // 3),
                             "sha256": hashlib.sha256(raw[p]).hexdigest(), "sig": signature(p, files[p])})
        return json.dumps({"spec": spec, "files": docs}, ensure_ascii=False, indent=2)

    # 起点用 0 而不是 100000：`manifest_of` 里 `self_bytes or len(str(pad))` 把 0 当
    # "还没有真实长度"，于是首轮按 pad 的**位数**起步、与磁盘模板态（若已固化过，
    # 清单里那个数就是真实字节数）一致。用 100000 起步会多走一轮，而 `solve` 的
    # 起点 `z + 4096` 会让第二轮解出不同的 pad（实测 48407 vs 48409）——
    # 盘上是 48437，两边都不改它，但**模块与工具必须从同一状态出发**才可能收敛到同一值。
    # 已固化磁盘上的主源时，用盘上的 `bytes` 字段当起点，直接对齐。
    m = re.search(r'"path":\s*"' + re.escape(MAIN) + r'",\s*"bytes":\s*(\d+)', files[MAIN])
    self_bytes = [int(m.group(1)) if m else 0]
    for _ in range(8):
        pad, final = solve(files[MAIN], manifest_of, code_text, self_bytes[0])
        real = len(final.encode("utf-8"))
        if real == self_bytes[0]:
            break
        self_bytes[0] = real
    else:
        raise RuntimeError(f"字节数未收敛（停在 {self_bytes[0]}）")
    # 复现判据：生成器读**磁盘上的主源**（`__file__`），所以除主源外的文件应与磁盘一致；
    # 主源那一项由 `solve` 内部的 `reproduces()` 保证（把候选当作磁盘内容再生成一遍）。
    ns = probe(final)  # 内存 exec：回落到 _PROBE_SOURCE，等价于"把 final 当磁盘内容"
    gen = ns["generate_framework"]()
    for p in FRAME_FILES:
        want = final if p == MAIN else files[p]
        if gen.get(p) != want:
            raise RuntimeError(f"复现失败: {p}")
    return final, pad


def main(argv: List[str]) -> int:
    global ROOT
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    files = {p: (ROOT / p).read_text(encoding="utf-8") for p in FRAME_FILES}
    if "--check" in argv:
        ns = probe(files[MAIN])
        ok, why = ns["verify_self_reproduction"]()
        print(("OK  " if ok else "FAIL ") + why)
        return 0 if ok else 1
    ns2: Dict = {"Any": object, "Dict": dict, "List": list}
    exec(re.search(r"^FRAMEWORK_SPEC: Dict\[str, Any\] = \{.*?^\}", files[MAIN], re.S | re.M).group(0), ns2)
    final, pad = build(files, ns2["FRAMEWORK_SPEC"])
    (ROOT / MAIN).write_text(final, encoding="utf-8")
    print(f"固化完成：主源 {len(final.encode('utf-8'))} 字节，pad={pad}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
