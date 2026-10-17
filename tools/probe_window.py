"""现场取证：打印 `pad -> z(pad)` 分布，说明「固定点不一定存在」。

用法：
    python tools/probe_window.py [lo] [hi]

z(pad) = gzip(渲染结果) 的字节数。严格不动点要求 z(pad) == pad。
实测 z(pad) 在真解附近几乎是常数（pad 变长 1，gzip 体积往往只涨 0~1），
所以 `z(pad) == pad` 可能**跨过去而不命中**——此时窗口内 z 在 pad-1 与 pad+1 之间跳。
这正是"自指填充"必须允许 ±1 容差、不能硬求等号的原因。
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixpoint as F  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    files = {p: (root / p).read_text(encoding="utf-8") for p in F.FRAME_FILES}
    ns = {"Any": object, "Dict": dict, "List": list}
    exec(re.search(r"^FRAMEWORK_SPEC: Dict\[str, Any\] = \{.*?^\}", files[F.MAIN], re.S | re.M).group(0), ns)
    spec = ns["FRAMEWORK_SPEC"]
    others = [p for p in F.FRAME_FILES if p != F.MAIN]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: F.b64of(raw[p]) for p in others}
    code_text = F.code_text_of(payloads)

    def manifest_of(pad: int, sha: str) -> str:
        docs = []
        for p in F.FRAME_FILES:
            if p == F.MAIN:
                docs.append({"path": p, "bytes": pad, "b64_len": pad,
                             "sha256": sha or "PENDING", "sig": F.signature(p, files[p])})
            else:
                docs.append({"path": p, "bytes": len(raw[p]), "b64_len": len(payloads[p]),
                             "sha256": hashlib.sha256(raw[p]).hexdigest(), "sig": F.signature(p, files[p])})
        return json.dumps({"spec": spec, "files": docs}, ensure_ascii=False, indent=2)

    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 32120
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 32230
    print(f"扫描窗口 [{lo}, {hi}]")
    print(" pad   z(pad)   d=z-pad")
    hits = []
    for cand in range(lo, hi + 1):
        sha = hashlib.sha256(F.render(files[F.MAIN], manifest_of(cand, ""), code_text, pad=cand).encode()).hexdigest()
        text = F.render(files[F.MAIN], manifest_of(cand, sha), code_text, pad=cand)
        z = len(F.gz(text.encode("utf-8")))
        if z == cand:
            hits.append(cand)
        print(f"{cand:>5} {z:>8} {z - cand:>+8} {'<<< 严格不动点' if z == cand else ''}")
    print(f"\n严格不动点: {hits if hits else '无（区间内 z(pad) 恒不等于 pad）'}")


if __name__ == "__main__":
    main()
