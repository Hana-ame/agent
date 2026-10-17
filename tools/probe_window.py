"""现场取证：打印 z(pad) 在候选窗口内的分布，证明"直接迭代会在两点间打转"。"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixpoint as F  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
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

    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 31945
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 31980
    print(f"扫描窗口 [{lo}, {hi})")
    print("pad   z(pad)  z==pad?")
    for cand in range(lo, hi):
        sha = hashlib.sha256(F.render(files[F.MAIN], manifest_of(cand, ""), code_text, pad=cand).encode()).hexdigest()
        text = F.render(files[F.MAIN], manifest_of(cand, sha), code_text, pad=cand)
        z = len(F.gz(text.encode("utf-8")))
        print(f"{cand}  {z}  {'<<< 不动点' if z == cand else ''}")


if __name__ == "__main__":
    main()
