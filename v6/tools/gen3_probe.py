"""三代生成探针：只从磁盘读一次 M0，之后全部在内存里自举（不碰磁盘）。"""
import hashlib, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MAIN = "vea_v6/parts/bootstrap.py"
src = (ROOT / MAIN).read_text(encoding="utf-8")

def load(text, tag):
    mod = types.ModuleType(tag)
    sys.modules[tag] = mod
    exec(compile(text, f"<{tag}>", "exec"), mod.__dict__)
    mod.__dict__["_PROBE_SOURCE"]["current"] = text
    return mod.__dict__

cur = src
for gen in range(1, 5):
    m = load(cur, f"_g{gen}")
    files = m["generate_framework"]()
    nxt = files[MAIN]
    same = nxt == cur
    print(f"M{gen}: {len(nxt.encode())}B sha={hashlib.sha256(nxt.encode()).hexdigest()[:16]} "
          f"files={len(files)} {'== 上一代' if same else '!= 上一代'}")
    if not same:
        import difflib
        d = list(difflib.unified_diff(cur.splitlines(), nxt.splitlines(), "cur", "next", lineterm="", n=1))
        print("\n".join(d[:40]))
        break
    cur = nxt
else:
    print("四代逐字节一致 ✔")
