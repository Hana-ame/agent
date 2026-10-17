"""临时探查：assemble 是否真的把模型产出内嵌进主源（跑完可留作诊断用）。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vea_v6.parts.bootstrap import _probe, assemble  # noqa: E402

def nb(s: str) -> int:
    """**字节数，不是字符数**。中文注释会让两者差出好几千，
    把 `len(str)` 当字节数比是 `dag.py` 27234 vs 29372 那次误判的来源。"""
    return len(s.encode("utf-8"))


g = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "bootstrap_runs/real02/gen0")
files = {p.relative_to(g).as_posix(): p.read_text(encoding="utf-8") for p in g.rglob("*.py")}
print("输入：")
for k, v in sorted(files.items()):
    print(f"  {k:32s} {nb(v):7d}B")

out = assemble(files)
rebuilt = _probe(out["vea_v6/parts/bootstrap.py"])["generate_framework"]()
print("\nassemble 后从主源重建：")
same_all = True
for k in sorted(files):
    ok = rebuilt[k] == files[k]
    same_all &= ok
    print(f"  {k:32s} {nb(rebuilt[k]):7d}B  内嵌==输入: {ok}")
print("\n全部一致:", same_all)
