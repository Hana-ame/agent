"""对一次 meta_dag 运行做完整验收（可复现，不依赖人眼）。

用法：
    python tools/accept_run.py bootstrap_runs/meta/s15

判据（全部满足才 exit 0，任一不过 exit 1）：
 1. 三代齐全：gen1/gen2/gen3 目录存在，7 个框架文件一个不少；
 2. 双过：RESULT.json 里每一代 demo=True 且 bootstrap=True；
 3. 凭据：gens[1:-1]（生成者）各有 ORIGIN.json，loaded 指向自己的 vea_v6
    且 stable=True（gen3 是末代，不生成任何东西，不要求有）；
 4. 互不相同：相邻两代总字节数不同（逐字节相同 = 回退复印件，s13 实测）；
 5. 回档链：gen0 为链根（parent=null），此后每一代 parent 等于上一代
    combined_sha256，且 combined_sha256 与磁盘重算一致；
 6. 代际闸门：每一代 REPORT.json 的 provider_ok=True；
 7. 逐步快照：每一代 steps/ 下 8 个步骤快照齐全。

设计纪律（本轮教训）：
 - 每条判据都必须在真实损坏输入上验过会红（见 tests/test_vea_v6_all.py
   与提交信息里的变异测试记录），不接受"看起来对"的断言；
 - 只读被验收的 run，不写、不删、不改任何文件。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._meta_dag_spec import FRAME_FILES  # noqa: E402

STEP_NAMES = (
    "expand", "plan_module", "normalize", "audit_contract",
    "supplement", "assemble", "measure", "passthrough",
)


def combined_sha(files: Dict[str, str]) -> str:
    """与 oneshot_chain.py / meta_dag.py::write_head 同一算法。"""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(files[p].encode("utf-8")).digest())
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"判据": name, "结果": "PASS" if ok else "FAIL", "说明": detail})
        return ok

    def all_ok(self) -> bool:
        return all(r["结果"] == "PASS" for r in self.rows)

    def print(self) -> None:
        width = max(len(r["判据"].encode("gbk", errors="ignore")) for r in self.rows)
        for r in self.rows:
            mark = "✓" if r["结果"] == "PASS" else "✗"
            print(f"  [{mark}] {r['判据']:<{width}}  {r['说明']}")
        print(f"\n  合计 {len(self.rows)} 项："
              f"{sum(1 for r in self.rows if r['结果'] == 'PASS')} PASS / "
              f"{sum(1 for r in self.rows if r['结果'] == 'FAIL')} FAIL")


def gen_bytes(gdir: Path) -> int:
    return sum(len((gdir / f).read_text(encoding="utf-8").encode("utf-8"))
               for f in FRAME_FILES if (gdir / f).exists())


def accept(run: Path) -> bool:
    rep = Report()
    gens = [run / f"gen{i}" for i in (1, 2, 3)]

    # ---- 1. 三代齐全 -----------------------------------------------------
    missing: List[str] = []
    for g in gens:
        for f in FRAME_FILES:
            if not (g / f).exists():
                missing.append(f"{g.name}/{f}")
    rep.check("三代齐全（3×7 文件）", not missing,
              "全部在位" if not missing else f"缺 {len(missing)} 个：{missing[:3]}…")

    # ---- 2. RESULT.json 双过 --------------------------------------------
    result_file = run / "RESULT.json"
    if not result_file.exists():
        rep.check("RESULT.json 存在", False, "没有 RESULT.json")
        return _finish(rep)
    history = json.loads(result_file.read_text(encoding="utf-8"))
    bad = [h["gen"] for h in history
           if not (h.get("demo") and h.get("bootstrap"))]
    rep.check("自选任务 + 自举双过（每代）", not bad and len(history) == 3,
              f"{len(history)} 代记录，失败代：{bad or '无'}")

    # ---- 3. ORIGIN.json 凭据（除末代外的每一代）--------------------------
    # 🔴 切片语义：这里 gens = [gen1, gen2, gen3]（不含 gen0），生成者是
    # **除末代外的全部**，即 gens[:-1] = [gen1, gen2]。
    # 曾错写成 gens[1:-1]（那是 test_19 里含 gen0 的四代列表的惯例），
    # 在 gen1/gen2/gen3 上只切出 [gen2]，gen1 的凭据缺失根本不报警
    # （变异测试实测：移走 gen1/ORIGIN.json 全绿）。
    origin_bad: List[str] = []
    for g in gens[:-1]:  # gen1、gen2 是生成者；gen3 是末代，不生成任何东西
        oj = g / "ORIGIN.json"
        if not oj.exists():
            origin_bad.append(f"{g.name} 无 ORIGIN.json")
            continue
        data = json.loads(oj.read_text(encoding="utf-8"))
        expect = str((g / "vea_v6").resolve())
        if Path(data.get("loaded", "")).resolve() != Path(expect).resolve():
            origin_bad.append(f"{g.name} 的 loaded 不指向自己")
        if not data.get("stable"):
            origin_bad.append(f"{g.name} 的 stable=False")
    rep.check("ORIGIN.json 凭据（除末代外的每一代）", not origin_bad,
              "gen1/gen2 各自 loaded 指向自己且 stable" if not origin_bad
              else "；".join(origin_bad))

    # ---- 4. 相邻两代字节数互不相同 ---------------------------------------
    sizes = [gen_bytes(g) for g in gens]
    dup = [f"{gens[i - 1].name}/{gens[i].name}" for i in range(1, len(sizes))
           if sizes[i] == sizes[i - 1] and sizes[i] > 0]
    rep.check("相邻两代字节数互不相同", not dup,
              f"字节数 {sizes}" if not dup else f"逐字节相同的相邻代：{dup}")

    # ---- 5. HEAD.json 回档链 ---------------------------------------------
    chain_ok = True
    chain_detail = ""
    prev_sha: Any = None
    for g in [run / "gen0"] + gens:
        head_file = g / "HEAD.json"
        if not head_file.exists():
            chain_ok = False
            chain_detail = f"{g.name} 没有 HEAD.json"
            break
        head = json.loads(head_file.read_text(encoding="utf-8"))
        if prev_sha is not None and head.get("parent") != prev_sha:
            chain_ok = False
            chain_detail = f"{g.name} 的 parent 断链"
            break
        srcs = {f: (g / f).read_text(encoding="utf-8")
                for f in FRAME_FILES if (g / f).exists()}
        if head.get("combined_sha256") != combined_sha(srcs):
            chain_ok = False
            chain_detail = f"{g.name} 的 combined_sha256 与磁盘重算不一致"
            break
        if head.get("gen") == 0 and head.get("parent") is not None:
            chain_ok = False
            chain_detail = "gen0 链根的 parent 不为 null"
            break
        prev_sha = head["combined_sha256"]
    rep.check("HEAD.json 回档链（gen0 链根 + parent 不断 + sha 重算一致）",
              chain_ok, chain_detail or "gen0→gen1→gen2→gen3 四环全部吻合")

    # ---- 6. 代际闸门：每一代 provider_ok ---------------------------------
    prov_bad: List[str] = []
    for g in gens:
        rj = g / "REPORT.json"
        if not rj.exists():
            prov_bad.append(f"{g.name} 无 REPORT.json")
            continue
        data = json.loads(rj.read_text(encoding="utf-8"))
        if not data.get("provider_ok", True):
            prov_bad.append(f"{g.name}：{str(data.get('provider_why'))[:60]}")
    rep.check("代际闸门（provider_ok 每代为真）", not prov_bad,
              "三代都是真造出来的" if not prov_bad else "；".join(prov_bad))

    # ---- 7. 每代 8 步快照齐全 --------------------------------------------
    step_bad: List[str] = []
    for g in gens:
        have = {p.name for p in (g / "steps").glob("*")} if (g / "steps").exists() else set()
        want = {f"{i + 1:02d}_{n}" for i, n in enumerate(STEP_NAMES)}
        if not want <= have:
            step_bad.append(f"{g.name} 缺 {sorted(want - have)}")
    rep.check("逐步快照（每代 8 步）", not step_bad,
              "gen1/gen2/gen3 各 8 个快照齐全" if not step_bad else "；".join(step_bad))

    return _finish(rep)


def _finish(rep: Report) -> bool:
    rep.print()
    return rep.all_ok()


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("用法：python tools/accept_run.py <run 目录>")
        print("示例：python tools/accept_run.py bootstrap_runs/meta/s15")
        return 2
    run = Path(argv[1])
    if not run.is_dir():
        print(f"目录不存在：{run}")
        return 2
    print(f"验收 {run}")
    return 0 if accept(run) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
