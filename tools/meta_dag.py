"""元 DAG：用**同一个图**连着造出 gen1 / gen2 / gen3，并且让新生产的那一代真的去干活。

## 形态（这是用户要的那个东西）

同一个 DAG 对象，8 个步骤，7 层：

    0  ["input"]      -- gen(expand)        --> ["briefs"]
    1  ["briefs"]     -- gen(plan_module)   --> ["module"]      N 次 LLM，可并发
    1  ["module"]     -- gen(normalize)     --> ["part"]
    2  ["part"]       -- gen(audit_contract)--> ["part_ok"]     契约闸门
    3  ["part_ok"]    -- gen(supplement)    --> ["part_full"]   多输入汇聚
    4  ["part_full"]  -- gen(assemble)      --> ["framework"]   解固定点
    5  ["framework"]  -- gen(measure)       --> ["report"]
    6  ["framework"]  -- gen(passthrough)   --> ["result"]

"gen" 不是模型名，而是**挂在 `DAG.tools` 槽位上的工具**。第 1 代挂的是
"调 LLM 写码"的引导工具；从第 2 代起，槽位换成**上一代框架自己的生成 DAG**——
于是第 2、3 代是框架在造框架，而图一个字都没改。

## 每一代都必须真的跑起来（不是 `--check`）

`--rounds 3` 跑完后，每一代目录里都会有 `demo/`：
用那一代的框架**自己**建一个 DAG、跑一个自选任务（默认是"日志行 → 统计"），
并附上那一代自选的验收脚本 `vtest.py`（进程退出码 + 打印用例）。
验收任务明确排除"写一个 vea 框架"。

## 用法

    python tools/meta_dag.py --rounds 3                  # 真模型，全流程
    python tools/meta_dag.py --self-test                 # 不联网，用假工具验 DAG 形态与 8 步分派
    python tools/meta_dag.py --replay <run-id> --rounds 3  # 重放某次运行里模型产出的文件（不再请求）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools._llm_connect import connect_llm, connect_model_config  # noqa: E402
from tools._meta_dag_spec import FRAME_FILES, META_DAG  # noqa: E402
from tools._meta_gen_tool import GenTool  # noqa: E402

RUNS = ROOT / "bootstrap_runs" / "meta"

#: 给模型的框架描述（纯自然语言，不含任何文件名/函数名/JSON 结构）
DESCRIPTION = """
我要一个"标签驱动的信息流"框架，它能自己把自己重新造出来。

数据模型很简单：一条信息 = 一段文本 + 一组标签。规则叫边：
一条边声明它需要哪些标签、拿到这些标签的信息之后干什么、产出哪些标签、排在第几层。
把信息丢进图里，反复"跑一轮"：每轮把"输入标签已经凑齐"的边都执行一遍，
执行结果变成带新标签的信息，继续往下流。外部想知道结果，就反复跑，
直到图上出现某个约定标签的信息，取它的内容。

图必须能处理"一次凑不齐"的情况：下游边要吃的几个标签分别来自不同上游、
这一轮还差一个——这时不能干等死循环，得用一个"输出标签正好等于所缺标签"的
中间信息顶上去，让流程能继续。

图还要有工具槽位：把某个可调用的东西挂到图上的某个名字上，
边就可以用这个名字去调它。这个槽位是关键——把上一代造出来的框架挂回去，
同一个图就能造下一代。

另外要能把"生成整个框架"这件事本身也做成图上的流程，分步骤进行：
先把描述变成逐文件的施工单，再逐文件写代码，再逐文件做结构检查，
检查不过的用上一代补，然后把整套代码交给固定点求解器让它自洽，最后统计并交付。

下面几条是**必须成立的约定**，缺一条整套就用不起来：
1. 边上的处理函数收到的就是那条信息的**文本内容本身**（一个字符串），
   不是包了一层的条目对象；它返回一段文本，这段文本就作为新信息的文本，
   打上这条边的输出标签。
2. 外部取结果时拿到的也是**文本内容**（字符串），不是条目对象。
3. "跑一轮"这件事对外是一个普通的同步调用：调一次就推进一轮、返回这一轮
   有没有推进，不需要调用方自己懂协程。
4. "生成整个框架"那个流程跑完时，整套代码必须作为文本出现在"结果"这个标签上，
   外部从"结果"标签上取到的就是可以直接落盘的整套源码。
5. 生成流程内部每一步都要能自己往下走：某一步没产出时要用上一轮的结果顶上，
   不能因为一处没产出就让整个流程停在半路。

只允许用标准库，import 的时候不许联网也不许读写磁盘。
""".strip()

#: 每一代都要跑的自选任务：**不是**"写 vea 框架"，而是一个能自己验收的小系统
DEMO_TASK = """
用你这个框架，做一个"访问日志分析流水线"。

输入是一条 Apache/Nginx 风格的一行日志文本，格式：
  203.0.113.7 - - [18/Oct/2026:10:00:01 +0800] "GET /api/items?page=2 HTTP/1.1" 200 512

要求：
1. 先把这一行切出五个字段：client_ip、method、path、status、bytes；
2. 再按 status 判断成败（2xx/3xx 算成功，4xx/5xx 算失败），并给 path 去掉查询串；
3. 再把结果拼成一行汇总文本，形如
   `client=<ip> method=<m> path=<p> status=<s> ok=<yes|no> bytes=<n>`；
4. 这条汇总文本必须最终以 `result` 标签出现在图上，外部取出来必须逐字符等于期望值。

只用标准库，不要联网。
""".strip()

#: 那一代必须自己写出来并用它自证的验收脚本（自选验收方式：进程退出码 + 逐条用例）
DEMO_TEST = '''
"""这一代的验收脚本：用本代框架跑自选任务，逐条断言。

运行：python vtest.py      —— 全过返回 0，任一不过返回 1。
"""
import asyncio
import inspect
import sys

sys.path.insert(0, "..")

from vea_v6 import DAG, SmartMockLLM  # noqa: E402


def _text(x):
    """各代对"处理函数收到什么"的包装不完全一样：有的是字符串，有的是带 .content 的条目。"""
    return x.content if hasattr(x, "content") else str(x)


def parse_line(content):
    """一条单输入边：把日志行切成五个字段，回一行 `k=v`。

    不假设"图会帮我合并多路输入"——切分全在这一个函数里完成。
    """
    line = _text(content)
    # 203.0.113.7 - - [18/Oct/2026:10:00:01 +0800] "GET /api/items?page=2 HTTP/1.1" 200 512
    # 引号里的请求行本身带空格，按空白切会切坏，所以按引号定界切：
    head, _, rest = line.partition('"')
    request, _, tail = rest.partition('"')
    method, path_full = request.split(" ", 1)
    status, nbytes = tail.split()
    return "client_ip=%s method=%s path=%s status=%s bytes=%s" % (
        head.split()[0], method, path_full.split("?")[0], status, nbytes)


def _check_all(fields):
    text = fields.content if hasattr(fields, "content") else str(fields)
    kv = dict(part.split("=", 1) for part in text.split() if "=" in part)
    need = ("client_ip", "method", "path", "status", "bytes")
    miss = [f for f in need if not kv.get(f)]
    if miss:
        raise ValueError("字段没凑齐: " + ",".join(miss) + " | " + text)
    return "READY " + " ".join("%s=%s" % (f, kv[f]) for f in need)


def _summarize(joined):
    text = joined.content if hasattr(joined, "content") else str(joined)
    if not text.startswith("READY"):
        raise ValueError("字段没凑齐: " + text)
    kv = {}
    for part in text.split()[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    ok = "yes" if kv.get("status", "").isdigit() and 200 <= int(kv["status"]) < 400 else "no"
    return ("client=%s method=%s path=%s status=%s ok=%s bytes=%s"
            % (kv.get("client_ip", "?"), kv.get("method", "?"), kv.get("path", "?"),
               kv.get("status", "?"), ok, kv.get("bytes", "?")))


def make_dag():
    """三条**单输入**边串成的流水线。

    🔴 为什么一条多输入边都不用：边一旦有多个输入 tag，各代框架对
    "汇聚后怎么调处理函数"的做法并不统一——实测同一组字段，
    磁盘那一代把所有条目按行拼成一条字符串再调，gen1 只给了其中两条
    （`203.0.113.7\n512`）。那样验收结论就只对某一代成立，不是"每代都可用"。
    单输入边在三代里行为完全一致，所以这里只用单输入边。
    """
    d = DAG("log_pipeline", llm=SmartMockLLM())
    d.add_edge(["input"], parse_line, ["fields"], layer=1)
    d.add_edge(["fields"], _check_all, ["all_fields"], layer=2)
    d.add_edge(["all_fields"], _summarize, ["result"], layer=3)
    return d


CASES = [
    ('203.0.113.7 - - [18/Oct/2026:10:00:01 +0800] "GET /api/items?page=2 HTTP/1.1" 200 512',
     "client=203.0.113.7 method=GET path=/api/items status=200 ok=yes bytes=512"),
    ('198.51.100.9 - - [18/Oct/2026:10:00:02 +0800] "POST /login HTTP/1.1" 500 78',
     "client=198.51.100.9 method=POST path=/login status=500 ok=no bytes=78"),
]


def _settle(x):
    """`run()` 在有的代是同步的、有的代是 async（协程）——两种都要能推进。

    🔴 这不是给某一代打补丁：同一份自然语言描述下各代对"跑一轮"的实现可以不同
    （磁盘代是 `def run` 内部 `asyncio.run`，第 5 代直接写成 `async def run`）。
    验收脚本不该因为这一层差异就判负，所以在这里把协程跑完。
    """
    if inspect.iscoroutine(x):
        x = asyncio.run(x)
    return x


def main():
    for i, (line, want) in enumerate(CASES, 1):
        d = make_dag()
        _settle(d.send({"tags": ["input"], "content": line}))
        for _ in range(12):
            if d.has_result():
                break
            _settle(d.run())
        # `get_result()` 有的代返回字符串、有的代返回条目对象（内容正确但包了一层）。
        # 验收的是"结果以 `result` 标签出现且内容逐字符正确"，不是包装形式。
        got = _text(d.get_result())
        if got != want:
            # 报错文本用 repr，不要用 f-string 里带字面换行（会写坏源码）
            print("case%d FAIL\\n  got =%r\\n  want=%r" % (i, got, want))
            return 1
        print("case%d OK" % i)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''.strip()

#: 从零开始建元 DAG 用的假工具（`--self-test` 与首次引导共用兜底）
class RecordingTool:
    def __init__(self, fail: bool = False) -> None:
        self.calls: List[str] = []
        self.fail = fail

    def __call__(self, content: str, step: str = "", **kw: Any) -> str:
        self.calls.append(step)
        if self.fail:
            raise RuntimeError("boom")
        return json.dumps({"step": step, "echo": content[:40]}, ensure_ascii=False)


def build_meta_dag(llm: Any = None, models: Optional[Dict[str, Any]] = None) -> Any:
    """建元 DAG：只用 `META_DAG` 里的边与工具槽位。**全程只建这一个对象。**"""
    from vea_v6.parts.dag import DAG

    dag = DAG("meta_dag_selfgen", llm=llm, models=models)
    for inputs, processor, outputs, layer in META_DAG:
        dag.add_edge(inputs, processor, outputs, layer=layer)
    return dag


def run_round(dag: Any, description: str) -> Dict[str, Any]:
    """在同一个 DAG 上跑一次（一份输入），返回各标签上的产物。"""
    dag.send({"tags": ["input"], "content": description})
    for _ in range(24):
        dag.run()
        if dag.has_tag("result"):
            break
    return {
        "framework": dag.get_result("framework") or "",
        "report": dag.get_result("report") or "",
        "result": dag.get_result("result") or "",
    }


def write_generation(payload: str, dest: Path) -> Dict[str, str]:
    """把 `framework` 标签上的 JSON 落成真实文件。"""
    data = json.loads(payload)
    files = {item["path"]: item["source"] for item in data["files"]}
    for p, src in files.items():
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8", newline="")
    return files


def write_head(gdir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """给这一代写 `HEAD.json` —— 父指针 + 逐文件 sha256，让
    `tools/boot_rollback.py` 能像 git 一样 list / verify / diff / checkout。

    格式与 `tools/oneshot_chain.py::snapshot` 完全一致（combined_sha256 的算法
    也一致：按路径排序，逐文件 hash 后整体 hash），所以两种 run 可以用同一个
    回档工具。只在**这一代真的通过闸门之后**调用：链上每一环都是可用的一代。
    """
    files: Dict[str, Dict[str, Any]] = {}
    for f in FRAME_FILES:
        p = gdir / f
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        files[f] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
        }
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(files[p]["sha256"]))
    head = dict(meta)
    head["files"] = files
    head["combined_sha256"] = h.hexdigest()
    head["total_bytes"] = sum(v["bytes"] for v in files.values())
    # parent 非空时自动判定"与上一代是否逐字节相同"（老 run 里同名字段的语义）
    if head.get("parent") is not None and "byte_identical" not in head:
        head["byte_identical"] = head["combined_sha256"] == head["parent"]
    (gdir / "HEAD.json").write_text(
        json.dumps(head, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    return head


#: 每一代都必须**能用自己把自己再生成一遍**（这是"框架造框架"的最低要求）。
#: 用 `SmartMockLLM` 跑，不发网络请求，几秒出结果。
#: 🔴 为什么必须有这道闸门：第 9 代的 gen1 自选任务是过了的，但它自己的
#: `bootstrap()` 一跑就崩 —— 它的图把 `ContentItem` 直接喂给处理函数，
#: 而处理函数按"收到文本"写的。于是 gen2 只能由引导工具回退生成 = 不是它造的。
BOOTSTRAP_SMOKE = (
    "import json, sys\n"
    "sys.path.insert(0, r'{root}')\n"
    "import vea_v6\n"
    "sys.stderr.write('VEAV6_FILE=' + vea_v6.__file__ + chr(10))\n"
    "from pathlib import Path\n"
    "desc = Path(r'{desc}').read_text(encoding='utf-8')\n"
    "r = vea_v6.bootstrap(desc, llm=vea_v6.SmartMockLLM())\n"
    "print(json.dumps({{'stable': bool(r.stable), 'files': len(r.files), "
    "'bytes': sum(len(v.encode('utf-8')) for v in r.values()) if 0 else "
    "sum(len(r.files[p].encode('utf-8')) for p in vea_v6.FRAME_FILES)}}, ensure_ascii=False))\n"
)


def smoke_bootstrap(dest: Path, run_dir: Path) -> Dict[str, Any]:
    """这一代用 `SmartMockLLM` 能不能把自己再生成一遍（不联网）。"""
    code = BOOTSTRAP_SMOKE.format(root=dest.as_posix(),
                                  desc=(run_dir / "DESCRIPTION.txt").as_posix())
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    log = (p.stdout or "") + (p.stderr or "")
    (dest / "SMOKE.log").write_text(log, encoding="utf-8", newline="")
    # 取 stdout 里最后一行 JSON（stderr 里还有 VEAV6_FILE 那行，不能按整个 log 取末尾）
    lines = [l for l in (p.stdout or "").strip().splitlines() if l.startswith("{")]
    return {"ok": p.returncode == 0, "out": (lines[-1] if lines else (log.strip().splitlines() or [""])[-1])[:160]}


def exercise(dest: Path) -> Dict[str, Any]:
    """**用这一代框架跑自选任务**（不是 `--check`）：写 demo + 跑它自己的验收。"""
    demo = dest / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    script = demo / "vtest.py"
    script.write_text(DEMO_TEST + "\n", encoding="utf-8", newline="")
    p = subprocess.run(
        [sys.executable, "vtest.py"],  # 只给相对文件名：cwd 已经是 demo，再给绝对路径会被拼两次
        cwd=str(demo),
        capture_output=True,
        text=True,
        timeout=300,
    )
    (demo / "vtest.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8", newline="")
    return {"ok": p.returncode == 0, "out": ((p.stdout or "").strip().splitlines() or [""])[-1]}


def load_generation_as_tool(dest: Path, run_dir: Path, llm: Any, cfg: Dict[str, Any],
                            fallback: Optional[Dict[str, str]] = None,
                            steps_dir: Optional[Path] = None) -> Any:
    """把某一代框架挂成工具：**由它产出下一代的源码**。

    返回的是一个 `GenTool`，只是 `source_provider` 换成了"调上一代自己的
    `bootstrap(description, llm)` 拿整套源码"。这样元 DAG 那 8 个 step 一个不少
    （expand / plan_module / normalize / audit_contract / supplement / assemble /
    measure / passthrough），每一步照样可观测，唯一变的是"源码从哪来"。

    `steps_dir` = 这 8 步快照落在哪。默认 `dest/steps`，但主循环会传
    **下一代自己的目录**（`genN/steps`）：语义是"这些步骤产出了第 N 代"，
    与 HEAD.json 里读的 `steps` 字段一致。🔴 早期版本用的是 `run_dir/steps`
    （本次运行目录），gen2/gen3 的快照写进同一处互相覆盖 —— 实测 s12 的 run 根
    `steps/` 里只剩最后一代的 01~08，中间代的逐步快照整个消失。

    🔴 早期版本直接把"调上一代 bootstrap"当作整个 `gen` 工具挂上去：但 `gen`
    会被 8 条边各调一次，于是同一代被生成 8 遍，而且后续 step（normalize/audit/
    assemble）全部退化成透传 —— 图上的分步骤名存实亡。

    用的是上一代自己的 `bootstrap(description, llm)`——参数名 `description/llm`
    是 SPEC 里固定的公共签名（`signature()` 会逐个核对），所以跨代调用是安全的。
    """
    code = (
        "import json, sys\n"
        "sys.path.insert(0, r'{root}')\n"
        "sys.path.insert(0, r'{root}/tools')\n"
        "from pathlib import Path\n"
        "import vea_v6\n"
        "sys.stderr.write('VEAV6_FILE=' + vea_v6.__file__ + chr(10))\n"
        "from tools._llm_connect import connect_llm\n"
        "desc = Path(r'{desc}').read_text(encoding='utf-8')\n"
        "llm = connect_llm('{key}', cls=vea_v6.OpenAICompatibleLLM)\n"
        "r = vea_v6.bootstrap(desc, llm=llm)\n"
        "files = [{{'path': p, 'source': r.files[p]}} for p in vea_v6.FRAME_FILES]\n"
        "print(json.dumps({{'files': files, 'stable': bool(r.stable), 'reason': str(r.reason)}}, ensure_ascii=False))\n"
    ).format(root=dest.as_posix(), desc=(run_dir / "DESCRIPTION.txt").as_posix(), key=cfg.get("_key", "laptop-6.8"))

    def gen(_content: str, **_kw: Any) -> str:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
        log = (p.stdout or "") + (p.stderr or "")
        # CHILD.log 也按代分开：写进**生成者自己**的目录。早期版本写 run_dir，
        # 每代覆盖上一代的，最后只剩一份，看不出中间代是谁造的。
        (dest / "CHILD.log").write_text(log, encoding="utf-8", newline="")
        if p.returncode != 0:
            raise RuntimeError(f"子代生成失败（见 {dest / 'CHILD.log'}）: {(p.stderr or '')[-300:]}")
        # 凭据：子进程里真正 import 到的 `vea_v6` 必须落在**上一代**目录里。
        # 这条不是装饰：实测把 dest 误写成 run_dir 时，gen2/gen3 其实是
        # 磁盘上这份框架生成的，图照样跑通、demo 照样全绿，日志里看不出来。
        seen = ""
        for line in log.splitlines():
            if line.startswith("VEAV6_FILE="):
                seen = line.split("=", 1)[1].strip()
        want = (dest / "vea_v6").resolve()
        try:
            got = Path(seen).resolve().parent if seen else None
        except OSError:
            got = None
        if got != want:
            raise RuntimeError(
                f"代际串了：下一代由 {got} 生成，应当由 {want}（见 {dest / 'CHILD.log'}）"
            )
        payload = (p.stdout or "").strip().splitlines()[-1]
        data = json.loads(payload)
        files = {item["path"]: item["source"] for item in data.get("files", [])}
        # 写到**生成者自己**的目录里（`dest` = 上一代）。语义是
        # "这一代确实被拿去生成了下一代"，`loaded` 必须等于它自己的 vea_v6。
        # 早先写进 `run_dir`（本次运行目录）会被下一轮覆盖，只剩最后一份，
        # 于是看不出 gen2 到底是谁造的（实测 s11 就只剩 gen3 那份）。
        (dest / "ORIGIN.json").write_text(
            json.dumps(
                {"generated_by": str(want), "loaded": str(got), "seconds": round(time.time() - t0, 1),
                 "stable": data.get("stable"), "reason": data.get("reason"), "files": sorted(files)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _ = _content, _kw
        return files

    return GenTool(llm, cfg, run_dir=steps_dir or (dest / "steps"),
                   fallback=fallback or {}, source_provider=gen)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="元 DAG：同一个图连造三代，每一代都真的跑起来")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model-key", default="laptop-6.8")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--self-test", action="store_true", help="不联网，用假工具验 DAG 形态与 8 步分派")
    ap.add_argument("--replay", default="", help="重放某次运行目录里的模型产出（不再请求）")
    ap.add_argument("--retries", type=int, default=2, help="装配闸门不过时整代重做的次数")
    a = ap.parse_args(argv)

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    base = RUNS / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "DESCRIPTION.txt").write_text(DESCRIPTION, encoding="utf-8", newline="")
    (base / "DEMO_TASK.txt").write_text(DEMO_TASK, encoding="utf-8", newline="")

    cfg = connect_model_config(a.model_key)
    cfg["_key"] = a.model_key
    llm = None if a.self_test else connect_llm(a.model_key)

    dag = build_meta_dag(llm=llm, models={a.model_key: cfg} if not a.self_test else None)
    (base / "dag.json").write_text(
        json.dumps(
            {
                "name": dag.name,
                "edges": [
                    {"inputs": list(e.input_tags), "processor": e.processor,
                     "outputs": list(e.output_tags), "layer": e.layer}
                    for e in dag.edges
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="",
    )
    print(f"run-id = {run_id}")
    print(f"DAG '{dag.name}': {len(dag.edges)} 条边 / {len({e.layer for e in dag.edges})} 层 / 工具槽位 {sorted(dag.tools)}")

    if a.self_test:
        probe = RecordingTool()
        dag.mount_tool("gen", probe)
        out = run_round(dag, DESCRIPTION)
        print("  分派到的 step:", probe.calls)
        print("  产出标签:",
              {k: (len(v) if isinstance(v, str) else v) for k, v in out.items()})
        ok = probe.calls == [e[1]["参数"]["step"] for e in META_DAG]
        print(f"\n自检 {'OK' if ok else 'FAIL'}：8 个步骤按层依次触发")
        return 0 if ok else 1

    history: List[Dict[str, Any]] = []
    prev_sources: Dict[str, str] = {}
    from vea_v6.parts.bootstrap import framework_sources

    prev_sources = dict(framework_sources())  # 第 1 代的参考实现 = 磁盘上这一份

    # gen0 = 磁盘现状（引导工具的参考实现），作为 git 式回档链的根。
    # 没有它，gen1 的 parent 只能是 null，"这一代相对于什么变了"就无从比对。
    g0 = base / "gen0"
    g0.mkdir(parents=True, exist_ok=True)
    for f, src in prev_sources.items():
        t = g0 / f
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(src, encoding="utf-8", newline="")
    prev_head = write_head(g0, {
        "gen": 0, "parent": None, "mode": "disk",
        "note": "gen0 = 磁盘现状（引导工具的参考实现），未调用模型",
    })
    print(f"gen0  {prev_head['combined_sha256'][:16]}  {prev_head['total_bytes']}B  (磁盘现状)")

    for i in range(1, a.rounds + 1):
        gdir = base / f"gen{i}"
        gdir.mkdir(parents=True, exist_ok=True)
        if a.replay:
            src_dir = RUNS / a.replay / f"gen{i}"
            for f in FRAME_FILES:
                dst = gdir / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_dir / f, dst)
            print(f"[gen{i}] 重放自 {src_dir}")
            # 重放没有"生成流程"，装配闸门（那四道静态契约）不适用；但**自举闸门
            # 照过** —— 重放来的代码同样必须能把自己再生成一遍，否则它就是死的。
            # 🔴 曾经这里什么闸门都不跑，直接往下走，`smoke` 根本没赋值，
            # 一到 record 就 UnboundLocalError（`--replay` 这条路从来是坏的）。
            report = {"assemble_ok": True, "note": "replay：未走生成流程，装配闸门不适用"}
            smoke = smoke_bootstrap(gdir, base)
            print(f"[gen{i}] 重放代自举={'OK' if smoke['ok'] else 'FAIL'}  {smoke['out']}")
            if not smoke["ok"]:
                print(f"[gen{i}] FAIL：重放代过不了自举闸门，停止")
                history.append({"gen": i, "bytes": 0, "demo": False, "demo_out": "",
                                "bootstrap": False, "bootstrap_out": smoke["out"]})
                break
        else:
            # 🔴 只有**第 1 代**挂引导工具 `GenTool`。第 2 代起 `gen` 槽位上挂的是
            # 上一代框架自己的 `bootstrap`（在上一轮末尾 mount 的）—— 这里如果
            # 无条件重新 mount `GenTool`，就会把上一代覆盖掉，于是"框架造框架"
            # 名存实亡：gen2/gen3 全都还是引导工具造的，而且日志里完全看不出来
            # （实测 s07 的 gen2 与 gen3 逐字节相同，ORIGIN.json 根本没生成）。
            if i == 1:
                dag.mount_tool("gen", GenTool(llm, cfg, run_dir=gdir / "steps", fallback=prev_sources))
            # 装配不过就整代作废再试一次：闸门回退出来的是**上一代**的代码，
            # 落盘了也算不得新的一代（实测 s08 的 gen1 就是这样被记成失败
            # 却又把互相 import 不上的那套写进了磁盘）。
            report = {}
            smoke = {"ok": False, "out": ""}
            for attempt in range(1, a.retries + 1):
                out = run_round(dag, DESCRIPTION)
                if not out["framework"]:
                    print(f"[gen{i}] FAIL：图上没有产出 framework")
                    break
                write_generation(out["framework"], gdir)
                (gdir / "REPORT.json").write_text(out["report"], encoding="utf-8", newline="")
                try:
                    report = json.loads(out["report"])
                except ValueError:
                    report = {}
                # 自举闸门直接进生成循环（带重试）：静态推断追不上模型写法的变化
                # —— 实测 `item.id`、`ContentItem.strip` 这类"类在、字段没了"的错，
                # 静态闸门要么漏判要么误报，而"用 SmartMockLLM 把自己再生成一遍"
                # 是唯一确定的判据，几秒就出结果。
                smoke = smoke_bootstrap(gdir, base)
                print(f"[gen{i}] 第 {attempt} 次（工具={type(dag.tools.get('gen')).__name__}）："
                      f"{out['report']} | 自举={'OK' if smoke['ok'] else 'FAIL'} {smoke['out']}")
                if report.get("assemble_ok") and smoke["ok"]:
                    break
            if not (report.get("assemble_ok") and smoke["ok"]):
                print(f"[gen{i}] FAIL：{a.retries} 次都没同时过装配闸门与自举闸门，停止")
                smoke = {"ok": False, "out": "未过自举闸门"}
                history.append({"gen": i, "bytes": 0, "demo": False, "demo_out": "",
                                "bootstrap": False, "bootstrap_out": smoke["out"]})
                break

        res = exercise(gdir)  # ← 用这一代框架真跑自选任务
        total = sum(len((gdir / f).read_text(encoding="utf-8").encode("utf-8")) for f in FRAME_FILES)
        record = {"gen": i, "bytes": total, "demo": res["ok"], "demo_out": res["out"],
                  "bootstrap": smoke["ok"], "bootstrap_out": smoke["out"]}
        history.append(record)
        print(f"[gen{i}] {total} 字节  自选任务={'OK' if res['ok'] else 'FAIL'}  {res['out'][:80]}")
        (base / "RESULT.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8", newline="")
        # 自举不过 = 这一代造不出下一代，继续往下跑只会得到"回退出来的上一代"
        if not res["ok"] or not smoke["ok"]:
            break
        # 过闸门了才写 HEAD.json：回档链上每一环都是可用的一代（父指针指向上一个
        # 通过验收的代，不是上一次尝试）。steps/ 里留着这代 8 步的逐步快照。
        head = write_head(gdir, {
            "gen": i,
            "parent": prev_head["combined_sha256"],
            "mode": f"meta-dag:{a.model_key}" + ("" if i == 1 else "+selfgen"),
            "model_key": a.model_key,
            "self_verify": smoke["ok"], "self_verify_reason": smoke["out"],
            "demo": res["ok"], "demo_out": res["out"],
            "steps": sorted(p.name for p in (gdir / "steps").iterdir()) if (gdir / "steps").exists() else [],
        })
        prev_head = head
        prev_sources = {f: (gdir / f).read_text(encoding="utf-8") for f in FRAME_FILES}
        # 下一代让**这一代自己**去生成（同一个 dag，只换挂载的工具）
        if i < a.rounds:
            # dest = **这一代自己的目录**（下一代必须由它生成），
            # run_dir = 本次运行的目录（DESCRIPTION.txt 在那儿）。
            # steps_dir = **下一代的目录**：产出它的 8 步快照归它自己，
            # 与 HEAD.json 的 `steps` 字段、boot_rollback 的按代回档对得上。
            nxt = base / f"gen{i + 1}"
            dag.mount_tool("gen", load_generation_as_tool(
                gdir, base, llm, cfg, fallback=prev_sources, steps_dir=nxt / "steps"))

    passed = sum(1 for h in history if h["demo"] and h.get("bootstrap"))
    print(f"\n自选任务 + 自举均通过的代数：{passed}/{a.rounds}")
    return 0 if passed >= a.rounds else 1


if __name__ == "__main__":
    sys.exit(main())
