"""同一个 DAG 连跑三代：输入是同一段自然语言描述，工具槽位每轮换挂"上一代框架"。

对应形态（用户口径）：
    1. 做一个 DAG；
    2. 用这个 DAG 生成 gen_next 框架；
    3. 用**同一个 DAG**（只把 gen_next 挂成工具）生成 gen_next+1 框架。

DAG 里只有两条边：
    ["input"]    -- {工具: "gen"} --> ["framework"]
    ["framework"] -- 透传 --> ["result"]

`gen` 工具由 DAG 自己的 `tools` 槽位提供。第 1 轮挂的是"用 LLM 按自然语言描述写框架"
这个引导工具；第 2 轮起挂的是**上一代生成的框架本身**——所以第 2、3 代是框架在造框架。

用法：
    python tools/dag_selfgen.py --rounds 3
    python tools/dag_selfgen.py --check gen1     # 只检查某一代能不能跑
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "bootstrap_runs"

DESCRIPTION = """
我需要一个标签驱动的信息流框架，请按下面的自然语言描述实现它。

数据只有两种东西：条目（一条内容，带一组标签）和边（处理规则）。
一条边包含：需要哪些输入标签、处理函数、产出哪些输出标签、在第几层。
条目沿边流动：某条边的输入标签齐了，就把这些条目交给它的处理函数，
处理函数返回新内容，带上这条边的输出标签，变成新条目。

框架要能这样用：建一个空的图对象，往里加边；放一条初始条目；
反复调用"跑一轮"，每轮把能触发的边执行一遍；外部检查图上有没有出现
某个特定标签的条目，出现就取它的内容当最终结果。

还要能自动补齐：当下游边要吃的多个标签分别来自不同上游、一次凑不齐时，
用一个"输出标签等于所需标签"的中间条目顶上，不能卡死。

再把"生成框架自己"这件事也做成图上的东西：图上有一个工具槽位，
把某个框架挂上去，它就能按这段描述产出一份新框架。换掉挂上去的框架，
同一个图就能产出下一代。

输出要求：只输出 JSON，结构为
{"files": [{"path": "vea_v6/parts/dag.py", "source": "完整源码"}, ...]}
必须含这 4 个文件：
  vea_v6/__init__.py
  vea_v6/parts/__init__.py
  vea_v6/parts/dag.py
  vea_v6/parts/llm.py
dag.py 必须能导出 ContentItem / Edge / DAG / DEFAULT_PRESET_EDGE / SmartMockLLM，
DAG 要有 add_edge / send / run / has_result / get_result / tools / mount_tool。
llm.py 要有 OpenAICompatibleLLM。只用标准库，import 期不做 IO。
不要 markdown 围栏，不要解释。
""".strip()

FILES = (
    "vea_v6/__init__.py",
    "vea_v6/parts/__init__.py",
    "vea_v6/parts/dag.py",
    "vea_v6/parts/llm.py",
)


# ---------------------------------------------------------------- 引导工具
def make_bootstrap_tool(model_key: str, run_dir: Path):
    """第 1 轮挂的工具：用 LLM 按自然语言描述写出框架源码。"""
    from vea_v6.parts.dag import load_models_file
    from vea_v6.parts.llm import OpenAICompatibleLLM

    models = load_models_file()
    cfg = models[model_key]
    llm = OpenAICompatibleLLM()

    def gen(description: str, **_kw: Any) -> str:
        """工具只吐一个 JSON；内部分文件发请求。

        不能一次要 4 个文件：实测整份约 4 万字符，网关在 100s 处直接 524
        （Cloudflare 的响应头等待上限），跟模型速度无关。拆成 4 次每 10~20s 就回来。
        `dag.run()` 看到的仍然是**一次工具调用 → 一份完整框架**。
        """
        prompt_dir = run_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        out_files: Dict[str, str] = {}
        for path in FILES:
            n = len(list(prompt_dir.glob("[0-9][0-9].txt")))
            ask = (
                f"{description}\n\n"
                f"本次只写这一个文件：{path}\n"
                f"输出 JSON：{{\"path\": \"{path}\", \"source\": \"完整源码\"}}，不要围栏、不要解释。"
            )
            (prompt_dir / f"{n:02d}.txt").write_text(ask, encoding="utf-8", newline="")
            t0 = time.time()
            text = llm._post_sync(
                ask,
                model=cfg["model"],
                base_url=cfg["api"],
                api_key=cfg.get("key") or None,
                max_tokens=(cfg.get("params") or {}).get("max_tokens"),
                temperature=(cfg.get("params") or {}).get("temperature", 0.2),
                stream=False,
            )
            (prompt_dir / f"{n:02d}.out.txt").write_text(text, encoding="utf-8", newline="")
            src = _one_source(text)
            if not src:
                raise ValueError(f"模型没给出 {path} 的源码（见 {prompt_dir/f'{n:02d}.out.txt'}）")
            out_files[path] = src
            print(f"    {path:32s} {len(src.encode('utf-8')):6d}B  {time.time()-t0:.0f}s")
        return json.dumps(
            {"files": [{"path": p, "source": out_files[p]} for p in FILES]}, ensure_ascii=False
        )

    gen.__name__ = "bootstrap_gen"
    return gen


def _one_source(text: str) -> Optional[str]:
    """从单文件响应里取源码。"""
    t = text.strip()
    if t.startswith("```"):
        i = t.find("\n")
        j = t.rfind("```")
        t = t[i + 1 : j] if j > i else t
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return t if t.strip() else None
    if isinstance(data, dict) and isinstance(data.get("source"), str):
        return data["source"]
    return None


def _to_files_json(text: str, run_dir: Path) -> str:
    """把模型输出规整成 {"files":[...]} 的 JSON 字符串；顺便落盘。"""
    t = text.strip()
    if t.startswith("```"):
        i = t.find("\n")
        j = t.rfind("```")
        t = t[i + 1 : j] if j > i else t
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        (run_dir / "RAW.txt").write_text(text, encoding="utf-8", newline="")
        raise ValueError(f"模型没返回合法 JSON（原文已存 {run_dir/'RAW.txt'}）")
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("模型返回的 JSON 里没有 files")
    out: Dict[str, str] = {}
    for item in files:
        if isinstance(item, dict) and item.get("path") and isinstance(item.get("source"), str):
            out[str(item["path"])] = item["source"]
    missing = [p for p in FILES if p not in out]
    if missing:
        raise ValueError(f"缺文件 {missing}（模型给了 {sorted(out)}）")
    return json.dumps({"files": [{"path": p, "source": out[p]} for p in out]}, ensure_ascii=False)


# ---------------------------------------------------------------- 落地与校验
def write_files(payload: str, dest: Path) -> Dict[str, str]:
    data = json.loads(payload)
    files = {item["path"]: item["source"] for item in data["files"]}
    for p, s in files.items():
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(s, encoding="utf-8", newline="")
    return files


def check_runs(gen_dir: Path) -> tuple[bool, str]:
    """这一代能不能真的跑起来：import 它、建 DAG、加边、send、run、出 result。"""
    probe = gen_dir / "_probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(gen_dir)!r})\n"
        "import vea_v6\n"
        "from vea_v6 import DAG, SmartMockLLM\n"
        "d = DAG('t', llm=SmartMockLLM())\n"
        "d.add_edge(['a'], lambda s: s.upper(), ['result'], layer=1)\n"
        "d.send({'tags': ['a'], 'content': 'hello'})\n"
        "for _ in range(5):\n"
        "    if d.has_result():\n"
        "        break\n"
        "    d.run()\n"
        "assert d.get_result() == 'HELLO', repr(d.get_result())\n"
        "has_tools = hasattr(d, 'tools') and hasattr(d, 'mount_tool')\n"
        "print('OK tools=' + str(has_tools))\n",
        encoding="utf-8",
        newline="",
    )
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=180)
    (gen_dir / "_probe.log").write_text(p.stdout + p.stderr, encoding="utf-8", newline="")
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-4:]
        return False, "; ".join(tail)
    return True, p.stdout.strip()


def load_gen_as_tool(gen_dir: Path):
    """把某一代框架当成可调用工具：调用它就返回它生成的下一代文件。

    这里不 import 上一代来"生成"——上一代如果有 `generate(description)` 就直接用，
    没有就走它自己 `bootstrap`/`create_generation_dag` 那条路；两者都拿不到就报错，
    不静默退化成"再调一次 LLM"（那会让三代其实是同一个工具在跑）。
    """
    import importlib.util

    for name in ("vea_v6",):
        for mod in list(sys.modules):
            if mod == name or mod.startswith(name + "."):
                del sys.modules[mod]
    sys.path.insert(0, str(gen_dir.parent))
    spec = importlib.util.spec_from_file_location(
        f"gen_{gen_dir.name}", gen_dir / "vea_v6" / "__init__.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pkg_dir = gen_dir 
    print(f"    已载入 {gen_dir.name} 的框架（{pkg_dir}）")
    return mod


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="同一个 DAG 连跑 N 代")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model-key", default="laptop-6.8")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args(argv)

    from vea_v6 import DAG, SmartMockLLM

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    base = RUNS / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "DESCRIPTION.txt").write_text(DESCRIPTION, encoding="utf-8", newline="")

    # ==== 1. 做一个 DAG（全程只用这一个）====
    dag = DAG("self_gen_dag", llm=SmartMockLLM())
    dag.add_edge(["input"], {"工具": "gen"}, ["framework"], layer=1)
    dag.add_edge(["framework"], lambda s: s, ["result"], layer=2)
    print(f"run-id = {run_id}")
    print(f"DAG '{dag.name}' 已建立：{len(dag.edges)} 条边，工具槽位 {sorted(dag.tools)}\n")

    history: List[Dict[str, Any]] = []
    tool = make_bootstrap_tool(a.model_key, base / "gen1")

    for i in range(1, a.rounds + 1):
        gdir = base / f"gen{i}"
        gdir.mkdir(parents=True, exist_ok=True)
        # ==== 2/3. 用同一个 DAG，换挂工具，跑一次 ====
        dag.mount_tool("gen", tool)
        dag.send({"tags": ["input"], "content": DESCRIPTION})
        for _ in range(6):
            if dag.has_result():
                break
            dag.run()
        payload = dag.get_result()
        if not payload:
            print(f"[gen{i}] FAIL DAG 没产出 result")
            break

        files = write_files(payload, gdir)
        total = sum(len(v.encode("utf-8")) for v in files.values())
        ok, why = check_runs(gdir)
        print(f"[gen{i}] {total} 字节  可运行={'OK' if ok else 'FAIL'}  {why[:90]}")
        history.append({"gen": i, "bytes": total, "runs": ok, "why": why})
        (base / "RESULT.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
        )
        if not ok:
            break
        # 下一代：把这一代挂成工具（同一个 dag，不重建）
        tool = load_gen_as_tool(gdir)
        if not callable(tool) and not hasattr(tool, "__call__"):
            print(f"[gen{i}] 警告：这一代没有可调用入口，下一代仍用引导工具")

    passed = sum(1 for h in history if h["runs"])
    print(f"\n可运行代数：{passed}/{a.rounds}")
    return 0 if passed >= a.rounds else 1


if __name__ == "__main__":
    sys.exit(main())
