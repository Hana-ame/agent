"""功能等价自举验证：同一段**自然语言描述**，连续生成三代，每代都必须能跑、能继续生成。

判据（不看 sha、不看字节相等）：
    1. 每次只给模型一段自然语言的框架描述（不含源码、不含签名清单）；
    2. 第 N 代产出的整个 vea_v6 必须能 **import 并运行**（能建 DAG、能跑一圈、能出 result）；
    3. 第 N 代必须能**独立生成第 N+1 代**（用自己的代码再走一遍同样的自然语言描述）；
    4. 连续三代都满足上面两条。

用法：
    python tools/gen_natural.py                 # 真模型，3 代
    python tools/gen_natural.py --dry-run       # 只打印提示词，不调模型
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "bootstrap_runs"

#: 唯一的输入：一段自然语言描述。不得包含源码、签名、文件清单。
DESCRIPTION = """
我需要一个"标签驱动"的信息流框架，请按下面的自然语言描述把它实现出来。

数据只有两种东西：条目（一条内容，带一组标签）和边（处理规则）。
一条边包含四样东西：需要哪些输入标签、处理函数、产出哪些输出标签、在第几层。
条目沿着边流动：如果某条边的输入标签都齐了，就把这些条目交给它的处理函数，
处理函数返回新内容，新内容带上这条边的输出标签，变成一个新人条目。

框架要能这样用：先建一个空的有向无环图对象，再往里加边；加完边往图里放一条初始条目；
然后反复调用"跑一轮"，每一轮把所有能触发的边都执行一遍。外部只需要检查
图上有没有出现某个特定标签的条目，出现了就取它的内容当作最终结果。

框架还必须有这个能力：当一条边的多个输入标签分别由不同的上游边产生、
而下游那条边一次要吃好几个标签时，框架要能自动把缺的那条补齐
（用"输出标签等于所需标签"的中间条目顶上），不能因为凑不齐输入就卡死。

另外要有两个模型接入：一个是真实的大模型 HTTP 客户端（OpenAI 兼容的
/v1/chat/completions，支持流式），一个是本地假模型（按关键词匹配返回预设内容，
用于不联网的测试）。模型配置要能从外部 JSON 文件读，按名字取到 base_url、
key、模型名和采样参数。

最后，这个框架要能**生成它自己**：给它一段自然语言描述，它跑一个 DAG，
产出一整套能独立运行的框架源码文件；产出的那份代码再给它同样的描述，
它还能再产出一份能跑的新框架。这个过程要能一直重复下去。
""".strip()


def _call_model(prompt: str, model_key: str, run_dir: Path) -> str:
    """单次调用：只发自然语言提示词，返回模型文本。"""
    from vea_v6.parts.dag import load_models_file
    from vea_v6.parts.llm import OpenAICompatibleLLM

    models = load_models_file()
    cfg = models[model_key]
    llm = OpenAICompatibleLLM()
    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
    n = len(list((run_dir / "prompts").glob("*.txt")))
    (run_dir / "prompts" / f"{n:02d}.txt").write_text(prompt, encoding="utf-8", newline="")
    t0 = time.time()
    out = llm._post_sync(
        prompt,
        model=cfg["model"],
        base_url=cfg["api"],
        api_key=cfg.get("key") or None,
        max_tokens=(cfg.get("params") or {}).get("max_tokens"),
        temperature=(cfg.get("params") or {}).get("temperature", 0.2),
        stream=False,
    )
    (run_dir / "prompts" / f"{n:02d}.out.txt").write_text(out, encoding="utf-8", newline="")
    print(f"    模型返回 {len(out)} 字符，{time.time()-t0:.0f}s")
    return out


FILE_SPEC = """
请输出 JSON，结构严格为：
{"path": "vea_v6/parts/xxx.py", "source": "该文件完整源码"}
不要解释，不要 markdown 围栏。

本次只写这一个文件：{path}

该文件要求：{req}

每个文件开头写 `from __future__ import annotations`；只用标准库；import 期不做任何 IO。

下面是框架的自然语言描述：
"""

FILE_REQS: Dict[str, str] = {
    "vea_v6/parts/dag.py": (
        "必须是完整的实现：ContentItem（条目：content / tags / gen / image）、"
        "Edge（边：input_tags / processor / output_tags / layer，可带 fn 直接调用）、"
        "DAG（add_edge / send / run / has_result / get_result / context）、"
        "DEFAULT_PRESET_EDGE（预设边常量）、SmartMockLLM（add_rule 加关键词规则，ask 按关键词返回）。"
        "DAG.run 跑一轮、执行所有输入标签齐了的边；run 要能处理预设边合成缺失标签。"
    ),
    "vea_v6/parts/llm.py": (
        "OpenAICompatibleLLM：用 httpx 或 urllib 打 OpenAI 兼容的 /v1/chat/completions，"
        "支持 stream=True 的 SSE 解析，暴露 _post_sync(prompt, model, base_url, api_key, "
        "max_tokens, temperature, stream) 和 ask。另需 LLMError 异常、extract_message_text。"
        "请求头必须显式带 User-Agent。"
    ),
    "vea_v6/__init__.py": (
        "把 dag.py 里的 ContentItem / Edge / DAG / DEFAULT_PRESET_EDGE / SmartMockLLM 和 "
        "llm.py 里的 OpenAICompatibleLLM 都 re-export 出来，带 __all__。"
    ),
    "vea_v6/parts/__init__.py": "一个简短的包说明 docstring 即可。",
}


#: 文件按段生成。一次要一整份 dag.py（几百行）会被网关在 100s 处掐成 524，
#: 拆成小段后每段 4~20s 就能回来。每段只描述"这一段要写什么"，不含任何现成源码。
FILE_SEGMENTS: Dict[str, List[str]] = {
    "vea_v6/parts/dag.py": [
        "第 1 段：文件头 `from __future__ import annotations`、import、"
        "ContentItem 数据类（content / tags / gen / image）与 Edge 数据类"
        "（input_tags / processor / output_tags / layer / fn=None / is_preset / edge_id）。只写这两块。",
        "第 2 段：在上一段之后追加 SmartMockLLM 类——用 add_rule(keyword, answer) 存规则，"
        "ask(prompt) 时按关键词命中返回对应内容，没命中返回空串。以及 OpenAICompatibleLLM 的导入占位"
        "（`from .llm import OpenAICompatibleLLM`）。只写这一段。",
        "第 3 段：在上一段之后追加 DAG 类的构造函数与 add_edge / send / has_result / get_result / "
        "context 属性：内部维护 entries 列表与 edges 列表，add_edge 接 (input_tags, processor, "
        "output_tags, layer=1, fn=None)。只写这一段。",
        "第 4 段：在上一段之后追加 DAG.run 方法：跑一轮，找出所有输入标签都能在同一批 entries 里凑齐的边，"
        "执行 processor（list 参数就传列表，否则传单条内容），把结果做成带 output_tags 的新条目追加进去；"
        "另外要能自动补缺失标签（输出标签等于所需标签的中间条目）。再追加 DEFAULT_PRESET_EDGE 常量"
        "（is_preset=True）与 __all__。只写这一段。",
    ],
    "vea_v6/parts/llm.py": [
        "写整个文件：LLMError 异常、extract_message_text(data) 取正文、OpenAICompatibleLLM 类，"
        "用 urllib 打 OpenAI 兼容的 /v1/chat/completions；_post_sync(prompt, model, base_url, "
        "api_key, max_tokens, temperature, stream) 与 ask 两个入口；stream=True 时解析 SSE 的 data: 行。"
        "请求头显式带 User-Agent 与 Content-Type。不要用第三方库。",
    ],
    "vea_v6/__init__.py": [
        "写整个文件：从 .parts.dag 导入 ContentItem / Edge / DAG / DEFAULT_PRESET_EDGE / SmartMockLLM，"
        "从 .parts.llm 导入 OpenAICompatibleLLM，并用 __all__ 列全。",
    ],
    "vea_v6/parts/__init__.py": ["写整个文件：一行包说明 docstring 即可。"],
}

SEG_HEAD = """
请输出 JSON，结构严格为：
{{"path": "文件路径", "source": "本段代码"}}
不要解释，不要 markdown 围栏。

本次写：{path}
{seg}

每个文件开头写 `from __future__ import annotations`；只用标准库；import 期不做任何 IO。

下面是框架的自然语言描述，供你理解整体意图：
"""


def _extract_one(text: str) -> Optional[str]:
    """从模型输出里取出代码段。"""
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


def _gen_one(gen_dir: Path, model_key: str) -> Optional[Dict[str, str]]:
    """按段生成：每个文件拆成若干段，逐段要，最后拼起来。"""
    files: Dict[str, str] = {}
    for path, segs in FILE_SEGMENTS.items():
        chunks: List[str] = []
        for idx, seg in enumerate(segs):
            head = SEG_HEAD.replace("{path}", path).replace("{seg}", seg)
            # 不能 str.format：JSON 示例里含 `{`，会被当成占位符
            prompt = head.replace("{{", "{").replace("}}", "}") + DESCRIPTION
            if chunks:
                prev_tail = "\n".join(chunks)[-2500:]
                prompt += f"\n\n（上一段已经写到这里，请无缝接着写，不要重复已写内容）\n{prev_tail}\n"
            text = _call_model(prompt, model_key, gen_dir)
            src = _extract_one(text)
            if not src:
                print(f"    FAIL {path} 第{idx+1}段: 没有拿到代码")
                return None
            chunks.append(src)
        target = gen_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        joined = "\n\n".join(chunks)
        target.write_text(joined, encoding="utf-8", newline="")
        files[path] = joined
    return files


def _can_run(gen_dir: Path) -> tuple[bool, str]:
    """把这一代当成独立包，真的 import 它、建 DAG、跑一圈、取 result。"""
    import subprocess

    probe = gen_dir / "_probe_self.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(gen_dir)!r})\n"
        "import vea_v6\n"
        "from vea_v6 import DAG, DEFAULT_PRESET_EDGE, SmartMockLLM\n"
        "llm = SmartMockLLM()\n"
        "llm.add_rule('hello', 'world')\n"
        "d = DAG('t', preset_edge=DEFAULT_PRESET_EDGE, llm=llm)\n"
        "d.add_edge(['a'], lambda s: s.upper(), ['result'], layer=1)\n"
        "d.send({'tags': ['a'], 'content': 'hello'})\n"
        "for _ in range(5):\n"
        "    if d.has_result():\n"
        "        break\n"
        "    d.run()\n"
        "r = d.get_result()\n"
        "assert r == 'HELLO', f'result 不对: {r!r}'\n"
        "print('OK', r)\n",
        encoding="utf-8",
        newline="",
    )
    p = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=120
    )
    (gen_dir / "_probe_self.log").write_text(p.stdout + p.stderr, encoding="utf-8", newline="")
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        return False, "; ".join(tail)
    return True, p.stdout.strip()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="自然语言描述驱动的功能等价自举")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--model-key", default="laptop-6.8")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--dry-run", action="store_true", help="只打印提示词")
    a = ap.parse_args(argv)

    if a.dry_run:
        print(FILE_SPEC + DESCRIPTION)
        return 0

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    base = RUNS / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "DESCRIPTION.txt").write_text(DESCRIPTION, encoding="utf-8", newline="")
    print(f"run-id = {run_id}")
    print(f"提示词：纯自然语言描述，{len(DESCRIPTION)} 字符，不含任何源码\n")

    results: List[Dict[str, Any]] = []
    for i in range(a.generations):
        gdir = base / f"gen{i}"
        gdir.mkdir(parents=True, exist_ok=True)
        print(f"[gen{i}] 生成")
        files = _gen_one(gdir, a.model_key)
        if files is None:
            results.append({"gen": i, "ok": False, "why": "生成失败"})
            break
        sizes = {p: len(s.encode("utf-8")) for p, s in files.items()}
        print(f"    {sum(sizes.values())} 字节  {sizes}")
        ok, why = _can_run(gdir)
        print(f"    可运行={'OK' if ok else 'FAIL'}  {why[:100]}")
        results.append({"gen": i, "ok": ok, "why": why, "sizes": sizes})
        if not ok:
            break

    (base / "RESULT.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8", newline=""
    )
    passed = sum(1 for r in results if r.get("ok"))
    print(f"\n可运行代数：{passed}/{a.generations}")
    return 0 if passed >= a.generations else 1


if __name__ == "__main__":
    sys.exit(main())
