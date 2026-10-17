from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from vea_v6.parts.dag import DEFAULT_PRESET_EDGE, DAG
from vea_v6.parts.llm import OpenAICompatibleLLM
from vea_v6.parts.primitives import ContentItem, Edge

__all__ = [
    "FRAMEWORK_SPEC",
    "FRAME_MANIFEST",
    "FRAME_CODE",
    "MAIN_SOURCE",
    "FRAME_FILES",
    "SPEC_FILES",
    "framework_sources",
    "manifest",
    "generate_framework",
    "verify_self_reproduction",
    "write_framework",
    "signature",
    "assemble",
    "BootstrapResult",
    "bootstrap",
    "create_generation_dag",
    "normalize_spec",
    "spec_to_text",
]

MAIN_SOURCE = "vea_v6/parts/bootstrap.py"
SELF_MARK = "BOOTSTRAP_B64"
SELF_FILL = "SELFB64FILL"

# =============================================================================
# 1. 框架 SPEC
# =============================================================================
FRAMEWORK_SPEC: Dict[str, Any] = {
    "name": "vea_v6",
    "version": "6.0.0",
    "summary": (
        "纯标签驱动的信息流 DAG 框架：信息条目 {tags, content, image?, gen} 在图上流转，"
        "处理边 (input_tags, processor, output_tags, layer) 声明数据依赖，缺失输入由预设边"
        "从相近信息合成，结果以 tag='result' 交付；框架能把自身 SPEC 渲染成自身源码，"
        "生成的框架可逐字节重建自身。"
    ),
    "core_invariants": [
        "DAG 结构只有 {contents: [], edges: []}",
        "处理边必须有输入 tag 与输出 tag（预设边除外）",
        "dag.run() 按 layer 升序跑一轮；终止由外部检测 tag='result' 决定",
        "缺输入 tag 时预设边从相近条目合成「输出 tag = 所需输入 tag」的中间条目",
        "模型注册表 KEY → {api, key, model, params}，边内联配置优先于注册表",
        "generate_framework() 的产物逐字节等于当前框架文件",
        "manifest() 的产物逐字节等于 FRAME_MANIFEST",
    ],
    "features": [
        "ContentItem: dict 子类，键访问与属性访问并存",
        "Edge: (input_tags, processor, output_tags, layer)，processor 支持 str/dict/callable",
        "预设边：单条或多条相近信息合成，按 tag 覆盖度与相似度打分",
        "OpenAI 兼容调用：流式 SSE、注册表/内联配置、推理字段回退",
        "元构建器：把自然语言边描述编译成可运行的 V6 DAG",
        "自举：SPEC → 框架源码；源码 → 逐字节复现自身",
    ],
    "public_api": [
        "ContentItem",
        "Edge",
        "DAG",
        "DEFAULT_PRESET_EDGE",
        "SmartMockLLM",
        "OpenAICompatibleLLM",
        "load_models_file",
        "create_dag_builder_dag",
        "MetaDAGBuilder",
        "create_generation_dag",
        "bootstrap",
        "generate_framework",
        "manifest",
        "verify_self_reproduction",
    ],
}

FRAME_FILES: Tuple[str, ...] = (
    "vea_v6/parts/__init__.py",
    "vea_v6/parts/primitives.py",
    "vea_v6/parts/llm.py",
    "vea_v6/parts/dag.py",
    "vea_v6/parts/builder.py",
    MAIN_SOURCE,
    "vea_v6/__init__.py",
)

SPEC_FILES: Tuple[str, ...] = ("vea_v6/__init__.py",)

# =============================================================================
# 2. 清单与源码块（占位形态；固化后由 tools/fixpoint.py 填入真实内容）
# =============================================================================
FRAME_MANIFEST = json.loads(r'''{
  "spec": {},
  "files": []
}''')

FRAME_CODE = json.loads(r'''{
  "files": []
}''')
BOOTSTRAP_B64 = "SELFB64FILL"


# =============================================================================
# 3. 工具
# =============================================================================
def _decode(text: str) -> bytes:
    return gzip.decompress(base64.b64decode(text.encode("ascii")))


def signature(path: str, source: str) -> List[str]:
    """提取公开签名：顶层 `def`/`class` 声明（跳过私有名，含参数名）。"""
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


def _b64of(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")


_MAN_RE = re.compile(r"^FRAME_MANIFEST = json\.loads\(r'''.*?'''\)", re.S | re.M)
_CODE_RE = re.compile(r"^FRAME_CODE = json\.loads\(r'''.*?'''\)", re.S | re.M)


def _render(source: str, manifest_text: str, code_text: str, pad: int = 0) -> str:
    out, n1 = _MAN_RE.subn(lambda _m: "FRAME_MANIFEST = json.loads(r'''" + manifest_text + "''')", source, count=1)
    if n1 != 1:
        raise ValueError("主源缺少 FRAME_MANIFEST 块")
    out, n2 = _CODE_RE.subn(lambda _m: "FRAME_CODE = json.loads(r'''" + code_text + "''')", out, count=1)
    if n2 != 1:
        raise ValueError("主源缺少 FRAME_CODE 块")
    return out.replace(SELF_FILL, "F" * pad)


def _probe(src: str) -> Dict[str, Any]:
    ns: Dict[str, Any] = {"__name__": "_vea_probe"}
    exec(compile(src, "<probe>", "exec"), ns)  # noqa: S102 - 只执行自家源码
    return ns


# =============================================================================
# 4. 只读视图
# =============================================================================
def _main_source() -> str:
    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


def framework_sources() -> Dict[str, str]:
    """框架当前源码 `{路径: 文本}`（来自 FRAME_CODE + 磁盘上的主源）。"""
    files = {item["path"]: _decode(item["code"]).decode("utf-8") for item in FRAME_CODE.get("files", [])}
    files[MAIN_SOURCE] = _main_source()
    return {p: files[p] for p in FRAME_FILES}


def manifest() -> str:
    """渲染框架清单（规范 JSON）；对同一份源码恒定。"""
    files = framework_sources()
    docs = []
    for p in FRAME_FILES:
        raw = files[p].encode("utf-8")
        docs.append(
            {
                "path": p,
                "bytes": len(raw),
                "b64_len": len(_b64of(raw)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "sig": signature(p, files[p]),
            }
        )
    return json.dumps({"spec": FRAMEWORK_SPEC, "files": docs}, ensure_ascii=False, indent=2)


def generate_framework(spec: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """重建整个框架源码 `{路径: 文本}`，逐字节等于被内嵌的那份。"""
    out = {item["path"]: _decode(item["code"]).decode("utf-8") for item in FRAME_CODE.get("files", [])}
    out[MAIN_SOURCE] = _render(
        _main_source(),
        json.dumps(FRAME_MANIFEST, ensure_ascii=False, indent=2),
        json.dumps(FRAME_CODE, ensure_ascii=False, indent=2),
        pad=len(BOOTSTRAP_B64),
    )
    return out


def verify_self_reproduction() -> Tuple[bool, str]:
    """`generate_framework()` 与 `manifest()` 是否都逐字节复现自身。"""
    cur = framework_sources()
    regen = generate_framework()
    diffs = [p for p in FRAME_FILES if cur.get(p) != regen.get(p)]
    if diffs:
        return False, f"源码不一致: {diffs}"
    if manifest() != json.dumps(FRAME_MANIFEST, ensure_ascii=False, indent=2):
        return False, "清单不一致"
    return True, f"{len(FRAME_FILES)} 个文件 / {sum(len(v.encode()) for v in regen.values())} 字节逐字节一致"


def write_framework(target_dir: str, files: Optional[Dict[str, str]] = None) -> List[str]:
    from pathlib import Path

    root = Path(target_dir)
    data = files if files is not None else generate_framework()
    written = []
    for path, text in data.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(str(p))
    return written


# =============================================================================
# 5. assemble：把任意文件集变成自洽框架（模型侧唯一写入口）
# =============================================================================
def assemble(parts: Dict[str, str], *, spec: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """补清单、解固定点、自校验，返回可逐字节复现的文件集。"""
    files = dict(parts)
    missing = [p for p in FRAME_FILES if p not in files]
    if missing:
        raise ValueError(f"assemble 缺少文件: {missing}")
    for p in FRAME_FILES:
        try:
            ast.parse(files[p])
        except SyntaxError as exc:
            raise ValueError(f"{p} 语法错误: {exc}") from exc

    spec_obj = spec or FRAMEWORK_SPEC
    others = [p for p in FRAME_FILES if p != MAIN_SOURCE]
    raw = {p: files[p].encode("utf-8") for p in others}
    payloads = {p: _b64of(raw[p]) for p in others}
    code_obj = {
        "files": [
            {"path": p, "key": p.replace("/", "_").replace(".", "_").upper() + "_B64", "code": payloads[p]}
            for p in others
        ]
        + [{"path": MAIN_SOURCE, "key": SELF_MARK, "code": "SELF"}]
    }
    code_text = json.dumps(code_obj, ensure_ascii=False, indent=2)

    def manifest_text(pad: int, self_bytes: int, self_sha: str) -> str:
        docs = []
        for p in FRAME_FILES:
            if p == MAIN_SOURCE:
                docs.append(
                    {"path": p, "bytes": self_bytes, "b64_len": pad, "sha256": self_sha, "sig": signature(p, files[p])}
                )
            else:
                docs.append(
                    {
                        "path": p,
                        "bytes": len(raw[p]),
                        "b64_len": len(payloads[p]),
                        "sha256": hashlib.sha256(raw[p]).hexdigest(),
                        "sig": signature(p, files[p]),
                    }
                )
        return json.dumps({"spec": spec_obj, "files": docs}, ensure_ascii=False, indent=2)

    pad = 1
    for _round in range(20):
        src = _render(files[MAIN_SOURCE], manifest_text(pad, 0, "P" * 64), code_text, pad=pad)
        need = len(_b64of(src.encode("utf-8")))
        if need != pad:
            pad = need
            continue
        size = len(gzip.compress(src.encode("utf-8"), 9, mtime=0))
        src = _render(files[MAIN_SOURCE], manifest_text(pad, size, hashlib.sha256(src.encode()).hexdigest()),
                      code_text, pad=pad)
        if len(_b64of(src.encode("utf-8"))) != pad:
            pad = len(_b64of(src.encode("utf-8")))
            continue
        break
    else:
        raise RuntimeError("assemble: 固定点未收敛")

    ns = _probe(src)
    gen = ns["generate_framework"]()
    want = {p: (files[p] if p != MAIN_SOURCE else src) for p in FRAME_FILES}
    bad = [p for p in FRAME_FILES if gen.get(p) != want[p]]
    if bad:
        raise RuntimeError(f"assemble: 复现失败 {bad}")
    out = {p: files[p] for p in FRAME_FILES if p != MAIN_SOURCE}
    out[MAIN_SOURCE] = src
    return out


# =============================================================================
# 6. 生成 DAG
# =============================================================================
@dataclass
class BootstrapResult:
    """一次生成的结果。"""

    files: Dict[str, str]
    manifest_text: str
    okay: bool
    reason: str = ""
    context: List[ContentItem] = field(default_factory=list)

    def write_to(self, target_dir: str) -> List[str]:
        return write_framework(target_dir, self.files)


def normalize_spec(spec: Union[Dict[str, Any], str, None]) -> Dict[str, Any]:
    """SPEC 规范化：键序固定、列表去重、缺键回落默认，保证渲染确定。"""
    if spec is None:
        base: Dict[str, Any] = dict(FRAMEWORK_SPEC)
    elif isinstance(spec, str):
        text = spec.strip()
        base = dict(FRAMEWORK_SPEC)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    base = dict(parsed)
            except json.JSONDecodeError:
                base["summary"] = text
    elif isinstance(spec, dict):
        base = dict(spec)
    else:
        raise TypeError(f"SPEC 必须是 dict/JSON/None，实际 {type(spec).__name__}")

    out: Dict[str, Any] = {
        "name": str(base.get("name") or FRAMEWORK_SPEC["name"]),
        "version": str(base.get("version") or FRAMEWORK_SPEC["version"]),
        "summary": str(base.get("summary") or FRAMEWORK_SPEC["summary"]).strip(),
    }
    for key in ("core_invariants", "features", "public_api"):
        raw_val = base.get(key, FRAMEWORK_SPEC[key])
        items = raw_val.splitlines() if isinstance(raw_val, str) else list(raw_val)
        out[key] = list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))
    return out


def spec_to_text(spec: Union[Dict[str, Any], str, None] = None) -> str:
    return json.dumps(normalize_spec(spec), ensure_ascii=False, indent=2)


#: 三段式提示词（生成 DAG 的每条边问一次模型）
PROMPTS: Dict[str, str] = {
    "spec": (
        "你在为下述框架写可执行规格（SPEC）。只输出 JSON，键为 name/version/summary/"
        "core_invariants/features/public_api，不要解释、不要 markdown 围栏。\n"
        "框架现状：\n{current}\n\n用户补充描述：\n{input}"
    ),
    "skeleton": (
        "框架文件与公开签名如下（JSON）。请输出你打算保留的签名，结构 "
        '{{"files":[{{"path":...,"sig":[...]}}]}}。\n'
        "硬约束：{spec_file} 的签名原样保留；public_api 中的每个名字都必须出现在某个文件的签名里。\n"
        "{sig_list}"
    ),
    "flesh": (
        "重写下面这个文件（只输出代码本体，不要 markdown 围栏）。\n"
        "约束：只用标准库与 `from vea_v6.parts…` 兄弟模块；import 期不做 IO；保留原签名。\n"
        "文件：{path}\n原源码：\n{source}"
    ),
}


def create_generation_dag(
    llm: Optional[Any] = None,
    *,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    model_key: str = "laptop-6.8",
    rewrite_files: Sequence[str] = (),
) -> DAG:
    """创建框架生成 DAG（标准 V6 DAG；三条 layer-1 边汇聚成 skeleton_code）。"""
    dag = DAG("V6_Framework_Generation_DAG", preset_edge=DEFAULT_PRESET_EDGE, llm=llm, models=models)
    base = dict(framework_sources())
    targets = [p for p in (rewrite_files or []) if p != MAIN_SOURCE]

    def _ask(prompt: str) -> str:
        return dag.ask_sync(prompt, model_key)

    def _edge_spec(content: str) -> str:
        current = spec_to_text(FRAMEWORK_SPEC)
        if not content.strip():
            return current
        return spec_to_text(_extract_code(_ask(PROMPTS["spec"].format(current=current[:4000], input=content)), "json"))

    def _edge_skeleton(spec_text: str) -> str:
        spec = normalize_spec(spec_text)
        sig_list = json.dumps({p: signature(p, base[p]) for p in FRAME_FILES}, ensure_ascii=False, indent=1)
        raw = _ask(PROMPTS["skeleton"].format(spec_file=SPEC_FILES[0], sig_list=sig_list))
        return json.dumps({"spec": spec, "plan": _merge_plan(spec, raw)}, ensure_ascii=False)

    def _edge_flesh(spec_text: str) -> str:
        spec = normalize_spec(spec_text)
        jobs = [
            {"path": p, "prompt": PROMPTS["flesh"].format(path=p, source=_clip(base[p]))}
            for p in targets
        ]
        return json.dumps({"spec": spec, "jobs": jobs}, ensure_ascii=False)

    def _edge_skeleton_code(merged: str) -> str:
        # 预设边把三路拼成一条文本，这里只取骨架计划（spec 与 jobs 分属各自 tag）
        data = json.loads(merged)
        return json.dumps({"spec": data.get("spec"), "plan": data.get("plan")}, ensure_ascii=False)

    def _edge_frame_code(plan_text: str) -> str:
        plan = json.loads(plan_text)
        parts = dict(base)
        for job in plan.get("flesh") or []:
            code = _extract_code(_ask(job["prompt"]), "python")
            if code.strip():
                parts[job["path"]] = code
        return json.dumps({"files": parts, "spec": plan["spec"]}, ensure_ascii=False)

    def _edge_manifest(payload: str) -> str:
        data = json.loads(payload)
        files = assemble(data["files"], spec=normalize_spec(data["spec"]))
        ns = _probe(files[MAIN_SOURCE])
        ok, why = ns["verify_self_reproduction"]()
        return json.dumps(
            {"okay": ok, "reason": why, "manifest": json.dumps(ns["FRAME_MANIFEST"], ensure_ascii=False, indent=2)},
            ensure_ascii=False,
        )

    dag.add_edge(["input"], _edge_spec, ["spec_plan"], layer=1)
    dag.add_edge(["input"], _edge_skeleton, ["skeleton_plan"], layer=1)
    dag.add_edge(["input"], _edge_flesh, ["flesh_plan"], layer=1)
    dag.add_edge(["spec_plan", "skeleton_plan", "flesh_plan"], _edge_skeleton_code, ["skeleton_code"], layer=2)
    dag.add_edge(["skeleton_code"], _edge_frame_code, ["frame_code"], layer=3)
    dag.add_edge(["frame_code"], _edge_manifest, ["manifest"], layer=4)
    dag.add_edge(["manifest"], lambda text: text, ["result"], layer=5)
    return dag


def _merge_plan(spec: Dict[str, Any], raw: str) -> Dict[str, List[str]]:
    """合并模型的签名计划：SPEC 文件只读；public_api 缺谁就从上一版补回同名声明。"""
    proposed: Dict[str, List[str]] = {}
    try:
        data = json.loads(_extract_code(raw, "json"))
        for item in data.get("files", []):
            proposed[str(item.get("path"))] = [str(x) for x in item.get("sig", [])]
    except Exception:
        proposed = {}
    base = framework_sources()
    plan: Dict[str, List[str]] = {}
    for p in FRAME_FILES:
        prev = signature(p, base[p])
        sigs = list(prev) if p in SPEC_FILES else list(proposed.get(p) or prev)
        for name in spec.get("public_api", []):
            if not any(_name_of(s) == name for s in sigs):
                sigs.extend(s for s in prev if _name_of(s) == name)
        plan[p] = sigs
    return plan


def _name_of(sig_line: str) -> str:
    m = re.match(r"^(?:def|class)\s+([A-Za-z_]\w*)", sig_line.strip())
    return m.group(1) if m else sig_line.strip()


def _extract_code(text: str, want: str = "python") -> str:
    """剥掉 markdown 围栏，返回代码/JSON 本体。"""
    t = (text or "").strip()
    m = re.search(r"```(?:python|json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return t if want == "json" else t + ("" if t.endswith("\n") else "\n")


def _clip(text: str, limit: int = 9000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n# ...(截断)\n"


def bootstrap(
    description: str = "",
    llm: Optional[Any] = None,
    *,
    model_key: str = "laptop-6.8",
    rewrite_files: Sequence[str] = (),
    max_ticks: int = 12,
) -> BootstrapResult:
    """跑一次生成 DAG：描述 → 框架文件集 + 清单 + 自校验结论。"""
    dag = create_generation_dag(llm=llm, model_key=model_key, rewrite_files=rewrite_files)
    dag.send({"tags": ["input"], "content": description, "gen": 0})
    ticks = 0
    while not dag.has_result() and ticks < max_ticks:
        dag.run()
        ticks += 1
    raw = dag.get_result()
    if not raw:
        raise RuntimeError("生成 DAG 没跑出 tag='result'")
    data = json.loads(raw)
    files = assemble(framework_sources(), spec=normalize_spec(description))
    return BootstrapResult(
        files=files,
        manifest_text=data.get("manifest", ""),
        okay=bool(data.get("okay")),
        reason=data.get("reason", ""),
        context=list(dag.context),
    )
