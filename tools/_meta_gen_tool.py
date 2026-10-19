"""元 DAG 的工具实现：把"写一个框架"拆成 8 个可观测的步骤。

`make_gen_tool(...)` 返回一个可调用对象，挂到 `DAG.tools["gen"]`。
它按 `edge.tool_args["step"]` 分派：

| step | 动作 | 调 LLM |
|---|---|---|
| expand          | 描述 → 逐文件施工单 | 1 次 |
| plan_module     | 施工单 → **逐文件原子生成**（每文件一次请求，产完即单文件过闸，不过带原因重做） | N 次 |
| normalize       | 剥围栏 / 去过短产出 | 否 |
| audit_contract  | 语法 + 顶层符号 + 体积 + SPEC 的 `__all__` | 否 |
| supplement      | 多份审计结果汇聚；缺的用 `fallback` 补 | 否 |
| assemble        | 交给当前框架的 `assemble()` 解固定点 | 否 |
| measure         | 统计字节数 / 文件数 / 闸门结论 | 否 |
| passthrough     | 原样透传（给图外读 `result`） | 否 |

**逐文件原子生成（更细的步骤）**：原先 `plan_module` 按字节凑批，一批一次请求产出
多个文件的完整源码 —— 单文件动辄 8 万字符，输出被 `MAX_TOKENS` 砍断还静默绕闸，且
一批里一个文件坏、整批重做。现在每个文件**单独一次请求**，产完立即单文件闸门
（`contract_violation`：语法 / 顶层符号不许少 / 体积不坍缩），不过带原因重做
（`FILE_RETRIES` 次），失败的文件用上一代补。参考实现只给这一个文件（全量 `REF_CHARS`，
不再截断），跨文件一致性靠 step 4 的 `audit_contract` + step 5 的 `supplement` 兜底。
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools._meta_dag_spec import FRAME_FILES, SOURCE_TASKS

#: 单次请求的输出上限。`deepseek-v4-flash` 上 `reasoning_effort=none` 才能让它
#: 把预算花在正文上（不开时 3000 token 全被思考吃掉，正文 0 字符，实测）。
#: 单次输出预算。🔴 原先 32000 不够：`bootstrap.py` 有 88742 字符，JSON 里再
#: 转义一遍就远超，响应被 max_tokens 砍断（实测 s17：dag.py 只生成 68%、
#: bootstrap.py 46%）。网关实测硬上限是 65536（65537 即 HTTP 400），取到上限。
MAX_TOKENS = 65536

#: 给模型的"参考实现"给多少字符。🔴 原先 20000 是**截断的元凶之一**：
#: 超过 2 万字符的大文件（bootstrap.py 88742、dag.py 32514）模型只看得到开头
#: 一截，剩下全靠自由发挥 —— 于是它按记忆里另一套 API 去写（`LLMClient`、
#: `LLMRegistry`、`make_graph`、`run_round` 这些名字在框架里**一个都不存在**，
#: 实测 s17 两次生成都因此"跨文件缺导出"被装配闸门整代否掉）。给全量即可。
REF_CHARS = 120000

#: 单文件生成的有界重试次数：产完单文件闸门不过，就带原因重做这么多次。
FILE_RETRIES = 2

#: 接口契约：由 SPEC 固定下来的名字。任何一代都必须提供，否则整套代码互相 import 不上。
#: 这不是"发明新词"，而是把磁盘上那一份框架**已经公开**的符号写死成共识。
API_CONTRACT = {
    "vea_v6/parts/primitives.py": [
        "ContentItem",
        "Edge",
        "LLMProtocol",
        "normalize_tags",
        "content_to_str",
        "merge_payload",
        "default_text_similarity",
        "ProcessorType",
    ],
    "vea_v6/parts/llm.py": ["OpenAICompatibleLLM", "SmartMockLLM", "extract_message_text"],
    "vea_v6/parts/dag.py": [
        "DAG",
        "DEFAULT_PRESET_EDGE",
        "load_models_file",
        "combine_contents",
    ],
    # DAG 的方法签名是"框架的公共接口"，跨代必须一致（demo 与外部脚本都按它调）。
    # 只查顶层名字抓不到"类在、方法丢了"这种情况：实测第 1 代 DAG 缺 `has_result`，
    # 顶层契约全过、固定点也解开了，demo 一跑就 AttributeError。
    "vea_v6/parts/dag.py::DAG": [
        "add_edge",
        "send",
        "run",
        "find_by_tag",
        "has_tag",
        "has_result",
        "get_result",
        "get_result_item",
        "mount_tool",
        "unmount_tool",
        "to_dict",
        # `bootstrap.py` 的生成 DAG 会调 `dag.ask_sync(prompt, key)`；第 3、4 代都漏了它。
        # 顶层名字契约抓不到（DAG 类在、方法丢了），只有列进这里才会被要求写。
        "ask_sync",
    ],
    "vea_v6/parts/builder.py": [
        "MetaDAGBuilder",
        "create_dag_builder_dag",
        "parse_edges_from_description",
        "parse_json_stream",
    ],
    "vea_v6/parts/bootstrap.py": [
        "FRAMEWORK_SPEC",
        "FRAME_FILES",
        "MAIN_SOURCE",
        "BOOTSTRAP_SLOT",
        "bootstrap",
        "assemble",
        "generate_framework",
        "manifest",
        "render_framework_code",
        "verify_fixed_point",
        "verify_self_reproduction",
        "create_generation_dag",
        "create_bootstrap_dag",
        "normalize_spec",
        "spec_to_text",
        "signature",
    ],
    "vea_v6/parts/__init__.py": [],
    "vea_v6/__init__.py": [],
}


def _import_map(sources: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    """解析 `from X import (a, b)` / `from .X import ...`，返回 `{模块名: {导入方: [名字...]}}`。

    只处理本框架自己的两种写法。**必须支持括号跨行**：实测模型写出来的都是
    `from .primitives import (\\n    ContentItem,\\n    Edge,\\n)` 这种，不认括号就一个
    名字都抓不到，闸门会静默放行（第一次跑就踩到了）。
    """
    wanted: Dict[str, Dict[str, List[str]]] = {}
    pattern = re.compile(
        r"^from\s+(?:vea_v6\.parts\.|\.)(\w+)\s+import\s+(?:\(([^)]*)\)|([^\n]*))",
        re.M,
    )
    for path, src in sources.items():
        if not src:
            continue
        for m in pattern.finditer(src):
            mod = m.group(1)
            body = m.group(2) if m.group(2) is not None else m.group(3)
            names = [n.strip().split(" as ")[0].strip() for n in body.replace("\\", "").split(",")]
            names = [n for n in names if n and n.isidentifier()]
            if not names:
                continue
            wanted.setdefault(mod, {}).setdefault(path, []).extend(names)
    return wanted


def _exported(mod_src: str) -> str:
    """模块对外提供的名字集合（顶层 class/def/赋值）。"""
    tops = _tops(mod_src)
    return "\n".join(sorted(tops))


def _class_methods(src: str, cls: str) -> List[str]:
    """某个类的全部方法名（含私有，供反查）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == cls:
            return [b.name for b in n.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return []


def _need_for(path: str) -> List[str]:
    """顶层契约 + 类方法契约（`path::Class` 那几条）合并后的要求。"""
    need = list(API_CONTRACT.get(path, []))
    for key, names in API_CONTRACT.items():
        head, _, cls = key.partition("::")
        if head == path and cls:
            need.extend(names)
    return need


#: 这些公共方法**必须是同步的**（`def` 不是 `async def`）。
#: 🔴 实测第 5 代的 `DAG.run` 被写成 `async def run`，而磁盘上那一份是
#: `def run(self) -> bool`（内部 `asyncio.run`）。顶层名字、方法名契约全过，
#: 验收脚本一调 `d.run()` 就拿到一个**没被 await 的协程**，返回 None。
#: 这是"同一份自然语言描述下各代行为不一致"的典型，必须写进契约。
SYNC_METHODS = {
    "vea_v6/parts/dag.py::DAG": ["run", "send", "add_edge", "mount_tool", "unmount_tool", "to_dict",
                                 "find_by_tag", "has_tag", "has_result", "get_result", "get_result_item"],
}


def _async_methods(src: str, cls: str) -> List[str]:
    """类里被定义成 `async def` 的方法名。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == cls:
            return [b.name for b in n.body if isinstance(b, ast.AsyncFunctionDef)]
    return []


def _check_class_methods(path: str, src: str) -> List[str]:
    """类方法契约：返回缺失列表（空 = 通过）。"""
    miss: List[str] = []
    for key, names in API_CONTRACT.items():
        head, _, cls = key.partition("::")
        if head != path or not cls:
            continue
        have = _class_methods(src, cls)
        if not have:
            miss.append(f"<类 {cls} 不存在>")
            continue
        miss.extend(f"{cls}.{n}" for n in names if n not in have)
        aio = _async_methods(src, cls)
        for n in SYNC_METHODS.get(key, []):
            if n in aio:
                miss.append(f"{cls}.{n} 是 async（要求同步）")
    return miss


def missing_exports(sources: Dict[str, str]) -> Dict[str, List[str]]:
    """哪些"被别处 import 的名字"在目标模块里根本没定义 → {模块: [缺的名字]}。"""
    by_mod = {p.split("/")[-1][:-3]: p for p in FRAME_FILES if p.endswith(".py")}
    out: Dict[str, List[str]] = {}
    for mod, users in _import_map(sources).items():
        target = by_mod.get(mod)
        if not target:
            continue
        have = set(_tops(sources.get(target, "")))
        need = sorted({n for names in users.values() for n in names} - have)
        if need:
            out[target] = need
    return out


import builtins as _builtins  # noqa: E402

_MODULES = {"vea_v6"}

_BUILTINS = frozenset(dir(_builtins)) | {"__name__", "__file__", "__doc__", "__all__", "__class__"}


def undefined_names(path: str, src: str, sources: Dict[str, str]) -> List[str]:
    """单文件里"被用到但没定义也没导入"的名字 —— 运行到那一行才炸的那种。

    🔴 为什么必须有这道闸门：第 10 代 `dag.py` 的 `__init__` 里写了
    `isinstance(llm, OpenAICompatibleLLM)`，但整个文件**根本没 import 它**
    （只在上面 `from .primitives import (...)` 里导了别的东西）。语法合法、
    `__all__` 齐全、顶层符号契约全过、固定点也解开了 —— 一 `DAG(...)` 就 NameError。
    只查"承诺导出的名字在不在"抓不到这种情况，方向正好相反。

    做法：`symtable` 拿到模块作用域的全部引用，减去
    (顶层定义 + import 名 + 局部/内建 + 本文件字面名字)，剩下的若不是
    本框架别的模块的公开符号，就是真悬空引用。
    """
    if not src.strip():
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    defined = set(_tops(src))
    imported: set = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported.add((a.asname or a.name.split(".")[0]))
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, (ast.Name,)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            defined.update(n.names)
    # 🔴 不能把"兄弟模块定义过这个名字"直接放行：实测 gen1 的 `dag.py` 在函数体里
    # 用了 `OpenAICompatibleLLM` 却没 import，而 `llm.py` 顶层**定义**了这个名字——
    # 一放行这道闸门就完全失效了。正确判据是：**只有 `X.Y` 属性访问里的 X**
    # （模块名/类名）才允许来自兄弟模块；裸名字一律必须在**本文件**里拿到。
    attr_base = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            v = n.value
            while isinstance(v, ast.Attribute):
                v = v.value
            if isinstance(v, ast.Name):
                attr_base.add(v.id)

    annotation_spans = set()
    for n in ast.walk(tree):
        ann = getattr(n, "annotation", None)
        if ann is not None and hasattr(ann, "lineno"):
            for sub in ast.walk(ann):
                if getattr(sub, "lineno", None) is not None:
                    annotation_spans.add((sub.lineno, sub.col_offset))
        args = getattr(n, "args", None)
        if args is not None:
            for a in list(getattr(args, "args", ()) or ()) + list(getattr(args, "kwonlyargs", ()) or ()):
                if getattr(a, "lineno", None) is not None:
                    annotation_spans.add((a.lineno, a.col_offset))

    missing = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
            continue
        if (n.lineno, n.col_offset) in annotation_spans:
            continue
        if n.id in defined or n.id in imported or n.id in _BUILTINS:
            continue
        # `X.Y` 里的 X 可以是兄弟模块名（`bootstrap.py` 有 `from . import dag` 这种）
        if n.id in attr_base and n.id in _MODULES:
            continue
        missing.append(n.id)
    return sorted(set(missing))


def _class_members(src: str, cls: str) -> set:
    """类体内定义的属性/方法名：赋值、注解、self.x = ...、property。"""
    out: set = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in tree.body:
        if not (isinstance(n, ast.ClassDef) and n.name == cls):
            continue
        for b in n.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(b.name)
            elif isinstance(b, ast.Assign):
                for t in b.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
            elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                out.add(b.target.id)
            # `self.x = ...` 也算实例属性，且在 __init__ 之外也要抓
            for sub in ast.walk(b):
                if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                    out.add(sub.attr)
    return out


#: 这些属性不是类契约的一部分（协议/魔法/内部），出现在别处也放行
_ATTR_OK = {
    "append", "extend", "get", "items", "keys", "values", "copy", "update", "pop", "setdefault",
    "format", "join", "split", "strip", "lstrip", "rstrip", "replace", "startswith", "endswith",
    "lower", "upper", "encode", "decode", "read", "write", "get", "sort", "sorted", "index",
    "group", "match", "findall", "sub", "search", "compile", "fullmatch",
}


def _method_return_cls(tree: ast.AST, call: ast.Call) -> str:
    """`x.m(...)` / `m(...)` 的返回类型名（只认写在注解里的简单名字）。"""
    f = call.func
    name = getattr(f, "attr", None) or getattr(f, "id", None)
    if not name:
        return ""
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name and n.returns is not None:
            r = n.returns
            if isinstance(r, ast.Name):
                return r.id
            if isinstance(r, ast.Subscript) and isinstance(r.value, ast.Name) and r.value.id == "Optional":
                inner = r.slice
                if isinstance(inner, ast.Name):
                    return inner.id
    return ""


def cross_class_attrs(sources: Dict[str, str]) -> Dict[str, List[str]]:
    """跨文件类属性契约：A 里 `x.attr` 用到的属性，定义该类型的类里必须有。

    🔴 为什么必须有这道闸门：第 4 代 `dag.py` 写了
    `self._executed_records.add((idx, result.id))`，而同一代的
    `primitives.ContentItem` **没有 `id` 字段**。跨文件导出名全都对得上、
    顶层契约全过、固定点也解开了，一 `run()` 就是 AttributeError。

    做法（只在能确定类型时判，不做类型推断）：
    - 找出 `sources` 里所有类的"方法名 → 该类成员集合"与"方法名 → 类名"；
    - 扫描每个文件里 `x.attr` 的属性访问，`x` 若是**某个已知类的实例**
      （`self` 在类内、`参数: 类名`、`var = 类名(...)`），就要求 `attr` 在该类成员里；
    - 太保守也胜过漏判：只判能唯一确定类型的那些点。
    """
    # 1. 建类表
    members: Dict[str, Dict[str, set]] = {}  # file -> cls -> members
    for path, src in sources.items():
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                members.setdefault(path, {})[n.name] = _class_members(src, n.name)

    # 2. 全框架"类名 -> 成员集合"（同名类只认第一个定义，避免歧义）
    by_cls: Dict[str, set] = {}
    for _p, cs in members.items():
        for cls, mem in cs.items():
            by_cls.setdefault(cls, set()).update(mem)

    out: Dict[str, List[str]] = {}
    for path, src in sources.items():
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        # 2a. 逐函数收集"变量 -> 类名"（只认注解与直接构造）
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            var_type: Dict[str, str] = {}
            if fn.args.args and fn.args.args[0].arg in ("self", "cls"):
                pass
            for a in list(fn.args.args) + list(fn.args.kwonlyargs):
                ann = getattr(a, "annotation", None)
                if isinstance(ann, ast.Name) and ann.id in by_cls:
                    var_type[a.arg] = ann.id
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Assign):
                    continue
                # 🔴 `result = await self._m(...)` 的右值是 `Await(Call(...))`，不剥掉
                # `Await` 就一个都认不出来（实测 s04 的 `result.id` 就是这样漏掉的）。
                value = sub.value
                while isinstance(value, (ast.Await,)):
                    value = value.value
                if not isinstance(value, ast.Call):
                    continue
                f = value.func
                name = getattr(f, "id", None) or getattr(f, "attr", None)
                if name in by_cls:
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            var_type[t.id] = name
                    continue
                # 类型来自**被调方法的返回注解**（`-> ContentItem`）。
                ret = _method_return_cls(tree, value)
                if ret:
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            var_type[t.id] = ret
            if not var_type:
                continue
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                    base = sub.value
                    if not isinstance(base, ast.Name) or base.id not in var_type:
                        continue
                    cls = var_type[base.id]
                    if sub.attr in _ATTR_OK or sub.attr.startswith("__"):
                        continue
                    if sub.attr not in by_cls[cls]:
                        out.setdefault(path, []).append(f"{cls}.{sub.attr}")
    return {k: sorted(set(v)) for k, v in out.items()}


def broken_imports(sources: Dict[str, str]) -> Dict[str, List[str]]:
    """`from X import a, b` 里，X 本模块没有的 a/b —— import 那一刻就 ImportError。

    与 `missing_exports` 互补：那个按"谁被 import"倒查，这个按"import 语句
    直接写的名字"正查，能抓到 `from .primitives import ContentItem` 而
    `primitives.py` 里只有 `Entry` 这种。
    """
    by_mod = {p.split("/")[-1][:-3]: p for p in FRAME_FILES if p.endswith(".py") and not p.endswith("__init__.py")}
    out: Dict[str, List[str]] = {}
    for path, src in sources.items():
        # 🔴 必须同时匹配**相对导入**和绝对导入：模型写 `vea_v6/parts/__init__.py`
        # 时常用 `from .primitives import (...)`，而原先的正则只认
        # `from vea_v6.parts.X import` —— 于是它臆造的一串名字（Graph/Info/
        # make_edge/run_round_sync…）在 audit 这一步**完全没被看见**，一路放到
        # assemble，而框架侧的 `missing_exports` 是"按谁被 import 倒查"的，会把账
        # 记到应当定义它们的 primitives.py 上 ⇒ 整代被否。实测 s17 两次尝试皆此。
        for m in re.finditer(r"^from\s+(?:vea_v6\.parts\.|\.)(\w+)\s+import\s+(?:\(([^)]*)\)|([^\n]*))", src or "", re.M):
            mod = m.group(1)
            body = m.group(2) if m.group(2) is not None else m.group(3)
            names = [n.strip().split(" as ")[0].strip() for n in body.replace("\\", "").split(",")]
            names = [n for n in names if n and n.isidentifier()]
            target = by_mod.get(mod)
            if not target:
                out.setdefault(path, []).append(f"模块 {mod} 不存在")
                continue
            have = set(_tops(sources.get(target, "")))
            # 模块里可能 `from .x import y` 再转手导出
            for m2 in re.finditer(r"^from\s+(?:vea_v6\.parts\.|\.)(\w+)\s+import\s+(?:\(([^)]*)\)|([^\n]*))",
                                  sources.get(target, ""), re.M):
                body2 = m2.group(2) if m2.group(2) is not None else m2.group(3)
                have |= {n.strip().split(" as ")[0].strip() for n in body2.replace("\\", "").split(",")}
            lost = [n for n in names if n not in have]
            if lost:
                out.setdefault(path, []).extend(f"{mod}.{n}" for n in lost)
    return out


def _strip_fence(text: str) -> str:
    """去掉 markdown 围栏（模型即使被要求"不要围栏"也常常加）。"""
    t = (text or "").strip()
    if "```" not in t:
        return t
    m = re.search(r"```(?:python|json)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t


def _tops(src: str) -> Dict[str, int]:
    """顶层定义名 → 源码行数（契约检查用）。"""
    out: Dict[str, int] = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in tree.body:
        span = (getattr(n, "end_lineno", None) or n.lineno) - n.lineno + 1
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = span
        elif isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = span
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out[n.target.id] = span
    return out


def contract_violation(path: str, code: str, base: str = "") -> str:
    """返回不合格原因（空串 = 通过）。判据只看顶层骨架，不看函数体。

    - 必须能过 `ast.parse`；
    - 若给了上一代同名文件（`base`）：顶层定义名一个都不许少，且每个定义行数
      不得跌破一半（掏空型改写的主要外形特征）；
    - 体积不得坍缩到上一代的 40% 以下；
    - SPEC 文件（包入口）必须仍带 `__all__`。
    """
    if not code.strip():
        return "空产出"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"语法错误: {exc.msg} (line {exc.lineno})"
    # 🔴 只对真正的包入口 `vea_v6/__init__.py` 强要 `__all__`；`parts/__init__.py`
    # 在真实框架里只是 `from . import ...` 的再导出、**没有** `__all__`，原先一刀切对
    # 所有 `__init__.py` 都要求 `__all__` 会把它永远判失败（实测每次都白重试 3 次再回退）。
    if path == "vea_v6/__init__.py" and "__all__" not in code:
        return "SPEC 文件缺 __all__"
    if not base.strip():
        return ""
    old, new = _tops(base), _tops(code)
    lost = sorted(set(old) - set(new))
    if lost:
        return f"顶层定义缺失 {len(lost)} 个（{', '.join(lost[:5])}）"
    shrunk = [k for k in old if old[k] >= 20 and new.get(k, 0) < old[k] // 2]
    if shrunk:
        return f"定义被削掉过半 {len(shrunk)} 个（{', '.join(shrunk[:4])}）"
    if len(code.encode("utf-8")) < len(base.encode("utf-8")) * 2 // 5:
        return f"体积坍缩到 {len(code.encode('utf-8'))}B（原 {len(base.encode('utf-8'))}B）"
    return ""


def _ask(llm: Any, prompt: str, cfg: Dict[str, Any]) -> str:
    """一次模型调用。统一走 `_post_sync`（三代框架都有），参数显式给全。

    🔴 `rpm exhausted`（HTTP 429）：这个网关按**每分钟请求数**限流，连续跑
    实验时很容易打满。框架自带的 `_post_sync` 只对 429 退避 3 次（2/4/6 秒），
    对于"每分钟 20 次"这种配额根本不够，于是整轮生成会因为一次限流报销。
    这里在**外层**再退避重试，退避步长按分钟级给（实测 20~60 秒后即可恢复）。
    """
    params = dict(cfg.get("params") or {})
    mt = min(int(params.pop("max_tokens", MAX_TOKENS) or MAX_TOKENS), MAX_TOKENS)
    params.pop("stream", None)
    last: Exception | None = None
    for attempt in range(6):
        try:
            return llm._post_sync(
                prompt,
                model=cfg["model"],
                base_url=cfg["api"],
                api_key=cfg.get("key") or None,
                max_tokens=mt,
                stream=True,
                # 关掉思考，否则 3000 token 预算会被思考吃光、正文 0 字符（实测）。
                reasoning_effort=params.pop("reasoning_effort", "none") if attempt == 0 else "none",
                **params,
            )
        except Exception as exc:  # noqa: BLE001 - 限流要重试
            msg = str(exc)
            if "429" not in msg and "rpm" not in msg and "quota" not in msg:
                raise
            last = exc
            time.sleep(min(90, 20 * (attempt + 1)))
    raise RuntimeError(f"限流重试 6 次仍失败: {last}")


class GenTool:
    """挂在 `DAG.tools["gen"]` 上的工具：按 step 分派。"""

    def __init__(
        self,
        llm: Any,
        cfg: Dict[str, Any],
        *,
        run_dir: Path,
        fallback: Optional[Dict[str, str]] = None,
        source_provider: Any = None,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.fallback = dict(fallback or {})
        #: 源码来源。默认 = 调 LLM 写；第 2 代起换成**上一代框架的 bootstrap**，
        #: 于是同一个 8 步图在"框架造框架"时一步一步照样可观测。
        self.source_provider = source_provider
        #: 代际闸门的状态位：非空 = 这一次的源码**不是**上一代真造出来的。
        #: 🔴 为什么必须有：provider 调用失败（网络瞬断等）时旧代码只往 self.log
        #: 记一笔就走回退，7 个文件全部变成上一代的副本；而副本本身合法，
        #: assemble/自举/自选任务三道闸门全绿 —— 实测 s13 的 gen2 就是这样
        #: 被当成"新的一代"验收（CHILD.log 里是 LLMError，ORIGIN.json 没写，
        #: gen2 与 gen1 逐字节相同）。空串 = 没有 provider 或 provider 完整产出。
        self.provider_status = ""
        #: 源码来源闸门（与 provider 闸门互补）：这一轮里**由上一代补进来**的文件
        #: 路径。空 = 全是新造的。🔴 补件是常态（实测 s12/s15 的 gen1 都是
        #: 5 个新造 + dag.py/bootstrap.py 契约没过回退补），所以判据**不能**是
        #: "一个都不许补"（那会否掉每一次真实运行），只能是
        #: "7 个全补 = 这一代没有一点新造的源码 = 复印件" —— 这在真实中可达：
        #: LLM 端点挂掉时 `_gen_one` 的异常被吞掉返回空，7 个文件全由上一代补，
        #: 拼出来的就是上一代的逐字节副本，而 assemble/自举/自选任务全绿。
        self.fallback_files: List[str] = []
        self.log: List[Dict[str, Any]] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    @property
    def source_ok(self) -> bool:
        """源码来源闸门：这一代是不是**至少有新造的源码**。

        只有"7 个文件全由上一代补"才算复印件 —— 补件本身是常态
        （实测 s12/s15 的 gen1 都是 5 新造 + 2 补）。
        """
        return len(self.fallback_files) < len(FRAME_FILES)

    @property
    def source_why(self) -> str:
        if self.source_ok:
            return ""
        return f"全部 {len(FRAME_FILES)} 个文件都由上一代补，这一代没有新造的源码"

    # ---- 落盘（每步可回看）-----------------------------------------------
    def _save(self, name: str, text: str) -> None:
        self._n += 1
        d = self.run_dir / f"{self._n:02d}_{name}"
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(text, encoding="utf-8", newline="")

    # ---- 入口 ------------------------------------------------------------
    def __call__(self, content: str, step: str = "", **kw: Any) -> str:
        fn = getattr(self, f"_step_{step}", None)
        if fn is None:
            raise KeyError(f"未知 step={step!r}（可用: expand / plan_module / normalize / audit_contract / supplement / assemble / measure / passthrough）")
        out = fn(content, **kw)
        self._save(step, out)
        self.log.append({"step": step, "chars": len(out)})
        return out

    # ---- 1. expand -------------------------------------------------------
    def _step_expand(self, description: str) -> str:
        """把自然语言描述变成逐文件施工单（纯 JSON，不写代码）。"""
        wants = "\n".join(
            f"- {p}：{SOURCE_TASKS.get(p, '')}"
            + (f" 必须提供：{', '.join(_need_for(p))}" if _need_for(p) else "")
            for p in FRAME_FILES
        )
        prompt = (
            "下面是一段对某个标签驱动信息流框架的自然语言描述。请把它翻译成"
            "**逐文件的施工单**，供后续逐步写码使用。\n"
            "只输出 JSON：{\"files\": [{\"path\": ..., \"duty\": 一句话职责, "
            "\"must\": [必须出现的类/函数/方法名], \"notes\": 实现要点(不超过 60 字)}]}\n"
            "不要围栏，不要解释。\n\n"
            f"【框架描述】\n{description}\n\n"
            f"【必须覆盖的文件】\n{wants}\n"
        )
        raw = _strip_fence(_ask(self.llm, prompt, self.cfg))
        briefs: Dict[str, Dict[str, Any]] = {}
        try:
            data = json.loads(raw)
            for item in data.get("files", []):
                p = str(item.get("path") or "")
                if p in FRAME_FILES:
                    briefs[p] = item
        except Exception:
            briefs = {}
        for p in FRAME_FILES:  # 模型漏的文件用静态施工单补齐（否则下游会缺件）
            briefs.setdefault(p, {"path": p, "duty": SOURCE_TASKS.get(p, ""), "must": [], "notes": ""})
        return json.dumps(
            {"description": description, "files": [briefs[p] for p in FRAME_FILES]},
            ensure_ascii=False,
        )

    # ---- 2. plan_module --------------------------------------------------
    def _step_plan_module(self, briefs_json: str) -> str:
        """每个文件一次（或每批一次）请求，写完整源码。并发跑。

        第 2 代起 `self.source_provider` 不是 None，源码由**上一代框架**产出，
        这里就只做归一化与落盘，不再请求模型。
        """
        payload = json.loads(briefs_json)
        description = payload.get("description", "")
        briefs = {item["path"]: item for item in payload.get("files", [])}
        if self.source_provider is not None:
            try:
                got = dict(self.source_provider(description))
            except Exception as exc:  # noqa: BLE001 - 上一代造不出来不能报销整轮
                # 但**必须留下痕迹**：旧版本只往 self.log 记一笔，report 里干干净净，
                # 于是回退出来的上一代副本被当成新的一代验收（s13 gen2 实测）。
                self.provider_status = f"provider 调用失败：{str(exc)[:200]}"
                self.log.append({"step": "source_provider_failed", "why": str(exc)[:200]})
                got = {}
            else:
                # 产出不完整同样算失败：缺的文件会被 supplement 用上一代补，
                # 拼出来的是一套"chimera"，不是这一代造的。
                holes = [p for p in FRAME_FILES if not (got.get(p) or "").strip()]
                self.provider_status = (f"provider 产出不完整，缺/空：{holes}" if holes else "")
            self.log.append({"step": "source_provider", "files": sorted(got),
                             "chars": sum(len(v) for v in got.values()),
                             "ok": not self.provider_status})
            for p in FRAME_FILES:
                got.setdefault(p, "")
            return json.dumps({"files": [{"path": p, "source": got[p]} for p in FRAME_FILES]}, ensure_ascii=False)
        # 🔴 逐文件原子生成（取代原先的"按字节凑批、一批多文件"）：
        # 原先一批产出多个文件的完整源码，单文件动辄 8 万字符，输出被 `MAX_TOKENS`
        # 砍断还静默绕闸；且一批里一个文件坏、整批得重做。现在每个文件一次请求，
        # 产完立即单文件闸门（`contract_violation`：语法 / 顶层符号不许少 / 体积不坍缩），
        # 不过带原因重做（`FILE_RETRIES` 次）。参考实现只给这一个文件（全量 `REF_CHARS`
        # 不再截断），跨文件一致性交给 step 4 `audit_contract` + step 5 `supplement` 兜底。
        got: Dict[str, str] = {}
        for p in FRAME_FILES:
            src, why = "", ""
            for attempt in range(1, FILE_RETRIES + 2):   # 第 1 次 + FILE_RETRIES 次重试
                try:
                    cand = self._gen_one(description, briefs, p, prev_why=why)
                except Exception as exc:  # noqa: BLE001 - 单文件失败不报销整轮
                    self.log.append({"step": "gen_file_failed", "file": p, "why": str(exc)[:200]})
                    cand = ""
                v = contract_violation(p, cand, self.fallback.get(p, ""))
                if not v:
                    src, why = cand, ""
                    break
                why = v
                self.log.append({"step": "gen_file_retry", "file": p,
                                 "attempt": attempt, "why": why[:160]})
            # 全失败留空串：`normalize` 照实往下传，`audit_contract` 判"空产出"，
            # `supplement` 再用上一代补 —— 每一步在产物里都看得见。
            got[p] = src
        return json.dumps({"files": [{"path": p, "source": got[p]} for p in FRAME_FILES]}, ensure_ascii=False)

    def _gen_one(self, description: str, briefs: Dict[str, Dict[str, Any]],
                 p: str, prev_why: str = "") -> str:
        """为**单个文件**写完整源码：一次请求、一个 JSON 对象、只一个文件。

        返回该文件源码字符串（空串表示解析失败）。单文件过闸在 `_step_plan_module`
        里做，这里只负责"按职责 + 参考实现写出这一份"。
        """
        b = briefs.get(p) or {}
        prev = self.fallback.get(p, "")
        item = {
            "path": p,
            "duty": b.get("duty", SOURCE_TASKS.get(p, "")),
            "must": list(dict.fromkeys([*b.get("must", []), *_need_for(p)])),
            "notes": b.get("notes", ""),
            # 参考实现只给这一个文件（全量，不再截断）：不要求照抄，但功能不低于它
            "reference": prev[:REF_CHARS],
        }
        feedback = (
            f"\n\n【上一轮没过单文件闸门，原因：{prev_why}】请据此修正后重出同一文件。"
            if prev_why else ""
        )
        prompt = (
            "你要按给定职责，为**下面这一个文件**写出**完整可用**的 Python 源码。\n"
            "硬性要求：\n"
            "1. 只输出一个 JSON 对象：{\"files\": [{\"path\": ..., \"source\": 完整源码}]}，"
            "里面**只能有这一个文件**；\n"
            "2. 不要 markdown 围栏，不要解释；\n"
            "3. 只用标准库；import 期不做网络/磁盘 IO；\n"
            "4. `must` 列出的名字必须**全部原样定义**在这份源码的顶层"
            "（别的文件会 `from ... import` 它们，名字差一个整套就跑不起来）；\n"
            "5. reference 是上一代同名实现，**功能不得少于它**（可以重写得更短）。\n\n"
            f"【整体描述】\n{description}\n\n"
            f"【本次要写的文件】\n{json.dumps([item], ensure_ascii=False)}\n"
            + feedback
        )
        t0 = time.time()
        raw = _strip_fence(_ask(self.llm, prompt, self.cfg))
        self.log.append({"step": "llm_file", "file": p, "seconds": round(time.time() - t0, 1)})
        try:
            data = json.loads(raw)
            return str((data.get("files") or [{}])[0].get("source") or "")
        except Exception:
            return ""

    # ---- 3. normalize ----------------------------------------------------
    def _step_normalize(self, payload: str) -> str:
        """剥围栏、补换行、记录每个文件的字节数。"""
        data = json.loads(payload)
        files = []
        for item in data.get("files", []):
            src = _strip_fence(item.get("source", ""))
            if src and not src.endswith("\n"):
                src += "\n"
            files.append({"path": item["path"], "source": src, "bytes": len(src.encode("utf-8"))})
        return json.dumps({"files": files}, ensure_ascii=False)

    # ---- 4. audit_contract ----------------------------------------------
    def _step_audit_contract(self, payload: str) -> str:
        """逐文件过契约闸门；不合格的记下原因（由 `supplement` 决定要不要回退）。

        两道闸门：
        1. 单文件契约（`contract_violation`）：语法 / 顶层符号不许少 / 体积不坍缩；
        2. **跨文件契约**（`API_CONTRACT` + `must`）：这个文件承诺提供的名字必须都在。
           实测第 1 代把 `primitives.py` 里的 `ContentItem` 写成 `Entry`，`__init__.py`
           还照样 `from .primitives import Entry` —— 单文件全都"语法合法"，一 import 就炸。
        """
        data = json.loads(payload)
        sources = {item["path"]: item.get("source", "") for item in data.get("files", [])}
        out = []
        for item in data.get("files", []):
            p = item["path"]
            src = item.get("source", "")
            why = contract_violation(p, src, self.fallback.get(p, ""))
            if not why:
                have = set(_tops(src))
                need = sorted(set(API_CONTRACT.get(p, [])) - have)
                if need:
                    why = f"接口契约缺名字 {need[:6]}"
                else:
                    miss = _check_class_methods(p, src)
                    if miss:
                        why = f"类方法契约缺 {miss[:6]}"
                    else:
                        # 第三道闸门：用到但没定义/没导入的名字（运行到才炸的 NameError）
                        und = undefined_names(p, src, sources)
                        if und:
                            why = f"悬空引用 {und[:6]}"
                        else:
                            bad = broken_imports(sources).get(p)
                            if bad:
                                why = f"导入不到 {bad[:6]}"
                            else:
                                # 第四道闸门：跨文件类属性（`result.id`，类里没有这个字段）
                                cattr = cross_class_attrs(sources).get(p)
                                if cattr:
                                    why = f"类属性不存在 {cattr[:6]}"
            out.append({"path": p, "source": src, "ok": not why, "why": why})
        return json.dumps({"files": out}, ensure_ascii=False)

    # ---- 5. supplement ---------------------------------------------------
    def _step_supplement(self, *payloads: str) -> str:
        """多份审计结果汇聚（每批一条边产出）。缺件或不合格的用上一代补。"""
        merged: Dict[str, Dict[str, Any]] = {}
        for payload in payloads:
            if not payload:
                continue
            for item in json.loads(payload).get("files", []):
                merged[item["path"]] = item
        # supplment 每轮跑一次，这里清空即等于"按轮重置"（重试时不累积上次的补件）。
        self.fallback_files = []
        files = []
        for p in FRAME_FILES:
            item = merged.get(p) or {"path": p, "source": "", "ok": False, "why": "缺失"}
            src, ok, why = item["source"], item["ok"], item["why"]
            if not ok and p in self.fallback:
                src, ok, why = self.fallback[p], True, f"回退上一代（{why}）"
                self.fallback_files.append(p)
            files.append({"path": p, "source": src, "ok": ok, "why": why})
        return json.dumps({"files": files}, ensure_ascii=False)

    # ---- 6. assemble -----------------------------------------------------
    def _step_assemble(self, payload: str) -> str:
        """交给**当前这一代框架**的 `assemble()` 解固定点；失败就回退上一代。

        `assemble(parts, spec=...)` 只收敛固定点、不写盘；它自己会用
        `_probe()` 复现一遍，不通过就抛 `RuntimeError`。

        一个补充闸门：`assemble` 只管"这套文件能否复现自己"，**不管它们互相 import
        得上不上**（实测第 1 代就是这样：`Entry` vs `ContentItem`，固定点解得很漂亮，
        一 import 就 ImportError）。所以这里再做一次跨文件导出检查。
        """
        data = json.loads(payload)
        files = {item["path"]: item["source"] for item in data.get("files", [])}
        verdict = {item["path"]: {"ok": item["ok"], "why": item["why"]} for item in data.get("files", [])}
        from vea_v6.parts import bootstrap as B

        bad = missing_exports(files)
        bad.update({p: (bad.get(p, []) + v) for p, v in broken_imports(files).items()})
        try:
            if bad:
                raise RuntimeError(f"跨文件缺导出 {bad}")
            built = B.assemble(files, spec=B.normalize_spec(""))
            ok, why = B.verify_self_reproduction(built[B.MAIN_SOURCE])
            if not ok:
                raise RuntimeError(why)
            verdict["_assemble"] = {"ok": True, "why": why}
        except Exception as exc:  # noqa: BLE001 - 失败要回退，不能整链炸掉
            # 🔴 **整代回退**，不是"哪个文件空了补哪个"。
            # `files.get(p) or self.fallback[p]` 只在源码为空串时才回退 —— 而实测
            # 坏的那一代每个文件都写满了内容，只是彼此 import 不上
            # （`from .llm import LLMRegistry` 而 llm.py 里没这个名）。
            # 于是"装配失败"被记进了 verdict，落盘的却仍是那套互相 import 不上的
            # 代码，下一步 demo 一 import 就炸。装配不过 = 这一代整体不可用。
            built = {p: self.fallback.get(p, "") for p in FRAME_FILES}
            verdict["_assemble"] = {"ok": False, "why": str(exc)[:200]}
        return json.dumps(
            {
                "files": [{"path": p, "source": built[p]} for p in FRAME_FILES],
                "verdict": verdict,
            },
            ensure_ascii=False,
        )

    # ---- 7. measure ------------------------------------------------------
    def _step_measure(self, payload: str) -> str:
        data = json.loads(payload)
        files = data.get("files", [])
        total = sum(len(f["source"].encode("utf-8")) for f in files)
        verdict = data.get("verdict") or {}

        def _parses(src: str) -> bool:
            try:
                ast.parse(src)
                return True
            except SyntaxError:
                return False

        report = {
            "files": len(files),
            "bytes": total,
            # 语法判据就是"能不能 parse"，空文件同样能 parse ⇒ 单独按体积兜一层，
            # 不能写成 `_tops(...) or endswith(...)` 那种只对 __init__ 成立的怪条件。
            "syntax_ok": all(_parses(f["source"]) and len(f["source"].strip()) > 0 for f in files),
            "assemble_ok": bool((verdict.get("_assemble") or {}).get("ok")),
            # 代际闸门：源码是不是上一代真造出来的。assemble_ok 为真**不代表**
            # 这一代是新造的 —— 回退副本同样能装配能自举（s13 gen2 实测：
            # assemble_ok=True、自举 stable、demo ALL OK，实际是 gen1 的复印件）。
            "provider_ok": not self.provider_status,
            "provider_why": self.provider_status,
            # 源码来源闸门：7 个文件全由上一代补 = 这一代没有新造的源码。
            # 补件清单照实写进报告，验收时能看出是"5 新造 + 2 补"还是"0 新造"。
            "source_ok": self.source_ok,
            "source_why": self.source_why,
            "fallback_files": sorted(self.fallback_files),
            "new_files": len(FRAME_FILES) - len(self.fallback_files),
            "rejected": {p: v.get("why") for p, v in verdict.items() if isinstance(v, dict) and not v.get("ok")},
        }
        self._save_file("REPORT.json", json.dumps(report, ensure_ascii=False, indent=2))
        return json.dumps(report, ensure_ascii=False)

    def _save_file(self, name: str, text: str) -> None:
        (self.run_dir / name).write_text(text, encoding="utf-8", newline="")

    # ---- 8. passthrough --------------------------------------------------
    def _step_passthrough(self, payload: str) -> str:
        return payload
