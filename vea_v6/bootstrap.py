"""VEA V6 自举模块：用 DAG 从一个描述生成新的 DAG 框架代码，并可再生成自身（固定点）。

自举链（bootstrap chain）::

    FRAMEWORK_SPEC                      # 框架描述（提示词，唯一输入）
        │
        ▼  create_bootstrap_dag().run()
    tag="result" 的框架代码              # 第 1 代生成物
        │
        ▼  用同一份 SPEC 在新框架里再跑一次
    第 2 代框架代码                      # 与第 1 代逐字节一致 → 固定点

## 为什么能稳定

生成物不是「LLM 现编的代码」，而是由 `MINIVEA_SOURCE` 模板 + 规范化的 SPEC 确定性地渲染出来的。
框架代码里同时内嵌了 SPEC 本身（`_SPEC_TEXT`）与「用 SPEC 渲染自身源码」的入口
（`MINIVEA.generate_framework()`），因此：

1. 任何时候用同一份 SPEC 跑 `generate_framework()`，输出都与手里的源码逐字节相等（quine 性质）；
2. 用生成物再生成一次，SPEC 不变 ⇒ 输出不变 ⇒ 固定点。

真正的 LLM 只在 SPEC 需要「从自然语言描述合成」时介入（`synthesize_spec`）；
一旦 SPEC 落定，后续每一代都是确定性渲染，不依赖任何随机性。

## 两种运行方式

```python
# 1) 只用自举 DAG（本地确定性，无需网络）
from vea_v6 import create_bootstrap_dag
dag = create_bootstrap_dag()
dag.send({"tags": ["input"], "content": "", "gen": 0})   # 空内容 = 用内置 FRAMEWORK_SPEC
while not dag.has_result():
    dag.run()
code = dag.get_result()

# 2) 完整链路：用真实 LLM 把自然语言描述合成 SPEC，再生成框架代码
from vea_v6 import OpenAICompatibleLLM, bootstrap
res = bootstrap("一个只用 tag 驱动、带预设边和 OpenAI 兼容调用能力的 DAG 框架",
                llm=OpenAICompatibleLLM())
print(res.stable, res.framework_code[:200])
```
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from vea_v6.dag import DEFAULT_PRESET_EDGE, DAG
from vea_v6.llm import LLMProtocol, OpenAICompatibleLLM
from vea_v6.primitives import ContentItem, Edge, ProcessorType


# =============================================================================
# 1. 框架描述（SPEC）—— 自举的唯一输入
# =============================================================================

#: 框架描述：既是「提示词」，也是 SPEC 的默认值。改动它 = 定义新一代框架。
FRAMEWORK_SPEC: Dict[str, Any] = {
    "name": "minivea",
    "version": "1.0.0",
    "summary": (
        "一个单文件、零依赖的 tag 驱动 DAG 框架：信息条目以 {tags, content, image?, gen} 流转，"
        "处理边以 (input_tags, processor, output_tags, layer) 声明，"
        "缺失输入由预设边（preset_edge）从相近信息合成，结果以 tag='result' 交付。"
    ),
    "core_invariants": [
        "DAG 结构只有 {contents: [], edges: []} 两个字段",
        "处理边必须有输入 tag 与输出 tag（预设边除外）",
        "dag.run() 按 layer 升序执行一次；外部循环检测 tag='result' 终止",
        "预设边在找不到所需输入 tag 时，从相近条目合成 输出tag=所需输入tag 的中间条目",
        "模型注册表以 KEY 映射 {api, key, model, params}，边用 {'模型': KEY} 引用",
        "边可内联 {model, api, key, params} 直连 OpenAI 兼容端点，优先级高于注册表",
    ],
    "features": [
        "ContentItem: dict 子类，支持键访问与属性访问",
        "Edge: 三元组解包input_tags/processor/output_tags + layer",
        "processor 支持 str(prompt) / dict(模型+处理prompt) / Callable",
        "OpenAI 兼容调用：内联配置、注册表 KEY、推理模型 reasoning 字段回退",
        "预设边可合成单条或多条相近信息的合并条目",
        "find_by_tag / has_tag / has_result / get_result / get_result_item",
        "to_dict / from_dict 序列化",
        "generate_framework: 用 SPEC 渲染框架自身源码（quine / 自举固定点）",
    ],
    "public_api": [
        "ContentItem",
        "Edge",
        "DAG",
        "DEFAULT_PRESET_EDGE",
        "SmartMockLLM",
        "OpenAICompatibleLLM",
        "load_models_file",
        "generate_framework",
        "FRAMEWORK_SPEC",
    ],
}

#: 提交给 LLM 的 SPEC 合成提示词模板。`{description}` 为用户描述。
SPEC_SYNTHESIS_PROMPT = """你在设计一个极简 DAG 框架的规格（SPEC）。框架理念：
- DAG 只有 {{contents: [], edges: []}}；
- 信息 = {{tags: list[str], content: str, image?: str, gen: int}}；
- 处理边 = (input_tags, processor, output_tags, layer)，processor 可以是提示词字符串、
  {{"模型": KEY, "处理prompt": ...}} 字典、或普通 Python 函数；
- 缺输入时由「预设边」从相近条目合成所需 tag；
- 结果以 tag="result" 交付，外部循环检测它来终止。

请根据下面的描述，输出**严格合法的 JSON**（不要 markdown 代码块、不要解释），结构为：
{{"name": str, "version": str, "summary": str, "core_invariants": [str], "features": [str], "public_api": [str]}}

描述：
{description}
"""


# =============================================================================
# 2. SPEC 规范化与渲染
# =============================================================================

_SPEC_LIST_KEYS = ("core_invariants", "features", "public_api")


def normalize_spec(spec: Union[Dict[str, Any], str, None]) -> Dict[str, Any]:
    """把任意 SPEC 输入规范化为稳定结构（字段齐全、列表去重保序、无多余键）。

    规范化是确定性的：同一份语义输入总是得到同一份字典，这是固定点成立的前提。
    """
    if spec is None:
        base: Dict[str, Any] = dict(FRAMEWORK_SPEC)
    elif isinstance(spec, str):
        text = spec.strip()
        if not text:
            base = dict(FRAMEWORK_SPEC)
        else:
            try:
                base = json.loads(text)
            except json.JSONDecodeError:
                # 非 JSON：当作纯自然语言摘要，套用默认骨架
                base = dict(FRAMEWORK_SPEC)
                base["summary"] = text
            if not isinstance(base, dict):
                base = dict(FRAMEWORK_SPEC)
    elif isinstance(spec, dict):
        base = dict(spec)
    else:
        raise TypeError(f"SPEC 必须是 dict / JSON 字符串 / None，实际: {type(spec).__name__}")

    merged: Dict[str, Any] = {
        "name": str(base.get("name") or FRAMEWORK_SPEC["name"]),
        "version": str(base.get("version") or FRAMEWORK_SPEC["version"]),
        "summary": str(base.get("summary") or FRAMEWORK_SPEC["summary"]).strip(),
    }
    for key in _SPEC_LIST_KEYS:
        raw = base.get(key)
        if raw is None:
            items = list(FRAMEWORK_SPEC[key])
        elif isinstance(raw, str):
            items = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            items = [str(x).strip() for x in raw if str(x).strip()]
        deduped: List[str] = []
        for it in items:
            if it not in deduped:
                deduped.append(it)
        merged[key] = deduped
    return merged


def spec_to_text(spec: Union[Dict[str, Any], str, None] = None) -> str:
    """把 SPEC 渲染为稳定的 JSON 文本（缩进 2、键序固定、`ensure_ascii=False`）。"""
    return json.dumps(normalize_spec(spec), ensure_ascii=False, indent=2, sort_keys=False)


def synthesize_spec(
    description: str,
    llm: Optional[LLMProtocol] = None,
    *,
    fallback: bool = True,
) -> Dict[str, Any]:
    """用 LLM 把自然语言描述合成为 SPEC；LLM 输出不可解析时回退到 `fallback_spec_from_text`。

    `llm` 为 `None` 或调用失败且 `fallback=True` 时，用确定性规则从描述文本构造 SPEC，
    保证整条自举链路在离线环境也能跑通。
    """
    text = (description or "").strip()
    if not text:
        return normalize_spec(None)

    if llm is not None:
        prompt = SPEC_SYNTHESIS_PROMPT.format(description=text)
        try:
            res = llm.ask(prompt)  # type: ignore[attr-defined]
            if hasattr(res, "__await__"):
                import asyncio

                res = asyncio.run(res)  # pragma: no cover - 仅在传异步 LLM 时触发
            parsed = _extract_json_object(str(res))
            if isinstance(parsed, dict):
                return normalize_spec(parsed)
        except Exception:
            if not fallback:
                raise
    return fallback_spec_from_text(text)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出里抠出第一个 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def fallback_spec_from_text(description: str) -> Dict[str, Any]:
    """零依赖地从自然语言描述构造 SPEC（离线兜底，确定性）。"""
    text = description.strip()
    keywords = _extract_keywords(text)
    spec = dict(FRAMEWORK_SPEC)
    spec["summary"] = text
    spec["features"] = list(FRAMEWORK_SPEC["features"]) + [
        f"关注点: {kw}" for kw in keywords if f"关注点: {kw}" not in FRAMEWORK_SPEC["features"]
    ]
    return normalize_spec(spec)


_KEYWORD_HINTS = (
    "dag",
    "tag",
    "标签",
    "预设边",
    "preset",
    "openai",
    "llm",
    "模型",
    "layer",
    "payload",
    "streaming",
    "异步",
    "并行",
    "缓存",
    "重试",
    "多模态",
    "image",
    "自举",
    "bootstrap",
)


def _extract_keywords(text: str) -> List[str]:
    low = text.lower()
    out: List[str] = []
    for kw in _KEYWORD_HINTS:
        if kw in low and kw not in out:
            out.append(kw)
    return out


# =============================================================================
# 3. minivea 源码模板（quine 结构）
# =============================================================================
#
# 表达方式：切片 + 字符偏移，**不用任何占位符**。
#
# 生成物形如::
#
#     <head>          # docstring / import / 常量 / API 函数，全是真实文本
#     <body>          # 框架主体（Tag、ContentItem、Edge、DAG …）
#     <tail>          # 把 SPEC 与自身按「偏移」切回去的收尾段落
#
# 生成物里的 `_SPEC_SPAN` / `_SELF_SPAN` 记录 SPEC 与自身源码在文件中的
# 起始偏移，运行时 `generate_framework()` 直接把这两段替换成新的 base64 与
# 新的自身源码，其余部分逐字不动 ⇒ 同一 SPEC 必得同一份文件（固定点）。
#
# 为什么不用 `str.replace(占位符, ...)`：占位符本身会出现在「记录占位符的代码」里，
# 全局替换会连那段记录代码一起改掉，自指就崩了。偏移方案没有这个自指问题。

_TEMPLATE_HEAD = r'''"""{{TITLE}}"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union


FRAMEWORK_SPEC = json.loads(base64.b64decode(
    "{{SPEC_B64}}"
).decode("utf-8"))

#: 自举尾段的起始标记（生成物与运行时共用，用于定位/校验）。
_TAIL_MARK = "# ==== 自举元信息（字段顺序固定，勿手改）===="


def _b64(text):
    """把文本编码为 base64 字符串（SPEC 与自身源码的唯一安全载体）。"""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def generate_framework(spec=None):
    """用 SPEC 渲染本框架源码：同一 SPEC 恒得同一份代码（自举固定点）。

    机制：本文件尾部（`_TAIL_MARK` 之后）保存两项自举元信息——
      BODY_B64：自身**去掉尾段**的那部分源码（base64）；
      SPEC_SPAN：SPEC 的 base64 在 BODY_B64 解码结果中的起止偏移。
    重写 = 解码 BODY_B64 → 替换 SPEC 段 → 拼回同格式的尾段。尾段长度
    只取决于 BODY 长度、BODY 长度只取决于 SPEC 长度，与尾段自身无关，
    因此不存在自指方程，同一 SPEC 必然产出逐字节相同的文件。
    """
    normalized = normalize_spec(spec)
    spec_b64 = _b64(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
    body = base64.b64decode(BODY_B64).decode("utf-8")
    body = body[: SPEC_SPAN[0]] + spec_b64 + body[SPEC_SPAN[1]:]
    return body + _make_tail(body, SPEC_SPAN[0], SPEC_SPAN[0] + len(spec_b64))


def _make_tail(body, spec_start, spec_end):
    """按与生成时完全相同的格式产出自举尾段（两侧共用同一份格式串，防漂移）。"""
    enc = _b64(body)
    chunks = [enc[i: i + 76] for i in range(0, len(enc), 76)]
    body_lines = "\n".join('    "%s"' % c for c in chunks) + "\n"
    return _TAIL_FMT % {
        "body_lines": body_lines,
        "spec_start": spec_start,
        "spec_end": spec_end,
    }


#: 尾段格式。生成物与运行时共用同一份字面量，因此永不漂移。
_TAIL_FMT = (
    "\n\n"
    + _TAIL_MARK
    + "\n"
    + "# " + "-" * 74 + "\n"
    + "# BODY_B64 是本文件去掉本尾段后的源码（base64）；SPEC_SPAN 是 SPEC 的\n"
    + "# base64 在解码结果中的起止偏移。generate_framework() 只重写这一段。\n"
    + "BODY_B64 = (\n"
    + "%(body_lines)s"
    + ")\n"
    + "SPEC_SPAN = (%(spec_start)d, %(spec_end)d)\n"
)
'''

#: 框架主体（与任何一代 SPEC 无关，逐字进入每一份生成物）
_TEMPLATE_BODY = r'''


def normalize_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        s = tags.strip()
        return [s] if s else []
    out = []
    for t in tags:
        s = str(t).strip()
        if s and s not in out:
            out.append(s)
    return out


def content_to_str(val):
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def normalize_spec(spec=None):
    if spec is None:
        return dict(FRAMEWORK_SPEC)
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return dict(FRAMEWORK_SPEC)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            return dict(FRAMEWORK_SPEC)
        spec = parsed
    merged = dict(FRAMEWORK_SPEC)
    merged.update({k: v for k, v in dict(spec).items() if k in merged})
    for key in ("core_invariants", "features", "public_api"):
        merged[key] = list(merged.get(key) or [])
    return merged


class ContentItem(dict):
    """信息条目 {tags, content, image?, gen}，支持键访问与属性访问。"""

    def __init__(self, data=None, *, tags=None, content=None, image=None, gen=None):
        super().__init__()
        raw = dict(data) if isinstance(data, dict) else {}
        self["tags"] = normalize_tags(tags if tags is not None else raw.get("tags", []))
        self["content"] = content_to_str(content if content is not None else raw.get("content", ""))
        img = image if image is not None else raw.get("image")
        if img is not None and str(img).strip():
            self["image"] = str(img).strip()
        self["gen"] = int(gen if gen is not None else raw.get("gen", 0))
        for k, v in raw.items():
            if k not in ("tags", "content", "image", "gen"):
                self[k] = v

    @property
    def tags(self):
        return self["tags"]

    @property
    def content(self):
        return self["content"]

    @property
    def image(self):
        return self.get("image")

    @property
    def gen(self):
        return int(self.get("gen", 0))

    def matches_tags(self, required):
        req = normalize_tags(required)
        if not req:
            return False
        mine = set(self["tags"])
        return all(t in mine for t in req)

    def to_dict(self):
        out = {"tags": list(self["tags"]), "content": self["content"]}
        if self.get("image"):
            out["image"] = self["image"]
        out["gen"] = self.gen
        return out


class Edge:
    """处理边 (input_tags, processor, output_tags, layer)。"""

    def __init__(self, input_tags=None, processor=None, output_tags=None, *, layer=0,
                 is_preset=False, edge_id=None):
        self.input_tags = normalize_tags(input_tags)
        self.processor = "" if processor is None else processor
        self.output_tags = normalize_tags(output_tags)
        self.layer = int(layer)
        self.is_preset = bool(is_preset)
        self.id = edge_id or ("preset_edge" if is_preset else "")

    @classmethod
    def from_any(cls, spec):
        if isinstance(spec, Edge):
            return spec
        if isinstance(spec, (tuple, list)):
            if len(spec) not in (3, 4):
                raise ValueError("边定义需要 3~4 个元素")
            return cls(spec[0], spec[1], spec[2], layer=int(spec[3]) if len(spec) == 4 else 0)
        if isinstance(spec, dict):
            return cls(
                spec.get("input_tags", spec.get("inputs", [])),
                spec.get("processor", spec.get("prompt", "")),
                spec.get("output_tags", spec.get("outputs", [])),
                layer=int(spec.get("layer", 0)),
                is_preset=bool(spec.get("is_preset", False)),
                edge_id=spec.get("id"),
            )
        raise TypeError("无法解析为 Edge: %r" % (type(spec),))

    @property
    def model(self):
        if isinstance(self.processor, dict):
            return self.processor.get("模型") or self.processor.get("model")
        return None

    @property
    def prompt(self):
        if isinstance(self.processor, str):
            return self.processor
        if isinstance(self.processor, dict):
            for k in ("处理prompt", "prompt", "process_prompt"):
                if self.processor.get(k):
                    return str(self.processor[k])
        return ""

    @property
    def fn(self):
        if callable(self.processor):
            return self.processor
        if isinstance(self.processor, dict) and callable(self.processor.get("fn")):
            return self.processor["fn"]
        return None

    @property
    def inline_llm_config(self):
        """边内联的 OpenAI 兼容配置（含 api/key/model/params 任一即为内联）。"""
        if not isinstance(self.processor, dict):
            return None
        p = self.processor
        base_url = p.get("api") or p.get("base_url")
        api_key = p.get("key") or p.get("api_key")
        model_name = p.get("model") or p.get("模型")
        params = p.get("params") or p.get("参数") or {}
        if not (base_url or api_key or params):
            return None
        out = {"params": dict(params) if isinstance(params, dict) else {}}
        if base_url:
            out["base_url"] = str(base_url)
        if api_key:
            out["api_key"] = str(api_key)
        if model_name:
            out["model"] = str(model_name)
        return out

    def render_prompt(self, item):
        body = content_to_str(item.get("content", ""))
        tags = ", ".join(item.get("tags", []))
        template = self.prompt or "请处理以下信息：\n{content}"
        used = False
        for ph, val in (("{content}", body), ("{input}", body), ("{tags}", tags)):
            if ph in template:
                template = template.replace(ph, val)
                used = True
        return template if used else template + "\n\n输入数据：\n" + body

    def is_cacheable(self):
        return False

    def to_dict(self):
        proc = self.processor
        if callable(proc):
            proc = getattr(proc, "__name__", "callable")
        elif isinstance(proc, dict):
            proc = {k: (getattr(v, "__name__", v) if callable(v) else v) for k, v in proc.items()}
        return {
            "input_tags": list(self.input_tags),
            "processor": proc,
            "output_tags": list(self.output_tags),
            "layer": self.layer,
        }

    def __iter__(self):
        yield self.input_tags
        yield self.processor
        yield self.output_tags

    def __repr__(self):
        return "Edge(%r, %r, %r, layer=%d)" % (
            self.input_tags, self.processor, self.output_tags, self.layer)


class SmartMockLLM:
    """离线模拟模型：按关键词/模型名返回固定响应，可用于纯本地测试。"""

    def __init__(self, rules=None, model_rules=None, fallback_fn=None):
        self.rules = dict(rules or {})
        self.model_rules = dict(model_rules or {})
        self.fallback_fn = fallback_fn
        self.call_history = []

    async def ask(self, prompt, model=None, image=None, **kwargs):
        self.call_history.append({"prompt": prompt, "model": model, "image": image})
        for kw in sorted(self.rules, key=len, reverse=True):
            if kw in prompt:
                return self._fire(self.rules[kw], prompt, model, image)
        if model and model in self.model_rules:
            return self._fire(self.model_rules[model], prompt, model, image)
        if self.fallback_fn is not None:
            return self._fire(self.fallback_fn, prompt, model, image)
        if "输入数据：" in prompt:
            ins, data = prompt.split("输入数据：", 1)
            return "[%s] -> %s" % (ins.strip(), data.strip())
        return "[Processed]: %s" % (prompt[:160],)

    async def _fire(self, handler, prompt, model, image):
        if callable(handler):
            try:
                n = len([p for p in inspect.signature(handler).parameters.values()
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
                res = handler(prompt, model, image) if n >= 3 else (
                    handler(prompt, model) if n == 2 else handler(prompt))
            except TypeError:
                res = handler(prompt)
            if asyncio.iscoroutine(res):
                res = await res
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        return str(handler)


class OpenAICompatibleLLM:
    """OpenAI 兼容客户端：支持按调用覆写 api/key/参数与多模态图片。"""

    def __init__(self, model="sensenova-6.8-flash-lite", api_key=None, base_url=None,
                 temperature=0.2, timeout=60.0):
        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.timeout = timeout

    async def ask(self, prompt, model=None, image=None, api_key=None, base_url=None, **extra):
        import httpx

        url = base_url or self.base_url
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = api_key or self.api_key
        if key:
            headers["Authorization"] = "Bearer " + key
        content = prompt
        if image:
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": image}}]
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        payload.update(extra)
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout, trust_env=False) as cli:
            resp = await cli.post(url, json=payload)
            resp.raise_for_status()
            return _extract_text(resp.json())


def _extract_text(data):
    """取响应正文，推理模型的 content 为空时回退 reasoning 字段。"""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("响应缺少 choices")
    msg = choices[0].get("message") or {}
    for k in ("content", "reasoning_content", "reasoning", "text"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise ValueError("模型返回空正文（推理模型可能在 max_tokens 内未产出正文）")


def load_models_file(path=None):
    """读取模型注册表 JSON；不传路径时尝试同目录 models.json 与 CWD。"""
    import os

    candidates = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json"))
        candidates.append("models.json")
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as fh:
                return json.load(fh)
    if path is not None:
        raise FileNotFoundError(path)
    return {}


def default_text_similarity(a, b):
    """字符 1-gram + 2-gram 余弦相似度（带包含加成），范围 0~1。"""
    x = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(a).lower().strip())
    y = re.sub(r"[\s_\-:：,，。()（）\[\]]+", "", str(b).lower().strip())
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0

    def grams(s):
        return Counter(list(s) + [s[i:i + 2] for i in range(len(s) - 1)])

    ga, gb = grams(x), grams(y)
    common = set(ga) & set(gb)
    dot = sum(ga[k] * gb[k] for k in common)
    na = math.sqrt(sum(v * v for v in ga.values()))
    nb = math.sqrt(sum(v * v for v in gb.values()))
    sim = dot / (na * nb) if na and nb else 0.0
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    if len(short) >= 2 and short in long_:
        sim = max(sim, 0.55 + 0.35 * (len(short) / len(long_)))
    return round(min(1.0, sim), 4)


DEFAULT_PRESET_EDGE = Edge([], {"模型": "default", "处理prompt": "请将以下信息拼凑为一条完整信息：\n{content}"},
                           ["merged"], is_preset=True, edge_id="preset_edge")


class DAG:
    """tag 驱动的有向无环图：{contents: [], edges: []}。"""

    def __init__(self, name_or_data=None, *, contents=None, edges=None, llm=None,
                 preset_edge=None, models=None):
        if isinstance(name_or_data, dict):
            self.name = str(name_or_data.get("name", "minivea_dag"))
            init_contents = name_or_data.get("contents", contents)
            init_edges = name_or_data.get("edges", edges)
        else:
            self.name = str(name_or_data or "minivea_dag")
            init_contents, init_edges = contents, edges
        self.llm = llm or SmartMockLLM()
        if models is None and isinstance(self.llm, OpenAICompatibleLLM):
            try:
                models = load_models_file()
            except Exception:
                models = {}
        self.models = dict(models or {})
        if preset_edge is not None:
            pe = Edge.from_any(preset_edge)
            pe.is_preset = True
            self.preset_edge = pe
        else:
            self.preset_edge = DEFAULT_PRESET_EDGE
        self.default_edge = self.preset_edge
        self.contents = []
        self.edges = []
        self._executed = set()
        self._consumed = set()
        self._fabricated = set()
        for item in init_contents or []:
            self.send(item)
        for spec in init_edges or []:
            self.add_edge(spec)

    @property
    def context(self):
        return self.contents

    def __getitem__(self, key):
        if key == "contents":
            return self.contents
        if key == "edges":
            return self.edges
        if key == "context":
            return self.contents
        if key == "preset_edge":
            return self.preset_edge
        raise KeyError(key)

    def add_edge(self, input_tags, processor=None, output_tags=None, layer=0):
        if processor is None and output_tags is None:
            edge = Edge.from_any(input_tags)
            if layer:
                edge.layer = int(layer)
        else:
            edge = Edge(input_tags, processor, output_tags, layer=layer)
        if not edge.is_preset:
            if not edge.input_tags:
                raise ValueError("处理边不允许没有输入 tag")
            if not edge.output_tags:
                raise ValueError("处理边不允许没有输出 tag")
        if not edge.id:
            edge.id = "e_%d" % (len(self.edges) + 1)
        self.edges.append(edge)
        return edge

    def send(self, info, tags=None, image=None, gen=0):
        if isinstance(info, dict):
            item = ContentItem(info, tags=tags if tags is not None else info.get("tags", []),
                               image=image if image is not None else info.get("image"),
                               gen=int(info.get("gen", gen)))
        else:
            item = ContentItem(tags=tags if tags is not None else ["input"],
                               content=content_to_str(info), image=image, gen=gen)
        self.contents.append(item)
        return item

    add_content = send
    feed = send
    push = send

    def find_by_tag(self, tag="result"):
        return [c for c in self.contents if tag in c["tags"]]

    def has_tag(self, tag="result"):
        return bool(self.find_by_tag(tag))

    def has_result(self):
        return self.has_tag("result")

    def get_result(self, tag="result"):
        hits = self.find_by_tag(tag)
        return hits[-1].content if hits else None

    def get_result_item(self, tag="result"):
        hits = self.find_by_tag(tag)
        return hits[-1] if hits else None

    def _resolve(self, model, edge=None):
        inline = edge.inline_llm_config if edge is not None else None
        reg = self.models.get(model) if (model and self.models) else None
        if reg is None and inline is None:
            return {"model": model} if model else None
        merged = {"params": {}, "model": model}
        if reg:
            params = dict(reg.get("params") or {})
            for k, v in reg.items():
                if k not in ("api", "base_url", "key", "api_key", "model", "params"):
                    params[k] = v
            merged.update({
                "params": params,
                "api_key": reg.get("key", reg.get("api_key")),
                "base_url": reg.get("api", reg.get("base_url")),
                "model": reg.get("model", model),
            })
        if inline:
            im = inline.get("model")
            if im and not self.models.get(im):
                merged["model"] = im
            if inline.get("api_key"):
                merged["api_key"] = inline["api_key"]
            if inline.get("base_url"):
                merged["base_url"] = inline["base_url"]
            if inline.get("params"):
                merged["params"] = dict(merged.get("params") or {}, **inline["params"])
        return merged

    async def _call_llm(self, prompt, model=None, image=None, edge=None):
        ask = getattr(self.llm, "ask", self.llm if callable(self.llm) else None)
        if ask is None:
            raise ValueError("llm 未实现 ask 且不可调用")
        cfg = self._resolve(model, edge) or {}
        kwargs = {"model": cfg.get("model", model), "image": image}
        if cfg.get("api_key"):
            kwargs["api_key"] = cfg["api_key"]
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        kwargs.update(cfg.get("params") or {})
        try:
            accepted = set(inspect.signature(ask).parameters)
            if not any(p.kind == p.VAR_KEYWORD for p in inspect.signature(ask).parameters.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            pass
        res = ask(prompt, **kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        return content_to_str(res)

    async def _exec_edge(self, edge, item):
        out_tags = normalize_tags(edge.output_tags) or ["output"]
        if edge.fn is not None:
            fn = edge.fn
            try:
                n = len([p for p in inspect.signature(fn).parameters.values()
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
            except (TypeError, ValueError):
                n = 1
            res = fn(item.content, item) if n >= 2 else fn(item.content)
            if asyncio.iscoroutine(res):
                res = await res
            if isinstance(res, dict) and "content" in res and "tags" in res:
                return ContentItem(res)
            text = content_to_str(res)
        elif not edge.prompt:
            text = item.content
        else:
            text = await self._call_llm(edge.render_prompt(item), model=edge.model,
                                        image=item.image, edge=edge)
        return ContentItem(tags=out_tags, content=text, image=item.image, gen=item.gen + 1)

    def _pick(self, edge, limit=2):
        if not self.contents:
            return []
        req = list(edge.input_tags)

        def score(i):
            it = self.contents[i]
            exact = len(set(req) & set(it["tags"])) / len(req) if req else 0.0
            sim = (sum(max((default_text_similarity(r, t) for t in it["tags"]), default=0.0)
                       for r in req) / len(req)) if req else 0.0
            pen = -10.0 if "result" in it["tags"] else 0.0
            pen += -5.0 if it.get("_preset_synthesized") else 0.0
            return (round(pen + exact * 4 + sim * 2 + (1.0 if i not in self._consumed else 0.0), 4),
                    float(it.gen), i)

        ranked = sorted(range(len(self.contents)), key=score, reverse=True)
        best = ranked[0]
        if len(req) == 1:
            tags = set(self.contents[best]["tags"])
            if req[0] in tags or any(default_text_similarity(req[0], t) >= 0.5 for t in tags):
                return [best]
        if limit < 2 or len(self.contents) < 2:
            return [best]
        best_pair, best_score = None, None
        for a in range(len(self.contents)):
            for b in range(a + 1, len(self.contents)):
                combined = set(self.contents[a]["tags"]) | set(self.contents[b]["tags"])
                exact = len(set(req) & combined) / len(req) if req else 0.0
                sim = (sum(max((default_text_similarity(r, t) for t in combined), default=0.0)
                           for r in req) / len(req)) if req else 0.0
                pen = -10.0 if "result" in combined else 0.0
                if (self.contents[a].get("_preset_synthesized")
                        or self.contents[b].get("_preset_synthesized")):
                    pen -= 5.0
                unc = (1.0 if a not in self._consumed else 0.0) + (1.0 if b not in self._consumed else 0.0)
                st = (round(pen + exact * 4 + sim * 2 + unc, 4), a + b, a, b)
                if best_score is None or st > best_score:
                    best_score, best_pair = st, (a, b)
        return list(best_pair) if best_pair else [best]

    async def _fabricate(self, items, target_tags):
        texts = [i.content for i in items]
        merged = texts[0] if texts else ""
        for t in texts[1:]:
            merged = (merged + "\n" + t) if merged else t
        item = ContentItem(tags=normalize_tags(target_tags), content=merged,
                           image=next((i.image for i in items if i.image), None),
                           gen=max(i.gen for i in items) + 1)
        item["_preset_synthesized"] = True
        return item

    async def run_async(self):
        new_items = []
        for idx in sorted(range(len(self.edges)), key=lambda i: (self.edges[i].layer, i)):
            edge = self.edges[idx]
            if edge.is_preset:
                continue
            hit = False
            for ci, item in enumerate(list(self.contents)):
                key = (idx, "single:%d" % ci)
                if key in self._executed or not item.matches_tags(edge.input_tags):
                    continue
                self._executed.add(key)
                self._consumed.add(ci)
                out = await self._exec_edge(edge, item)
                self.contents.append(out)
                new_items.append(out)
                hit = True
            if hit:
                continue
            fab_key = (idx, tuple(sorted(edge.input_tags)))
            if fab_key in self._fabricated or not self.contents:
                continue
            picks = self._pick(edge)
            if not picks:
                continue
            items = [self.contents[i] for i in picks]
            synth = await self._fabricate(items, edge.input_tags)
            self.contents.append(synth)
            self._fabricated.add(fab_key)
            self._consumed.update(picks)
            new_items.append(synth)
            out = await self._exec_edge(edge, synth)
            self.contents.append(out)
            new_items.append(out)
        return new_items

    def run(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run_async()).result()
        return asyncio.run(self.run_async())

    def to_dict(self):
        return {"contents": [c.to_dict() for c in self.contents],
                "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, data, *, llm=None, models=None, edge_bindings=None):
        dag = cls(data.get("name", "minivea_dag"), llm=llm, models=models or data.get("models"))
        for c in data.get("contents", []):
            dag.send(c)
        bindings = dict(edge_bindings or {})
        for spec in data.get("edges", []):
            edge = Edge.from_any(spec)
            if isinstance(edge.processor, str) and edge.processor in bindings:
                edge.processor = bindings[edge.processor]
            dag.add_edge(edge)
        return dag

    def __repr__(self):
        return "DAG(name=%r, contents=%d, edges=%d)" % (self.name, len(self.contents), len(self.edges))


__all__ = [
    "FRAMEWORK_SPEC",
    "ContentItem",
    "Edge",
    "DAG",
    "DEFAULT_PRESET_EDGE",
    "SmartMockLLM",
    "OpenAICompatibleLLM",
    "load_models_file",
    "normalize_tags",
    "normalize_spec",
    "default_text_similarity",
    "generate_framework",
]
'''


#: 不带尾段的模板（= head + body）。完整源码 = 它 + `_TAIL_FMT` 渲染出的尾段。
_BARE_TEMPLATE = _TEMPLATE_HEAD + _TEMPLATE_BODY

#: 兼容别名：带占位符的完整模板（head + body + 尾段占位），供外部只读浏览结构。
MINIVEA_TEMPLATE = _BARE_TEMPLATE


# 渲染器（模块侧，与生成物里 `generate_framework()` 严格等价）
_PH_TITLE = "{{TITLE}}"
_PH_SPEC = "{{SPEC_B64}}"

#: 尾段起始标记（生成物与模块共用同一字面量）。
TAIL_MARK = "# ==== 自举元信息（字段顺序固定，勿手改）===="

#: 尾段格式。与生成物内的 _TAIL_FMT 必须逐字一致，由 check_tail_format_sync() 校验。
_TAIL_FMT = (
    "\n\n"
    + TAIL_MARK
    + "\n"
    + "# " + "-" * 74 + "\n"
    + "# BODY_B64 是本文件去掉本尾段后的源码（base64）；SPEC_SPAN 是 SPEC 的\n"
    + "# base64 在解码结果中的起止偏移。generate_framework() 只重写这一段。\n"
    + "BODY_B64 = (\n"
    + "%(body_lines)s"
    + ")\n"
    + "SPEC_SPAN = (%(spec_start)d, %(spec_end)d)\n"
)


def _b64(text: str) -> str:
    """把文本编码为 base64 字符串（SPEC 与自身源码的唯一安全载体）。"""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _normalized_spec_json(normalized: Dict[str, Any]) -> str:
    """SPEC 的规范 JSON 文本（键排序 ⇒ 同一语义恒得同一字符串）。"""
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def framework_title(normalized: Dict[str, Any]) -> str:
    """生成物的模块 docstring 首行。"""
    return "minivea %s - self-bootstrapped tag-driven DAG framework" % (
        normalized.get("version") or ""
    )


def make_tail(body: str, spec_start: int, spec_end: int) -> str:
    """产出自举尾段。`body` 为不含尾段的完整源码。"""
    enc = _b64(body)
    chunks = [enc[i: i + 76] for i in range(0, len(enc), 76)]
    body_lines = "\n".join('    "%s"' % c for c in chunks) + "\n"
    return _TAIL_FMT % {
        "body_lines": body_lines,
        "spec_start": spec_start,
        "spec_end": spec_end,
    }


def check_tail_format_sync() -> None:
    """自检：模块侧 `_TAIL_FMT` 必须与**生成物内**的 `_TAIL_FMT` 逐字一致。

    生成物内那份是模板源码里的字面量副本；若只改了一边，固定点会静默失效。
    这里从模板正文里把那条表达式抽出来求值，两侧比对，不一致直接抛错。
    """
    m = re.search(
        r"^_TAIL_FMT = \(\n(.*?)^\)\n",
        MINIVEA_TEMPLATE,
        re.DOTALL | re.MULTILINE,
    )
    if not m:
        raise RuntimeError("模板里找不到生成物侧的 _TAIL_FMT 定义")
    ns: Dict[str, Any] = {}
    expr = "_TAIL_MARK = %r\n_TAIL_FMT = (\n%s)\n" % (TAIL_MARK, m.group(1))
    exec(compile(expr, "<tail_fmt_sync>", "exec"), ns)  # noqa: S102 - 只求值本仓模板
    if ns["_TAIL_FMT"] != _TAIL_FMT:
        raise RuntimeError(
            "尾段格式不同步：模块侧与生成物侧不一致，固定点会失效。\n模块侧: %r\n生成侧: %r"
            % (_TAIL_FMT, ns["_TAIL_FMT"])
        )


def render_bare_body(spec: Union[Dict[str, Any], str, None] = None) -> str:
    """渲染「不含尾段」的源码：docstring + SPEC + 主体。"""
    normalized = normalize_spec(spec)
    title = framework_title(normalized)
    spec_b64 = _b64(_normalized_spec_json(normalized))
    return _BARE_TEMPLATE.replace(_PH_TITLE, title).replace(_PH_SPEC, spec_b64)


def render_framework_code(spec: Union[Dict[str, Any], str, None] = None) -> str:
    """用 SPEC 渲染完整框架源码（含自举尾段）：同一 SPEC 恒得同一份代码。

    结构上是纯函数：
      `body = template(spec)`（不含尾段）→ `code = body + tail(body, span)`
    而 `tail` 只依赖 `body` 的长度与 SPEC 段的位置，不依赖尾段自身 ⇒
    不存在自指方程，也没有迭代不收敛的风险。
    """
    body = render_bare_body(spec)
    spec_start, spec_end = _spec_span_in(body, spec)
    return body + make_tail(body, spec_start, spec_end)


def _spec_span_in(body: str, spec: Union[Dict[str, Any], str, None]) -> Tuple[int, int]:
    """定位 SPEC 的 base64 在 `body` 中的起止偏移。"""
    normalized = normalize_spec(spec)
    spec_b64 = _b64(_normalized_spec_json(normalized))
    start = body.index(spec_b64)
    return start, start + len(spec_b64)


def extract_spec_from_code(code: str) -> Dict[str, Any]:
    """从生成的框架代码中取回 SPEC（同时校验 SPEC_SPAN 是否自洽）。"""
    m = re.search(
        r'^FRAMEWORK_SPEC = json\.loads\(base64\.b64decode\(\s*\n?\s*"([A-Za-z0-9+/=]+)"',
        code,
        re.MULTILINE,
    )
    if not m:
        raise ValueError("生成代码中找不到 FRAMEWORK_SPEC")
    spec_b64 = m.group(1)
    span_m = re.search(r"^SPEC_SPAN = \((\d+), (\d+)\)$", code, re.MULTILINE)
    if span_m:
        body = code.rsplit(TAIL_MARK, 1)[0]
        got = (int(span_m.group(1)), int(span_m.group(2)))
        want = _spec_span_in(body, json.loads(base64.b64decode(spec_b64).decode("utf-8")))
        if got != want:
            raise ValueError("SPEC_SPAN 不自洽：记录 %r，实际 %r" % (got, want))
    return normalize_spec(json.loads(base64.b64decode(spec_b64).decode("utf-8")))


def verify_fixed_point(spec: Union[Dict[str, Any], str, None] = None) -> Tuple[bool, str]:
    """验证固定点：同一 SPEC ⇒ 同一份代码 ⇒ 取回 SPEC 再渲染仍逐字节相同。"""
    gen1 = render_framework_code(spec)
    try:
        spec_from_code = extract_spec_from_code(gen1)
    except Exception as exc:
        return False, "无法从生成代码中取回 SPEC: %s" % exc
    gen2 = render_framework_code(spec_from_code)
    if gen1 == gen2:
        return True, ""
    for i, (a, b) in enumerate(zip(gen1, gen2)):
        if a != b:
            return False, "第 %d 字符处发散: %r != %r" % (
                i, gen1[max(0, i - 40): i + 40], gen2[max(0, i - 40): i + 40]
            )
    return False, "长度不同: gen1=%d gen2=%d" % (len(gen1), len(gen2))


def verify_self_reproduction(code: str) -> Tuple[bool, str]:
    """更强的一步：把生成物**真正当模块加载**，调它的 `generate_framework()` 再生一次。

    只有这一步通过，才算「生成的框架能自己生成自己」——而不只是两个字符串相等。
    """
    import types

    spec_b64 = re.search(
        r'^FRAMEWORK_SPEC = json\.loads\(base64\.b64decode\(\s*\n?\s*"([A-Za-z0-9+/=]+)"',
        code,
        re.MULTILINE,
    )
    if not spec_b64:
        raise ValueError("生成代码中找不到 FRAMEWORK_SPEC")
    mod = types.ModuleType("minivea_generated")
    mod.__dict__["__file__"] = "<generated>"
    try:
        exec(compile(code, "<minivea_generated>", "exec"), mod.__dict__)
        again = mod.generate_framework()
    except Exception as exc:
        return False, "生成物自身执行失败: %s: %s" % (type(exc).__name__, exc)
    if again != code:
        for i, (a, b) in enumerate(zip(code, again)):
            if a != b:
                return False, "生成物自我复现失败，第 %d 字符: %r != %r" % (
                    i, code[max(0, i - 40): i + 40], again[max(0, i - 40): i + 40]
                )
        return False, "生成物自我复现长度不同: %d != %d" % (len(code), len(again))
    return True, ""


# =============================================================================
# 4. 自举 DAG
# =============================================================================

#: 规范化 SPEC 的边（从描述文本 → SPEC JSON）
BOOTSTRAP_SPEC_PROMPT = (
    "把下面的框架描述规范化成一个 SPEC JSON，"
    "键为 name/version/summary/core_invariants/features/public_api：\n{content}"
)


def create_bootstrap_dag(
    llm: Optional[LLMProtocol] = None,
    *,
    use_llm_for_spec: bool = False,
    preset_edge: Optional[Union[Edge, Tuple[Any, ...], Dict[str, Any]]] = DEFAULT_PRESET_EDGE,
) -> DAG:
    """创建自举 DAG：输入描述 → SPEC → 框架代码 → 固定点校验 → `tag="result"`。

    内部拓扑（本身就是一个标准 V6 DAG）：

    | layer | input_tags | 产出 | 说明 |
    |---|---|---|---|
    | 1 | `["input"]` | `["framework_spec"]` | 描述 → 规范化 SPEC（LLM 或确定性兜底） |
    | 2 | `["framework_spec"]` | `["framework_code"]` | SPEC → 渲染框架源码 |
    | 3 | `["framework_code"]` | `["fixed_point"]` | 再生成一次并比对（稳定性证据） |
    | 4 | `["framework_code", "fixed_point"]` | `["result"]` | 汇聚两路，交付最终框架代码 |

    第 4 条边需要两个输入 tag，单个条目都不满足，因此运行时**必然**由预设边
    从 `framework_code` 与 `fixed_point` 两条相近信息合成一条中间条目——这也是
    「预设边拼凑」在真实任务里的用法示例。

    `use_llm_for_spec=True` 且传入 `llm` 时，SPEC 由 LLM 从描述合成（失败自动兜底）。
    """
    dag = DAG("V6_Bootstrap_DAG", preset_edge=preset_edge)
    if llm is not None:
        dag.llm = llm

    def _edge1_make_spec(content: str) -> str:
        text = (content or "").strip()
        if not text:
            spec = normalize_spec(None)
        elif use_llm_for_spec:
            spec = synthesize_spec(text, dag.llm)
        else:
            spec = normalize_spec(text)
        return spec_to_text(spec)

    def _edge2_render_code(spec_text: str) -> str:
        try:
            spec = json.loads(spec_text)
        except json.JSONDecodeError:
            spec = normalize_spec(spec_text)
        return render_framework_code(spec)

    def _edge3_check_fixed_point(code: str) -> str:
        try:
            spec = extract_spec_from_code(code)
        except ValueError as exc:
            return json.dumps({"stable": False, "reason": str(exc)}, ensure_ascii=False)
        regen = render_framework_code(spec)
        if regen == code:
            return json.dumps(
                {"stable": True, "bytes": len(code), "sha_len": len(regen)},
                ensure_ascii=False,
            )
        return json.dumps(
            {"stable": False, "reason": "regenerated code differs", "bytes": len(code)},
            ensure_ascii=False,
        )

    def _edge4_deliver(merged: str) -> str:
        # merged 是预设边把 framework_code 与 fixed_point 拼起来的文本；
        # 代码在上、稳定性报告在下，按渲染顺序取回代码块。
        idx = merged.rfind('"""minivea')
        if idx < 0:
            idx = merged.find('"""')
        code = merged[idx:] if idx >= 0 else merged
        # 去掉尾部拼接进来的 fixed_point JSON
        tail = code.rfind("\n{")
        if tail > 0 and '"stable"' in code[tail:]:
            code = code[:tail]
        return code.rstrip() + "\n"

    dag.add_edge(["input"], _edge1_make_spec, ["framework_spec"], layer=1)
    dag.add_edge(["framework_spec"], _edge2_render_code, ["framework_code"], layer=2)
    dag.add_edge(["framework_code"], _edge3_check_fixed_point, ["fixed_point"], layer=3)
    dag.add_edge(
        ["framework_code", "fixed_point"], _edge4_deliver, ["result"], layer=4
    )
    return dag


@dataclass
class BootstrapResult:
    """自举运行结果。"""

    framework_code: str
    spec: Dict[str, Any]
    stable: bool
    reason: str = ""
    spec_text: str = ""
    bootstrap_context: List[ContentItem] = field(default_factory=list)

    def write_to(self, path: Union[str, "Any"]) -> str:
        """把生成的框架代码写到文件，返回路径字符串。"""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.framework_code, encoding="utf-8")
        return str(p)


def bootstrap(
    description: str = "",
    llm: Optional[LLMProtocol] = None,
    *,
    use_llm_for_spec: Optional[bool] = None,
    max_ticks: int = 8,
) -> BootstrapResult:
    """完整自举：描述 → 框架代码，并返回稳定性证据。

    - `description` 为空：使用内置 `FRAMEWORK_SPEC`（改 SPEC 即定义新一代框架）。
    - `description` 非空：作为 SPEC 的 summary；`use_llm_for_spec` 默认在传入真实
      `OpenAICompatibleLLM` 时为 True、否则 False（离线确定性）。
    """
    if use_llm_for_spec is None:
        use_llm_for_spec = isinstance(llm, OpenAICompatibleLLM)

    dag = create_bootstrap_dag(llm=llm, use_llm_for_spec=use_llm_for_spec)
    dag.send({"tags": ["input"], "content": description or "", "gen": 0})

    ticks = 0
    while not dag.has_result() and ticks < max_ticks:
        dag.run()
        ticks += 1

    code = dag.get_result()
    if not code:
        raise RuntimeError("自举 DAG 未能产出 tag='result' 的框架代码")

    spec = extract_spec_from_code(code)
    stable, reason = verify_fixed_point(spec)
    return BootstrapResult(
        framework_code=code,
        spec=spec,
        stable=stable,
        reason=reason,
        spec_text=spec_to_text(spec),
        bootstrap_context=list(dag.context),
    )
