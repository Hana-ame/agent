"""VEA V6 数据原语：`ContentItem`（信息条目）与 `Edge`（处理边）。

V6 不再维护命名顶点状态机，只保留两条基本结构：

1. `ContentItem` = `{"tags": list[str], "content": str, "image"?: path|url, "gen": int}`
2. `Edge` = `(input_tags, processor, output_tags, layer)`

`processor` 支持三种形态：

- `str`：提示词模板（`{content}` 占位，用 `Edge.render_prompt` 渲染）
- `dict`：`{"模型": KEY, "处理prompt": ...}`，可附加内联连接配置 `api/key/model/params`
- `Callable`：确定性 Python 函数（用于测试与纯本地转换）
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "normalize_tags",
    "content_to_str",
    "ContentItem",
    "Edge",
    "ProcessorType",
    "merge_payload",
]


def normalize_tags(tags: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """把标签输入规范化为去重保序的 `list[str]`。"""
    if tags is None:
        return []
    if isinstance(tags, str):
        cleaned = tags.strip()
        return [cleaned] if cleaned else []
    out: List[str] = []
    for t in tags:
        s = str(t).strip()
        if s and s not in out:
            out.append(s)
    return out


def content_to_str(val: Any) -> str:
    """把任意内容安全转成字符串；`dict/list` 用 JSON 表示。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def merge_payload(c1: str, c2: str) -> str:
    """把两段内容合成一段"下游可直接当数据用"的正文。

    - 都是 JSON 且都是对象 → 合并成一个对象（`{"a":1}` + `{"b":2}` ⇒ `{"a":1,"b":2}`）；
    - 否则按行拼接（保留各自的原始文本，不下判断、不做语义压缩）。
    """
    try:
        d1, d2 = json.loads(c1), json.loads(c2)
    except (json.JSONDecodeError, TypeError):
        return c1 + "\n" + c2
    if isinstance(d1, dict) and isinstance(d2, dict) and not (set(d1) & set(d2)):
        merged = dict(d1)
        merged.update(d2)
        return json.dumps(merged, ensure_ascii=False)
    return c1 + "\n" + c2


class ContentItem(dict):
    """V6 标准信息条目，`dict` 子类，同时支持键访问与属性访问。"""

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], "ContentItem"]] = None,
        *,
        tags: Optional[Union[str, Sequence[str]]] = None,
        content: Optional[Any] = None,
        image: Optional[str] = None,
        gen: Optional[int] = None,
    ) -> None:
        super().__init__()
        raw = dict(data) if isinstance(data, dict) else {}
        self["tags"] = normalize_tags(tags if tags is not None else raw.get("tags", []))
        self["content"] = content_to_str(content if content is not None else raw.get("content", ""))
        resolved_image = image if image is not None else raw.get("image")
        if resolved_image is not None and str(resolved_image).strip() != "":
            self["image"] = str(resolved_image).strip()
        self["gen"] = int(gen if gen is not None else raw.get("gen", 0))
        for k, v in raw.items():
            if k not in ("tags", "content", "image", "gen"):
                self[k] = v

    @property
    def tags(self) -> List[str]:
        return self["tags"]

    @tags.setter
    def tags(self, value: Union[str, Sequence[str]]) -> None:
        self["tags"] = normalize_tags(value)

    @property
    def content(self) -> str:
        return self["content"]

    @content.setter
    def content(self, value: Any) -> None:
        self["content"] = content_to_str(value)

    @property
    def image(self) -> Optional[str]:
        return self.get("image")

    @image.setter
    def image(self, value: Optional[str]) -> None:
        if value is None or str(value).strip() == "":
            self.pop("image", None)
        else:
            self["image"] = str(value).strip()

    @property
    def gen(self) -> int:
        return int(self.get("gen", 0))

    @gen.setter
    def gen(self, value: int) -> None:
        self["gen"] = int(value)

    def has_tag(self, tag: str) -> bool:
        return tag in self["tags"]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def __repr__(self) -> str:
        preview = self["content"]
        if len(preview) > 60:
            preview = preview[:57] + "..."
        return f"ContentItem(tags={self['tags']}, gen={self['gen']}, content={preview!r})"


ProcessorType = Union[str, Dict[str, Any], Callable[..., Any]]

#: processor 字典里被视为「内联连接配置」的键
_INLINE_KEYS = ("api", "base_url", "key", "api_key", "model", "params", "参数")
_INLINE_ALIASES = {"参数": "params"}


class Edge:
    """V6 处理边 `(input_tags, processor, output_tags, layer)`。"""

    def __init__(
        self,
        input_tags: Optional[Union[str, Sequence[str]]] = None,
        processor: Optional[ProcessorType] = None,
        output_tags: Optional[Union[str, Sequence[str]]] = None,
        layer: int = 0,
        *,
        edge_id: str = "",
        is_preset: bool = False,
    ) -> None:
        self.input_tags: List[str] = normalize_tags(input_tags)
        self.processor: ProcessorType = processor if processor is not None else ""
        self.output_tags: List[str] = normalize_tags(output_tags)
        self.layer = int(layer)
        self.id = str(edge_id or "")
        self.is_preset = bool(is_preset)
        self.last_prompt: Optional[str] = None

    # ---- 解包（三元组语义）------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        return iter((self.input_tags, self.processor, self.output_tags))

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Any:
        return (self.input_tags, self.processor, self.output_tags)[index]

    # ---- processor 派生属性 ---------------------------------------------
    @property
    def prompt(self) -> str:
        """处理提示词（`str` processor 或字典里的 `处理prompt`/`prompt`）。"""
        if isinstance(self.processor, str):
            return self.processor
        if isinstance(self.processor, dict):
            for key in ("处理prompt", "prompt", "处理提示词"):
                val = self.processor.get(key)
                if isinstance(val, str) and val:
                    return val
        return ""

    @property
    def model(self) -> Optional[str]:
        """边引用的模型 KEY 或内联模型名；没有则返回 `None`。"""
        if isinstance(self.processor, dict):
            val = self.processor.get("模型") or self.processor.get("model")
            return str(val) if val else None
        return None

    @property
    def fn(self) -> Optional[Callable[..., Any]]:
        return self.processor if callable(self.processor) else None

    @property
    def tool(self) -> Optional[str]:
        """边引用的**工具名**（`{"工具": 名字}` / `{"tool": 名字}`）；没有则 `None`。

        工具由 `DAG.tools` 提供，是图自身的机制：同一个 DAG 只要换掉 `tools`
        里挂的框架，就能拿上一代的产出再跑一次下一代。
        """
        if isinstance(self.processor, dict):
            for key in ("工具", "tool"):
                val = self.processor.get(key)
                if isinstance(val, str) and val:
                    return val
        return None

    @property
    def tool_args(self) -> Dict[str, Any]:
        """传给工具的参数（`{"工具": 名字, "参数": {...}}` 里的那部分）。"""
        if isinstance(self.processor, dict):
            for key in ("参数", "args", "kwargs"):
                val = self.processor.get(key)
                if isinstance(val, dict):
                    return dict(val)
        return {}

    @property
    def inline_llm_config(self) -> Optional[Dict[str, Any]]:
        """提取边内联的连接配置；没有任何内联键时返回 `None`（表示走注册表）。

        `{"模型": KEY, "参数": {...}}` 这种「引用注册表 + 覆盖参数」的写法也算内联，
        但 `model` 字段只在它**不是注册表 KEY**时才作为模型名覆盖（见 `DAG._resolve_model_config`）。
        """
        if not isinstance(self.processor, dict):
            return None
        found: Dict[str, Any] = {}
        for key in _INLINE_KEYS:
            if key in self.processor and self.processor[key] not in (None, ""):
                found[_INLINE_ALIASES.get(key, key)] = self.processor[key]
        # 统一同义键：内联配置的规范形态是 `{base_url, api_key, model, params}`
        if "api" in found and "base_url" not in found:
            found["base_url"] = found.pop("api")
        if "key" in found and "api_key" not in found:
            found["api_key"] = found.pop("key")
        return found or None

    def render_prompt(self, item: "ContentItem") -> str:
        """把信息条目渲染进提示词模板（`{content}`/`{tags}` 占位，未知占位原样保留）。

        模板里没有内容占位时，把条目内容以 **`输入数据：`** 段追加在末尾 ——
        这是"提示词模板 + 实际数据"的 V6 约定，很多下游（含测试与真实模型）都靠它取原始数据。
        """
        template = self.prompt or "请处理以下信息：\n{content}"
        mapping = {
            "content": item.content,
            "input": item.content,
            "tags": ", ".join(item.tags),
            "gen": str(item.gen),
            "image": item.image or "",
        }
        out = template
        used_placeholder = False
        for k, v in mapping.items():
            key = "{" + k + "}"
            if key in out:
                out = out.replace(key, v)
                if k in ("content", "input"):
                    used_placeholder = True
        if item.image and item.image not in out:
            out = f"{out}\n[附带图片: {item.image}]"
        if not used_placeholder:
            out = f"{out}\n\n输入数据：\n{item.content}"
        self.last_prompt = out
        return out

    def render_pair_prompt(self, item1: "ContentItem", item2: "ContentItem") -> str:
        """预设边专用：把两条信息拼成一条合并提示词。

        模板里没写 `{content_1}` 这类占位时，两条内容以 **`信息1 [tags]：\\n内容`** 直接追加，
        并在末尾保留 **`输入数据：`** 段 —— 下游 lambda / 模型都依赖这个分隔符取原始 JSON，
        删掉它会让"合并后的数据"变成一句自然语言（历史回归点，勿再简化）。
        """
        c1, c2 = item1.content, item2.content
        t1, t2 = ", ".join(item1.tags), ", ".join(item2.tags)
        template = self.prompt
        if any(k in template for k in ("{content_1}", "{content_2}", "{input_1}", "{input_2}")):
            return (
                template.replace("{content_1}", c1)
                .replace("{content_2}", c2)
                .replace("{input_1}", c1)
                .replace("{input_2}", c2)
                .replace("{tags_1}", t1)
                .replace("{tags_2}", t2)
            )
        joined = merge_payload(c1, c2)
        return (
            template
            + f"\n\n信息1 [{t1}]：\n{c1}"
            + f"\n\n信息2 [{t2}]：\n{c2}"
            + f"\n\n输入数据：\n{joined}"
        )

    # ---- 序列化 ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        proc: Any = self.processor
        if callable(proc):
            proc = {"__callable__": getattr(proc, "__name__", "anonymous")}
        data: Dict[str, Any] = {
            "input_tags": list(self.input_tags),
            "processor": proc,
            "output_tags": list(self.output_tags),
            "layer": self.layer,
        }
        if self.id:
            data["id"] = self.id
        if self.is_preset:
            data["is_preset"] = True
        return data

    def to_tuple(self) -> Tuple[List[str], ProcessorType, List[str], int]:
        return (list(self.input_tags), self.processor, list(self.output_tags), self.layer)

    @classmethod
    def from_any(cls, spec: Any) -> "Edge":
        """从 `Edge` / 四元组 / 三元组 / 字典构造边。"""
        if isinstance(spec, Edge):
            return spec
        if isinstance(spec, dict):
            processor = spec.get("processor", spec.get("处理方式", ""))
            return cls(
                spec.get("input_tags", spec.get("inputs", [])),
                processor,
                spec.get("output_tags", spec.get("outputs", [])),
                int(spec.get("layer", 0)),
                edge_id=str(spec.get("id", "")),
                is_preset=bool(spec.get("is_preset", False)),
            )
        if isinstance(spec, (tuple, list)):
            layer = int(spec[3]) if len(spec) > 3 else 0
            return cls(spec[0], spec[1], spec[2], layer)
        raise TypeError(f"无法把 {spec!r} 解析为 Edge")

    def __repr__(self) -> str:
        return f"Edge({self.input_tags} -> {self.output_tags}, layer={self.layer})"
