"""VEA V6 核心数据原语模块 (Primitives).

在 VEA V6 中，图不再维护繁琐的命名顶点状态机，而是回归极简的信息流模型：
1. **信息条目 (`ContentItem`)**：
   结构为 `{"tags": list[str], "content": str, "image"?: path|url, "gen": int}`（初始送入时 `gen: 0`）。
   继承自 `dict`，既支持标准字典键访问（`item["tags"]`, `item["content"]`, `item["gen"]`），
   也支持对象属性访问（`item.tags`, `item.content`, `item.image`, `item.gen`）。
2. **处理边 (`Edge`)**：
   三元组结构 `(input_tags: list[str], processor: str | dict | Callable, output_tags: list[str])`。
   其中 `processor`（处理方式）支持：
   - `prompt: str`（提示词字符串）
   - `{"模型": str, "处理prompt": str}` 或 `{"model": str, "prompt": str}`（指定模型与处理提示词的字典）
   - `Callable`（Python 可调用函数，便于确定性转换与测试）
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union


def normalize_tags(tags: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """将标签输入规范化为去重保序的 `list[str]`。"""
    if tags is None:
        return []
    if isinstance(tags, str):
        cleaned = tags.strip()
        return [cleaned] if cleaned else []
    result: List[str] = []
    for t in tags:
        s = str(t).strip()
        if s and s not in result:
            result.append(s)
    return result


def content_to_str(val: Any) -> str:
    """将任意内容安全转换为字符串。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


class ContentItem(dict):
    """V6 标准信息条目对象.

    结构：`{"tags": list[str], "content": str, "image"?: path|url, "gen": 0}`
    同时兼容 `dict` 键访问与属性访问。
    """

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

        resolved_tags = normalize_tags(tags if tags is not None else raw.get("tags", []))
        resolved_content = content_to_str(content if content is not None else raw.get("content", ""))
        resolved_image = image if image is not None else raw.get("image")
        resolved_gen = int(gen if gen is not None else raw.get("gen", 0))

        self["tags"] = resolved_tags
        self["content"] = resolved_content
        if resolved_image is not None and str(resolved_image).strip() != "":
            self["image"] = str(resolved_image).strip()
        self["gen"] = resolved_gen

        # 保留其它额外扩展字段（如有）
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
        """检查当前信息是否包含指定 tag。"""
        return tag in self["tags"]

    def matches_tags(self, required_tags: Sequence[str]) -> bool:
        """检查当前信息的 tags 是否包含 `required_tags` 中的全部标签。"""
        req = normalize_tags(required_tags)
        if not req:
            return False
        my_tags = set(self["tags"])
        return all(t in my_tags for t in req)

    def to_dict(self) -> Dict[str, Any]:
        """导出为标准字典。"""
        out: Dict[str, Any] = {
            "tags": list(self["tags"]),
            "content": self["content"],
        }
        if "image" in self and self["image"] is not None:
            out["image"] = self["image"]
        out["gen"] = int(self["gen"])
        for k, v in self.items():
            if k not in out and k != "image" and not str(k).startswith("_"):
                out[k] = v
        return out


ProcessorType = Union[str, Dict[str, Any], Callable[..., Any]]


class Edge:
    """VEA V6 处理边.

    每条边仅由四个核心要素构成：
    1. `input_tags`: 输入的 tag `list[str]`（处理边不允许为空）
    2. `processor`: 处理方式（`prompt: str` | `{"模型": ..., "处理prompt": ...}` | `Callable`）
    3. `output_tags`: 输出的 tag `list[str]`（处理边不允许为空）
    4. `layer`: 执行层级 `int`，`dag.run()` 总是从 layer 小的边开始跑起
    """

    def __init__(
        self,
        input_tags: Optional[Union[str, Sequence[str]]] = None,
        processor: Optional[ProcessorType] = None,
        output_tags: Optional[Union[str, Sequence[str]]] = None,
        *,
        layer: int = 0,
        is_preset: bool = False,
        edge_id: Optional[str] = None,
    ) -> None:
        self.input_tags: List[str] = normalize_tags(input_tags)
        self.processor: ProcessorType = processor if processor is not None else ""
        self.output_tags: List[str] = normalize_tags(output_tags)
        self.layer: int = int(layer)
        self.is_preset: bool = is_preset
        self.id: str = edge_id or ("preset_edge" if is_preset else "")

    @classmethod
    def from_any(cls, spec: Union["Edge", Tuple[Any, ...], List[Any], Dict[str, Any]]) -> "Edge":
        """从 Edge 实例、三元组/四元组 `(input_tags, processor, output_tags[, layer])` 或字典构造 `Edge`。"""
        if isinstance(spec, Edge):
            return spec
        if isinstance(spec, (tuple, list)):
            if len(spec) not in (3, 4):
                raise ValueError(
                    f"边定义元组必须包含 3~4 个元素 (input_tags, processor, output_tags[, layer])，实际长度: {len(spec)}"
                )
            return cls(
                input_tags=spec[0],
                processor=spec[1],
                output_tags=spec[2],
                layer=int(spec[3]) if len(spec) == 4 else 0,
            )
        if isinstance(spec, dict):
            in_tags = spec.get("input_tags", spec.get("inputs", spec.get("in_tags", [])))
            proc = spec.get("processor", spec.get("prompt", spec.get("action", "")))
            out_tags = spec.get("output_tags", spec.get("outputs", spec.get("out_tags", [])))
            return cls(
                input_tags=in_tags,
                processor=proc,
                output_tags=out_tags,
                layer=int(spec.get("layer", 0)),
                is_preset=bool(spec.get("is_preset", False)),
                edge_id=spec.get("id"),
            )
        raise TypeError(f"无法将类型 {type(spec).__name__} 解析为 V6 Edge")

    @property
    def model(self) -> Optional[str]:
        """提取处理方式中指定的模型名称（若 `processor` 为字典）。"""
        if isinstance(self.processor, dict):
            return (
                self.processor.get("模型")
                or self.processor.get("model")
                or self.processor.get("model_name")
            )
        return None

    @property
    def prompt(self) -> str:
        """提取处理方式中的处理 prompt 文本。"""
        if isinstance(self.processor, str):
            return self.processor
        if isinstance(self.processor, dict):
            p = (
                self.processor.get("处理prompt")
                or self.processor.get("prompt")
                or self.processor.get("process_prompt")
                or self.processor.get("instruction")
                or ""
            )
            return str(p)
        if callable(self.processor):
            return getattr(self.processor, "__name__", "callable_processor")
        return str(self.processor or "")

    @property
    def fn(self) -> Optional[Callable[..., Any]]:
        """如果处理方式是 Python 可调用函数（或字典中携带了 `fn`），返回该函数。"""
        if callable(self.processor):
            return self.processor
        if isinstance(self.processor, dict) and callable(self.processor.get("fn")):
            return self.processor["fn"]
        return None

    def render_prompt(self, item: Union[ContentItem, Dict[str, Any]]) -> str:
        """将输入信息条目渲染进当前边的处理 prompt 中。"""
        content_str = content_to_str(item.get("content", ""))
        tags_str = ", ".join(item.get("tags", []))
        image_str = str(item.get("image", "") or "")
        template = self.prompt or "请处理以下信息：\n{content}"

        rendered = template
        used_placeholder = False
        for placeholder, val in (
            ("{content}", content_str),
            ("{input}", content_str),
            ("{tags}", tags_str),
            ("{image}", image_str),
        ):
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, val)
                if placeholder in ("{content}", "{input}"):
                    used_placeholder = True

        if image_str and image_str not in rendered:
            rendered = f"{rendered}\n[附带图片: {image_str}]"
        if not used_placeholder:
            rendered = f"{rendered}\n\n输入数据：\n{content_str}"
        return rendered

    def render_pair_prompt(
        self,
        item1: Union[ContentItem, Dict[str, Any]],
        item2: Union[ContentItem, Dict[str, Any]],
    ) -> str:
        """用于预设边（`preset_edge`）：将两条信息条目拼凑渲染为合并提示词。"""
        c1 = content_to_str(item1.get("content", ""))
        c2 = content_to_str(item2.get("content", ""))
        t1 = ", ".join(item1.get("tags", []))
        t2 = ", ".join(item2.get("tags", []))
        template = self.prompt or (
            "请将以下两条信息拼凑整合为一条完整信息：\n"
            "信息1 [{tags_1}]: {content_1}\n"
            "信息2 [{tags_2}]: {content_2}"
        )
        if any(k in template for k in ("{content_1}", "{content_2}", "{input_1}", "{input_2}")):
            return (
                template.replace("{content_1}", c1)
                .replace("{content_2}", c2)
                .replace("{input_1}", c1)
                .replace("{input_2}", c2)
                .replace("{tags_1}", t1)
                .replace("{tags_2}", t2)
            )
        return f"{template}\n\n信息1 [{t1}]：\n{c1}\n\n信息2 [{t2}]：\n{c2}"

    def __iter__(self) -> Iterator[Any]:
        """支持三元组解包：`input_tags, processor, output_tags = edge`。"""
        yield self.input_tags
        yield self.processor
        yield self.output_tags

    def __len__(self) -> int:
        return 3

    def __getitem__(self, key: Union[int, str]) -> Any:
        """同时支持按下标 `[0, 1, 2]` 访问与按字典键 `['input_tags', 'processor', 'output_tags']` 访问。"""
        if isinstance(key, int):
            return (self.input_tags, self.processor, self.output_tags)[key]
        if key in ("input_tags", "inputs", "in_tags"):
            return self.input_tags
        if key in ("processor", "処理方式", "处理方式"):
            return self.processor
        if key in ("output_tags", "outputs", "out_tags"):
            return self.output_tags
        if key in ("model", "模型"):
            return self.model
        if key in ("prompt", "处理prompt"):
            return self.prompt
        if key == "is_preset":
            return self.is_preset
        if key == "layer":
            return self.layer
        raise KeyError(key)

    def to_tuple(self) -> Tuple[List[str], ProcessorType, List[str]]:
        return (list(self.input_tags), self.processor, list(self.output_tags))

    def to_dict(self) -> Dict[str, Any]:
        proc_serialized: Any
        if isinstance(self.processor, dict):
            proc_serialized = {
                k: (getattr(v, "__name__", str(v)) if callable(v) else v)
                for k, v in self.processor.items()
            }
        elif callable(self.processor):
            proc_serialized = getattr(self.processor, "__name__", "callable")
        else:
            proc_serialized = self.processor

        return {
            "input_tags": list(self.input_tags),
            "processor": proc_serialized,
            "output_tags": list(self.output_tags),
            "layer": self.layer,
        }

    def __repr__(self) -> str:
        return f"Edge({self.input_tags!r}, {self.processor!r}, {self.output_tags!r}, layer={self.layer})"
