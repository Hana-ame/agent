"""Data schemas and JSON persistence for DAG Agent (`models.json`, `contents.json`, `edges.json`).

High-cohesion data layer with zero execution logic.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union


def normalize_tags(tags: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalize a tag or sequence of tags into a deduplicated, order-preserving list of strings."""
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


def serialize_content(value: Any) -> str:
    """Convert arbitrary payload into a clean string (JSON for dict/list)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


@dataclass
class ModelConfig:
    """Represents a single entry in `models.json`: `{key: {endpoint, apikey, ...other_params}}`."""

    key: str
    endpoint: str = "https://api.openai.com/v1"
    apikey: str = ""
    model: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> "ModelConfig":
        raw = dict(data)
        endpoint = (
            raw.pop("endpoint", None)
            or raw.pop("api", None)
            or raw.pop("base_url", None)
            or "https://api.openai.com/v1"
        )
        apikey = (
            raw.pop("apikey", None)
            or raw.pop("api_key", None)
            or raw.pop("key", None)
            or ""
        )
        model_name = str(raw.pop("model", None) or key)
        explicit_params = dict(raw.pop("params", None) or {})
        # Any remaining keys in `{endpoint, apikey, other params}` are merged into params
        other_params = {k: v for k, v in raw.items() if not str(k).startswith("_")}
        merged_params = {**other_params, **explicit_params}
        return cls(
            key=key,
            endpoint=str(endpoint),
            apikey=str(apikey),
            model=model_name,
            params=merged_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "apikey": self.apikey,
            "model": self.model,
        }
        if self.params:
            out["params"] = dict(self.params)
        return out


class ModelRegistry(dict):
    """Manages `models.json` (`{key: {endpoint, apikey, other params}}`)."""

    def __init__(self, models: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.path: Optional[Path] = Path(path) if path else None
        if self.path and self.path.exists() and models is None:
            self.load(self.path)
        elif models:
            self.load_dict(models)

    def load_dict(self, raw: Dict[str, Any]) -> None:
        self.clear()
        for k, v in raw.items():
            if str(k).startswith("_") or not isinstance(v, dict):
                continue
            self[k] = ModelConfig.from_dict(k, v)

    def load(self, path: Union[str, Path]) -> "ModelRegistry":
        p = Path(path)
        self.path = p
        data = json.loads(p.read_text(encoding="utf-8"))
        self.load_dict(data)
        return self

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path else self.path
        if not target:
            raise ValueError("No file path specified for saving models.json")
        payload = {k: v.to_dict() if isinstance(v, ModelConfig) else v for k, v in self.items()}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def register(self, key: str, endpoint: str, apikey: str = "", model: str = "", **other_params: Any) -> ModelConfig:
        cfg = ModelConfig(key=key, endpoint=endpoint, apikey=apikey, model=model or key, params=other_params)
        self[key] = cfg
        return cfg

    def resolve(self, key: Optional[str]) -> Optional[ModelConfig]:
        if not key or key == "default":
            if "default" in self:
                val = self["default"]
                return val if isinstance(val, ModelConfig) else ModelConfig.from_dict("default", val)
            if self:
                first_cfg = next(iter(self.values()))
                return first_cfg if isinstance(first_cfg, ModelConfig) else ModelConfig.from_dict(next(iter(self.keys())), first_cfg)
            return None
        if key in self:
            val = self[key]
            return val if isinstance(val, ModelConfig) else ModelConfig.from_dict(key, val)
        return next(iter(self.values()), None) if self else None

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: (v.to_dict() if isinstance(v, ModelConfig) else dict(v)) for k, v in self.items()}


@dataclass
class ToolSpec:
    """Represents a tool definition registered in `tools.json`."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Union[str, Callable[..., Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ToolSpec":
        raw = dict(data)
        tool_name = str(raw.pop("name", name) or name)
        desc = str(raw.pop("description", "") or raw.pop("desc", ""))
        params = dict(raw.pop("parameters", {}) or raw.pop("params", {}))
        handler = raw.pop("handler", None) or raw.pop("fn", None)
        return cls(name=tool_name, description=desc, parameters=params, handler=handler, extra=raw)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if isinstance(self.handler, str):
            out["handler"] = self.handler
        elif callable(self.handler):
            out["handler"] = getattr(self.handler, "__name__", "custom_callable")
        if self.extra:
            out.update(self.extra)
        return out


class ToolRegistry(dict):
    """Manages `tools.json` (`{tool_name: {name, description, parameters, handler}}`)."""

    def __init__(self, tools: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.path: Optional[Path] = Path(path) if path else None
        if self.path and self.path.exists() and tools is None:
            self.load(self.path)
        elif tools:
            self.load_dict(tools)

    def load_dict(self, raw: Dict[str, Any]) -> None:
        self.clear()
        for k, v in raw.items():
            if str(k).startswith("_") or not isinstance(v, dict):
                continue
            self[k] = ToolSpec.from_dict(k, v)

    def load(self, path: Union[str, Path]) -> "ToolRegistry":
        p = Path(path)
        self.path = p
        data = json.loads(p.read_text(encoding="utf-8"))
        self.load_dict(data)
        return self

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path else self.path
        if not target:
            raise ValueError("No file path specified for saving tools.json")
        payload = {k: v.to_dict() if isinstance(v, ToolSpec) else v for k, v in self.items()}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def register(
        self,
        name: str,
        handler: Optional[Union[str, Callable[..., Any]]] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=handler,
            extra=extra,
        )
        self[name] = spec
        return spec

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name with arguments."""
        if tool_name not in self:
            raise KeyError(f"Tool {tool_name!r} not found in ToolRegistry. Available: {list(self.keys())}")
        spec = self[tool_name]
        handler = spec.handler

        if callable(handler):
            return handler(**kwargs)

        # Builtin handlers
        h_str = str(handler or "").strip()
        if h_str in ("builtin:eval_math", "eval_math", "calculator"):
            expr = str(kwargs.get("expression", kwargs.get("expr", ""))).strip()
            # Safe basic math evaluation
            allowed = set("0123456789+-*/(). %")
            if not all(c in allowed for c in expr):
                raise ValueError(f"Unsafe math expression: {expr!r}")
            return eval(expr, {"__builtins__": None}, {})

        if h_str in ("builtin:search", "search_web", "mock_search"):
            q = str(kwargs.get("query", kwargs.get("q", "")))
            return f"[ToolSearch Result for {q!r}]: Found relevant documentation and status records."

        if h_str in ("builtin:format_json", "format_json"):
            data = kwargs.get("data", {})
            return json.dumps(data, ensure_ascii=False, indent=2)

        return f"[Executed Tool {tool_name} with params {kwargs}]"

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: (v.to_dict() if isinstance(v, ToolSpec) else dict(v)) for k, v in self.items()}



class ContentItem(dict):
    """Represents a content entry in `contents.json`: `{tags, contents, gen: int}`.

    Subclasses `dict` so Python code can read/modify via either `item["contents"]` / `item["tags"]` / `item["gen"]`
    or attribute syntax `item.contents` / `item.content` / `item.tags` / `item.gen`.
    """

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], "ContentItem"]] = None,
        *,
        tags: Optional[Union[str, Sequence[str]]] = None,
        contents: Optional[Any] = None,
        content: Optional[Any] = None,
        gen: Optional[int] = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        raw = dict(data) if isinstance(data, dict) else {}
        resolved_tags = tags if tags is not None else raw.get("tags", [])
        resolved_contents = (
            contents
            if contents is not None
            else (content if content is not None else raw.get("contents", raw.get("content", "")))
        )
        resolved_gen = gen if gen is not None else raw.get("gen", 0)

        self["tags"] = normalize_tags(resolved_tags)
        self["contents"] = serialize_content(resolved_contents)
        self["gen"] = int(resolved_gen)

        for k, v in raw.items():
            if k not in ("tags", "contents", "content", "gen"):
                self[k] = v
        for k, v in extra.items():
            self[k] = v

    def __getitem__(self, key: str) -> Any:
        if key == "content":
            return super().__getitem__("contents")
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "content":
            super().__setitem__("contents", serialize_content(value))
        elif key == "contents":
            super().__setitem__("contents", serialize_content(value))
        elif key == "tags":
            super().__setitem__("tags", normalize_tags(value))
        elif key == "gen":
            super().__setitem__("gen", int(value))
        else:
            super().__setitem__(key, value)

    @property
    def tags(self) -> List[str]:
        return self["tags"]

    @tags.setter
    def tags(self, value: Union[str, Sequence[str]]) -> None:
        self["tags"] = normalize_tags(value)

    @property
    def contents(self) -> str:
        return self["contents"]

    @contents.setter
    def contents(self, value: Any) -> None:
        self["contents"] = serialize_content(value)

    @property
    def content(self) -> str:
        """Alias for `contents` for ergonomic compatibility."""
        return self["contents"]

    @content.setter
    def content(self, value: Any) -> None:
        self["contents"] = serialize_content(value)

    @property
    def gen(self) -> int:
        return int(self["gen"])

    @gen.setter
    def gen(self, value: int) -> None:
        self["gen"] = int(value)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def __repr__(self) -> str:
        preview = self.contents if len(self.contents) <= 50 else self.contents[:47] + "..."
        return f"ContentItem(tags={self.tags}, gen={self.gen}, contents={preview!r})"


class ContentStore(list):
    """Manages `contents.json` (`list[{tags, contents, gen}]`) with Python context read/modify/write support."""

    def __init__(
        self,
        items: Optional[Sequence[Any]] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.path: Optional[Path] = Path(path) if path else None
        if self.path and self.path.exists() and items is None:
            self.load(self.path)
        elif items:
            for item in items:
                self.add(item)

    def add(
        self,
        item_or_contents: Any = None,
        *,
        tags: Optional[Union[str, Sequence[str]]] = None,
        contents: Optional[Any] = None,
        gen: int = 0,
        **extra: Any,
    ) -> ContentItem:
        if isinstance(item_or_contents, ContentItem) and tags is None and contents is None:
            entry = item_or_contents
        elif isinstance(item_or_contents, dict) and tags is None and contents is None:
            entry = ContentItem(item_or_contents, **extra)
        else:
            payload = contents if contents is not None else item_or_contents
            entry = ContentItem(tags=tags or ["input"], contents=payload, gen=gen, **extra)
        super().append(entry)
        return entry

    def append(self, item: Any) -> None:
        self.add(item)

    def by_tag(self, tag: str) -> List[ContentItem]:
        return [item for item in self if tag in item.tags]

    def filter(
        self,
        *,
        tag: Optional[str] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        min_gen: Optional[int] = None,
        max_gen: Optional[int] = None,
    ) -> List[ContentItem]:
        results: List[ContentItem] = []
        for item in self:
            if tag and tag not in item.tags:
                continue
            if tags_any and not (set(tags_any) & set(item.tags)):
                continue
            if tags_all and not set(tags_all).issubset(set(item.tags)):
                continue
            if min_gen is not None and item.gen < min_gen:
                continue
            if max_gen is not None and item.gen > max_gen:
                continue
            results.append(item)
        return results

    @contextmanager
    def edit(self, auto_save: bool = True) -> Iterator["ContentStore"]:
        """Context manager to read/modify/write contents in Python and optionally persist to `contents.json`."""
        yield self
        for idx, item in enumerate(self):
            if not isinstance(item, ContentItem):
                super().__setitem__(idx, ContentItem(item))
        if auto_save and self.path:
            self.save(self.path)

    def load(self, path: Union[str, Path]) -> "ContentStore":
        p = Path(path)
        self.path = p
        raw = json.loads(p.read_text(encoding="utf-8"))
        self.clear()
        items = raw.get("contents", raw) if isinstance(raw, dict) else raw
        for item in items:
            self.add(item)
        return self

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path else self.path
        if not target:
            raise ValueError("No file path specified for saving contents.json")
        target.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_list(self) -> List[Dict[str, Any]]:
        return [item.to_dict() if isinstance(item, ContentItem) else ContentItem(item).to_dict() for item in self]


@dataclass
class Processor:
    """Represents `Processor(model, prompt)` or a Python callable for an Edge."""

    model: str = "default"
    prompt: str = "{contents}"
    fn: Optional[Callable[..., Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, spec: Any) -> "Processor":
        if isinstance(spec, Processor):
            return spec
        if callable(spec):
            return cls(model="callable", prompt="", fn=spec)
        if isinstance(spec, str):
            spec_clean = spec.strip()
            if spec_clean.startswith("builtin:meta_plan") or spec_clean.startswith("meta_plan") or spec_clean in ("_plan_modification", "plan_meta_modification"):
                from v7.meta import plan_meta_modification
                return cls(model="callable", prompt=spec, fn=plan_meta_modification)
            if spec_clean.startswith("builtin:meta_exec") or spec_clean.startswith("meta_exec") or spec_clean in ("_execute_and_validate", "execute_and_validate_meta_dag"):
                from v7.meta import execute_and_validate_meta_dag
                return cls(model="callable", prompt=spec, fn=execute_and_validate_meta_dag)
            return cls(model="default", prompt=spec)
        if isinstance(spec, dict):
            raw = dict(spec)
            c_name = raw.get("callable_name", "")
            model = str(raw.pop("model", None) or raw.pop("模型", None) or "default")
            prompt = str(
                raw.pop("prompt", None)
                or raw.pop("处理prompt", None)
                or raw.pop("处理提示词", None)
                or "{contents}"
            )
            fn = raw.pop("fn", None)
            if not fn and (
                c_name in ("_plan_modification", "plan_meta_modification", "meta_plan")
                or model.startswith("builtin:meta_plan")
                or model.startswith("meta_plan")
                or prompt.startswith("builtin:meta_plan")
                or prompt.startswith("meta_plan")
            ):
                from v7.meta import plan_meta_modification
                fn = plan_meta_modification
            elif not fn and (
                c_name in ("_execute_and_validate", "execute_and_validate_meta_dag", "meta_exec")
                or model.startswith("builtin:meta_exec")
                or model.startswith("meta_exec")
                or prompt.startswith("builtin:meta_exec")
                or prompt.startswith("meta_exec")
            ):
                from v7.meta import execute_and_validate_meta_dag
                fn = execute_and_validate_meta_dag
            return cls(model=model, prompt=prompt, fn=fn if callable(fn) else None, extra=raw)
        raise TypeError(f"Unsupported processor specification: {spec!r}")

    def format_prompt(self, items: Sequence[ContentItem]) -> str:
        """Render a clean prompt with the given content items."""
        if len(items) == 1:
            joined_contents = items[0].contents
            joined_tags = ", ".join(items[0].tags)
        else:
            joined_contents = "\n\n".join(
                f"[{', '.join(it.tags)}]: {it.contents}" for it in items
            )
            joined_tags = ", ".join(t for it in items for t in it.tags)

        template = self.prompt or "{contents}"
        rendered = template
        used_var = False
        for placeholder, val in (
            ("{contents}", joined_contents),
            ("{content}", joined_contents),
            ("{input}", joined_contents),
            ("{tags}", joined_tags),
        ):
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, val)
                if placeholder in ("{contents}", "{content}", "{input}"):
                    used_var = True
        if not used_var:
            rendered = f"{rendered}\n\n{joined_contents}" if rendered.strip() else joined_contents
        return rendered

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.fn is not None:
            out["callable_name"] = getattr(self.fn, "__name__", "callable")
        if self.extra:
            out.update(self.extra)
        return out


@dataclass
class EdgeSpec:
    """Represents a single edge in `edges.json`: `{input_tags, processor, output_tags, layer: int, count: int}`."""

    input_tags: List[str]
    processor: Processor
    output_tags: List[str]
    layer: int = 0
    count: int = 0
    id: str = ""

    def __init__(
        self,
        input_tags: Union[str, Sequence[str]],
        processor: Any,
        output_tags: Union[str, Sequence[str]],
        layer: int = 0,
        count: int = 0,
        id: str = "",
    ) -> None:
        self.input_tags = normalize_tags(input_tags)
        self.processor = Processor.from_any(processor)
        self.output_tags = normalize_tags(output_tags)
        self.layer = int(layer)
        self.count = int(count)
        self.id = str(id)

    @classmethod
    def from_any(cls, data: Any, default_id: str = "") -> "EdgeSpec":
        if isinstance(data, EdgeSpec):
            return copy.deepcopy(data)
        if isinstance(data, dict):
            proc = data.get("processor") or data.get("Processor") or {
                "model": data.get("model", "default"),
                "prompt": data.get("prompt", "{contents}"),
            }
            return cls(
                input_tags=data.get("input_tags", data.get("inputs", [])),
                processor=proc,
                output_tags=data.get("output_tags", data.get("outputs", [])),
                layer=int(data.get("layer", 0)),
                count=int(data.get("count", 0)),
                id=str(data.get("id", default_id)),
            )
        if isinstance(data, (tuple, list)):
            in_tags = data[0]
            proc = data[1]
            out_tags = data[2]
            layer = int(data[3]) if len(data) > 3 else 0
            count = int(data[4]) if len(data) > 4 else 0
            return cls(in_tags, proc, out_tags, layer=layer, count=count, id=default_id)
        raise TypeError(f"Cannot parse EdgeSpec from: {data!r}")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "input_tags": list(self.input_tags),
            "processor": self.processor.to_dict(),
            "output_tags": list(self.output_tags),
            "layer": self.layer,
            "count": self.count,
        }
        if self.id:
            out["id"] = self.id
        return out
