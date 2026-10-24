"""v8 输出契约：给边声明「输出必须长什么样」，把格式问题挡在落库之前。

配置示例（edge 级）：

```json
{
  "id": "extract",
  "expect": {"type": "json", "required_keys": ["topics", "count"]},
  "repair": 1
}
```

支持的约束（可组合）：

| 字段 | 含义 |
|---|---|
| `type` | `json` / `text`（默认 text） |
| `required_keys` | JSON 必须包含的顶层键 |
| `schema` | 嵌套 JSON Schema 子集（type/properties/required/items/enum/additionalProperties/minItems/maxItems/minimum/maximum） |
| `pattern` | 正则（`re.search`） |
| `min_chars` / `max_chars` | 正文长度 |
| `min_sections` | Markdown 标题（`#`~`######`）数量下限 |

`repair` 是失败后的重试次数：调度器会把失败原因回灌给 pi 节点重跑。
校验器只做确定性检查，不调模型，适合当便宜的质量闸门。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

__all__ = ["normalize_spec", "validate_output"]

_ALLOWED = {"type", "required_keys", "pattern", "min_chars", "max_chars", "min_sections", "schema"}
_TYPES = {"text", "json"}


def normalize_spec(raw: Any) -> Dict[str, Any]:
    """把配置里的 expect 规范成 dict；非法配置直接抛 ValueError（配置错误要早暴露）。"""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        if raw == "json":
            return {"type": "json"}
        if raw == "text":
            return {}
        raise ValueError(f"expect 字符串只支持 'json' / 'text'，收到 {raw!r}")

    if not isinstance(raw, dict):
        raise ValueError(f"expect 必须是对象或 'json'，收到 {type(raw).__name__}")

    spec = dict(raw)
    unknown = set(spec) - _ALLOWED
    if unknown:
        raise ValueError(f"expect 有不支持的字段：{sorted(unknown)}")

    kind = spec.get("type", "text")
    if kind not in _TYPES:
        raise ValueError(f"expect.type 只支持 {sorted(_TYPES)}，收到 {kind!r}")
    spec["type"] = kind

    if "required_keys" in spec:
        keys = spec["required_keys"]
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError("expect.required_keys 必须是字符串数组")
    for key in ("min_chars", "max_chars", "min_sections"):
        if key in spec and (not isinstance(spec[key], int) or spec[key] < 0):
            raise ValueError(f"expect.{key} 必须是非负整数")
    if "schema" in spec:
        if not isinstance(spec["schema"], dict):
            raise ValueError("expect.schema 必须是对象")
    if "pattern" in spec:
        if not isinstance(spec["pattern"], str):
            raise ValueError("expect.pattern 必须是字符串")
        try:
            re.compile(spec["pattern"])
        except re.error as exc:
            raise ValueError(f"expect.pattern 正则非法：{exc}") from exc
    return spec


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```$", t, re.S)
    return m.group(1).strip() if m else t


def _parse_json(text: str) -> Any:
    raw = _strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 允许模型在 JSON 前后夹带解释：取第一个 { 到最后一个 }
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _check_schema(value: Any, schema: Any, path: str = "$") -> str:
    """JSON Schema 子集校验；返回空串 = 通过。"""
    if not isinstance(schema, dict):
        return f"{path}: schema 必须是对象"

    kind = schema.get("type")
    if kind:
        checks = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }
        if kind not in checks:
            return f"{path}: 未知 schema.type={kind!r}"
        if not checks[kind](value):
            return f"{path}: 期望 {kind}，实际 {type(value).__name__}"

    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: 值 {value!r} 不在 enum 中"
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        return f"{path}: {value} < minimum={schema['minimum']}"
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        return f"{path}: {value} > maximum={schema['maximum']}"

    if isinstance(value, dict):
        for req in schema.get("required", []) or []:
            if req not in value:
                return f"{path}.{req}: 缺少必需键"
        props = schema.get("properties") or {}
        for key, sub in props.items():
            if key in value:
                err = _check_schema(value[key], sub, f"{path}.{key}")
                if err:
                    return err
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                return f"{path}: 不允许的键 {extra}"

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{path}: 数组长度 {len(value)} < minItems={schema['minItems']}"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path}: 数组长度 {len(value)} > maxItems={schema['maxItems']}"
        if "items" in schema:
            for i, item in enumerate(value):
                err = _check_schema(item, schema["items"], f"{path}[{i}]")
                if err:
                    return err
    return ""


def validate_output(text: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    """返回 (是否通过, 失败原因)。空 spec 直接通过。"""
    if not spec:
        return True, ""

    body = text or ""
    if "min_chars" in spec and len(body.strip()) < spec["min_chars"]:
        return False, f"长度 {len(body.strip())} < min_chars={spec['min_chars']}"
    if "max_chars" in spec and len(body.strip()) > spec["max_chars"]:
        return False, f"长度 {len(body.strip())} > max_chars={spec['max_chars']}"
    if "pattern" in spec and not re.search(spec["pattern"], body, re.S):
        return False, f"未匹配 pattern={spec['pattern']!r}"
    if "min_sections" in spec:
        n = len(re.findall(r"(?m)^#{1,6}\s+\S", body))
        if n < spec["min_sections"]:
            return False, f"Markdown 标题数 {n} < min_sections={spec['min_sections']}"

    if spec.get("type") == "json" or "schema" in spec:
        try:
            data = _parse_json(body)
        except json.JSONDecodeError as exc:
            return False, f"不是合法 JSON：{exc}"
        if "schema" in spec:
            err = _check_schema(data, spec["schema"])
            if err:
                return False, err
        if spec.get("type") == "json" and not isinstance(data, dict):
            return False, f"JSON 顶层必须是对象，实际是 {type(data).__name__}"
        missing = [k for k in spec.get("required_keys", []) if k not in data]
        if missing:
            return False, f"JSON 缺少键：{missing}"

    return True, ""
