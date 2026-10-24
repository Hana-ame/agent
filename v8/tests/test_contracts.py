"""输出契约校验器的离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from dag8.contracts import normalize_spec, validate_output  # noqa: E402


def test_json_direct_and_fenced():
    spec = normalize_spec({"type": "json", "required_keys": ["a"]})
    assert validate_output('{"a": 1}', spec)[0]
    assert validate_output('```json\n{"a": 1}\n```', spec)[0]


def test_json_with_surrounding_text():
    spec = normalize_spec({"type": "json", "required_keys": ["a"]})
    ok, _ = validate_output('结果如下：\n{"a": 1}\n以上', spec)
    assert ok


def test_json_missing_key():
    spec = normalize_spec({"type": "json", "required_keys": ["a"]})
    ok, why = validate_output('{"b": 1}', spec)
    assert not ok and "缺少键" in why


def test_json_top_level_must_be_object():
    ok, why = validate_output("[1, 2]", normalize_spec({"type": "json"}))
    assert not ok and "对象" in why


def test_min_sections():
    spec = normalize_spec({"min_sections": 2})
    ok, why = validate_output("# 只有一节\n短", spec)
    assert not ok and "标题" in why
    assert validate_output("## A\n内容\n## B\n再加一点", spec)[0]


def test_length_bounds():
    assert not validate_output("短", normalize_spec({"min_chars": 10}))[0]
    ok, why = validate_output("太长了啊", normalize_spec({"max_chars": 3}))
    assert not ok and "max_chars" in why


def test_regex_pattern():
    spec = normalize_spec({"pattern": r"^答案[:：]"})
    assert validate_output("答案：42", spec)[0]
    assert not validate_output("42", spec)[0]


@pytest.mark.parametrize("raw", [
    {"type": "yaml"},
    {"bogus": 1},
    "yaml",
    {"min_chars": -1},
    {"pattern": "["},
])
def test_invalid_specs_raise(raw):
    with pytest.raises(ValueError):
        normalize_spec(raw)


def test_empty_spec_always_passes():
    assert validate_output("任意内容", {})[0]


NESTED = {"schema": {
    "type": "object", "required": ["entities", "count"],
    "properties": {
        "entities": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["name"],
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False}},
        "count": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}}


def test_schema_nested_ok():
    ok, reason = validate_output('{"entities": [{"name": "a"}], "count": 2}', NESTED)
    assert ok, reason


def test_schema_nested_type_error():
    ok, reason = validate_output('{"entities": [{"name": 1}], "count": 2}', NESTED)
    assert not ok
    assert "entities[0].name" in reason


def test_schema_required_and_items():
    ok, reason = validate_output('{"entities": [], "count": 0}', NESTED)
    assert not ok and "minItems" in reason

    ok, reason = validate_output('{"count": 0}', NESTED)
    assert not ok and "entities" in reason


def test_schema_enum_and_additional_properties():
    spec = {"schema": {
        "type": "object",
        "properties": {"kind": {"enum": ["a", "b"]}},
        "additionalProperties": False,
    }}
    assert validate_output('{"kind": "a"}', spec)[0]
    assert not validate_output('{"kind": "c"}', spec)[0]
    assert not validate_output('{"kind": "a", "x": 1}', spec)[0]


def test_schema_without_type_json_parses():
    spec = {"schema": {"type": "array", "items": {"type": "integer"}}}
    assert validate_output("[1, 2, 3]", spec)[0]
    ok, reason = validate_output("[1, \"x\"]", spec)
    assert not ok and "[1]" in reason
