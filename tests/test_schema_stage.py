import json

from parsicall.repair import repair_schema
from parsicall.repair.types import ToolCall


def _call(arguments: dict) -> ToolCall:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "book", "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _args(result) -> dict:
    return json.loads(result.call["function"]["arguments"])


def test_s01_jalali_date():
    call = _call({"date": "۱۴۰۴/۰۷/۱۷"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "format": "date"}},
        "required": ["date"],
    }
    res = repair_schema(call, schema)
    assert _args(res)["date"] == "2025-10-09"
    assert [m.rule_id for m in res.mutations] == ["S01"]
    assert res.mutations[0].path == "arguments.date"
    assert res.mutations[0].before == "۱۴۰۴/۰۷/۱۷"
    assert res.confidence == "high"


def test_s02_persian_digits_to_integer():
    call = _call({"count": "۱۲۳"})
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    res = repair_schema(call, schema)
    assert _args(res)["count"] == 123
    assert [m.rule_id for m in res.mutations] == ["S02"]
    assert res.mutations[0].before == "۱۲۳"
    assert res.mutations[0].after == "123"
    assert res.confidence == "high"


def test_s02_number_with_thousands_separator():
    call = _call({"amount": "۲٬۵۰۰"})
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}}
    res = repair_schema(call, schema)
    assert _args(res)["amount"] == 2500
    assert [m.rule_id for m in res.mutations] == ["S02"]


def test_s03_enum_snap():
    call = _call({"status": " paid "})
    schema = {"type": "object", "properties": {"status": {"enum": ["paid", "due"]}}}
    res = repair_schema(call, schema)
    assert _args(res)["status"] == "paid"
    assert [m.rule_id for m in res.mutations] == ["S03"]


def test_s03_enum_snap_zwnj_variant():
    call = _call({"city": "می\u200cخواهم"})
    schema = {"type": "object", "properties": {"city": {"enum": ["می خواهم", "نمی خواهم"]}}}
    res = repair_schema(call, schema)
    assert _args(res)["city"] == "می خواهم"
    assert [m.rule_id for m in res.mutations] == ["S03"]


def test_s03_known_values_maps_to_enum_index():
    call = _call({"status": "پرداخت‌شده"})
    schema = {"type": "object", "properties": {"status": {"enum": ["paid", "due"]}}}
    known = {"status": ["پرداخت شده", "سررسید"]}
    res = repair_schema(call, schema, known_values=known)
    assert _args(res)["status"] == "paid"
    assert [m.rule_id for m in res.mutations] == ["S03"]


def test_valid_args_noop():
    call = _call({"date": "2025-10-09", "count": 3})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "format": "date"}, "count": {"type": "integer"}},
    }
    res = repair_schema(call, schema)
    assert res.mutations == []
    assert res.confidence == "high"
    assert _args(res) == {"date": "2025-10-09", "count": 3}


def test_never_adds_key():
    call = _call({"date": "2026-01-01"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string"}, "note": {"type": "string"}},
        "required": ["date"],
    }
    res = repair_schema(call, schema)
    assert set(_args(res)) == {"date"}


def test_unresolvable_date_leaves_value_and_does_not_crash():
    call = _call({"date": "۱۴۰۴/۰۷/۳۱"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "format": "date"}},
        "required": ["date"],
    }
    res = repair_schema(call, schema)
    assert res.mutations == []
    assert _args(res)["date"] == "۱۴۰۴/۰۷/۳۱"


def test_invalid_args_no_rule_covers_returns_original_with_low_confidence():
    call = _call({"date": "۱۴۰۴/۰۷/۳۱"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "integer"}},
        "required": ["date"],
    }
    res = repair_schema(call, schema)
    assert res.mutations == []
    assert _args(res)["date"] == "۱۴۰۴/۰۷/۳۱"
    assert res.confidence == "low"
