import json

from parsicall.repair import repair_persian
from parsicall.repair.types import ToolCall


def _call(arguments: dict) -> ToolCall:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "pay", "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _args(result) -> dict:
    return json.loads(result.call["function"]["arguments"])


def test_p01_persian_digits_to_ascii():
    call = _call({"phone": "۰۹۱۲۳۴۵۶۷۸۹"})
    schema = {"type": "object", "properties": {"phone": {"type": "string"}}}
    res = repair_persian(call, schema)
    assert _args(res)["phone"] == "09123456789"
    assert [m.rule_id for m in res.mutations] == ["P01"]
    assert res.mutations[0].path == "arguments.phone"
    assert res.mutations[0].before == "۰۹۱۲۳۴۵۶۷۸۹"
    assert res.mutations[0].after == "09123456789"
    assert res.confidence == "high"


def test_p01_fires_even_when_schema_says_number():
    # P canonicalizes text; the string->number coercion belongs to stage S (S02)
    call = _call({"count": "۱۲۳"})
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    res = repair_persian(call, schema)
    assert _args(res)["count"] == "123"
    assert [m.rule_id for m in res.mutations] == ["P01"]


def test_p02_jalali_to_iso():
    call = _call({"date": "1404/07/17"})
    schema = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}}}
    res = repair_persian(call, schema)
    assert _args(res)["date"] == "2025-10-09"
    assert [m.rule_id for m in res.mutations] == ["P02"]
    assert res.mutations[0].path == "arguments.date"
    assert res.confidence == "high"


def test_p02_named_jalali_month():
    call = _call({"date": "۱۷ مهر ۱۴۰۴"})
    schema = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}}}
    res = repair_persian(call, schema)
    assert _args(res)["date"] == "2025-10-09"
    assert [m.rule_id for m in res.mutations] == ["P01", "P02"]


def test_p02_key_name_marks_property_date_ish():
    call = _call({"created_at": "1404/07/17"})
    schema = {"type": "object", "properties": {"created_at": {"type": "string"}}}
    res = repair_persian(call, schema)
    assert _args(res)["created_at"] == "2025-10-09"
    assert [m.rule_id for m in res.mutations] == ["P02"]


def test_p02_unparseable_date_in_date_field_fails_closed():
    call = _call({"date": "۱۴۰۴/۰۷/۳۱"})  # day 31 does not exist in Mehr
    schema = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}}}
    res = repair_persian(call, schema)
    # P01 still canonicalized the digits; P02 could not, so the stage is not confident
    assert [m.rule_id for m in res.mutations] == ["P01"]
    assert _args(res)["date"] == "1404/07/31"
    assert res.confidence == "low"  # in-scope date that no rule could canonicalize


def test_p03_arabic_variants_folded():
    call = _call({"name": "علي كاظمی"})
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    res = repair_persian(call, schema)
    assert _args(res)["name"] == "علی کاظمی"
    assert [m.rule_id for m in res.mutations] == ["P03"]
    assert res.mutations[0].before == "علي كاظمی"
    assert res.mutations[0].after == "علی کاظمی"


def test_p04_zwnj_variant_snaps_to_known_value():
    call = _call({"status": "پرداخت‌شده"})
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    known = {"status": ["پرداخت شده", "سررسید"]}
    res = repair_persian(call, schema, known_values=known)
    assert _args(res)["status"] == "پرداخت شده"
    assert [m.rule_id for m in res.mutations] == ["P04"]
    assert res.confidence == "high"


def test_p04_no_match_is_noop():
    call = _call({"status": "در انتظار بررسی"})
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    known = {"status": ["پرداخت شده", "سررسید"]}
    res = repair_persian(call, schema, known_values=known)
    assert res.mutations == []
    assert _args(res)["status"] == "در انتظار بررسی"
    assert res.confidence == "low"


def test_p05_rial_to_toman_with_rate():
    call = _call({"amount": "50000 ریال"})
    schema = {"type": "object", "properties": {"amount": {"type": "integer"}}}
    res = repair_persian(call, schema, toman_rate=10)
    assert _args(res)["amount"] == 5000
    assert [m.rule_id for m in res.mutations] == ["P05"]
    assert res.mutations[0].before == "50000 ریال"
    assert res.mutations[0].after == "5000"
    assert res.confidence == "high"


def test_p05_toman_amount_only_strips_currency_wording():
    call = _call({"amount": "۵۰۰ تومان"})
    schema = {"type": "object", "properties": {"amount": {"type": "integer"}}}
    res = repair_persian(call, schema, toman_rate=10)
    assert _args(res)["amount"] == 500
    assert [m.rule_id for m in res.mutations] == ["P01", "P05"]


def test_p05_prose_that_merely_mentions_a_price_is_not_collapsed():
    call = _call({"note": "پرداخت 50000 ریال بابت قسط"})
    schema = {"type": "object", "properties": {"note": {"type": "string"}}}
    res = repair_persian(call, schema, toman_rate=10)
    assert res.mutations == []
    assert _args(res)["note"] == "پرداخت 50000 ریال بابت قسط"
    assert res.confidence == "low"


def test_p05_without_rate_fails_closed():
    call = _call({"amount": "50000 ریال"})
    schema = {"type": "object", "properties": {"amount": {"type": "integer"}}}
    res = repair_persian(call, schema)
    assert res.mutations == []
    assert _args(res)["amount"] == "50000 ریال"
    assert res.confidence == "low"


def test_p05_fail_closed_forces_low_confidence_for_other_mutations():
    call = _call({"phone": "۰۹۱۲۳۴۵۶۷۸۹", "amount": "50000 ریال"})
    schema = {
        "type": "object",
        "properties": {"phone": {"type": "string"}, "amount": {"type": "integer"}},
    }
    res = repair_persian(call, schema)
    assert [m.rule_id for m in res.mutations] == ["P01"]
    assert res.confidence == "low"


def test_p05_lossy_rial_conversion_fails_closed():
    call = _call({"amount": "105 ریال"})  # 10.5 toman at rate 10: never drop rial silently
    schema = {"type": "object", "properties": {"amount": {"type": "integer"}}}
    res = repair_persian(call, schema, toman_rate=10)
    assert res.mutations == []
    assert _args(res)["amount"] == "105 ریال"
    assert res.confidence == "low"


def test_valid_args_noop():
    call = _call({"date": "2025-10-09", "count": 3})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "format": "date"}, "count": {"type": "integer"}},
    }
    res = repair_persian(call, schema)
    assert res.mutations == []
    assert res.confidence == "low"  # clean pass-through is never "high": high iff a mutation landed
    assert _args(res) == {"date": "2025-10-09", "count": 3}


def test_already_iso_date_is_not_reparsed_as_jalali():
    call = _call({"date": "2025-10-09", "count": "۵"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string", "format": "date"}, "count": {"type": "string"}},
    }
    res = repair_persian(call, schema)
    assert _args(res)["date"] == "2025-10-09"
    assert [m.rule_id for m in res.mutations] == ["P01"]
    assert res.confidence == "high"


def test_untouched_call_arguments_are_byte_identical():
    call: ToolCall = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "pay", "arguments": '{"date":"2025-10-09"}'},  # compact form
    }
    res = repair_persian(call, {"type": "object", "properties": {"date": {"type": "string"}}})
    assert res.mutations == []
    assert res.call["function"]["arguments"] == '{"date":"2025-10-09"}'


def test_never_adds_key():
    call = _call({"date": "2026-01-01"})
    schema = {
        "type": "object",
        "properties": {"date": {"type": "string"}, "note": {"type": "string"}},
        "required": ["date"],
    }
    res = repair_persian(call, schema)
    assert set(_args(res)) == {"date"}


def test_ordering_digits_converted_before_jalali_parse():
    call = _call({"date": "۱۴۰۴/۰۷/۱۷"})
    schema = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}}}
    res = repair_persian(call, schema)
    assert _args(res)["date"] == "2025-10-09"  # ISO, not merely ASCII
    assert [m.rule_id for m in res.mutations] == ["P01", "P02"]


def test_malformed_arguments_json_is_low_confidence():
    call: ToolCall = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "pay", "arguments": "{not json"},
    }
    res = repair_persian(call, {})
    assert res.mutations == []
    assert res.confidence == "low"
