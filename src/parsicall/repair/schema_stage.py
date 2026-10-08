import json
import re
from datetime import date
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError

from parsicall.norm import digits_to_ascii, jalali_to_iso, normalize_zwnj
from parsicall.repair.types import Confidence, Mutation, RepairResult, ToolCall

_JALALI = re.compile(r"^\d{3,4}[/-]\d{1,2}[/-]\d{1,2}$")
_NUMERIC_TYPES = frozenset({"integer", "number"})
_DATE_FORMATS = frozenset({"date", "date-time"})
_FA_DIGITS = frozenset("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩")


def _show(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _norm(text: str) -> str:
    return normalize_zwnj(text).casefold()


def _valid(args: dict[str, Any], schema: dict[str, Any]) -> bool:
    # no FormatChecker: `format` (e.g. "date") is not enforced, so "valid" != "well-formed date"
    try:
        validate(instance=args, schema=schema)
    except (ValidationError, SchemaError):
        return False
    return True


def _prop(schema: dict[str, Any], key: str) -> dict[str, Any]:
    properties = schema.get("properties")
    prop = properties.get(key) if isinstance(properties, dict) else None
    return prop if isinstance(prop, dict) else {}


def _s01(value: object, prop: dict[str, Any]) -> object | None:
    if not isinstance(value, str) or prop.get("format") not in _DATE_FORMATS:
        return None
    if not _JALALI.match(digits_to_ascii(value).strip()):
        return None
    try:  # already a valid ISO date: treat as Gregorian, not Jalali
        date.fromisoformat(digits_to_ascii(value).strip())
    except ValueError:
        return jalali_to_iso(value)
    return None


def _s02(value: object, prop: dict[str, Any]) -> object | None:
    kind = prop.get("type")
    if not isinstance(value, str) or kind not in _NUMERIC_TYPES:
        return None
    if not _FA_DIGITS.intersection(value):
        return None
    text = digits_to_ascii(value)
    try:
        return int(text)
    except ValueError:
        if kind != "number":
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _s03(
    value: object, prop: dict[str, Any], key: str, known: dict[str, list[str]]
) -> object | None:
    enum = prop.get("enum")
    if not isinstance(value, str) or not isinstance(enum, list):
        return None
    target = _norm(value)
    matches = [e for e in enum if isinstance(e, str) and _norm(e) == target]
    if len(matches) == 1:
        return matches[0]
    known_values = known.get(key)
    if not isinstance(known_values, list) or len(known_values) != len(enum):
        return None
    hits = [i for i, e in enumerate(known_values) if isinstance(e, str) and _norm(e) == target]
    return enum[hits[0]] if len(hits) == 1 else None

def _apply(
    value: object, key: str, prop: dict[str, Any], known: dict[str, list[str]]
) -> tuple[str, object] | None:
    for rule_id, new in (
        ("S01", _s01(value, prop)),
        ("S02", _s02(value, prop)),
        ("S03", _s03(value, prop, key, known)),
    ):
        if new is not None and new != value:
            return rule_id, new
    return None


def repair_schema(
    call: ToolCall,
    schema: dict[str, Any],
    *,
    known_values: dict[str, list[str]] | None = None,
) -> RepairResult:
    result: ToolCall = {
        "id": call["id"],
        "type": call["type"],
        "function": {
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        },
    }
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        return RepairResult(call=result, mutations=[], confidence="low")
    if not isinstance(args, dict):
        return RepairResult(call=result, mutations=[], confidence="low")

    known = known_values or {}
    fixed = dict(args)
    mutations: list[Mutation] = []
    for key, value in args.items():  # top-level arguments only; nested objects stay untouched
        prop = _prop(schema, key)
        if not prop:
            continue
        applied = _apply(value, key, prop, known)
        if applied is None:
            continue
        rule_id, new = applied
        mutations.append(Mutation(rule_id, f"arguments.{key}", _show(value), _show(new)))
        fixed[key] = new

    # "high" only when at least one mutation landed and the result validates; anything else is "low"
    confidence: Confidence = "low"
    if mutations and _valid(fixed, schema):
        result["function"]["arguments"] = json.dumps(fixed, ensure_ascii=False)
        confidence = "high"
    return RepairResult(call=result, mutations=mutations, confidence=confidence)
