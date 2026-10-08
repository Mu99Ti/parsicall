import json
import re
from datetime import date
from typing import Any

from parsicall.norm import (
    digits_to_ascii,
    fold_persian_variants,
    jalali_to_iso,
    normalize_zwnj,
    parse_persian_amount,
)
from parsicall.repair.types import Confidence, Mutation, RepairResult, ToolCall

_DATE_FORMATS = frozenset({"date", "date-time"})
_DATE_KEY = re.compile(r"date|created|birth|day", re.IGNORECASE)
_ARABIC_VARIANTS = ("\u064a", "\u0643", "\u06c0")  # ي ك ۀ
_PERSIAN_DIGITS = frozenset("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩")
_AMOUNT_FILLER = re.compile(r"[\d\s.,٫٬،+-]|ریال|تومان|هزار|میلیون")


def _show(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _norm(text: str) -> str:
    return normalize_zwnj(fold_persian_variants(text)).casefold()


def _prop(schema: dict[str, Any], key: str) -> dict[str, Any]:
    properties = schema.get("properties")
    prop = properties.get(key) if isinstance(properties, dict) else None
    return prop if isinstance(prop, dict) else {}


def _date_ish(key: str, prop: dict[str, Any]) -> bool:
    return prop.get("format") in _DATE_FORMATS or bool(_DATE_KEY.search(key))


def _p01(value: object) -> object | None:
    if isinstance(value, str) and _PERSIAN_DIGITS.intersection(value):
        return digits_to_ascii(value)
    return None


def _p03(value: object) -> object | None:
    if isinstance(value, str) and any(ch in value for ch in _ARABIC_VARIANTS):
        return fold_persian_variants(value)
    return None


def _bare_amount(text: str) -> bool:
    # only digits, punctuation and currency/unit words may remain once those are
    # stripped, so prose that merely mentions a price is never rewritten to a number
    return _AMOUNT_FILLER.sub("", digits_to_ascii(text).casefold()) == ""


def _to_toman(value: int, currency: str, rate: int) -> int | None:
    # target currency is always toman: the schema states no currency, so this stage
    # cannot know which one the callee expects — toman is the Persian-side default
    if currency == "toman":
        return value
    whole, remainder = divmod(value, rate)
    return None if remainder else whole  # fractional toman would silently drop rial


def _repair_key(
    key: str,
    value: object,
    prop: dict[str, Any],
    known: dict[str, list[str]],
    toman_rate: int | None,
) -> tuple[object, list[Mutation], bool]:
    current = value
    mutations: list[Mutation] = []
    degraded = False

    def record(rule_id: str, after: object | None) -> None:
        nonlocal current
        if after is not None and after != current:  # None = rule did not apply, never a value
            mutations.append(Mutation(rule_id, f"arguments.{key}", _show(current), _show(after)))
            current = after

    # P01 first: the P02/P05 parsers must see ASCII digits
    record("P01", _p01(current))
    # P03 next: one alphabet before anything compares or parses text
    record("P03", _p03(current))

    if _date_ish(key, prop) and isinstance(current, str):
        try:  # already a valid ISO date: leave it, do not re-parse as Jalali
            date.fromisoformat(current.strip())
        except ValueError:
            iso = jalali_to_iso(current)
            if iso is None:
                degraded = True  # date-ish field we could not canonicalize
            else:
                record("P02", iso)

    if isinstance(current, str):
        amount = parse_persian_amount(current)
        if amount is not None and _bare_amount(current):
            if toman_rate is None or toman_rate <= 0:
                degraded = True  # no configured rate: never guess a conversion
            else:
                toman = _to_toman(amount[0], amount[1], toman_rate)
                if toman is None:
                    degraded = True  # lossy: fail closed rather than round away rial
                else:
                    record("P05", toman)

    entries = known.get(key)
    if isinstance(current, str) and entries:
        target = _norm(current)
        hits = [entry for entry in entries if isinstance(entry, str) and _norm(entry) == target]
        if len(hits) == 1:  # exact post-fold equality only; ambiguous or absent -> no-op
            record("P04", hits[0])

    return current, mutations, degraded


def repair_persian(
    call: ToolCall,
    schema: dict[str, Any],
    *,
    known_values: dict[str, list[str]] | None = None,
    toman_rate: int | None = None,
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
    degraded = False
    for key, value in args.items():  # top-level arguments only; nested objects stay untouched
        new, step, step_degraded = _repair_key(key, value, _prop(schema, key), known, toman_rate)
        fixed[key] = new
        mutations += step
        degraded = degraded or step_degraded

    if mutations:  # a call nothing touched stays byte-identical
        result["function"]["arguments"] = json.dumps(fixed, ensure_ascii=False)
    # "high" iff a rule landed AND no rule hit a fail-closed branch; anything else is "low"
    confidence: Confidence = "high" if mutations and not degraded else "low"
    return RepairResult(call=result, mutations=mutations, confidence=confidence)