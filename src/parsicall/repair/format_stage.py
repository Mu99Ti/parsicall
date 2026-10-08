import json
import re
from json import JSONDecodeError

from json_repair import loads as repair_json

from parsicall.repair.types import Confidence, Mutation, RepairResult, ToolCall

_FENCE = re.compile(r"```(?:json)?\s*\{", re.DOTALL)
_BARE = re.compile(r"\{\s*\"name\"", re.DOTALL)
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(?P<name>[A-Za-z0-9_.-]+)\s*\{")
_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}": "{", "]": "["}


def _encode(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _make_call(name: str, arguments: str) -> ToolCall:
    return {"id": "", "type": "function", "function": {"name": name, "arguments": arguments}}


def _object_span(text: str, start: int) -> int | None:
    """Index just past the bracket opened at `start`, or None if it never closes."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return None
    return None


def _load(text: str) -> object:
    try:
        return json.loads(text)
    except JSONDecodeError:
        return repair_json(text)


def _closers(stack: list[str]) -> str:
    return "".join(_OPENERS[opener] for opener in reversed(stack))


def _open_stack(text: str) -> tuple[list[str], bool, bool]:
    """Return (unclosed openers, ended inside a string, ended on a dangling key)."""
    stack: list[str] = []
    in_string = False
    escaped = False
    dangling_key = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                dangling_key = True
        elif ch == '"':
            in_string = True
        elif ch in _OPENERS:
            stack.append(ch)
            dangling_key = False
        elif ch in _CLOSERS:
            if not stack or stack[-1] != _CLOSERS[ch]:
                return [], False, False  # crossed closers: ambiguous, refuse to guess
            stack.pop()
            dangling_key = False
        elif ch in ":,":
            dangling_key = ch == ","
    return stack, in_string, dangling_key and bool(stack) and stack[-1] == "{"


def _f02(name: str, arguments: object, snippet: str) -> RepairResult:
    wire = _encode({"name": name, "arguments": arguments})
    return RepairResult(
        call=_make_call(name, _encode(arguments)),
        mutations=[Mutation("F02", "call", snippet, wire)],
        confidence="high",
    )


def _call_from_object(text: str, start: int) -> RepairResult | None:
    end = _object_span(text, start)
    if end is None:
        return None
    obj = _load(text[start:end])
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return _f02(name, arguments, text[start:end])


def extract_tool_call(content: str) -> RepairResult | None:
    for pattern in (_FENCE, _BARE):
        for match in pattern.finditer(content):
            found = _call_from_object(content, content.index("{", match.start()))
            if found is not None:
                return found
    mistral = _MISTRAL.search(content)
    if mistral is not None:
        start = content.index("{", mistral.start())
        end = _object_span(content, start)
        if end is not None:
            arguments = _load(content[start:end])
            if isinstance(arguments, dict):
                return _f02(mistral.group("name"), arguments, content[start:end])
    return None


def repair_format(text: str, call: ToolCall) -> RepairResult:
    result: ToolCall = {
        "id": call["id"],
        "type": call["type"],
        "function": {
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        },
    }
    mutations: list[Mutation] = []
    confidence: Confidence = "high"

    if not result["function"]["arguments"].strip():
        found = extract_tool_call(text)
        if found is not None:
            mutations += found.mutations
            result["function"]["name"] = found.call["function"]["name"]
            result["function"]["arguments"] = found.call["function"]["arguments"]

    before = result["function"]["arguments"]
    stack, mid_string, mid_key = _open_stack(before)
    if stack and not mid_string and not mid_key:
        closed = before + _closers(stack)
        try:
            if isinstance(json.loads(closed), dict):
                mutations.append(Mutation("F03", "function.arguments", before, closed))
                result["function"]["arguments"] = closed
                confidence = "low"
        except JSONDecodeError:
            pass

    if not mutations:
        try:
            json.loads(before)
        except JSONDecodeError:
            salvaged = repair_json(before)
            if isinstance(salvaged, dict):
                mutated = _encode(salvaged)
                if mutated != before:
                    mutations.append(Mutation("F01", "function.arguments", before, mutated))
                    result["function"]["arguments"] = mutated

    return RepairResult(call=result, mutations=mutations, confidence=confidence)