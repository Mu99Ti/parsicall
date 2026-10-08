import json

from parsicall.repair import extract_tool_call, repair_format
from parsicall.repair.types import ToolCall


def _call(arguments: str = "", name: str = "book_flight") -> ToolCall:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_extract_from_fenced_json_block():
    text = '```json\n{"name": "book_flight", "arguments": {"city": "تهران"}}\n```'
    result = extract_tool_call(text)
    assert result is not None
    assert result.call["function"]["name"] == "book_flight"
    assert json.loads(result.call["function"]["arguments"]) == {"city": "تهران"}
    assert [m.rule_id for m in result.mutations] == ["F02"]
    assert result.mutations[0].path == "call"
    assert result.confidence == "high"


def test_extract_from_bare_json_object_in_text():
    text = 'Sure! {"name": "get_weather", "arguments": {"city": "شیراز"}} done'
    result = extract_tool_call(text)
    assert result is not None
    assert result.call["function"]["name"] == "get_weather"
    assert json.loads(result.call["function"]["arguments"]) == {"city": "شیراز"}


def test_extract_from_mistral_style():
    text = '[TOOL_CALLS]book_flight{"city": "تبریز"}'
    result = extract_tool_call(text)
    assert result is not None
    assert result.call["function"]["name"] == "book_flight"
    assert json.loads(result.call["function"]["arguments"]) == {"city": "تبریز"}


def test_extract_returns_none_when_nothing_extractable():
    assert extract_tool_call("I cannot help with that.") is None


def test_f01_resalvages_unparseable_arguments():
    result = repair_format("noise", _call('{"city": "تهران",}'))
    assert [m.rule_id for m in result.mutations] == ["F01"]
    mutation = result.mutations[0]
    assert mutation.path == "function.arguments"
    assert mutation.before == '{"city": "تهران",}'
    assert result.call["function"]["arguments"] == '{"city": "تهران"}'
    assert result.confidence == "high"


def test_valid_arguments_are_noop():
    result = repair_format("noise", _call('{"city": "تهران"}'))
    assert result.mutations == []
    assert result.confidence == "high"


def test_f03_closes_unbalanced_brackets_with_low_confidence():
    result = repair_format("noise", _call('{"stops": ["a", "b"'))
    assert [m.rule_id for m in result.mutations] == ["F03"]
    assert result.call["function"]["arguments"] == '{"stops": ["a", "b"]}'
    assert json.loads(result.call["function"]["arguments"]) == {"stops": ["a", "b"]}
    assert result.confidence == "low"


def test_f03_no_mutation_when_ending_mid_string():
    result = repair_format("noise", _call('{"city": "تهرا'))
    assert [m.rule_id for m in result.mutations] != ["F03"]


def test_repair_format_uses_extracted_call_when_arguments_empty():
    text = '```json\n{"name": "get_weather", "arguments": {"city": "کرج"}}\n```'
    result = repair_format(text, _call(""))
    assert result.call["function"]["name"] == "get_weather"
    assert json.loads(result.call["function"]["arguments"]) == {"city": "کرج"}
    assert "F02" in [m.rule_id for m in result.mutations]


def test_repair_format_with_extracted_call_and_valid_args_is_noop():
    text = '[TOOL_CALLS]get_weather{"city": "کرج"}'
    result = repair_format(text, _call('{"city": "کرج"}'))
    assert result.mutations == []
    assert result.confidence == "high"