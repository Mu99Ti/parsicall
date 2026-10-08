import json
from pathlib import Path

import pytest

from parsicall.policy import AuditLog, PolicyConfig, decide
from parsicall.repair import Mutation, RepairResult, repair_call
from parsicall.repair.types import Confidence, ToolCall

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "malformed.jsonl"
ROWS = [
    json.loads(line)
    for line in FIXTURES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
FIXTURES = [(row, {row["tool"]: row["schema"]}) for row in ROWS]
IDS = [f"{i:02d}-{row['tool']}" for i, row in enumerate(ROWS)]
KNOWN_VALUES = {"status": ["پرداخت شده", "سررسید", "paid", "cancelled"]}
TOMAN_RATE = 10


def _call(row: dict) -> ToolCall:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": row["tool"], "arguments": row["arguments_raw"]},
    }


def test_fixture_file_has_fifty_rows():
    assert len(ROWS) == 50


@pytest.mark.parametrize("row,tools_schemas", FIXTURES, ids=IDS)
def test_idempotent(row, tools_schemas):
    once = repair_call(_call(row), tools_schemas=tools_schemas, known_values=None, toman_rate=None)
    twice = repair_call(once.call, tools_schemas=tools_schemas, known_values=None, toman_rate=None)
    assert twice.mutations == []
    assert twice.call == once.call


@pytest.mark.parametrize("row,tools_schemas", FIXTURES, ids=IDS)
def test_idempotent_with_known_values_and_toman_rate(row, tools_schemas):
    # the same fixpoint property must hold once P04/P05 have inputs to work with
    kwargs = {"known_values": KNOWN_VALUES, "toman_rate": TOMAN_RATE}
    once = repair_call(_call(row), tools_schemas=tools_schemas, **kwargs)
    twice = repair_call(once.call, tools_schemas=tools_schemas, **kwargs)
    assert twice.mutations == []
    assert twice.call == once.call


def test_stages_run_in_order_and_accumulate_rule_ids():
    row = {
        "tool": "order",
        "arguments_raw": '{"note": "علي كاظمی", "count": "۵",}',
        "schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}, "count": {"type": "integer"}},
        },
    }
    result = repair_call(_call(row), tools_schemas={row["tool"]: row["schema"]})
    assert [m.rule_id for m in result.mutations] == ["F01", "S02", "P03"]
    assert result.confidence == "high"
    args = json.loads(result.call["function"]["arguments"])
    assert args == {"note": "علی کاظمی", "count": 5}


def test_unknown_tool_skips_schema_and_persian_but_still_runs_format():
    call = _call({"tool": "mystery", "arguments_raw": '{"count": 1,}', "schema": {}})
    result = repair_call(call, tools_schemas={})
    assert [m.rule_id for m in result.mutations] == ["F01"]
    assert result.confidence == "low"


def test_unknown_tool_persian_values_are_left_alone():
    call = _call({"tool": "mystery", "arguments_raw": '{"phone": "۰۹۱۲۳۴۵"}', "schema": {}})
    result = repair_call(call, tools_schemas={})
    assert result.mutations == []
    assert json.loads(result.call["function"]["arguments"])["phone"] == "۰۹۱۲۳۴۵"
    assert result.confidence == "low"


def test_result_that_cannot_be_verified_against_schema_is_low_confidence():
    # P05 fails closed without a rate: nothing mutated, but nothing verified either
    row = {"tool": "pay", "arguments_raw": '{"amount": "50000 ریال"}', "schema": {}}
    schema = {"type": "object", "properties": {"amount": {"type": "integer"}}}
    result = repair_call(_call(row), tools_schemas={"pay": schema})
    assert result.mutations == []
    assert result.confidence == "low"


def _repaired(rule_ids: list[str], confidence: Confidence) -> RepairResult:
    call: ToolCall = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "pay", "arguments": '{"amount": 5000}'},
    }
    mutations = [
        Mutation(rule_id, "arguments.amount", '"50000 ریال"', "5000") for rule_id in rule_ids
    ]
    return RepairResult(call=call, mutations=mutations, confidence=confidence)


def test_decide_rejects_beyond_the_mutation_budget():
    result = _repaired(["P01"] * 9, "high")
    assert decide(result, PolicyConfig()) == "reject"


def test_decide_rejects_strict_low_confidence_with_a_guessing_rule():
    assert decide(_repaired(["P05"], "low"), PolicyConfig(mode="strict")) == "reject"


def test_decide_emits_in_permissive_mode():
    assert decide(_repaired(["P05"], "low"), PolicyConfig(mode="permissive")) == "emit"


def test_audit_log_writes_one_line_per_call_and_appends(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    record = {
        "tool": "pay",
        "original": '{"amount": "50000 ریال"}',
        "final": '{"amount": 5000}',
        "mutations": [{"rule_id": "P05", "path": "arguments.amount"}],
        "decision": "emit",
    }
    log.append(record)
    log.append(record | {"tool": "book"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["tool"] == "pay"
    assert first["original"] == '{"amount": "50000 ریال"}'
    assert first["mutations"] == [{"rule_id": "P05", "path": "arguments.amount"}]
    assert first["decision"] == "emit"
    assert isinstance(first["ts"], str)
    assert json.loads(lines[1])["tool"] == "book"


def test_idempotency_sweep_is_not_vacuous():
    # the property is only meaningful if some fixtures actually get repaired
    repaired = [
        row
        for row, tools_schemas in FIXTURES
        if repair_call(
            _call(row), tools_schemas=tools_schemas, known_values=None, toman_rate=None
        ).mutations
    ]
    assert len(repaired) >= 20


def test_strict_policy_rejects_a_guess_that_does_not_validate():
    schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "integer"},
            "status": {"type": "string", "enum": ["پرداخت شده", "سررسید"]},
        },
    }
    row = {
        "tool": "pay",
        "arguments_raw": '{"amount": "5000 تومان", "status": "پرداخت"}',
        "schema": schema,
    }
    result = repair_call(_call(row), tools_schemas={"pay": schema}, toman_rate=10)
    assert "P05" in [m.rule_id for m in result.mutations]
    assert result.confidence == "low"  # status never landed in the enum
    assert decide(result, PolicyConfig(mode="strict")) == "reject"
    assert decide(result, PolicyConfig(mode="permissive")) == "emit"


def test_fixture_corpus_exercises_every_rule_the_pipeline_can_emit():
    # F02 is deliberately absent: repair_call passes content "", so extract_tool_call
    # (which needs empty arguments AND assistant content) can never fire in the pipeline.
    expected = {"F01", "F03", "S01", "S02", "S03", "P01", "P02", "P03", "P04", "P05"}
    contexts = (
        {"known_values": None, "toman_rate": None},
        {"known_values": KNOWN_VALUES, "toman_rate": TOMAN_RATE},
    )
    seen = {
        mutation.rule_id
        for context in contexts
        for row, tools_schemas in FIXTURES
        for mutation in repair_call(
            _call(row), tools_schemas=tools_schemas, **context
        ).mutations
    }
    assert expected <= seen
