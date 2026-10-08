import json
from pathlib import Path

import pytest

from parsicall.policy import AuditLog, PolicyConfig, decide
from parsicall.repair.types import Confidence, Mutation, RepairResult, ToolCall


def _result(rule_ids: list[str], confidence: Confidence = "high") -> RepairResult:
    call: ToolCall = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "pay", "arguments": '{"amount": 5000}'},
    }
    mutations = [
        Mutation(rule_id, "arguments.amount", '"50000 ریال"', "5000") for rule_id in rule_ids
    ]
    return RepairResult(call=call, mutations=mutations, confidence=confidence)


def test_policy_config_defaults():
    cfg = PolicyConfig()
    assert cfg.mode == "strict"
    assert cfg.max_mutations_per_call == 8
    assert cfg.max_reprompt == 1


@pytest.mark.parametrize("mode", ["strict", "permissive"])
def test_budget_rejects_in_both_modes(mode):
    cfg = PolicyConfig(mode=mode, max_mutations_per_call=2)
    assert decide(_result(["P01", "P03", "P05"]), cfg) == "reject"
    assert decide(_result(["P01", "P03"]), cfg) == "emit"


def test_default_budget_is_eight_mutations():
    assert decide(_result(["P01"] * 8), PolicyConfig()) == "emit"
    assert decide(_result(["P01"] * 9), PolicyConfig()) == "reject"


@pytest.mark.parametrize("rule_id", ["P04", "P05"])
def test_strict_rejects_guessing_rule_under_low_confidence(rule_id):
    assert decide(_result([rule_id], "low"), PolicyConfig(mode="strict")) == "reject"


def test_strict_emits_the_same_rule_when_confidence_is_high():
    assert decide(_result(["P05"], "high"), PolicyConfig(mode="strict")) == "emit"


def test_strict_emits_low_confidence_for_non_guessing_rules():
    # a structural guess (F03) is recoverable downstream; a value guess is not
    assert decide(_result(["F03"], "low"), PolicyConfig(mode="strict")) == "emit"


def test_permissive_emits_low_confidence_guessing_rule():
    assert decide(_result(["P04"], "low"), PolicyConfig(mode="permissive")) == "emit"


def test_decide_without_mutations_emits():
    assert decide(_result([], "low"), PolicyConfig()) == "emit"


def test_audit_log_creates_one_jsonl_line(tmp_path):
    path = tmp_path / "nested" / "audit.jsonl"
    path.parent.mkdir()
    AuditLog(path).append({"tool": "pay", "decision": "emit"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"ts", "tool", "decision"}
    assert isinstance(record["ts"], str)


def test_audit_log_is_append_only_across_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append({"tool": "pay", "decision": "emit"})
    AuditLog(path).append({"tool": "book", "decision": "reject"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["tool"] for line in lines] == ["pay", "book"]


def test_audit_log_keeps_a_caller_supplied_timestamp(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append({"ts": "2026-10-09T00:00:00+00:00", "tool": "pay"})
    assert json.loads(path.read_text(encoding="utf-8"))["ts"] == "2026-10-09T00:00:00+00:00"


def test_audit_log_accepts_str_path(tmp_path):
    AuditLog(str(tmp_path / "audit.jsonl")).append({"tool": "pay"})
    assert Path(tmp_path / "audit.jsonl").exists()
