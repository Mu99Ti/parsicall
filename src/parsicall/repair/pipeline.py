import json
from typing import Any

from parsicall.repair.format_stage import repair_format
from parsicall.repair.persian_stage import repair_persian
from parsicall.repair.schema_stage import _valid, repair_schema
from parsicall.repair.types import Confidence, Mutation, RepairResult, ToolCall


def _verifies(call: ToolCall, schema: dict[str, Any]) -> bool:
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        return False
    return isinstance(args, dict) and _valid(args, schema)


def repair_call(
    call: ToolCall,
    *,
    tools_schemas: dict[str, dict[str, Any]],
    known_values: dict[str, list[str]] | None = None,
    toman_rate: int | None = None,
) -> RepairResult:
    """Run F(arguments) -> S -> P exactly once each over `call`.

    Mutations accumulate in stage order. Confidence is "high" only when no stage
    mutated into a fail-closed branch and the result validates against the tool
    schema; an unknown tool can never be verified, so it stays "low".
    """
    mutations: list[Mutation] = []
    degraded = False

    def run(stage: RepairResult) -> ToolCall:
        nonlocal degraded
        mutations.extend(stage.mutations)
        degraded = degraded or (stage.confidence == "low" and bool(stage.mutations))
        return stage.call

    current = run(repair_format("", call))
    schema = tools_schemas.get(current["function"]["name"])
    if schema is None:
        return RepairResult(call=current, mutations=mutations, confidence="low")

    current = run(repair_schema(current, schema, known_values=known_values))
    current = run(repair_persian(current, schema, known_values=known_values, toman_rate=toman_rate))

    confidence: Confidence = "high" if not degraded and _verifies(current, schema) else "low"
    return RepairResult(call=current, mutations=mutations, confidence=confidence)
