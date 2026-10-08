from dataclasses import dataclass, field
from typing import Literal, TypedDict


class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolFunction


Confidence = Literal["high", "low"]


@dataclass(frozen=True)
class Mutation:
    rule_id: str
    path: str
    before: str
    after: str


@dataclass
class RepairResult:
    call: ToolCall
    mutations: list[Mutation] = field(default_factory=list)
    confidence: Confidence = "high"