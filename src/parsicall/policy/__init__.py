import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from parsicall.repair.types import RepairResult

_GUESSING_RULES = frozenset({"P04", "P05"})  # value guesses the callee must confirm
Decision = Literal["emit", "reject"]


@dataclass
class PolicyConfig:
    mode: Literal["strict", "permissive"] = "strict"
    max_mutations_per_call: int = 8
    max_reprompt: int = 1


def decide(result: RepairResult, cfg: PolicyConfig) -> Decision:
    if len(result.mutations) > cfg.max_mutations_per_call:
        return "reject"
    if cfg.mode == "strict" and result.confidence == "low":
        if any(m.rule_id in _GUESSING_RULES for m in result.mutations):
            return "reject"
    return "emit"


class AuditLog:
    """Append-only JSONL log: one line per tool call, never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, object]) -> None:
        payload = {"ts": datetime.now(UTC).isoformat(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
