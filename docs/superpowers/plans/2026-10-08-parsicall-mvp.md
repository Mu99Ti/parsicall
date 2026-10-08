# ParsiCall MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-compatible proxy that repairs malformed and Persian-argument tool calls from small local LLMs, plus a 300-case Persian tool-calling benchmark that measures the repair.

**Architecture:** A stateless Starlette gateway sits between an agent app and a local model server. Responses containing `tool_calls` pass through three pure repair stages (Format rescue → Schema repair → Persian canonicalization), each recording mutations to a JSONL audit log under a strict policy. A separate benchmark runner scores raw vs repaired outputs against a gold corpus.

**Tech Stack:** Python 3.13 (uv-managed), Starlette + uvicorn, httpx, json-repair, jsonschema, persiantools, pytest, ruff, mypy.

## Global Constraints

- Python floor: 3.12 (dev via Homebrew 3.13 at `/opt/homebrew/bin/python3.13`)
- Package name: `parsicall`, src layout under `src/parsicall/`
- No GPU required; all tests run CPU-only. Live-model tests are opt-in via `PARSICALL_LIVE=1`
- Dependency budget: only `starlette`, `uvicorn`, `httpx`, `json-repair`, `jsonschema`, `persiantools`, `pydantic` (+ dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `respx`)
- Repair must be idempotent: `repair(repair(x)) == repair(x)` for every stage
- Every mutation logged with rule ID; repair never creates an absent argument key
- Commit after every task, conventional-commit messages, push after each task

---

### Task 1: Repo + scaffold + CI baseline

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `Makefile`
- Create: `src/parsicall/__init__.py` (version only), `tests/test_smoke.py`

**Interfaces:**
- Produces: installable package `parsicall` with `__version__ == "0.1.0"`, `make lint` / `make test` targets.

- [ ] **Step 1: Init git repo in `/Users/muhamadtalebi/parsicall`, write pyproject with deps above, src layout, ruff+mypy+pytest config**

```toml
[project]
name = "parsicall"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["starlette>=0.40", "uvicorn>=0.30", "httpx>=0.27", "json-repair>=0.30", "jsonschema>=4.22", "persiantools>=6.0", "pydantic>=2.7"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.6", "mypy>=1.11", "respx>=0.21"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/parsicall"]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
[tool.mypy]
python_version = "3.12"
strict = true
packages = ["parsicall"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: `uv venv --python 3.13 && uv pip install -e '.[dev]'`, write smoke test `assert parsicall.__version__`**

- [ ] **Step 3: Run `uv run pytest -q` → PASS; `uv run ruff check .` → clean**

- [ ] **Step 4: README (one-paragraph pitch + status = MVP in progress), Makefile with `lint: ruff check . && mypy`, `test: pytest -q`**

- [ ] **Step 5: Commit `chore: scaffold parsicall package` + push; create GitHub repo `Mu99Ti/parsicall` public via `gh repo create --source . --public --push`**

---

### Task 2: Persian normalization library (`parsicall.norm`)

**Files:**
- Create: `src/parsicall/norm/__init__.py`, `src/parsicall/norm/digits.py`, `src/parsicall/norm/dates.py`, `src/parsicall/norm/text.py`, `src/parsicall/norm/money.py`
- Test: `tests/test_norm.py`

**Interfaces:**
- Produces (all pure, deterministic):
  - `digits_to_ascii(s: str) -> str` — maps ۰-۹ and ٠-٩ to ASCII, strips thousands separators (٬ ، space) inside numbers
  - `jalali_to_iso(s: str) -> str | None` — parses `1404/07/17`, `۱۴۰۴/۰۷/۱۷`, `۱۷ مهر ۱۴۰۴`, `دهم مهر ۱۴۰۴` (ordinals یکم..سی‌ام); returns `YYYY-MM-DD` or `None` if incomplete/ambiguous
  - `fold_persian_variants(s: str) -> str` — ي→ی, ك→ک, ۀ→ه, ء variants, Arabic-Indic already handled by digits; strips tashkeel
  - `normalize_zwnj(s: str) -> str` — ZWNJ→space then collapse whitespace, trim (canonical *form* used for matching only)
  - `parse_persian_amount(s: str) -> tuple[int, str] | None` — `"۲٬۵۰۰٬۰۰۰ ریال"` → `(2500000, "rial")`; toman detected separately; returns None when currency absent or both present

- [ ] **Step 1: Write failing tests (table-driven) for each function**

```python
import pytest
from parsicall.norm import digits_to_ascii, jalali_to_iso, fold_persian_variants, normalize_zwnj, parse_persian_amount

@pytest.mark.parametrize("raw,expected", [
    ("۱۴۰۴", "1404"),
    ("١٤٠٤", "1404"),
    ("۲٬۵۰۰", "2500"),
    ("2500", "2500"),
])
def test_digits_to_ascii(raw, expected):
    assert digits_to_ascii(raw) == expected

@pytest.mark.parametrize("raw,expected", [
    ("1404/07/17", "2026-10-08"),
    ("۱۴۰۴/۰۷/۱۷", "2026-10-08"),
    ("۱۷ مهر ۱۴۰۴", "2026-10-08"),
    ("", None),
    ("1404/07", None),
])
def test_jalali_to_iso(raw, expected):
    assert jalali_to_iso(raw) == expected

def test_fold_persian_variants():
    assert fold_persian_variants("علي رضايي") == "علی رضایی"

def test_normalize_zwnj():
    assert normalize_zwnj("می\u200cخواهم") == "می خواهم"

@pytest.mark.parametrize("raw,expected", [
    ("۲٬۵۰۰٬۰۰۰ ریال", (2500000, "rial")),
    ("2,500,000 ریال", (2500000, "rial")),
    ("۲۵۰ هزار تومان", (250000, "toman")),
    ("۲۵۰۰۰۰", None),
])
def test_parse_persian_amount(raw, expected):
    assert parse_persian_amount(raw) == expected
```

- [ ] **Step 2: Run → FAIL (ImportError)**

- [ ] **Step 3: Implement. `jalali_to_iso` uses `persiantools.jdatetime.JalaliDate`; month-name table `فروردین..اسفند`; ordinal words `یکم دوم سوم ... بیستم سی‌ام` — parse what maps cleanly, else `None`. Never guess missing components.**

- [ ] **Step 4: Run → PASS; `ruff` + `mypy` clean**

- [ ] **Step 5: Commit `feat(norm): Persian digit/date/variant/amount normalization` + push**

---

### Task 3: Rule registry + Format rescue stage F

**Files:**
- Create: `src/parsicall/repair/types.py`, `src/parsicall/repair/format_stage.py`, `src/parsicall/repair/__init__.py`
- Test: `tests/test_format_stage.py`

**Interfaces:**
- Produces:
  - `@dataclass Mutation: rule_id: str; path: str; before: str; after: str`
  - `@dataclass RepairResult: call: ToolCall; mutations: list[Mutation]; confidence: str  # "high"|"low"`
  - `ToolCall = dict` shaped `{"id": str, "type": "function", "function": {"name": str, "arguments": str}}` (arguments always a JSON *string* on the wire)
  - `repair_format(text: str, call: ToolCall) -> RepairResult` handling:
    - F01: `function.arguments` unparseable JSON → `json_repair.loads` salvage; if salvage yields dict, re-serialize; rule `F01`
    - F02: tool-call text found in a content string (` ```json ` fence containing `"name"`/`"arguments"`, `<tool_call>...</tool_call>`, Mistral `[TOOL_CALLS]name{args}`) → extract into ToolCall; rule `F02`. Exposed as `extract_tool_call(content: str) -> RepairResult | None`
    - F03: truncated args (unbalanced braces/brackets) → close only if exactly one way to close (no ambiguity between `}`/`]` at any point); else no-op; rule `F03`

- [ ] **Step 1: Failing tests: fenced JSON extraction, tool_call tag extraction, truncated `{"a": "b` salvage, valid-args no-op (mutations==[])**

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement (regex-based extraction; `json_repair` for salvage)**

- [ ] **Step 4: PASS + lint + mypy**

- [ ] **Step 5: Commit `feat(repair): format rescue stage F (F01-F03)` + push**

---

### Task 4: Schema repair stage S

**Files:**
- Create: `src/parsicall/repair/schema_stage.py`
- Test: `tests/test_schema_stage.py`

**Interfaces:**
- Produces: `repair_schema(call: ToolCall, schema: dict, *, known_values: dict[str, list[str]] | None = None) -> RepairResult`
  - S01: value matches `^\d{3,4}[/-]\d{1,2}[/-]\d{1,2}$` AND schema format is `date`/`date-time` → `jalali_to_iso`; rule `S01`
  - S02: schema type `integer`/`number`, value is Persian/Arabic-digit string → `digits_to_ascii` + parse; rule `S02`
  - S03: schema `enum` → trim + casefold match against enum; Persian→English enum matched only via `known_values` mapping if provided; rule `S03`
  - Validation via `jsonschema` on resulting args dict; returns original when schema invalid in a way not covered by rules (no guessing)
  - Never adds keys (S04 documented refusal: not implemented)

- [ ] **Step 1: Failing tests (one per rule + no-op on valid + never-add-key assertion)**

```python
def test_s01_jalali_date():
    call = tc({"date": "۱۴۰۴/۰۷/۱۷"})
    schema = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}}, "required": ["date"]}
    res = repair_schema(call, schema)
    assert json.loads(res.call["function"]["arguments"])["date"] == "2026-10-08"
    assert [m.rule_id for m in res.mutations] == ["S01"]

def test_never_adds_key():
    call = tc({"date": "2026-01-01"})
    schema = {"type": "object", "properties": {"date": {"type": "string"}, "note": {"type": "string"}}, "required": ["date"]}
    res = repair_schema(call, schema)
    assert set(json.loads(res.call["function"]["arguments"])) == {"date"}
```

- [ ] **Step 2-5: FAIL → implement → PASS → lint → commit `feat(repair): schema repair stage S (S01-S03)` + push**

---

### Task 5: Persian canonicalization stage P

**Files:**
- Create: `src/parsicall/repair/persian_stage.py`
- Test: `tests/test_persian_stage.py`

**Interfaces:**
- Produces: `repair_persian(call: ToolCall, schema: dict, *, known_values: dict[str, list[str]] | None = None, *, toman_rate: int | None = None) -> RepairResult`
  - P01: any string arg containing ۰-۹/٠-٩ → `digits_to_ascii` (type string or number); rule `P01`
  - P02: string arg that `jalali_to_iso` parses (and result non-None) in a date-ish property (format date, or property name matches `(?i)date|created|birth|day`) → ISO; rule `P02`
  - P03: Arabic ي/ك/ۀ present → `fold_persian_variants`; rule `P03`
  - P04: value normalizes (`fold` + `normalize_zwnj` + casefold) to equal an entry of `known_values[property]` (or any known value for non-enum free-text when explicitly enabled by passing the key) → snap to canonical entry; else no-op; confidence `high` only on exact post-fold equality; rule `P04`
  - P05: `parse_persian_amount` returns currency → if only one currency and `toman_rate` configured, convert to it and strip wording; if toman_rate is None → fail closed (no mutation, confidence `low`); rule `P05`
  - Ordering inside P: P01 → P03 → P02 → P05 → P04 (digits first so date parser sees ASCII)

- [ ] **Step 1: Failing tests per rule, incl. P05 fail-closed-without-rate, P04 no-match no-op**

- [ ] **Step 2-5: FAIL → implement → PASS → lint → commit `feat(repair): Persian canonicalization stage P (P01-P05)` + push**

---

### Task 6: Pipeline composition, idempotency property, policy + audit

**Files:**
- Create: `src/parsicall/repair/pipeline.py`, `src/parsicall/policy/__init__.py` (audit writer + config)
- Test: `tests/test_pipeline.py`, `tests/test_policy.py`

**Interfaces:**
- Produces:
  - `repair_call(call, *, tools_schemas: dict[str, dict], known_values, toman_rate) -> RepairResult` — runs F(arguments) → S → P in order using the schema for `call["function"]["name"]`; collects mutations with rule IDs
  - `@dataclass PolicyConfig: mode: Literal["strict","permissive"] = "strict"; max_mutations_per_call: int = 8; max_reprompt: int = 1`
  - `AuditLog.append(record: dict) -> None` — JSONL append-only file; record = `{ts, tool, original, final, mutations, decision}`
  - `decide(result, cfg) -> Literal["emit","reject"]` — reject if `len(mutations) > max_mutations_per_call` or confidence low in strict mode for P04/P05

- [ ] **Step 1: Failing tests:**

```python
@pytest.mark.parametrize("call,args_schemas", FIXTURES)  # 50 fixtures built from tests/fixtures/malformed.jsonl
def test_idempotent(call, args_schemas):
    once = repair_call(call, tools_schemas=args_schemas, known_values=None, toman_rate=None)
    twice = repair_call(once.call, tools_schemas=args_schemas, known_values=None, toman_rate=None)
    assert twice.mutations == []
    assert twice.call == once.call
```

  (Fixture file: 50 rows `{"tool", "arguments_raw", "schema"}` — hand-written, covers all rules + valid no-ops.)

- [ ] **Step 2-5: FAIL → implement → PASS → commit `feat(repair): pipeline composition, idempotency, policy audit` + push**

---

### Task 7: Gateway (Starlette) with passthrough + streaming repair

**Files:**
- Create: `src/parsicall/gateway/app.py`, `src/parsicall/gateway/config.py`
- Test: `tests/test_gateway.py` (respx-mocked upstream)

**Interfaces:**
- Produces: `create_app(upstream: str, cfg: GatewayConfig) -> Starlette`
  - `POST /v1/chat/completions`: forward body to upstream; if response has no `tool_calls` → return untouched; else run `repair_call` per call, apply `decide`; strict-reject → OpenAI-shaped error `{"error": {"message": "tool_call_repair_failed", "type": "invalid_tool_call", "code": rule_ids}}` HTTP 400
  - Header `x-parsicall-repaired: true` set only when a mutation was emitted
  - Streaming: parse SSE deltas; if `tool_calls` fragments present, accumulate, repair at `finish_reason`, re-emit one corrected delta then `[DONE]`; non-tool streams pass through byte-for-byte
  - `GET /healthz` → `{"status": "ok"}`
  - Audit written per tool-call response via `AuditLog`

- [ ] **Step 1: Failing tests with respx: passthrough-no-toolcalls (untouched body), repaired-body-on-F01, strict-reject-400, header set, healthz**

- [ ] **Step 2-5: FAIL → implement → PASS → commit `feat(gateway): OpenAI-compatible repair proxy` + push**

---

### Task 8: Benchmark corpus (300 cases) + scorer

**Files:**
- Create: `bench/corpus/*.jsonl` (6 files by category), `src/parsicall/bench/scorer.py`, `src/parsicall/bench/__init__.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Corpus row: `{"id": "arg-date-001", "category": "arg_semantic|structural|tool_choice", "user": "…", "tool": "book_appointment", "schema": {…}, "gold_args": {…}, "tools": [{…}]}`
- Produces: `score(raw_call, gold_args, repaired_call) -> {"parse_ok": bool, "exact": bool, "semantic": bool}` where semantic compares args after canonicalizing *gold* with the same norm functions; `summarize(rows) -> dict` with rates listed in spec §7 (parse_fail_rate, arg_exact_match, arg_semantic_match, tool_choice_acc, over_repair_rate, repair_latency_ms p50/p95, mutation_attribution)

- [ ] **Step 1: Write scorer tests first (known raw/repaired pairs → expected metrics)**
- [ ] **Step 2-5: FAIL → implement scorer → PASS → commit `feat(bench): scoring metrics incl. over-repair` + push**

---

### Task 9: Corpus authoring 300 cases

**Files:**
- Create: 6 JSONL files under `bench/corpus/` — `dates.jsonl` (75), `digits_money.jsonl` (60), `entities.jsonl` (45), `structural.jsonl` (75), `tool_choice.jsonl` (45)

**Interfaces:** row format as Task 8; `gold_args` always in canonical ASCII/ISO form.

- [ ] **Step 1: Author rows (persian, realistic Iranian domains: booking, CRM, invoice, support). Dates spread across all 5 Jalali forms.**
- [ ] **Step 2: Validation test `tests/test_corpus.py`: unique ids, required keys, gold_args schema-valid against `schema`, ≥300 rows, each category ≥ its quota**
- [ ] **Step 3: PASS → commit `feat(bench): 300-case Persian tool-call corpus` + push**

---

### Task 10: Benchmark runner (live models) + results pipeline

**Files:**
- Create: `src/parsicall/bench/runner.py`, `src/parsicall/cli.py`, `results/.gitkeep`

**Interfaces:**
- Produces: CLI `parsicall bench --endpoint http://localhost:8000/v1 --model M --mode {raw,parsicall} --corpus bench/corpus --out results/M_raw.json` and `parsicall report results/*.json --markdown results/RESULTS.md`
- Runner: OpenAI-compatible chat call per row, tools=[row.tools], temperature 0, records latency; respects `PARSICALL_LIVE` guard (without it, CLI errors clearly: live model required)
- Manifest in each results JSON: model, endpoint, timestamp, corpus hash

- [ ] **Step 1: Unit-test runner against a respx fake endpoint (no live model)**
- [ ] **Step 2-5: PASS → commit `feat(bench): runner + markdown report CLI` + push**

---

### Task 11: Live validation + real results (kill-gate check)

- [ ] **Step 1: Install llama.cpp (`brew install llama.cpp`) or `uv pip install llama-cpp-python`; download a small GGUF (Qwen3-4B Q4_K_M, mirror-aware; if HF blocked, use an alternate mirror e.g. ModelScope)**
- [ ] **Step 2: `llama-server -hf ... --jinja -c 8192 --port 8000`**
- [ ] **Step 3: Run `parsicall bench --mode raw` and `--mode parsicall` (gateway on :8080); then forge comparison if installable, else record "baseline unavailable" honestly**
- [ ] **Step 4: Kill-gate evaluation per spec §11: raw Persian-arg failure ≥5%? repair beats raw on arg_semantic_match? over_repair < 1%? Write findings into `results/RESULTS.md` verbatim including negatives**
- [ ] **Step 5: Commit `bench: initial live results (qwen3-4b)` + push**

---

### Task 12: Demo (Docker compose) + docs

**Files:**
- Create: `docker-compose.yml` (llama.cpp service + gateway + demo agent script), `examples/demo_agent.py`, `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`, final `README.md`

- [ ] **Step 1: compose stack with a tiny model for CI-friendliness (`Qwen/Qwen3-0.6B-GGUF` or smallest available), `examples/demo_agent.py` = ~60-line OpenAI-SDK script issuing a Persian booking request showing broken-without/good-with gateway**
- [ ] **Step 2: METHODOLOGY (metrics definitions, protocol) + LIMITATIONS (spec §14 verbatim-ish) + README rewrite (results table above the fold once Task 11 data exists)**
- [ ] **Step 3: `docker compose up` smoke locally; commit `docs: demo stack, methodology, limitations` + push**

---

### Task 13: Final polish + release

- [ ] **Step 1: `ruff check .`, `mypy`, `pytest -q` all green; fix**
- [ ] **Step 2: VERSION/tag `v0.1.0`, `gh release create v0.1.0 --notes-file docs/RELEASE_NOTES.md`**
- [ ] **Step 3: Push final state; verify GitHub Actions CI green on the pushed ref**

## Self-Review

1. **Spec coverage:** norm (T2), F/S/P rules (T3-5, incl. documented S04 refusal), pipeline idempotency+policy+audit (T6), gateway+streaming+strict mode (T7), benchmark scorer incl. over_repair (T8), 300-case corpus with quotas (T9), runner+report (T10), live results + kill gates (T11), demo+docs+limitations (T12), release (T13). Gaps: spec says "forge comparison" — T11 marks it optional with honest fallback, acceptable. Arabic stretch (§11 7-10wk) explicitly out of MVP.
2. **Placeholders:** none; all steps carry code or exact commands.
3. **Type consistency:** `RepairResult`/`Mutation`/`ToolCall` defined T3, reused T4-T7; `repair_call` signature fixed T6 and used by T7/T10.
