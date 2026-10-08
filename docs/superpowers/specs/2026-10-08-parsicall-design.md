# ParsiCall: Persian Tool-Call Repair and Benchmark Harness for Small Local LLMs

Date: 2026-10-08
Status: draft for review
Working name: **ParsiCall** (package `parsicall`). Verify PyPI/GitHub name availability before publishing; rename is a mechanical change if taken.

## 1. Problem

Small open-weight models (roughly 4B–14B) served locally on CPU or consumer GPUs are the practical option for Iranian teams: foreign APIs and Hugging Face are unreliable behind sanctions, and enterprise data cannot leave the premises. But these models fail at tool calling in two distinct ways:

1. **Structural failure.** Malformed JSON in `arguments`, tool calls emitted as prose or XML-ish text (`<tool_call>`) instead of the `tool_calls` field, truncated arguments, missing quotes, wrong types. A single malformed call aborts the agent turn with an HTTP 500 from llama.cpp or a silent no-op.
2. **Persian-argument failure.** The model picks the right tool but writes arguments that are semantically right for a human and wrong for the API:
   - `۱۴۰۴/۰۷/۱۷` (Persian digits, Jalali) where the schema wants `2026-10-08`
   - `تهران` typed with Arabic ي / ك instead of Persian ی / ک
   - `tehran` or `تهران ` with ZWNJ/space drift in multi-word values
   - `۲٬۵۰۰٬۰۰۰ ریال` in a numeric field; toman vs rial confusion
   - enum values written in Persian (`"آبی"`) against English enums (`blue`)

Failure (1) is a known ecosystem problem with existing tooling. Failure (2) has no open-source tooling in Persian — and it is the failure that silently corrupts business data: the call succeeds, the wrong date or wrong entity is stored.

Real cost: a booking or CRM agent that writes `1404/07/17` into a Gregorian column, or matches the wrong customer because of a ي/ی difference, fails hours later as a data bug that no trace shows. Teams currently discover this in production, one record at a time.

## 2. Target users

- Primary: Iranian engineers building agents on local models (llama.cpp, Ollama, vLLM, LM Studio) who need tool calling to stop dropping calls.
- Secondary: international teams serving non-English agent traffic on small local models (the repair pipeline is language-parameterized; Persian is the first and best-tested pack).
- Tertiary: hiring managers reading the repo as evidence of production engineering.

## 3. Competitive landscape (verified 2026-10-08)

| Project | Stars | What it does | What it does not do |
|---|---|---|---|
| `antoinezambelli/forge` (MIT, active 2026-09) | 2253 | OpenAI-compatible proxy; rescues tool calls emitted in Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, code fences into canonical `tool_calls` | No schema repair, no Persian/argument canonicalization, no Persian benchmark |
| `langchain-dev-utils` `ToolCallRepairMiddleware` | small | Repairs invalid JSON arguments against schema | LangChain-bound, English-only assumptions, no benchmark |
| praisonai `max_tool_repairs` budget | large | Retry budget for malformed local tool calls | Retries only; no argument semantics |
| `reactive-agents-ts` healing pipeline | 28 | ToolNameHealer/ParamNameHealer/TypeCoercer, dialect calibration | TypeScript, English-oriented, not a benchmark |
| ParsBench | 87 | Persian model benchmark (13 tasks) + app-eval with Persian-aware golden matching including `ToolCall("search_flights", date="2026-09-27")` | Does not repair tool calls; has no Persian **tool-calling model benchmark** — its 13 tasks are NLU/MMLU/math style |
| MLCL (arXiv 2601.05366), Arabic tool-call papers, BabelArena, ITC | papers | Multilingual tool-calling failure analysis | zh/hi/igbo/Arabic — no Persian; research artifacts, not deployable repair tooling |
| BFCL / EvalScope | large | De facto tool-calling eval | English prompts, no Persian digit/date/entity semantics |

**Gap (stated precisely):** no project combines (a) a deployable OpenAI-compatible repair layer that canonicalizes Persian arguments into schema-valid values, with (b) an auditable record of every mutation it made, with (c) a Persian tool-calling benchmark that measures whether the whole loop — model raw, model + repair — actually works. Each half exists somewhere; the combination does not.

**Honest overlap:** format-level rescue (failure 1) is already solved by forge et al. ParsiCall must not re-sell that as novelty. The contribution is failure 2 + the benchmark that proves it.

## 4. Non-goals

- Not an agent framework. No graph, no planner, no memory. It sits between an existing agent app and an existing model server.
- Not a model. No fine-tuning (optionally benchmark may *compare* fine-tuned models, but the project ships no weights).
- Not a Persian translation layer for prompts or tool names. Tool names and schemas stay as the API defines them; only *argument values* are canonicalized, and only when the model produced a Persian surface form of a value the schema accepts.
- Not a general LLM gateway (no provider routing, no cost management, no billing).
- No GPU required. Development, CI, and the benchmark run on CPU with llama.cpp serving a 1.7B–4B model; a larger model may be run opportunistically.
- English repair stays as a baseline path (needed anyway for the benchmark's A/B), but English is not the selling point.

## 5. Architecture

```text
Agent app (LangGraph / LangChain / any OpenAI SDK)
        │  OpenAI-compatible (streaming + non-streaming)
        ▼
┌─────────────────────────────── parsicall serve ───────────────────────────────┐
│ Gateway (stateless HTTP)                                                      │
│   ├─ passthrough when response is already schema-valid                       │
│   └─ RepairPipeline (only on tool_calls):                                     │
│        Stage F  Format rescue: fences, <tool_call>/Mistral text forms,          │
│                 truncation, json-repair                                       │
│        Stage S  Schema repair: type coercion, required-arg fill attempts,     │
│                 enum/int/float/date format checks against JSON Schema         │
│        Stage P  Persian canonicalization: digit scripts, Jalali→ISO,          │
│                 Yeh/Kaf/Alef folding, ZWNJ normalization,                     │
│                 rial/toman→number, entity matching against known values       │
│        Every mutation recorded: path, before, after, rule id                  │
│   Policy: mutation budget, fail-open/fail-closed mode, JSONL audit log        │
└───────────────────────────────────────────────────────────────────────────────┘
        │  OpenAI-compatible
        ▼
Local model server (llama.cpp / Ollama / vLLM / LM Studio)
```

Components:

1. **`parsicall.gateway`** — Starlette app exposing `/v1/chat/completions` (and `/v1/completions` passthrough). Forwards request untouched; intercepts the response's `tool_calls` for repair. Streaming: buffers only if a tool call is present in the delta stream (tool-call streaming in OpenAI format arrives as argument fragments; re-emit repaired arguments in a final delta, then finish). passthrough latency budget: < 2 ms p50 when no repair is needed.
2. **`parsicall.repair`** — three pure stages, each `tool_call -> RepairResult` with `mutations: list[Mutation]`, `confidence`, `repaired: bool`. Stages compose; a stage that cannot improve returns input unchanged (idempotent by construction: repair(repair(x)) == repair(x)).
3. **`parsicall.norm`** — Persian normalization library (dependency-light; uses `persiantools` for Jalali conversion, own folding tables). Public functions: `digits_to_ascii`, `jalali_to_iso`, `fold_persian_variants`, `normalize_zwnj`, `parse_persian_amount`. Pure, deterministic, unit-testable without any model.
4. **`parsicall.policy`** — decides whether a repaired call is emitted, retried, or failed: max mutations per call, max re-prompt (one retry with a targeted error message if repair fails), audit write (JSONL, append-only, includes original and final argument bytes).
5. **`parsicall.bench`** — dataset + runner + scorer (Section 9).
6. **`parsicall.cli`** — `parsicall serve`, `parsicall bench`, `parsicall check` (dry-run one message through the pipeline), `parsicall inspect <audit-log>`.

Data flow for one request: app → gateway → model server → response → if no `tool_calls`, return; else run stages F→S→P → compare to original → if identical, return original (zero-copy fast path); else apply policy → log mutation record → return repaired response with `x-parsicall-repaired: true` header and `repair` metadata object (optional, controlled by config).

Error handling:
- Stage S cannot produce a schema-valid call → one repair re-prompt to the model (configurable, default 1), appending the JSON Schema error as a tool-role message. If still invalid: **fail-closed** (return OpenAI-shaped error object `tool_call_repair_failed`) in strict mode; **fail-open** (return the raw call plus a warning) in permissive mode. Default: strict.
- Model server unreachable/timeouts: passthrough semantics — return 502 with upstream body; never hang > upstream timeout + 500 ms.
- Audit log write failure: log to stderr, never block the response.

## 6. Repair rules (the core design table)

Every rule has an ID, so the audit log, benchmark attribution, and docs all reference the same names.

| ID | Stage | Trigger | Rule | Reversible |
|---|---|---|---|---|
| F01 | F | JSON parse error | `json-repair` style salvage | yes (original kept) |
| F02 | F | tool call in `content` (`` / `[TOOL_CALLS]` / fenced JSON) | extract + re-emit as `tool_calls` | yes |
| F03 | F | truncated arguments | close brackets/strings only if trailing structure is unambiguous; else no-op | yes |
| S01 | S | `"1404/07/17"` in `format: date` | convert to ISO | yes |
| S02 | S | numeric field, Persian/Arabic digits or thousands separators | ASCII-ize and parse | yes |
| S03 | S | string field, untrimmed / wrong-case enum | trim + case-fold enum match | yes |
| S04 | S | missing optional arg with obvious single candidate | **never auto-fill** (skip rule — listed to document refusal) | n/a |
| P01 | P | digit script mismatch (`۰-۹`, `٠-٩`) | → ASCII | yes |
| P02 | P | Jalali date/datetime (numeric, month-name, or both) | → `YYYY-MM-DD` per schema format | yes |
| P03 | P | Arabic ي ك ۀ ء variants in values | → Persian ی ک | yes |
| P04 | P | ZWNJ/space drift in multi-word values | normalize to canonical form; only replace if a **known-value index match** exists | only with match |
| P05 | P | rial/toman wording in numeric fields | convert per configured rate flag; if rate unspecified and both appear, **do not guess** — fail closed | conditional |
| P06 | P | free-text value matches a known entity (city, product, agent name) after folding | snap to stored canonical form | only with match |

Guardrails (these are requirements, not niceties):
- P04/P06 fire only against an explicitly configured **known-value index** (CSV/JSON the user supplies) or enum lists. No fuzzy auto-repair of free text without a match; fuzzy-only matches are logged as `confidence: low` and dropped in strict mode.
- Every mutation is logged with rule ID and before/after. Nothing is silently changed.
- Repair never invents an argument that was absent (no P-rule creates a key).

## 7. Scope of the benchmark

**Name:** FaTool-Bench (working name) — Persian tool-calling evaluation for small local models, runnable on CPU.

**Corpus (MVP: 300 cases, target 500 by 1.0):**
- 60% argument-semantic cases: Jalali dates (5 forms: `۱۴۰۴/۰۷/۱۷`, `۱۷ مهر ۱۴۰۴`, `دهم مهر`, mixed-script), digit scripts, rial/toman amounts, entity variants (Yeh/Kaf/spaced), enum-in-Persian.
- 25% structural cases: malformed JSON, text-form calls, truncation, wrong types.
- 15% tool-selection cases: correct tool, wrong tool, no-tool-when-asked (measures over-eagerness).

**Metrics (all reported per stage: raw / raw+repair):**
- `parse_fail_rate` — no valid `tool_calls` emitted
- `arg_exact_match` — canonicalized args equal gold (AST comparison, BFCL-style)
- `arg_semantic_match` — equal after *gold* canonicalization only (isolates repair's contribution from model sloppiness)
- `tool_choice_acc`
- `over_repair_rate` — pipeline mutated a correct call (must be ~0; this is the false-positive metric)
- `repair_latency_ms` p50/p95 (budget: < 5 ms p95 without re-prompt)
- `mutation_attribution` — fixes by rule ID (tells users which rules earn their keep)

**Protocol:** temperature 0, seed fixed, 1 sample per case for the headline table, 3 samples for a stability appendix. Two runs of the same model must not drift more than the reported CI.

**Models in the published table (CPU-runnable):** Qwen3-1.7B, Qwen3-4B, Gemma-3-4B or Llama-3.2-3B via llama.cpp. Larger models (8B–14B) added only if hardware allows; explicitly labeled when run elsewhere.

**Baselines:** (1) raw model, (2) model + forge (format rescue only), (3) model + ParsiCall. ParsiCall must beat forge on `arg_semantic_match` and tie on `parse_fail_rate`. If it does not, that is published as-is.

## 8. Technology stack

- Python 3.12, Starlette + uvicorn (gateway), httpx (upstream client)
- `json-repair` for F01 (existing dependency, not reinvented)
- `persiantools` for Jalali conversion (existing, MIT-adjacent, 6.2.0 on PyPI)
- `jsonschema` for Stage S
- Pydantic for config and the CLI
- pytest for unit tests; `llama-cpp-python` or plain llama-server for CI smoke (tiny model, CPU)
- No database: audit is JSONL; benchmark results are JSON + Markdown.
- Deployment: single container; Dockerfile + `docker compose` with a llama.cpp server and a demo agent script.

## 9. Repository structure

```text
parsicall/
  README.md  METHODOLOGY.md  LIMITATIONS.md
  src/parsicall/
    gateway/        # Starlette app, passthrough, streaming repair
    repair/         # stages F/S/P, rule registry
    norm/           # persian normalization (pure functions)
    policy/         # budget, fail modes, audit writer
    bench/          # corpus loader, runner, scorer, report
    cli.py
  bench/corpus/     # 300+ JSONL cases + gold args
  examples/         # LangGraph agent + llama-server compose stack
  tests/            # unit (norm, rules), integration (gateway), golden (audit logs)
  results/          # published benchmark JSON + generated table
  docs/
```

## 10. Testing strategy

- **Unit:** every norm function and rule gets table-driven tests including the tricky digit/date/ZWNJ cases; idempotency property test (`repair(repair(x)) == repair(x)`) across a fixture set of 50 malformed calls.
- **Golden:** full gateway request/response fixtures, including streaming, with expected audit-log JSON (schema-validated).
- **Mutation tests on the scorer:** deliberately corrupt gold answers and assert the benchmark catches it — the scorer is part of the product.
- **CI:** lint + typecheck (`ruff`, `mypy`) + unit + a 10-case live smoke against llama.cpp with a 1.7B model on CPU (skipped with clear message if no model available; never silently passes).
- Non-trivial logic leaves runnable checks: `pytest` is the one runnable check.

## 11. Milestones and kill gates (6-week MVP; 10-week full)

| Week | Deliverable | Kill gate (stop or pivot if…) |
|---|---|---|
| 1 | Norm library + rule table + 50-case corpus hand-written; run raw Qwen3-4B on corpus | Raw model shows < 5% Persian-arg failures → the premise is false; stop |
| 2 | Stage F + S + gateway passthrough, audit log | Repair cannot reach ≥ 90% parse success on structural cases without over-repair > 1% → pivot to benchmark-only |
| 3 | Stage P + known-value index, policy modes | Over-repair rate > 2% or Jalali conversion wrong on month-name dates → freeze P04/P06 behind opt-in |
| 4 | Benchmark runner + scorer + baselines (raw, forge) | ParsiCall does not beat forge on `arg_semantic_match` → publish honestly, pivot emphasis to benchmark |
| 5 | Streaming, re-prompt path, Docker compose demo | — |
| 6 | Full corpus 300, published results table, README, METHODOLOGY | — |
| 7–10 (stretch) | Second language pack as proof of parameterization (Arabic fold) or vLLM server-side integration; Hugging Face dataset publication of the corpus | — |

If week 1's gate fails, the fallback project is the Persian-safe data agent (already scoped in conversation) — the norm library transfers directly.

## 12. Risks

1. **"Yet another proxy" perception.** Mitigation: benchmark-first README (results table above the fold), and forge explicitly credited for format rescue.
2. **Small models fail regardless of repair.** Mitigation: the benchmark separates `parse_fail` (repair's domain) from `arg_semantic` (model's domain); a bad result is still a publishable measurement.
3. **ParsBench absorbs the benchmark.** Mitigation: they evaluate apps and rank models on NLU; we measure tool-call pipelines. Possible future: contribute the corpus to ParsBench as a task — that is a feature, not a defeat.
4. **Over-repair corrupting good calls.** Mitigation: `over_repair_rate` is a headline metric; strict mode defaults; mutations always logged and reversible by replaying the original from the audit log.
5. **Jalali ambiguity** (e.g. missing year). Rule P02 refuses incomplete dates; documented, not guessed.

## 13. Definition of done (MVP)

- `parsicall serve` in front of llama.cpp; an existing LangGraph agent works unchanged by changing `base_url`.
- 300-case corpus, published table for ≥ 3 models with raw/forge/parsicall columns.
- Over-repair rate < 1% reported from the run (not asserted in prose).
- Docker compose demo that runs the benchmark end-to-end on CPU in under 30 minutes.
- `LIMITATIONS.md` states what the tool does not do (Section 4 plus measured ceilings).

## 14. What this does not claim

- Not the first tool-call repair layer (forge, praisonai, langchain-dev-utils exist).
- Not a fix for model capability: it raises a floor, it does not make a 1.7B model good at tool selection.
- Not production-proven: no users yet; do not write "used in production" anywhere.
- Benchmark numbers apply to the exact model/server versions recorded in `results/`.
- Persian-only correctness claims are scoped to the documented rule table; unfuzzed free text is intentionally untouched.
