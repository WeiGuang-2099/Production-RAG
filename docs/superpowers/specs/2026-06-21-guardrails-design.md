# Guardrails — Design

**Status:** approved (design phase)
**Date:** 2026-06-21
**Scope:** Subsystem #3 of 4 (Agentic RAG → model routing → **guardrails** → MCP server).
This spec covers guardrails only.

## Goal

Add safety at the API edge: sanitize what comes in and what goes out, with transparent,
heuristic-first checks (no heavy framework). Three guards:

1. **Input — prompt-injection detection** → block the request (HTTP 400).
2. **Output — PII redaction** → redact the answer before returning.
3. **Output — toxicity** → flag the response (annotate, do not block).

All detection is pure functions over regex/wordlists — deterministic, unit-testable, no LLM calls.

## Module structure (`app/guardrails/`)

- `__init__.py`
- `injection.py` — `detect_injection(text: str) -> list[str]` (matched rule labels; empty = clean)
- `pii.py` — `redact_pii(text: str) -> tuple[str, list[str]]` (redacted text, sorted PII types found)
- `toxicity.py` — `TOXIC_TERMS` constant + `detect_toxicity(text: str) -> list[str]` (matched terms)
- `service.py` — orchestration, gated by `GUARDRAILS_ENABLED`:
  - `check_input(question: str) -> list[str]` — injection labels, or `[]` when disabled
  - `apply_output(answer: str) -> dict` — `{"answer": redacted, "pii_redacted": [...], "flags": [...]}`; returns the answer unchanged with empty lists when disabled

## Detection methods (v1)

**Injection** — case-insensitive regex for common attacks; `detect_injection` returns the labels of
matched rules:

- `ignore_previous`: `ignore (all |the )?(previous|prior|above) (instruction|prompt)s?`
- `disregard_above`: `disregard (the |all )?(previous|prior|above)`
- `system_prompt`: `system prompt`
- `you_are_now`: `you are now`
- `reveal_prompt`: `reveal (your )?(system )?(prompt|instructions)`
- `override_rules`: `override (your |the )?(instruction|rule|setting)s?`
- `pretend`: `pretend (to be|you are)`
- `jailbreak`: `jailbreak`
- `dan`: `\bDAN\b`
- `new_instructions`: `new instructions\s*:`

(Generic phrases like "act as" are deliberately excluded to avoid false positives on benign prompts.)

**PII** — `redact_pii` applies these in order (broad/structured first) and replaces matches with
`[REDACTED_<TYPE>]`:

- `email`: `[\w.+-]+@[\w-]+\.[\w.-]+` → `[REDACTED_EMAIL]`
- `credit_card`: `\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b` → `[REDACTED_CC]`
- `ssn`: `\b\d{3}-\d{2}-\d{4}\b` → `[REDACTED_SSN]`
- `ipv4`: `\b(?:\d{1,3}\.){3}\d{1,3}\b` → `[REDACTED_IP]`
- `phone`: `(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}` → `[REDACTED_PHONE]`

Returns the redacted string and the sorted unique list of types found.

**Toxicity** — `TOXIC_TERMS` is a small, conservative, illustrative wordlist of abusive terms
(e.g. `idiot`, `moron`, `scumbag`); `detect_toxicity` returns word-boundary, case-insensitive
matches. Intentionally minimal to limit false positives.

## Integration (API edge)

Applied in `app/api/routes_chat.py` and `app/api/routes_agent.py` for all four endpoints:

- **Input:** at the top of each handler, `labels = check_input(body.question)`; if non-empty, raise
  `HTTPException(400, detail={"error": "blocked by input guardrails", "patterns": labels})`. For the
  stream endpoints this check runs before the `StreamingResponse` is created (so a blocked request
  gets a clean 400, not a streamed error).
- **Output:** before returning, `g = apply_output(answer)`; set `answer = g["answer"]` and include
  `guardrails = {"pii_redacted": g["pii_redacted"], "flags": g["flags"]}` in the response. For the
  stream endpoints, apply to the final answer inside the `done` event.

## Config

- `GUARDRAILS_ENABLED: bool = True` (safety feature, on by default; `false` makes both helpers pass through).

## Response shape

`ChatResponse` and `AgentResponse` gain `guardrails: dict = {}`, e.g.
`{"pii_redacted": ["email"], "flags": []}`.

## Defaults (confirmed)

- Toxicity → **flag** (not block).
- `GUARDRAILS_ENABLED` → **on**.
- Redaction applies to the **answer only**, not `sources` (sources are the user's own ingested corpus).

## Error handling

Detection helpers are pure and do not raise. Input blocking is expressed only as an `HTTPException`
at the route layer. `apply_output` never raises; redaction is always safe.

## Testing (TDD)

- **Unit:** `detect_injection` (each rule matches its attack; benign text → `[]`); `redact_pii`
  (each type redacted; clean text unchanged; types list correct); `detect_toxicity` (listed term
  flagged; benign → `[]`).
- **Service:** `check_input` / `apply_output` honor `GUARDRAILS_ENABLED` (active when on, passthrough when off).
- **API:** injection question → 400; an answer containing PII → redacted in the response with
  `guardrails.pii_redacted` populated; a toxic answer → `guardrails.flags` populated (not blocked);
  `GUARDRAILS_ENABLED=false` → passthrough.

## Trade-offs & limitations (stated honestly)

- Heuristic injection detection is bypassable — follow-up: an LLM classifier escalation.
- Wordlist toxicity is incomplete and false-positive-prone — follow-up: OpenAI moderation API.
- Streaming redacts only the final `done` answer, not individual tokens — follow-up: buffered
  streaming redaction.
- PII regexes (esp. phone) are best-effort — follow-up: a dedicated library (e.g. Presidio).

## Follow-up (out of scope)

- LLM/moderation-API upgrades for injection + toxicity.
- Subsystem #4 (MCP server) builds on this.
