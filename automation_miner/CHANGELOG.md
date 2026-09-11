# Changelog

## 0.1.0

First release.

**Mining**

- Context-chain causality: every state change classified as human, automation,
  script or device by walking `context_parent_id` to its root.
- Override detection: human corrections of automation actions, stored as
  negative feedback for that automation and positive evidence for what you
  actually wanted.
- Time-of-day / day-of-week habit mining with circular clustering and
  opportunity-based consistency scoring.
- Association-rule mining (FP-Growth) and PrefixSpan sequence mining.
- Conditional mining against external signals, via a single-threshold decision
  stump for numeric signals and lift-based selection for categorical ones.
- Numeric threshold / matrix-profile motif mining (stumpy used when available,
  deterministic threshold-crossing search otherwise).
- Energy load-shifting suggestions against a dynamic price signal.
- Stale-automation and never-used-entity housekeeping.

**Trust**

- Shadow-mode backtesting with precision, recall, missed actions and an
  unwanted-fires-per-week nuisance budget. Nothing is surfaced without passing.
- Conflict detection: value inconsistency, dependency loops, redundancy and
  shared-device races, plus an audit view of your existing automations.
- Persistent dismissals keyed to the rule, so a dismissed suggestion never
  returns.

**Data access**

- Recorder auto-discovery: `recorder: db_url:` with `!secret` resolution,
  falling back to the default SQLite database opened read-only and WAL-aware.
- SQLite, MariaDB/MySQL and PostgreSQL through SQLAlchemy, using pure-Python
  drivers so the image needs no compiler.
- Long-term statistics used for numeric sensors, so seasonality survives
  `purge_keep_days`.
- Entity resolver unioning the registries with `/api/states`, recovering
  entities that have no `unique_id`.
- Three-step degradation: `.storage` → WebSocket registry API → `/api/states`.

**Output**

- Deterministic YAML rendering from the mined schema by default.
- Optional local-first LLM generation (Ollama auto-detected; cloud opt-in),
  behind a mandatory validation gate: reference existence, schema, and
  `check_config`.
- Ingress web UI restricted to `172.30.32.2`, with suggestion, gap, audit,
  archive and status views.

**Packaging**

- The image installs only from wheels: every pinned dependency has a musllinux
  build for amd64 and aarch64, so the base needs no compiler. A CI job enforces
  it against the interpreter the image actually runs.
- build.yaml pins a base-python image, so the Python version is explicit instead
  of tracking whatever the current Alpine release ships.
- mlxtend is installed with `--no-deps`; only `mlxtend.frequent_patterns` is
  used, which needs nothing beyond numpy/pandas/scipy. This keeps scikit-learn
  (no musl wheels, never imported here) and matplotlib out of the image.
- A miner that raises no longer aborts the analysis; the run completes, is
  marked `partial`, and reports which miner failed and why.

**Optional AI assistance (all off by default)**

- `llm_entity_classification`: the model labels signal roles the regex detector
  misses. Additive only, cached on an inventory fingerprint, and every returned
  entity id and role is validated before use.
- `llm_hypotheses`: for backtest-rejected rules the model proposes explanatory
  conditions, each of which is rebuilt as a normal candidate and put through the
  same backtester. Only verified proposals are surfaced, carrying their
  provenance. Proposals naming unknown entities, unsimulatable condition kinds,
  or no actual constraint are rejected before measurement.
- `llm_triage`: advisory plausibility verdicts that can demote and annotate a
  suggestion but never promote, hide or alter its evidence.
- Each feature runs in isolation: a failure is reported and skipped rather than
  affecting the deterministic run.
- The Anthropic provider now forces JSON with an assistant prefill; it was the
  only path without machine-enforced structured output.

**Fixed**

- The synthetic fixture declared an analysis window that did not cover the rows
  it generated, so a window-bounded query returned a different first row
  depending on the day of the week the tests ran.
