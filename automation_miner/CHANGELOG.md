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
