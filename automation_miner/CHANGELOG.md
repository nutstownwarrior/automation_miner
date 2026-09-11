# Changelog

## 0.2.0

An adversarial review of the whole codebase. Most of what follows is a
correction: something that was measured and then not acted on, claimed and not
checked, or checked in a way that could not fail.

**Behaviour changes worth knowing about before you update**

Suggestions that passed under 0.1.0 can be rejected under 0.2.0. That is the
point of the release, but it means the list may get shorter.

- The backtest gained an evidence floor. A rule that fired once, correctly, had
  100% precision and zero unwanted fires per week — it cleared every threshold
  on a single observation. `backtest_min_true_fires` (default 4) must now be met
  before those percentages are allowed to speak for it.
- `recall` was computed, displayed and never gated. `backtest_min_recall`
  (default 0.25) now applies: three correct fires out of the twenty times you
  really did something is precise and useless.
- `nuisance_fires` — unwanted fires landing where you had already reached over
  and undone an automation — was computed, called the metric that matters most
  in its own docstring, and then not consulted. `backtest_max_nuisance_fires`
  (default 0) now applies.
- Actions on `lock`, `cover`, `valve`, `alarm_control_panel`, `water_heater`,
  `siren`, `lawn_mower` and `vacuum` must clear 95% precision, 12 correct fires
  and zero unwanted fires. These are deliberately not your own thresholds, so
  lowering `backtest_min_precision` to see more light suggestions does not also
  lower the bar for your front door.
- Unlocking, opening a cover or valve, and disarming an alarm are refused before
  any statistics are consulted. There is no precision at which a correlation
  becomes a reason to unsecure a house. `allow_security_actions` (default false)
  lifts the refusal, not the thresholds.
- Association rules must now show their antecedent actually led. A basket is a
  set, so FP-Growth yields `A → B` and `B → A` with identical support,
  confidence and lift for every strong pair — "turn the pump on because the
  kitchen light came on" carried exactly the evidence of the rule that was true,
  and accepting both cards built a loop.
- An error-severity conflict now blocks Apply until you are shown it and say to
  go ahead. The check was being run and counted in the report, then ignored by
  the one operation it exists to inform.
- `hassio_api` is no longer requested. Nothing called a Supervisor endpoint; the
  two methods that would have had no callers.

**Fixed: things that made a rule look better than it was**

- Overnight time conditions never held. "after 22:00 and before 06:00" was
  tested as two independent bounds, and nothing on a clock face is both, so the
  condition was unsatisfiable at every instant — every evening and night-time
  candidate simulated zero fires and was rejected for never firing.
- A `numeric_state` trigger with both `above` and `below` is one band, not two
  thresholds. It fired on a value that jumped clean over the band and landed
  outside it, and again on the way out.
- Burst collapsing counted a flapping trigger as one fire. `mode: single` only
  suppresses a re-trigger while the action is still running, and turning a light
  on is not. Swallowed fires are counted, reported, and disqualifying when they
  outnumber the counted ones.
- The sequence miner divided by the wrong denominator: a routine that followed
  the front door opening 10 times out of 10 was reported at 20% confidence, and
  every genuine routine's score fell as unrelated activity in the house grew.
- The conditional miner compared action-time values against a uniform grid, so
  any signal with a daily shape separated the samples. A clock habit at 18:00
  and a server-load sensor dipping during a nightly backup produced "turn on the
  light when server load is above 50" at 100% purity. The background is now
  drawn at the same times of day, on days you did not act.
- The time-of-day miner tried all days, weekdays and weekends and reported the
  best as though it were the only hypothesis. On a day-independent process, 109
  of 200 trials were accepted as day-conditioned habits; with a margin required,
  11 of 200. It also picked the cluster centre before collapsing to one event
  per day, so one evening of fiddling with a dimmer could pull the chosen time.
- `_numeric_stump` required three distinct values, discarding the cleanest
  signal there is: one value whenever you act and another whenever you do not.
- Long-term statistics are hourly buckets stamped at the hour they open, and
  were read at that timestamp — so "was it below 8 degrees at 14:05?" was
  answered with the mean of 14:00–15:00, including readings that had not
  happened yet. They also replaced raw history whenever they were longer, which
  they almost always are.
- The signal detector matched patterns against the integration that produced an
  entity. "template" contains "temp", so on any instance with template sensors a
  garden gnome counter was an outdoor thermometer. Patterns were also bare
  substrings: `go_?e` matched "man(go_e)thanol".
- Causality read a user id as proof a person acted, but Home Assistant carries
  the starting user's id down a whole script run — so every light a hand-started
  script touched looked like a manual habit worth automating.
- A change with no context at all was recorded as `device`. That is a guess
  written down as a fact; it is `unknown` now and takes no part in mining.
- The conflict checks had three blind spots, each verified by running them: a
  target written as `data: {entity_id: ...}` parsed to an automation that
  touched nothing; `target: {area_id: ...}` was parsed and never consulted; and
  a clock-driven rule and a state-driven rule could fight over one lamp every
  evening without ever being compared.

**Fixed: the validation gate**

- The gate proved the model named things that exist, which an entirely different
  automation built from real entities also does. Output is now reduced to a
  canonical form of its triggers, conditions, actions and mode and compared
  against the deterministic rendering; only the alias and description may be
  reworded.
- Apply regenerated from scratch, asking a non-deterministic model the same
  question a second time and writing the second answer. Previews are persisted
  and applied verbatim — still fully re-validated, `check_config` included.
- Three configs Home Assistant accepts reached the end of the gate unchecked,
  each by naming nothing for the existence check to reject: a templated service,
  a templated target, and `entity_id: all`.
- The service check was skipped entirely when the service index came back empty
  — which is what happens when Core is unreachable, i.e. when an unverified
  service name matters most. It now falls back to the closed set this add-on can
  emit.

**Fixed: operational**

- Shutdown closed the database five seconds after signalling the scheduler,
  while an analysis could still be writing: `Cannot operate on a closed
  database`, every unpersisted candidate lost, and the run row left saying
  "running" for good. Any run still open at the next start is marked
  interrupted.
- Only the miners were isolated. Backtesting, conflict checking, gap analysis,
  the audit and the persistence loop fell through to the outer handler, so one
  exception after mining discarded every candidate already produced.
- "Shadow-test only" set a status and logged nothing — `log_shadow_fire` had no
  callers, so the page promised a count that stayed at zero forever. Each run
  now replays watched rules over the history since it last looked.
- "Restore" set the status back to `new` and left the dismissal row in place, so
  the suggestion was dropped again on the next run and pruned. It reappeared,
  then quietly vanished.
- Every web handler was `async def` while doing synchronous SQLite, HTTP and
  model calls with a 180-second timeout, holding the only event loop. Measured:
  a `/health` request alongside a two-second page completed after 2.04s; as sync
  handlers, 0.01s.
- The Google provider passed its API key as `?key=...`, and httpx quotes the
  failing URL — so the key travelled into the run report, the status page and
  the database. It authenticates with a header now, and provider error text is
  scrubbed regardless.
- The signal store and the entity resolver were each built twice per run, and
  the resolver the UI and every preview used was never the one that was mined
  against.
- `Options.from_mapping` silently dropped unknown keys while warning about
  mistyped values, so a typo was the quieter mistake. Non-string path options
  reached `Path()` and raised before logging was configured.

**Testing**

- 453 tests, 87% coverage, gated at 85% in CI.
- Nine thresholds that decide what you are shown could each be deleted, or moved
  by orders of magnitude, with the suite staying green. Each now has a test that
  fails when the check is removed, verified by re-running the deletion.
- The shared fixture was date-dependent as well as seed-dependent: weekday
  branches consume different amounts of randomness, so the same seed gave
  different ground truth depending on the day the suite ran. It ends on a fixed
  date, with a test walking all seven end-weekdays.
- The suite assumed the machine was in UTC and CI never set `TZ`, so the non-UTC
  case the code is explicitly written for was never exercised. `TZ` is pinned,
  and a dedicated module runs the mining path under four real zones.
- A `messy=True` fixture mode adds what a real recorder is full of — restarts
  leaving `unavailable`/`unknown` then a context-free restored state, flapping
  contacts, human changes with no user id — and the whole pipeline is asserted
  against it.
- `apply.py`, the one module that writes to a live configuration, had no direct
  tests.
- `pyproject.toml`, `config.yaml` and `version.py` are now asserted to agree, so
  a version bump cannot be half-applied.

**Documentation**

- README and DOCS claimed three validation gates (there are five), and that the
  same SQL is tested against MariaDB and PostgreSQL — one MariaDB test exists
  and asserts no parity, and nothing runs Postgres, which is now stated.

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
