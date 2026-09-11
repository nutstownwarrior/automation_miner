# Automation Miner

A Home Assistant **add-on** that reads your own history and proposes automations
worth having — each one explained, backtested against what you actually did, and
checked against the automations you already run.

It is suggest-only by default. Nothing is enabled, and nothing acts, until you
press **Apply**.

```
Recorder history ─┐
Registries + states ─┼─▶ causality ─▶ miners ─▶ enrichment ─▶ backtest ─▶ conflict check ─▶ you decide
External signals ─┘      (who did it)            (why)         (would it work?)   (does it clash?)
```

---

## Why this is not another "you turned the light on at 06:30" add-on

Four things make the difference:

**It knows who did what.** Every recorded state change is classified as caused by
a *human*, an *automation/script*, or a *device*, by walking Home Assistant's
`context_parent_id` chains back to their root. Home Assistant exposes no native
"root cause", so Automation Miner reconstructs it. The result: it never suggests
automating something an automation already does.

**It learns from your corrections.** When you undo what an automation just did —
within 120 seconds, on the same entity — that is an **override**. Overrides are
negative evidence against that automation and positive evidence for what you
actually wanted. They are stored, shown in the audit view, and counted against
suggestions that would misfire the same way.

**It backtests before it suggests.** Every candidate is replayed against your
real history: how often would it have fired correctly, how often would it have
fired when you did *not* want it, how many of your actions would it have missed.
A rule has to clear all five of these to reach you, and the numbers are shown on
the card:

- **enough evidence** — it must have been right at least `backtest_min_true_fires`
  times. This one comes first because it is the one a percentage hides: a rule
  that fired once, correctly, has 100% precision and zero unwanted fires per
  week, and has shown nothing at all.
- **precision** — of the times it would have fired, how many you wanted.
- **recall** — how much of the real behaviour it accounts for. Firing correctly
  three times out of the twenty you actually did something is precise and
  useless.
- **nuisance budget** — unwanted fires per week.
- **no fires where you have already said no** — an unwanted fire that lands next
  to a moment when you reached over and undid an automation is not merely
  unnecessary, and is disqualifying on its own.

Rules that trigger on an event rather than a clock time are matched more
strictly, per trigger: "some time in the next quarter hour" is a fair reading of
a daily habit and a dishonest one for a rule claiming the door opening caused
the light.

**Not everything gets the same bar.** A rule that moves a physical barrier or
secures a building is not the same kind of suggestion as one that turns on a
lamp, and every miner in this project would otherwise apply identical thresholds
to both. Actions on `lock`, `cover`, `valve`, `alarm_control_panel`,
`water_heater`, `siren`, `lawn_mower` and `vacuum` must clear 95% precision, at
least 12 correct fires, and zero unwanted fires. Actions that leave the home
*less* secured — unlocking, opening a cover or valve, disarming — are not
proposed at all unless you turn on `allow_security_actions`: a correlation is
never a reason to unlock a door, and there is no precision at which it becomes
one.

**It tells you when it finds something.** Suggestions would otherwise sit in the
add-on's own database until you went looking. A run posts a Home Assistant
notification, and optionally calls a notify service of your choosing, for
*genuinely new* findings only — a run re-surfaces every rule that still holds,
and repeating those nightly is how a notification becomes something you turn
off. Dismissed suggestions are never announced again.

**It checks for conflicts.** Before surfacing a rule it looks for value
inconsistencies, dependency loops, redundancy with an existing automation, and
races between entities on the same physical device. A conflicting rule is never
auto-enabled and the reason is shown.

---

## Zero-config path

On a stock Home Assistant OS install — SQLite recorder, default 10-day
retention, no extra integrations — installing the add-on and starting it is the
whole setup:

1. **Settings → Add-ons → Add-on Store → ⋮ → Repositories**, add
   `https://github.com/nutstownwarrior/automation_miner`.
2. Install **Automation Miner**, then **Start**.
3. Open the panel from the sidebar.

The first analysis runs at startup and then nightly at 03:00. With no
configuration at all it will:

- find the recorder database (`recorder: db_url:` if set, otherwise
  `/homeassistant/home-assistant_v2.db`, opened read-only and WAL-aware),
- read the entity/device/area/label registries and **union them with
  `/api/states`** so template and YAML entities without a `unique_id` are not
  invisible,
- auto-select an analysis window of `min(available history, 60 days)`,
- mine time-of-day habits, conditional patterns and numeric-threshold rules,
- detect whichever external signals you happen to have (sun, weather, workday,
  presence, price, solar forecast, carbon intensity, DWD warnings),
- backtest and conflict-check everything,
- tell you, on the Status page, exactly what it could **not** do and why.

With 10 days of history you get time-of-day and numeric patterns. Sequence and
association mining need at least 7 days and are much better with weeks — see the
next section.

---

## The MariaDB retention upgrade

SQLite with `purge_keep_days: 10` is the single biggest limit on what can be
mined. Association rules and multi-step routines need weeks of history to be
statistically meaningful, and backtest precision on 10 days is a small sample.

Install the **MariaDB** add-on and point the recorder at it:

```yaml
# configuration.yaml
recorder:
  db_url: !secret recorder_db_url
  purge_keep_days: 60
  # optional but recommended: keep the noisiest entities out of the database
  exclude:
    entities:
      - sensor.time
      - sensor.date
```

```yaml
# secrets.yaml
recorder_db_url: mysql://homeassistant:YOUR_PASSWORD@core-mariadb/homeassistant?charset=utf8mb4
```

Restart Home Assistant. Automation Miner picks the new database up on its next
run — no add-on configuration changes needed. `!secret` is resolved, and the
credentials are never shown in the UI or the logs.

The add-on connects with pure-Python drivers (PyMySQL for MariaDB/MySQL, pg8000
for PostgreSQL), so no compiler is needed in the image regardless of your
recorder backend.

**Note on long-term statistics:** hourly statistics are *not* subject to
`purge_keep_days`. Automation Miner reads them for numeric sensors, so seasonal
temperature patterns remain minable even on a short-retention SQLite install.

---

## What it mines

| Miner | Finds | Needs |
|---|---|---|
| **Time of day** | "Turn on the kitchen light at 06:30 on weekdays" | any history |
| **Conditional** | "Turn on the heater when it is below 8 °C outside" | one external signal |
| **Numeric threshold / motif** | "Close the blinds when lux drops below 600" | a numeric sensor |
| **Association rules** | "When Alex gets home, the hallway light goes on" | ≥ 7 days |
| **Sequences** | "Arrive home → hall light → thermostat up" | ≥ 7 days |
| **Energy shifting** | "You run the dishwasher in the expensive window; shift it" | a dynamic price sensor + a deferrable load |
| **Housekeeping** | stale automations, never-used entities | — |

Clustering is **circular**: 23:59 and 00:05 are six minutes apart, not
twenty-three hours. Consistency is measured against *opportunities* (eligible
days), not against occurrences, so "6 of 7 weekdays" scores far above "6 hits
scattered over 60 days".

---

## Integration and hardware gaps

Beyond automations, the add-on reports capability gaps your own behaviour
implies — each with the evidence that produced it:

- lots of manual light switching + PIR only → mmWave presence (Aqara FP2,
  Everything Presence) or ESPresense/Bermuda room presence,
- deferrable loads + no price signal → Tibber / Nordpool / ENTSO-e / aWATTar /
  Energi Data Service, and EMHASS for full optimisation,
- solar inverter + no forecast → Forecast.Solar or Solcast,
- climate entities + no outdoor temperature → Met.no or a temperature sensor,
- weekday-conditioned rules + no Workday sensor → the Workday integration,
- no presence entities at all → the companion app or FRITZ!Box device trackers,
- short SQLite history → the MariaDB upgrade above.

---

## Generating YAML (optional, local-first)

By default (`llm_provider: none`) the automation YAML is rendered
**deterministically** from the mined schema. This is not a lesser fallback — the
internal schema maps one-to-one onto Home Assistant triggers, conditions and
actions, so the output is correct every time.

An LLM is used at exactly one point: rewriting an *already mined and already
backtested* rule with a nicer alias and description. **Raw history is never sent
to a model** — the prompt contains the neutral rule schema plus the specific
entity ids it is allowed to use, and nothing else.

- `llm_provider: ollama` auto-detects a local Ollama server (the Ollama add-on,
  the Docker host, `localhost`). Recommended models: `qwen3:8b` for structured
  output, `qwen3:4b` or `gemma3:4b` on CPU-only hardware. Reasoning
  ("think-mode") models are skipped — they are worse at emitting bare JSON.
- Cloud providers (OpenAI, Anthropic, Google, OpenRouter) are **strictly opt-in**
  and require an API key.

### Optional AI assistance (all off by default)

Nine further features use the model for things statistics cannot do. Each is a
separate switch, each defaults to **off**, and none of them can put a suggestion
in front of you that has not passed the same deterministic gates as every other
suggestion.

| Option | What it does | What it is not allowed to do |
|---|---|---|
| `llm_entity_classification` | Labels signal roles the pattern matcher misses — a price sensor called `sensor.stroomprijs`, a dishwasher called `switch.geschirr`. Cached until your entities change. | Remove a role the deterministic detector found. It is additive only. |
| `llm_hypotheses` | For rules the backtest **rejected**, proposes conditions that might explain when the action really happens — "only when it's cold out", "only on workdays". Each proposal is then re-backtested. | Surface anything. Only proposals that clear the same precision and nuisance thresholds are shown, and they carry a note saying the condition was suggested and then verified. |
| `llm_gaps` | Integrations or hardware the fixed detector rules have no case for, judged with world knowledge. | Remove or reword a detected gap, repeat one, cite an entity you do not have, or propose anything without naming the real-world precondition it depends on. |
| `llm_audit` | For findings about your **existing** automations, judges whether a flagged pair is a real conflict — given both rules in full, conditions included. | Raise a severity or invent a finding. It can only hide or soften what the deterministic audit already produced, and the before/after counts are reported. |
| `llm_triage` | Flags rules that are statistically real but semantically absurd — two things that merely happen at the same time of day. | Promote or hide anything. It can lower a score and attach a visible reason; the evidence and backtest stay exactly as they were. |
| `llm_preferences` | Reads the reasons you typed when dismissing things and generalises them into standing preferences — "nothing in the guest room" — then hides new suggestions that match one. Every preference is listed on the Archive page as an editable field: rewrite it, switch it off, delete it, or write your own — by hand, or by telling the model what to change ("only the lamp, not the whole room"). | Hide anything without saying so, or keep a wording you have disagreed with. Every hidden suggestion is listed with the rule that hid it; a preference needs two of your own dismissals behind it; and once you rewrite one, the text is yours and no later run puts the model's version back. The wording helper writes nothing: it fills in the text box, and the rule is stored only when you press Save. It cannot promote or reorder anything, and it never overrules something you have already accepted, dismissed or shadow-tested. |
| `llm_explain` | Rewrites the evidence line as one plain sentence: "you switched this on at about 06:30 on 30 of the 34 weekdays" instead of "consistency 88%, lift 2.00". | State a number the evidence does not support. A sentence containing one is dropped rather than corrected, and the original figures stay on the card underneath. |
| `llm_scenes` | Proposes that several suggestions are really one routine — Bedtime, Leaving the house — and names it. | Inherit its parts' scores or replace them. The members must share a trigger the backtester would still credit each of their actions against, the consolidated rule is backtested as one rule from a score of zero, and the individual suggestions stay exactly where they were. |
| `llm_areas` | Works out which room an entity is in when the registry does not say — `sensor.hue_motion_kitchen_2` is the kitchen. | Touch an area you assigned, invent a room you do not have, or present a guess as fact. It only sees entities with no area, may only answer with an area that already exists, and every inference is marked as a guess wherever it is shown. |

`llm_hypotheses` is the one that changes what the tool can *find*: the built-in
conditional miner only tests one signal at a time against a single threshold, so
a combination like "dark **and** a workday" is outside its reach. The model
supplies the guess, your history decides.

Turning these on costs more calls than the YAML step: classification is one call
per ~60 entities (cached), hypotheses one call per rejected rule, triage one per
25 suggestions, explanations one per 20, areas one per ~80 unplaced entities, and
preferences and scenes one each per run. The Status page reports exactly what each one did, including how
many invented entity ids were discarded.

### The validation gate

Nothing reaches your configuration without passing all five checks:

1. **Equivalence** — the model's automation is reduced to a canonical form of
   its triggers, conditions, actions and mode, and compared against the same
   reduction of the deterministic rendering. The prompt allows it to reword the
   alias and description; anything else it changes is a rejection that names the
   field. Checks 2 and 3 only prove that what the model named *exists* — an
   entirely different automation built from real entities and real services
   passes both. This one is what makes "the rule you read the evidence for" and
   "the rule that gets written" the same rule.
2. **Knowable targets** — a Jinja template names nothing, so an existence check
   on `service: "{{ svc }}"` passes vacuously; `entity_id: all` names everything.
   Both defer the decision to runtime, where no gate can see it, and neither is
   something this add-on ever generates, so both are refused outright.
3. **Existence** — every `entity_id`, `device_id`, `area_id` and service must
   exist in the registry-union-states set. This is what catches hallucination;
   Home Assistant's own config check does not.
4. **Schema** — the YAML must parse and match Home Assistant's automation schema
   (mirrored in voluptuous).
5. **`POST /api/config/core/check_config`** must pass.

Every one of them fails closed. If Home Assistant is unreachable and the live
service list is unavailable, the service check does not quietly become a no-op:
it falls back to the closed set of services the miners are capable of emitting,
which is narrower than your real instance. An unreachable Core can cost you a
legitimate suggestion; it cannot turn "unverified" into "fine".

If the LLM's output fails the gate, it is rejected, the reason is shown
(including which entity ids it invented, or which field it rewrote), and you are
given the deterministic rendering instead.

**Apply writes the artifact you previewed.** The automation shown on the detail
page is stored when it is rendered, and Apply writes that stored object rather
than asking the model the same question a second time — a second answer would be
an automation you never saw. It is re-validated in full, `check_config`
included, before it is written; reuse means no new content, not no new checks.
If the finding has since been re-mined into a different rule, the stale preview
is discarded and regenerated rather than applied.

---

## Configuration

Everything below has a working default. The add-on is designed to be started
without touching any of it.

| Option | Default | What it does |
|---|---|---|
| `analysis_window_days` | `0` (auto) | `0` = `min(available history, 60)` |
| `min_consistency` | `0.6` | how reliably a habit must repeat |
| `min_occurrences` | `5` | minimum times a habit must have happened |
| `time_cluster_minutes` | `30` | half-width of the time cluster |
| `min_support` / `min_confidence` / `min_lift` | `0.02` / `0.6` / `1.5` | association-rule thresholds |
| `override_window_seconds` | `120` | how soon a correction counts as an override |
| `backtest_min_precision` | `0.7` | minimum backtest precision to surface a rule |
| `backtest_min_recall` | `0.25` | how much of the real behaviour a rule must account for |
| `backtest_min_true_fires` | `4` | times a rule must have been right before its percentages count |
| `backtest_max_false_fires_per_week` | `3` | nuisance budget |
| `backtest_max_nuisance_fires` | `0` | unwanted fires allowed where you previously overrode an automation |
| `notify_on_new_suggestions` | `true` | post a Home Assistant notification when a run finds something new |
| `notify_service` | `""` | also call a notify service, e.g. `notify.mobile_app_your_phone` |
| `allow_security_actions` | `false` | let suggestions unlock, open or disarm things |
| `excluded_domains` / `excluded_entities` / `excluded_users` | `[]` | added to sensible built-in exclusions; entities accept globs |
| `llm_provider` | `none` | `none`, `ollama`, or a cloud provider (opt-in) |
| `schedule` | `0 3 * * *` | cron for the nightly analysis |
| `run_on_start` | `true` | analyse once at startup |

---

## Graceful degradation

The add-on never stops working because something is missing. Whatever it could
not do is listed on the Status page and at the top of the suggestions list.

| Missing | Consequence |
|---|---|
| < 7 days of history | time-of-day and numeric patterns only; sequence/association disabled; MariaDB recommended |
| No `context_user_id` | human and automated changes cannot be separated; everything runs at reduced confidence, flagged in the UI |
| `.storage` unreadable | falls back to the WebSocket registry API, then to `/api/states` alone (no area/device/label context) |
| No Home Assistant API | mines from the recorder alone; suggestions cannot be validated or applied, only copied |
| No external signals | intrinsic patterns only |
| No LLM | deterministic blueprint YAML you can paste |
| No recorder at all | the run is marked *degraded* and says why |
| A miner raising | only that miner's findings are lost; the run is marked *partial* and names the failure |
| An AI feature enabled without a provider, or failing | the feature is skipped and says so; the deterministic run is unaffected |

---

## Privacy and safety

- History is read **read-only**. SQLite is opened `mode=ro` so Home Assistant
  never loses a write lock. (`immutable=1` is deliberately *not* used — it would
  hide the WAL and silently drop the most recent hours.)
- The add-on's own state lives in its **private** `/config` (`addon_config`),
  never in Home Assistant's database.
- The web UI only accepts connections from the ingress address `172.30.32.2`.
- No raw history is sent anywhere. With `llm_provider: none` nothing leaves the
  instance at all.
- Applying a suggestion is always an explicit click, and always validated first.

---

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r automation_miner/requirements.txt
pip install --no-deps -r automation_miner/requirements-nodeps.txt
pip install pytest
pytest
```

Install both files, exactly as the image does — developing against a different
resolution than the pinned one is how a dependency breaks in CI but not locally.

### Dependency footprint

The add-on image installs **only from wheels**: every pinned dependency has a
musllinux wheel for amd64 and aarch64, so the base needs no compiler and the
build is fast. CI enforces this in a dedicated job.

`build.yaml` pins a Home Assistant **base-python** image, so the interpreter
version is explicit rather than whatever Python the current Alpine release
happens to ship — that floats, and a base bump would invalidate every pinned
wheel without a line of this repository changing. The CI wheel check targets the
same interpreter, and a test asserts the two agree.

`mlxtend` is installed with `--no-deps`. It declares scikit-learn, matplotlib
and joblib, but the only part used here — `mlxtend.frequent_patterns`
(FP-Growth and association rules) — needs nothing beyond numpy, pandas and
scipy. scikit-learn publishes no musllinux wheels for any version, so depending
on it would force a from-source build inside the image; nothing in this package
imports it. `amminer.miners.association` therefore provides its own one-hot
encoder rather than using `mlxtend.preprocessing.TransactionEncoder`, which is
the module that pulls scikit-learn in. If a future miner genuinely needs
scikit-learn, add it to `requirements.txt` and move `build.yaml` to the Debian
base images at the same time.

Run it outside a container against a copy of your config:

```bash
export AMMINER_HA_CONFIG_DIR=/path/to/a/copy/of/homeassistant
export AMMINER_STATE_DIR=/tmp/amminer
export AMMINER_ALLOW_ANY_HOST=1     # skip the ingress source check
export PYTHONPATH=automation_miner
python -m amminer                   # http://localhost:8099
```

Generate a synthetic recorder database with known patterns in it:

```bash
python -m amminer.testing.synthetic /tmp/rec.db --days 45
```

### Testing

- **Synthetic recorder** — `amminer.testing.synthetic` writes the *real*
  post-2023 normalised schema (`states_meta`, binary context columns,
  `statistics`) and injects known patterns: a 06:30 weekday habit, an
  arrive-home sequence, a temperature-driven heater habit, and deliberate
  override events. The miners are asserted to recover exactly those.
- **Dialects** — every test runs the production SQL against SQLite. Setting
  `AMMINER_TEST_MYSQL_URL` adds one MariaDB test (`pytest -m mariadb`) that
  asserts the same queries execute there and return context ids; it does not
  compare results between the two. PostgreSQL is supported by the same
  SQLAlchemy code path and the DDL translator in `testing/mysql_loader.py`, but
  nothing in CI exercises it — treat Postgres as untested rather than verified.
- **Timezones** — the suite pins `TZ=UTC` so the synthetic history and the
  miners agree, and `tests/test_timezones.py` deliberately runs the mining path
  under several real zones, which is the case a Home Assistant instance is
  actually in.
- **Messy history** — the same generator, with `messy=True`, adds what a real
  recorder is full of and the clean fixture never had: restarts leaving
  `unavailable`/`unknown` and then a restored state with no context at all,
  flapping contacts, and human changes carrying no `context_user_id`. The whole
  pipeline runs against it and the same injected patterns must still come out,
  so the degraded paths are exercised by the flagship test rather than only by
  small unit tests.
- **Mutation checks** — the thresholds that decide what a user is shown
  (`min_consistency`, the conditional miner's lift/purity/staleness bounds, the
  association lift and human-consequent filters, the stale-automation cutoff)
  each have a test that fails when the check is deleted. They were added because
  every one of them could be removed with the suite staying green.
- **Public datasets** — `amminer.testing.datasets` maps CASAS, ARAS and
  Kasteren into the internal schema. The archives are not redistributable, so
  the tests run on data generated in those exact on-disk formats; point
  `AMMINER_CASAS_FILE` at a real CASAS file to run against the genuine archive.
  CI does not set it, so CI validates the format handling, not the archives.
- **Registry snapshots** — including entities with no `unique_id` (recovered via
  the states union) and large `deleted_entities` sections (which must be ignored).
- **Golden LLM tests** — a deliberately hallucinating model is asserted to be
  rejected by the gate, and a `check_config` failure is asserted to block apply.

---

## Prior art

Shadow execution as a correctness check comes from **HAWatcher** (USENIX
Security 2021). The conflict taxonomy follows the trigger-action-programming
conflict-detection literature (SHACR, AutoIoT and the TAP conflict surveys). The
staged-trust UX — suggest, explain, let the user accept or dismiss, and learn
from dismissals — follows **Alexa Hunches**, with the Nest thermostat's
auto-schedule as the cautionary tale about acting without asking.

## License

MIT
