# Automation Miner

Mines your Home Assistant history for automation suggestions that are worth
having — explained, backtested, and conflict-checked. Suggest-only by default.

## Installation

1. **Settings → Add-ons → Add-on Store → ⋮ → Repositories** and add this
   repository's URL.
2. Install **Automation Miner**.
3. Press **Start**. No configuration is required.
4. Open the panel from the sidebar (or **Open Web UI**).

The first analysis starts immediately and then runs nightly at 03:00.

## First run

The **Suggestions** page shows what was found. Each card carries:

- the rule in plain language,
- **Backtest** — how it would have performed against your real history:
  correct fires, unwanted fires per week, and missed actions,
- **Evidence** — occurrences, consistency, confidence/lift, and the data window,
- any **conflicts** with automations you already have.

You can **Review** (see the YAML and the full explanation), **Shadow-test**
(log when it would fire without acting), or **Dismiss** (permanently — it will
never be suggested again).

Shadow-testing does not run continuously: each nightly analysis replays the
rules you are watching over the history since it last looked and records every
would-be fire, marking whether you really did that thing around then. So the
count on the suggestion's page fills in a run at a time, not minute by minute.

If the list is empty, open **Status**. It says exactly what limited the run.

## Pages

- **Suggestions** — new candidates plus housekeeping findings.
- **Gaps** — integrations or hardware your behaviour suggests you want.
- **Audit** — conflicts and redundancy among your *existing* automations, plus
  every time you overrode an automation.
- **Archive** — dismissed and applied suggestions; dismissals can be restored.
- **Status** — data sources, entity resolution, causality breakdown, detected
  signals, LLM status, run history, and every degradation.

## Applying a suggestion

Press **Apply to Home Assistant** on a suggestion's detail page. What gets
written is the automation you were shown: it is stored when the preview is
rendered, and Apply writes that stored object rather than generating a fresh
one. Before anything is written it must pass all five gates:

1. it still means what the mined rule meant - same triggers, conditions,
   actions and mode (an LLM may reword the alias and description, nothing else),
2. everything it touches is named outright - no templates, no `entity_id: all`,
3. every entity, device, area and service it references exists,
4. the YAML parses and matches Home Assistant's automation schema,
5. `POST /api/config/core/check_config` succeeds.

Only then is it written through the config API and `automation.reload` called.
If any check fails, the reason is shown and nothing is written.

Applied automations get a stable id (`amminer_<hash>`), so re-applying an
updated version replaces it rather than creating a duplicate.

## Options

Every option has a working default; you can leave all of them alone.

### `log_level`
`trace` | `debug` | `info` (default) | `notice` | `warning` | `error` | `fatal`

### `schedule`
Cron expression for the nightly analysis. Default `0 3 * * *` (03:00 daily).

### `run_on_start`
Analyse once when the add-on starts. Default `true`.

### `analysis_window_days`
How far back to look. `0` (default) means auto: `min(available history, 60 days)`.
A larger value than your history allows is clamped, and the UI says so.

### `min_consistency`
How reliably a habit must repeat to count, measured against *eligible days*.
Default `0.6`. Raise it for fewer, stronger suggestions.

### `min_occurrences`
Minimum number of times a habit must have happened. Default `5`.

### `time_cluster_minutes`
Half-width of the time cluster — "within ±30 minutes of 06:30". Default `30`.

### `min_support`, `min_confidence`, `min_lift`
Association-rule thresholds. Defaults `0.02`, `0.6`, `1.5`.

### `override_window_seconds`
How soon after an automation acts a human correction counts as an *override*.
Default `120`.

### `backtest_min_true_fires`
How many times a rule must have been *right* during the analysis window before
its percentages are allowed to speak for it. A rule that fired once, correctly,
scores 100% precision with no unwanted fires and rests on a single observation.
Default `4`.

### `backtest_min_precision`
Minimum backtest precision for a rule to be surfaced. Default `0.7`.

### `backtest_min_recall`
How much of the real behaviour the rule has to account for. A rule that fires
correctly three times out of the twenty you actually did something is precise
and useless. Default `0.25`.

### `backtest_max_false_fires_per_week`
Nuisance budget. A rule that would have fired more often than this when you did
not want it is rejected regardless of its precision. Default `3`.

### `backtest_max_nuisance_fires`
Unwanted fires allowed at moments where you have previously reached over and
undone an automation on the same entity. These are not merely unnecessary - they
land exactly where you have already said no - so the default is `0`.

### `llm_gaps`
Off by default. Lets the model propose integrations or hardware the built-in
detector has no rule for.

Every proposal must name the real-world condition that makes it worth doing -
the thing your devices cannot tell it. A proposal without one is discarded, not
shown with a guess attached. It can only add to the list: it cannot change,
reorder or remove anything the detector found, cannot repeat a detector
suggestion under a new name, and cannot mention an entity you do not have.
Proposals are marked **model-proposed** on the card and rank below detected
gaps.

### `llm_audit`
Off by default. Lets the model review the audit's findings about your existing
automations and say whether each is a real conflict.

The audit itself already refuses to report a pair whose conditions provably
cannot both hold — "when I am home" against "when I am out", "below 20 lux"
against "above 500". This is for the rest: two rules conditioned on different
entities, where whether they ever coincide depends on what those entities mean
in your house.

It can only **hide** a finding or **lower** its severity. It can never raise one
and never invent one — a warning the deterministic audit did not produce is
never shown. The Status page reports how many findings there were before and
after, so nothing can be hidden quietly.

### `notify_on_new_suggestions`
On by default. Posts a notification in Home Assistant when a run finds something
new, listing the first few and linking back here.

Only *new* suggestions are announced. Every run re-surfaces the rules that still
hold, so announcing all of them would repeat the same list nightly; a suggestion
you have dismissed is never announced again. Each run replaces its own previous
notification rather than adding another one.

### `notify_service`
Empty by default. A notify service to call as well, e.g.
`notify.mobile_app_your_phone`, so new suggestions reach your phone without you
writing an automation. Run **Developer Tools → Actions** and search for `notify.`
to see what your instance offers.

Notifying happens after everything is saved, and never at the run's expense: an
unreachable Home Assistant or a service that does not exist costs you the
message and nothing else. The Status page reports what was sent and what was not.

### `allow_security_actions`
Off by default. While it is off, no suggestion whose action would unlock a door,
open a cover or valve, or disarm an alarm is ever surfaced, no matter how strong
the pattern behind it. Turning it on lets those suggestions through; they are
still held to the stricter thresholds below.

Actions in the `lock`, `cover`, `valve`, `alarm_control_panel`, `water_heater`,
`siren`, `lawn_mower` and `vacuum` domains always face a higher bar than a lamp:
95% precision, at least 12 correct fires, and no unwanted fires at all. These
are deliberately not tied to the options above, so lowering
`backtest_min_precision` to see more light suggestions does not also lower the
bar for your front door.

### `excluded_domains`, `excluded_entities`, `excluded_users`
Added to sensible built-in exclusions (`sensor.time`, `update.*`, …). Entities
accept glob patterns such as `sensor.*_battery`. Put long-lived-token "service
accounts" in `excluded_users` so their changes are not mistaken for a human's.

### `llm_provider`
`none` (default) renders YAML deterministically from the mined schema.
`ollama` auto-detects a local Ollama server. `openai`, `anthropic`, `google`
and `openrouter` are opt-in and need `llm_api_key`.

### `llm_entity_classification`, `llm_hypotheses`, `llm_triage`
Three optional AI features, all `false` by default and all requiring
`llm_provider` to be set to something other than `none`.

- **`llm_entity_classification`** — lets the model read your entity inventory
  (names, areas, device models, units — never states or history) and label
  signal roles the built-in pattern matcher missed, such as a price sensor or a
  dishwasher named in your own language. Additive only: it can add a signal,
  never remove one. Cached until your entities change.
- **`llm_hypotheses`** — for rules the backtest rejected, the model proposes
  conditions that might explain when the action really happens. Every proposal
  is re-backtested against your history with the same thresholds; only ones that
  pass are shown, labelled as suggested-then-verified.
- **`llm_triage`** — the model flags rules that are statistically real but make
  no sense, and they are ranked lower with the reason shown. It cannot promote a
  rule, hide one, or change any evidence.

If a feature is switched on while `llm_provider` is `none`, or the provider is
unreachable, the run says so on the Status page and continues normally.

Tuning: `llm_classification_batch` (entities per call, default 60),
`llm_hypothesis_candidates` (rejected rules to attempt, default 10),
`llm_hypotheses_per_candidate` (default 3), `llm_triage_penalty` (score
multiplier for an implausible verdict, default 0.5).

### `llm_model`, `llm_base_url`, `llm_api_key`
Optional overrides. With Ollama, leaving `llm_model` empty picks the best
installed model (`qwen3:8b` preferred; think-mode models are skipped).

## Getting more out of it

**Longer history is the biggest single improvement.** With the default 10-day
SQLite retention, sequence and association mining are disabled. Install the
MariaDB add-on and set:

```yaml
recorder:
  db_url: mysql://homeassistant:PASSWORD@core-mariadb/homeassistant?charset=utf8mb4
  purge_keep_days: 60
```

Use `!secret` for the URL if you prefer; it is resolved automatically and
credentials are never displayed.

**External signals make rules smarter.** Sun, a weather integration, the Workday
sensor, presence, and a dynamic price sensor all become conditions in mined
rules, which sharply reduces false fires. The **Gaps** page recommends the ones
your setup is missing and explains what each would buy you.

## Troubleshooting

**No suggestions at all.** Check **Status**. Common causes: too little history,
an unreadable recorder, or thresholds set too high. Try lowering
`min_consistency` to `0.5` or `backtest_min_precision` to `0.6`.

**"No entities could be resolved".** The add-on could read neither
`/homeassistant/.storage` nor `/api/states`. Confirm `homeassistant_api` is
granted (it is by default) and that Home Assistant is running. Mining continues
from the recorder alone, but suggestions cannot be validated or applied.

**"No context_user_id found".** This window of history has no user attribution,
so human and automated changes cannot be separated. It usually means the history
predates the changes being made through the UI, or came from a restored backup.
Mining continues at reduced confidence.

**Everything is rejected by the backtest.** That is the gate doing its job — the
patterns exist but would misfire too often. The Status page reports how many
were rejected. More history usually helps more than lowering the thresholds.

**Suggestions I do not want keep coming back.** Use **Dismiss**, not just
ignore. Dismissals are permanent and keyed to the rule, not its wording.

## Data and privacy

- Your history is read **read-only**; the SQLite database is opened `mode=ro`.
- The add-on's own database lives in its private `/config`, never in Home
  Assistant's.
- The web UI accepts connections only from the ingress address `172.30.32.2`.
- With `llm_provider: none`, nothing leaves your instance. With an LLM
  configured, only the mined rule schema and the entity ids it may use are sent
  — never raw history.
