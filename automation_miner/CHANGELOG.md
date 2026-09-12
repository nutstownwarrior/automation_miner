# Changelog

## 0.7.0

**Added: learned ordering — `ranking_enabled` (on by default)**

Every miner has always scored its own candidates with its own arithmetic -
association rules by `confidence * lift`, time-of-day habits by `consistency *
hits`, staleness by raw age. `Evidence`'s own docstring has said for a while
that these numbers are not comparable across miners, and the index page sorted
them against each other anyway.

Suggestions are now ordered by one calibrated number instead: the estimated
probability that *you* accept this particular suggestion, learned from your own
accept/dismiss history. Each miner's own score and evidence stay on the card
exactly as before - this replaces the ordering, not the explanation.

- A brand-new instance has no history to learn from, so a hand-set prior
  (documented, weight by weight, in `amminer/learn/ranking.py`) provides a
  sane starting order: a holdout-validated candidate outranks an in-sample
  one, more conflicts and riskier action domains rank lower, and so on.
- As you accept and dismiss things, a personal model is fit *towards* that
  prior rather than towards zero, so with little data the fit barely moves
  and it takes real, consistent evidence to pull it away
  (`ranking_prior_strength` controls how hard - lower means your own
  decisions move things sooner).
- On top of that, the personal fit is cross-validated against the prior
  before it is ever used, with a statistical test rather than a bare
  smaller-number comparison. If a handful of noisy decisions would not
  *clearly* make the ordering better than the prior alone, the prior is used
  instead and the card says so.
- This is ranking and display only. It cannot rescue a suggestion that failed
  its backtest, and it cannot hide one that passed - those gates are entirely
  unaffected, the same way the existing AI triage and classification features
  can only demote or add, never decide.
- Every card says plainly how much its number rests on general patterns versus
  your own decisions, rather than showing a precise-looking percentage earned
  by three data points.

Turn it off with `ranking_enabled: false` to go back to sorting by each
miner's own score, unchanged.

## 0.6.0

Four more optional AI features, all `false` by default and all requiring
`llm_provider`. As before, none of them can put a suggestion in front of you
that has not passed the same deterministic gates as every other suggestion.

**Added: `llm_preferences` — learning from the reasons you already gave**

`dismissals.reason` has been written since the first release and read by
nothing. Dismissing was therefore a mute keyed to one exact rule: reword the
rule and it came back, and the sentence you typed explaining *why* — the only
place you say anything about your own home in your own words — was stored and
ignored.

Those sentences are now read back and generalised into standing preferences
("nothing in the guest room"), which hide new suggestions that match one. This
is the only optional feature that can make a suggestion disappear, so it is the
most constrained one here:

- a preference needs at least **two** of your own dismissals behind it, cited by
  the model and checked against the real ones; a citation that does not exist
  does not count;
- a suppression must name a preference you can read, or it is refused;
- everything hidden is listed under **Archive** with the rule that hid it and a
  button to show it anyway;
- a suggestion you have already accepted, dismissed or asked to shadow-test is
  never touched.

A learned preference is a guess about what someone meant, made from a sentence
they typed in a hurry, so it is not allowed to be permanent. The Archive page
lists preferences as **editable fields**, not as verdicts:

- **rewriting** one makes the wording the user's. `edited` is what stops the
  next run's relearning from quietly restoring the model's phrasing — without
  it an amendment would have lasted until the following night and no longer —
  and an amended preference is never withdrawn by relearning either. The row
  keeps its id, because that is what the suggestions it hid point at;
- **switching off** keeps it listed but inert, and survives relearning;
- **deleting** forgets it. The UI says plainly that a learned one may be
  generalised again from the same dismissals, and that switching it off is the
  option that sticks;
- **writing one by hand** needs no dismissals at all, and is never touched by
  relearning;
- **rewording with the model's help** turns "only the lamp, not the whole room"
  into the amended sentence. The helper writes nothing: it fills in the same box
  a hand edit uses, and the rule reaches the store only once a person has read it
  and pressed Save. A model that could change a stored preference directly would
  be able to reword the rules that hide things, which is the one power this
  feature exists to withhold.

Its endpoint is declared ahead of `/preferences/{id}` — routes match in order, so
the parameterised one would otherwise have tried to edit a preference called
"draft" — and the provider call runs in a threadpool rather than on the event
loop, because a timeout measured in minutes awaited there stalls every other
request, `/health` included.

Switching off, rewriting or deleting brings back everything the preference was
hiding, immediately. Rewriting does so because a rule the user has just
disagreed with is not a rule to keep hiding things by.

What the matcher is given is read back from the store rather than taken from
what the model just said, so an amendment changes what is actually hidden and
not merely what is displayed.

Only the titles you dismissed and the reasons you gave are sent — never history.

**Added: `llm_explain` — the evidence as a sentence**

A card's evidence line read `30 of 34, consistency 88%, confidence 100%, lift
2.00, ±6 min`. Worse, as `amminer.miners.base` documents, `confidence` and
`consistency` carry *different quantities* depending on which miner filled them
in, so the numbers were not even comparable between two cards on the same page.

The model now writes the "why" in plain language. It cannot change a score, a
verdict, a backtest or any evidence value, and the original figures stay on the
card underneath. A sentence containing a number the evidence does not support is
**dropped, not corrected**: a wrong figure in the sentence explaining why to
trust something is the one error that cannot be tolerated here.

**Added: `llm_scenes` — several suggestions that are really one routine**

Six cards that all say "at about 22:40" are one habit split six ways, and no
miner has a vocabulary for that. The model proposes the grouping and names it;
everything it claims is then checked by something that is not the model:

- the members must exist, and one member cannot be spent on two scenes;
- they must share a trigger the **backtester would still credit each member's own
  action against** — the compatibility window is
  `backtest_match_tolerance_seconds` itself, so a scene can never be measured
  against a moment its parts never happened at;
- a group whose members fight over the same entity is refused;
- only conditions *every* member carries survive into the consolidated rule;
- the consolidated rule is backtested **as a unit, from a score of zero**. It
  inherits nothing from its parts, because firing all of those actions together
  is a different rule from any one of them.

A scene that fails the gate is not surfaced. A scene that passes is added
alongside its members and never replaces them.

**Added: `llm_areas` — a room for entities the registry never placed**

Area is the only structural fact this add-on has about a home, and it is the one
most often left half-filled. The room is usually right there in the entity id or
the device name, which is a reading task: `sensor.hue_motion_kitchen_2` is the
kitchen, `binary_sensor.0x00158d` is nothing.

The model is shown **only entities whose area is unset** — an area you assigned
is never sent, never questioned and never overwritten — and may only answer with
a room that already exists in your registry. If you have no areas at all it says
so and does nothing, because proposing a set of rooms would be inventing the
structure the feature exists to read. Every entity it places is flagged as
inferred and reads "(guessed)" wherever it is shown, including in the prompts
built from it.

## Found by review, before release

An adversarial review of the four features above, partitioned across the
preference module, the store, the pipeline, scenes, explanations, the web layer
and the tests themselves. Every fix below is pinned by a test that was confirmed
to fail without it.

**Things that could hide or lose a suggestion**

- Undoing a preference read the hidden suggestions back a page at a time,
  ordered by score, across *every* preference - so a preference hiding
  low-scored suggestions could have all of them left behind because other
  preferences were hiding higher-scored ones. Those rows then stayed hidden with
  nothing pointing at them. It is now one statement scoped to the preference,
  with no limit.
- A run decides what to hide from a snapshot of the preferences taken minutes
  earlier. If the user switched that preference off in between, the write landed
  behind the undo. The suppression is now conditional on the preference still
  being active, decided inside the write.
- A provider timeout made `learn` return an empty list, which
  `save_preferences` read as "the model withdrew every preference" - deleting
  every learned preference the user had, silently. Only a successful call may
  now rewrite the stored set, and a failed one is reported as a degradation.
- A dismissed scene was rebuilt with the same id every run and kept being
  counted as surfaced. Scenes are built after the dismissal filter runs, so they
  are now filtered on their own.
- Applying an AI result - area inferences, scenes, explanations - ran outside
  the isolation that wraps the model call, so a bug there ended the whole run
  and discarded every mined and backtested candidate. All three are isolated.

**Things a model could say that got through**

- A preference citing the same dismissal twice cleared the "at least two
  dismissals" bar on the strength of one.
- A citation list that was a bare number, or contained a nested object, crashed
  `learn`; an unhashable `preference` value crashed the matcher. Both claim
  never to raise.
- The `MAX_PREFERENCES` cap counted proposals rather than valid ones, silently
  discarding good preferences further down the list.
- An explanation could state a number that is real for one field while asserting
  it about another - quoting the window length as the number of occurrences,
  say. "N of M" is now checked as a pair. A raw ratio no longer permits its own
  rounding, which had been putting "1" into the allowed set for almost every
  card. Numbers written as words are refused rather than waved through, negative
  numbers no longer pass on the strength of their magnitude, and a correct
  figure written with a thousands separator is no longer rejected.
- The evidence sent for an explanation is now an allowlist. Popping `samples`
  was not enough: `extra` carries raw epoch timestamps of its own for some
  miners, and the window bounds are exact timestamps too.
- Two scene members calling the same service on the same entity with different
  data - `light.turn_on` at brightness 30 and at 255 - were not a conflict, and
  both survived into one scene that fired them back to back. Services with no
  binary state, like `climate.set_temperature`, were invisible here for the same
  reason. `toggle` alongside anything else on the same entity now counts too.
- Two groupings could consolidate to the identical rule and be stored under one
  id, the second silently erasing the first's name.
- Two areas whose names differ only in case were collapsed, so a guess could be
  written to whichever of them survived. Neither is offered now.

**Things that were measured wrong**

- Scene members were compared along a number line, so 23:58 and 00:02 looked
  nearly a day apart and every routine that straddles midnight was refused - the
  module's own motivating example. Distances are measured around the clock now.
- A shared condition was dropped whenever two members spelled it differently:
  `weekday` is built in whatever order it was read, and `source` records which
  miner found it. Conditions are compared on meaning now.
- A sun-triggered grouping was built, sent through the gate and rejected there,
  because the backtester cannot simulate a sun trigger. It is refused up front
  and reported as what it is.

**Things that blocked or leaked**

- Building the provider for the wording helper ran on the event loop. The Ollama
  provider probes several candidate hosts with synchronous HTTP at a two-second
  timeout each, so every click froze the whole UI - `/health` included - for up
  to fourteen seconds whenever Ollama was not running. The original test could
  not see this: it replaced `build_provider`, which is the call that blocked.
- The count of suggestions an undo restored was computed and discarded. It is
  what the UI now tells the user.
- `area_inferred_reason` was collected and readable by nothing.

**Tests that were not holding the line**

- "A hidden suggestion is not announced" passed because the suggestion had been
  seen twice, not because it was hidden - removing the status filter entirely
  left it green. It now suppresses on a first sighting.
- The isolation test for scenes never reached the step it was breaking, because
  the stub proposed no grouping to apply.
- Several assertions checked a method's return type rather than the state that
  changed, and broke on a legitimate refactor while the guarantee held.

**Also**

- `prune_suggestions` now sweeps hidden suggestions on the same terms as new
  ones. Hidden is not decided, so a stale hidden row was as much litter as a
  stale new one — and it would have accumulated forever.
- The Status page now describes what `llm_audit` and `llm_gaps` did. Both had
  been reported as a bare feature name with no summary since they were added.

## 0.5.0

**Fixed: gap suggestions that assumed things about your life**

Every gap is inferred from entities, and entities cannot see an electricity
contract, a roof, a car or a job. "Add a dynamic electricity price sensor" was
recommended to someone on a fixed-price tariff, where it saves exactly nothing -
and the suggestion gave no hint that it depended on the tariff at all.

`GapSuggestion` now carries a **`requires`** field naming the real-world
condition that makes the suggestion worth acting on, shown on the card as
*"Only worth it if"*. Six of the ten detector gaps state one:

- the two energy-price gaps name the variable tariff they assume, and say
  plainly that on a fixed-price contract there is nothing to shift loads
  towards;
- the solar forecast gap names the panels;
- the workday sensor gap names a schedule that follows public holidays;
- the presence gap names carrying a tracked device;
- the carbon-intensity gap says outright that it is a preference rather than a
  saving, because the cleanest hour and the cheapest hour are often different.

The other four have no precondition beyond the evidence already on the card, and
deliberately state none - filler would make the field meaningless on the ones
that matter.

**Added: `llm_gaps` (default off)**

The detector's rules are a fixed list of named patterns. Noticing that someone
with a heat pump and no energy dashboard might want one takes knowing what those
things are for, which is world knowledge, so a model is asked - and held to the
rule that prompted this release: **every proposal must state its precondition**,
or it is rejected rather than patched up. "requires": "that you want cheaper
electricity" is not a precondition; "an electricity contract whose price varies
through the day" is.

It is additive only. It cannot remove, reword or reorder anything the detector
produced, cannot restate a detector gap under a new name, cannot cite an entity
that does not exist, and cannot claim you *have* anything - it proposes, and
names the condition under which the proposal applies. Everything it suggests is
labelled **model-proposed** on the card and ranks below every detected gap. It
is sent no raw history: only which signal roles were detected, how much manual
activity there was per domain, and the titles the detector already used.

The run report records every rejection reason, so a model producing unusable
proposals is visible on the Status page rather than silently doing nothing.

## 0.4.0

**Fixed: the audit read triggers and targets, and claimed to have read conditions**

`audit_existing` compared two automations by their trigger entities, their
trigger times and the entities they act on. It never looked at conditions — yet
both of its messages ended "under overlapping conditions", asserting something
nothing had checked.

Conditions are how people say *this one is for when I am out, that one for when
I am in*. Ignoring them meant complementary automations were reported as
conflicts:

- "turn the light off when motion, **if nobody is home**" against "turn it on
  when motion, **if somebody is home**" was an **error**-severity value
  inconsistency, though the two can never both apply;
- "turn the hall light on **below 20 lux**" against "**above 500 lux**" was
  reported as redundancy.

The new `amminer.conditions` module decides the part of this that is decidable:
whether two condition sets can *provably* never both hold. Opposite required
states, disjoint numeric ranges, disjoint weekday sets, non-overlapping time
windows (including ones crossing midnight), and sun-up against sun-down are all
proven exclusive, and a pair proven exclusive is no longer reported at all. The
same blind spot existed when checking a mined candidate against an existing
automation, and is fixed there too.

Where exclusivity **cannot** be proven — two rules conditioned on different
entities, or on a template — the finding is still reported, but honestly: as a
warning rather than an error, reading "their conditions differ, so they may
never both apply". "I cannot prove these are exclusive" is not the same claim as
"these overlap", and the audit no longer conflates them. Rules with identical or
no conditions still read "under overlapping conditions", because there it is
true.

**Added: `llm_audit` (default off)**

For the findings that remain genuinely ambiguous, the model is shown both rules
in full — triggers, conditions and actions — and asked one question: in a real
home, can these two ever actually apply at once?

It is given no authority. It may **dismiss** a finding, which hides it with its
reasoning recorded, or **soften** one from error to warning. It may never raise
a severity and never invent a finding: a warning the deterministic audit did not
produce is never shown, whatever the model returns. A finding it does not
mention is untouched, an unusable verdict is counted and ignored, and a provider
outage leaves every finding exactly as it was.

The run report keeps the finding count from before and after the review, so a
model quietly dismissing real conflicts shows up on the Status page rather than
disappearing.

## 0.3.0

**Added: you no longer have to go and look**

Suggestions lived in this add-on's own database, so the only way to learn a
nightly run had found anything was to open its page. A nightly analysis nobody
is told about is a nightly analysis nobody reads.

- **`notify_on_new_suggestions`** (default `true`) posts a notification in Home
  Assistant when a run finds something, listing the first few titles and linking
  to the add-on. It replaces its own previous notification rather than stacking
  a new one beside it, and it says plainly that nothing has been applied.
- **`notify_service`** (default empty) additionally calls a notify service you
  name, e.g. `notify.mobile_app_your_phone`, so it reaches your phone without
  you writing an automation.

Only genuinely new suggestions are announced. A run re-surfaces every rule that
still holds, so announcing "what this run produced" would announce the same
rules every night — which is the fastest way to make a notification something
people switch off. A suggestion you have dismissed is never announced again.

Announcing is the last thing a run does and the least important thing it does:
it happens after everything is persisted, so a notification can only ever
describe suggestions that are really there to read, and every failure in it is
reported as a degradation and swallowed. An unreachable Home Assistant, a
notify service that does not exist, or a malformed `notify_service` value costs
you the message and nothing else — never a suggestion, never the run.

## 0.2.1

**Fixed**

- A cloud endpoint typed or pasted into `llm_base_url` was used exactly as
  given, so a stray space, a trailing newline, or a host with no `https://`
  made every request fail with "Request URL is missing an 'http://' or
  'https://' protocol". None of that was visible afterwards: the status page
  renders the value into HTML, which collapses surrounding whitespace, so the
  endpoint looked correct and the failure had no apparent cause. Endpoints are
  now trimmed, given a scheme if they lack one, and stripped of a trailing
  slash. The Ollama path had always done this; the cloud path never did.
- A Google endpoint set to the bare host posted to the host and got a bare 404.
  Google addresses the model in the path, so an endpoint without the `{model}`
  placeholder is now completed with the standard
  `/v1beta/models/{model}:generateContent` path rather than being sent as-is.
- The model was substituted into a Google endpoint with `str.format()`, so any
  other brace in a user-supplied URL raised `KeyError`. It is a plain
  replacement now.
- API keys and model names are trimmed, so a pasted key with a trailing newline
  authenticates.
- A failed cloud request reported the status and the URL and dropped the
  response body — but the body is the half that says why. "404 Not Found" now
  reads "404 Not Found - models/gemini-9 is not found for API version v1beta",
  and "400 Bad Request" says "API key not valid". The key is still scrubbed
  from all of it.

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
