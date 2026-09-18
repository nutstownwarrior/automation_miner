"""Infer a small number of unobserved household "modes" from raw activity.

The problem this exists to fix: the single most predictive variable in a home
- whether the household is asleep, out, winding down, cooking, hosting guests -
has no entity.  ``conditional.py`` can only condition on signals that already
exist (weather, presence, workday...); ``time_of_day.py`` only ever sees the
clock and a weekday/weekend split.  A habit that is really driven by "the
evening has wound down" looks, to both of them, like noisy clock behaviour: a
light that goes on inconsistently around 22:00-23:00 clears neither miner's
bar, when conditioned on the right latent state it would be highly consistent.

This module fits a small hidden Markov model (HMM) over binned household
activity with EM (Baum-Welch), in pure numpy - no scikit-learn, no hmmlearn
(deliberately not dependencies: neither ships musllinux wheels for aarch64,
see the CI ``wheels`` job) - and exposes the result as an ordinary
:class:`~amminer.enrich.signals.SignalSeries`, in exactly the step-function
shape every other signal (weather, presence, workday...) already has.  That is
the entire point: once the inferred mode is a series with a ``value_at(ts)``,
``conditional.py`` tests it exactly like it tests outdoor temperature or a
workday sensor, through the code path that already exists - see
``condition_entities`` in ``amminer/miners/conditional.py``.  No change to the
signal layer itself was needed; :class:`SignalSeries`/:class:`SignalStore`
already make no assumption that an entity id names something Home Assistant
actually has.

Design decisions, and why
--------------------------

**Bin width - 15 minutes.**  Short enough to catch the kind of regime change
this model exists to find ("dinner starts", "everyone has gone to bed") without
averaging it away, long enough that a full year of history is at most a few
tens of thousands of bins - see ``MAX_TRAIN_BINS`` below for how that is
bounded regardless.

**Observation representation - per-bin counts of currently-active entities,
bucketed by a small, fixed domain vocabulary, with diagonal-covariance
Gaussian emissions on their log1p.**  A multinomial/count emission model is a
reasonable
alternative, but it needs more free parameters per state (a full distribution
over event *types*, not just how many) for the same amount of data, and this
project already leans on log1p + Gaussian for exactly this kind of heavy-tailed
count data (see ``amminer/learn/ranking.py``'s ``_LOG_SCALE`` and its own
reasoning).  Six buckets - light, switch, media_player, climate,
other_actionable, presence - keep the parameter count small: with ``d=6`` and
``k`` up to 5, a state has 12 free emission parameters, not hundreds.  Clock
time is deliberately **not** a feature: feeding the hour of day back into the
model that is supposed to discover something beyond the hour of day would make
"the light turns on in the evening" and "the home enters its winding-down
pattern" the same finding wearing a different name.  The model only ever sees
*what* is happening, never *when*; that a mode still correlates with certain
hours (it should - human routines do) is discovered after the fact, from the
decoded segmentation, and reported as a description, never fed in as truth.

**State count - selected from a small range, never hardcoded, by whichever of
this project's two sanctioned criteria the data can support.**  Every
candidate in ``STATE_CANDIDATES`` is fit (with restarts); when there is enough
training history to spare a slice of it, the number of states is the one
whose *held-out* likelihood on that slice clears a noise bar over the next
smaller candidate (see :func:`select_model` and :data:`SELECTION_Z_SCORE`),
falling back to in-sample BIC only when there is not enough history for that
slice to mean anything. Either way a spurious extra "mode" that just fits
noise, or a shape a Gaussian is a poor match for, is not worth its added
complexity, and a quiet home legitimately comes back with 2.

Clearing that noise bar is necessary but not sufficient: a larger candidate
must *also* introduce a state whose emission is not a near-duplicate of one
a smaller candidate already has (see :func:`_larger_fit_adds_a_novel_state`).
A first-order Markov chain's dwell time in a state is memoryless; real
activity durations often are not (a burst that lasts "about an hour" is
much closer to constant than exponential), and splitting one true regime
into two states with identical emissions but different transition dynamics
is a real, reproducible way to fit that non-geometric dwell time better -
clearing *any* held-out significance bar, however strict, because it is
not sampling noise. Nothing about *how significant* that gain is can tell
it apart from a genuine extra regime; what the new state's emission
actually looks like can, which is why both conditions are required
together, not either alone.

It is not, however, a hard floor: the state count actually reported is the
number of states the *final* fit still uses after refitting on every
training bin (a component with no support left in that refit can end up
empty rather than merging into a neighbour - see
:func:`_drop_unoccupied_states`), so a home whose activity does not
distinguish into separate regimes at all can honestly come back as 1,
described plainly rather than split into two groups neither of which means
anything (see "Honesty" below).

**Restarts and determinism - fixed, seeded, best-of-N.**  EM only finds a local
optimum, so each candidate ``k`` is fit :data:`RESTARTS` (or, for ranking
candidates against each other, the cheaper :data:`SELECTION_RESTARTS`) times
from independent initialisations - and "best" means two *different* things
for two different jobs, deliberately not the same criterion for both (see
:func:`_select_best_restart`'s own docstring for the full reasoning and the
regression each half guards against). Choosing which restart's *model* to
keep, once ``k`` is already settled, prefers whichever restart causally
distinguishes the most states once decoded the same causal way this module
is ever deployed - EM's own smoothed training objective can otherwise
prefer a restart that looks better on paper and collapses in practice.
Ranking *different* candidate ``k`` values against each other uses plain
best log-likelihood instead, with no such preference: applying the first
rule there does not fix anything (a single-state restart was never the
best-scoring one for its own k anyway) and instead systematically rewards
a restart that fragments one true regime into several near-duplicate
states, at every k, pulling the whole ranking toward the top of
:data:`STATE_CANDIDATES` regardless of what the data actually supports.
Every restart's random state comes from
``numpy.random.default_rng`` seeded with a fixed base constant plus ``(k,
restart_index)`` - never from wall-clock time, never from Python's global
``random`` module, never from iteration over a ``dict``/``set`` whose order
could differ across processes (feature bucket order is a fixed tuple,
:data:`FEATURE_NAMES`, and bins are built by walking sorted, already-ordered
``StateChange`` timestamps). Given the same ``StateChange`` rows and the same
window, this fits bit-for-bit the same model in a fresh Python process - see
``tests/test_home_mode.py::test_fit_is_deterministic_across_processes``,
which does exactly that across a subprocess boundary.

**Bounding the work for a nightly cron on a Raspberry Pi.**  The forward-backward
recursion is sequential in time by construction, so its cost is dominated by
Python-level loop overhead over bins, not raw FLOPs - a fact an earlier
version of this module's own docstring got wrong by reasoning only about
floating-point operation counts, before actually timing it.  Two things keep
that bounded: :data:`MAX_TRAIN_BINS` caps ``T`` at 5760 (60 days of 15-minute
bins) regardless of how large a window is configured, and :func:`select_model`
only ever runs the full :data:`RESTARTS`:code:`x`:data:`MAX_EM_ITERS` budget
on *one* candidate state count - the one either BIC or held-out likelihood
actually chose - scoring the other candidates against each other first at a
deliberately cheaper :data:`SELECTION_RESTARTS`:code:`x`:data:`SELECTION_MAX_ITERS`
(see that constant's own comment: fitting every candidate at full quality
twice, once to compare them and again to keep the winner, was measured as the
single largest cost here). A user who points this at a year of dense history
does **not** get an EM run over 35,000 bins: bins beyond
:data:`MAX_TRAIN_BINS` are dropped from the *start* of the training window
(the most recent activity is kept) - a deliberate, documented subsampling
policy, not an accident of whatever the window happened to be. Recent
behaviour is also the more relevant behaviour for "what mode is the household
in tonight", which is the only thing this model needs to be good at.

Below ``options.home_mode_min_train_days`` (see that field's own docstring in
``amminer.config`` for the evidence behind its default) :func:`fit` does not
run at all - a home with too little history to trust a latent-mode model does
not pay for one either, and the same "not enough history yet" honesty this
project already gives ``backtest_min_train_days``/``health_min_days`` applies
here too (see :func:`fit`'s own early return and ``report.degradations`` in
``amminer.pipeline``). :func:`select_model` also never fits a candidate ``k``
the data plainly cannot support - see :data:`MIN_BINS_PER_CANDIDATE_STATE` -
so a home with, say, nine days of history is not asked to distinguish five
states from a few hundred bins just because it cleared the floor above.

Restarts of the *same* candidate ``k`` are fit together, one shared,
still-sequential-in-``t`` sweep instead of one sweep per restart - see
:func:`_forward_backward_batch`'s own docstring for the mechanism (the same
"numpy's per-call overhead dwarfs the work at k<=5" fact behind
:func:`_logsumexp_row`, paid once per batch instead of once per restart) and
:func:`_fit_em_batch`'s for why this changes nothing about what any one
restart converges to - each restart's own trajectory, and its own
independent convergence check, depend only on its own history, never on
another restart sharing the same sweep.

**The holdout discipline.**  :func:`fit` takes only the rows and window the
caller decides to hand it - it has no way to reach into a wider window itself,
and it is the caller's job (``amminer.pipeline``) to call it with
``train_changes``/``train_window`` alone, exactly as every other miner already
receives.  Once fit, the model's parameters are frozen; :func:`decode_series`
replays them forward over a *longer* range (train + holdout) using **causal
filtering only** (each bin's label depends on that bin's own activity and
everything before it, never on what comes after - see :func:`_filter_states`).
That is deliberately not the usual Viterbi/smoothed decoding, which would let a
holdout bin's label be informed by holdout activity *after* it - a real HA
install computing this live could never do that either, and this project's own
holdout doctrine (``amminer/pipeline.py``, ``amminer/backtest.py``) treats
"could a live system have known this at the time" as the line that matters.
Each bin's label is stamped at the bin's *close*, not its start, for the same
reason ``amminer/enrich/signals.py``'s long-term-statistics fallback stamps an
hourly mean where its interval closes: the label is not knowable until the
activity inside the bin has actually happened.

**Mode identity across runs.**  The model is refit from scratch every night
(see "The holdout discipline" above and ``amminer.pipeline``, which persists
the result via ``Store.save_home_mode_model``) - there is nothing incremental
about a from-scratch EM fit to carry forward.  But EM's states are
unlabelled: nothing about a fit itself prefers calling one cluster "state 0"
over "state 1", so two fits of even *identical* underlying behaviour, one
restart-draw apart, are free to number their states differently.  Left alone,
that makes "mode_2" mean something different from one night to the next -
unstable for a person trying to build an understanding of their own home's
modes, and for a mined candidate that conditions on a specific mode, whose
meaning would silently shift under it between the run that found it and the
run that reports it.  :func:`fit` therefore accepts the *previous* persisted
model (``amminer.pipeline`` reads it back before calling :func:`fit`) purely
to relabel tonight's states to best match last time's, by the same
scale-aware emission distance :attr:`HomeModeModel.weakly_separated` already
uses (see :func:`_align_state_order`) - never to seed or otherwise influence
the fit itself, so this has no effect whatsoever on what was found, only on
which integer names it. When the state count itself changes night to night,
no relabelling is attempted: there is no bijection between two differently-
sized sets of states that could preserve every identity, and a state count
that changed is real information, reported honestly as
:attr:`HomeModeModel.n_states` changing rather than papered over by a
best-effort partial relabelling.

Honesty about what the modes are
---------------------------------

States are unlabelled and unsupervised.  Nothing here ever writes "asleep" or
"away" as fact.  :meth:`HomeModeModel.state_summaries` describes each state by
what was actually observed while the model was in it - a typical time-of-day
centre and spread (via the same circular statistics ``time_of_day.py`` already
uses), the most active domains, and (when a resolver is given) the most active
areas - and nothing stronger.  A home whose activity does not really split
into distinct regimes is not hidden behind a confident-looking two-mode
story: :attr:`HomeModeModel.weakly_separated` is set when the states the
fit found overlap heavily (their means are close relative to their spread),
and every caller that renders a description is expected to say so plainly
(see ``describe()``'s own wording) rather than name two "lifestyles" that
are actually one blurred together.

What this deliberately does not do
------------------------------------

The mode series this module produces is usable by the *mining and backtesting*
layers immediately - ``conditional.py`` can discover and validate "you do X
when the home is in mode M" exactly as it validates any other conditional
habit, entirely offline, against history.  It does **not** publish a live,
continuously-updated entity into Home Assistant.  That matters at *apply*
time: ``amminer.llm.validate.validate_references`` correctly refuses to ship
any trigger or condition naming an entity id Home Assistant does not actually
have, and ``amminer.entities``'s resolver has no way to know about
``HOME_MODE_ENTITY_ID`` because nothing has registered it as a real entity.
So a candidate that conditions on the inferred mode is real, mined, and
backtested evidence about the household's behaviour - and today it cannot be
one-click-applied as a live automation until a follow-up feature actually
publishes the current mode into Home Assistant as a helper entity that is kept
up to date between nightly runs.  This is a genuine, known gap, not an
oversight papered over: ``amminer.llm.validate``'s own test suite already
covers that refusal for any unknown entity id, this module's id included; no
special case was added, and none was needed.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from ..config import ACTIONABLE_DOMAINS, Options
from ..enrich.detect import SignalSet
from ..enrich.signals import SignalSeries
from ..recorderdb.models import Cause, StateChange
from ..util.timeutil import circular_mean_minutes, circular_std_minutes, hhmm, local_tz

_LOGGER = logging.getLogger(__name__)

#: Bumped whenever the feature vector's shape/meaning or the persisted model
#: shape changes - the same "a stale fit is meaningless, not stale data"
#: discipline ``amminer/learn/ranking.py`` uses for its own schema version.
HOME_MODE_SCHEMA_VERSION = 1

#: The synthetic entity id the inferred mode is exposed under, in the exact
#: shape enrich/signals.py already produces. The "amminer." prefix is
#: deliberate: it can never collide with a real Home Assistant domain, and it
#: marks - to a human reading a candidate's payload - that nothing in Home
#: Assistant's registry backs this id (see the module docstring's "What this
#: deliberately does not do").
HOME_MODE_ENTITY_ID = "amminer.home_mode"

#: One activity bin. See module docstring ("Bin width").
BIN_SECONDS = 900.0

#: Hard cap on bins one fit ever processes. See module docstring ("Bounding
#: the work"). 60 days at 15-minute bins.
MAX_TRAIN_BINS = 5760

#: The actual minimum-training-history floor :func:`fit` checks against is
#: ``options.home_mode_min_train_days`` - a config option, not a constant
#: here, precisely so it can be tuned like every other "enough history to
#: trust this" bar in this project (``backtest_min_train_days``,
#: ``health_min_days``...). See that field's own docstring (``amminer.config``)
#: for the evidence behind its default.

#: Candidate numbers of hidden states to try; BIC picks among these, never a
#: fixed one. Bounded at 5 so a "quiet home" question never becomes an
#: unbounded search, and because past five unlabelled, low-data household
#: regimes stop being distinguishable from each other in practice.
STATE_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5)

#: Independent, seeded restarts used for the model that is actually kept -
#: either the single candidate the BIC fallback fits, or (see
#: SELECTION_RESTARTS below) the one candidate state count held-out
#: selection actually chose.
RESTARTS = 3
MAX_EM_ITERS = 25
EM_TOL = 1e-4

#: Cheaper restarts/iterations used only to *rank* candidate state counts
#: against each other (see select_model's held-out branch) - not to produce
#: anything that is ever kept as a model. Comparing four candidate state
#: counts at full RESTARTS/MAX_EM_ITERS twice (once to rank them, again to
#: refit the winner on all the data) was the single largest cost in this
#: module: on a Raspberry Pi, a 60-day fit at full cost both times measured
#: in low minutes, not the seconds a nightly job should spend on this. Ranking
#: candidates more cheaply and only spending the full budget on the one
#: candidate that is actually kept cuts that by roughly (RESTARTS /
#: SELECTION_RESTARTS) * (MAX_EM_ITERS / SELECTION_MAX_ITERS) without
#: changing which candidate wins in practice - the ranking only needs to be
#: good enough to compare four numbers against each other, not to produce a
#: publication-quality fit for all four.
#:
#: ``SELECTION_RESTARTS`` was raised from 2 to 4 after measuring that 2 was
#: not enough restarts for the ranking fits themselves to be reliable: a
#: candidate as small as ``k=2`` can, at only 2 cheap restarts, land on a
#: genuinely degenerate fit (its states not causally distinguishable at
#: all - see :func:`_causal_occupied_count`) purely by bad luck, which then
#: makes *every* larger candidate look like a dramatic, highly "significant"
#: improvement for the wrong reason: not because it found more real
#: structure, but because the baseline it is being compared against never
#: represented the data it already had. That, not fragmentation being
#: rewarded, was the actual mechanism behind
#: ``build_two_regime_activity``'s seeds returning 4-5 states for a
#: genuinely two-regime household even with occupied-count preference
#: correctly confined to :func:`_fit_best_of_restarts`'s *other* job (see
#: that function's own docstring) - a degenerate ``k=2`` baseline makes
#: :func:`_larger_fit_adds_a_novel_state` too, since a fit that never
#: distinguished its own states looks "novel" relative to almost anything.
#: 4 restarts was measured sufficient for every synthetic fixture this
#: module's own test suite exercises to reliably find a properly-converged
#: fit at every candidate k; raising it further had no further effect
#: (the degeneracy is a discrete "did a restart happen to land near the
#: right optimum" event, not something more EM iterations at the *same*
#: restarts fixes - see :data:`SELECTION_MAX_ITERS`, left unchanged and
#: confirmed by the same measurement to already be enough once a restart
#: does land well). Doubling it also did not make the module slower
#: end-to-end in practice: a ranking stage that reliably tells small
#: candidates apart from large ones now more often *keeps* the count small,
#: and the (far more expensive) full-quality refit on the winning k is
#: correspondingly cheaper - measured net effect on this module's own
#: dense-synthetic-year benchmark was a wash to a modest improvement, not a
#: regression.
SELECTION_RESTARTS = 4
SELECTION_MAX_ITERS = 10

#: Absolute floor under any state's emission variance, regardless of scale.
#: Prevents a division by (near) zero when a feature never varies at all.
VAR_FLOOR = 1e-3

#: The *effective* floor used while fitting is the larger of :data:`VAR_FLOOR`
#: and this fraction of the whole dataset's own feature variance, computed
#: once per fit. A flat, tiny constant floor is not enough on its own: EM can
#: still raise the likelihood without bound by carving a state around a
#: handful of near-identical bins and shrinking its variance towards that
#: constant - a textbook Gaussian-mixture singularity, and exactly the kind
#: of "EM silently converges to something degenerate" failure this project's
#: pre-flight checklist calls out by name (see ``tests/test_home_mode.py``,
#: which fits on data with genuinely only two regimes and asserts this does
#: *not* happen). Scaling the floor to the data's own spread means a handful
#: of outlier bins can no longer buy an implausibly tight, and therefore
#: implausibly persuasive, "mode" - the model still fits real minority
#: patterns, but only a state that captures a *broad* enough share of the
#: variance for that to be a genuine explanation rather than four data points
#: standing in a memorised huddle.
VAR_FLOOR_FRACTION = 0.05

#: Fixed forever - part of what makes a fit reproducible. Not a secret, not
#: tied to anything; just a constant so restarts are deterministic without
#: depending on wall-clock time or process-specific hash randomisation.
_SEED_BASE = 1_753_014_001

#: The activity feature vocabulary. Six buckets, in a fixed order, so the
#: feature vector has a stable width and meaning across every fit forever -
#: the same "small, fixed vocabulary" discipline as
#: ``amminer.learn.ranking.DOMAIN_VOCAB``.
FEATURE_NAMES: tuple[str, ...] = (
    "light",
    "switch",
    "media_player",
    "climate",
    "other_actionable",
    "presence",
)

_DOMAIN_BUCKET: dict[str, str] = {
    "light": "light",
    "switch": "switch",
    "input_boolean": "switch",
    "media_player": "media_player",
    "climate": "climate",
    "water_heater": "climate",
    "cover": "other_actionable",
    "lock": "other_actionable",
    "fan": "other_actionable",
    "humidifier": "other_actionable",
    "input_number": "other_actionable",
    "input_select": "other_actionable",
    "select": "other_actionable",
    "number": "other_actionable",
    "siren": "other_actionable",
    "valve": "other_actionable",
    "lawn_mower": "other_actionable",
    "scene": "other_actionable",
    "script": "other_actionable",
    "vacuum": "other_actionable",
}
# Every actionable domain must land in exactly one bucket - a domain added to
# ACTIONABLE_DOMAINS later and forgotten here would silently vanish from the
# activity signal instead of loudly failing at import time.
assert set(_DOMAIN_BUCKET) == set(ACTIONABLE_DOMAINS), (
    "_DOMAIN_BUCKET and ACTIONABLE_DOMAINS have drifted apart"
)


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def _bin_bounds(start_ts: float, end_ts: float, bin_seconds: float) -> tuple[int, int]:
    """The half-open range ``[first_idx, last_idx)`` of bin indices covering the window."""
    first_idx = int(start_ts // bin_seconds)
    last_idx = int(-(-end_ts // bin_seconds))  # ceil division
    return first_idx, max(last_idx, first_idx)


def _presence_entities(signals: SignalSet) -> frozenset[str]:
    """Entities whose *any* transition is a proxy for someone being present.

    Deliberately reuses signals ``enrich/detect.py`` already found (motion,
    occupancy, person, device trackers, room-level presence) instead of
    inventing a second regex layer - the whole point of taking ``signals`` as
    an input here rather than raw entity ids.
    """
    return frozenset(
        signals.motion
        + signals.occupancy
        + signals.person
        + signals.device_tracker
        + signals.room_presence
    )


#: States that mean "not doing anything" for the level snapshot below - a
#: light in any other state (``"on"``, a colour, a brightness-as-string...)
#: counts as active, a media_player ``"playing"`` or ``"paused"`` does not
#: mean the same thing as ``"idle"``, and so on.  Deliberately a denylist, not
#: an allowlist: this project's other heuristics (``amminer.enrich.detect``)
#: already show that Home Assistant's live vocabulary of "on-like" values is
#: too varied to enumerate positively, and a value this misses just does not
#: register as activity, one bin, which is a far smaller mistake than
#: excluding a real one.
_INACTIVE_STATES = frozenset(
    {
        "off",
        "closed",
        "idle",
        "standby",
        "unavailable",
        "unknown",
        "",
        "paused",
        "not_home",
        "away",
        "docked",
        "none",
        "locked",
    }
)


def _is_active_state(state: str | None) -> bool:
    return (state or "").strip().lower() not in _INACTIVE_STATES


def build_feature_matrix(
    changes: Sequence[StateChange],
    signals: SignalSet,
    options: Options,
    start_ts: float,
    end_ts: float,
    bin_seconds: float = BIN_SECONDS,
) -> tuple[np.ndarray, int]:
    """Per-bin counts of *currently active* entities over ``[start_ts, end_ts)``.

    Returns ``(counts, first_bin_index)`` where ``counts`` is ``(T, D)`` with
    ``D == len(FEATURE_NAMES)`` and ``T`` bins of width ``bin_seconds`` -
    every bin in range gets a row, including all-zero ones (a bin with no
    activity at all is itself informative: it is what "asleep" mostly looks
    like).  Counts are raw, un-transformed; :func:`_to_features` applies the
    log1p used for fitting.

    Each bin's row is a **level**, not a rate: for every domain bucket
    (:data:`_DOMAIN_BUCKET`) and for presence (:func:`_presence_entities`),
    it counts how many distinct entities were in an active state
    (:func:`_is_active_state`) *at the bin's close*, not how many transitions
    happened during it. An earlier version of this counted transitions - a
    ``light.turn_on`` was +1 in the bin it happened, and the next three hours
    the same light stayed on were worth nothing - and it materially
    under-detects exactly the regimes this module cares about most: a media
    session where ``media_player.playing`` is set once and stays that way for
    two hours looks, to a transition count, like one blip of activity
    surrounded by silence identical to genuinely nobody-home silence. A level
    snapshot keeps registering that entity as active in every bin the session
    actually spans, which is what lets a multi-hour regime look, to the model,
    like a multi-hour regime rather than one instant plus noise (see
    ``tests/test_home_mode.py``'s wind-down fixture, built specifically
    because a transition-count version of this function could not tell a
    real three-hour "watching something" pattern from clock noise either).

    Cause is deliberately not filtered here (unlike, say,
    ``amminer.miners.time_of_day.human_action_events``, which only wants
    changes a person caused): a level feature is describing *what state the
    home is in*, not attributing credit for who put it there, and an
    automation-controlled light that is on is still a light that is on.

    One deliberate non-guard worth naming: bucketing is by *domain*
    (:data:`_DOMAIN_BUCKET`), not by individual entity, so a single entity's
    own activity is one contribution among every other entity in its domain,
    not a bucket of its own. A specific entity's habit later tested against
    the mode this produces (``amminer.miners.conditional``) is therefore not
    immune to a mild circularity - in a home with very few lights, one
    light's own state is a non-trivial share of the whole "light" bucket that
    also helped decide what mode that moment was in. This is not treated as
    leakage requiring a special case, for the same reason
    ``amminer.miners.time_of_day`` clustering a habit's own trigger time from
    the same occurrences it then reports consistency for is not: the
    project's actual safeguard against "found a pattern in the data it was
    shaped by" is the train/holdout backtest every candidate goes through
    regardless of which miner or signal produced it (``amminer.backtest``),
    not preventing every possible correlation between a signal and the action
    it might end up explaining.
    """
    first_idx, last_idx = _bin_bounds(start_ts, end_ts, bin_seconds)
    n_bins = last_idx - first_idx
    counts = np.zeros((n_bins, len(FEATURE_NAMES)), dtype=np.float64)
    if n_bins <= 0:
        return counts, first_idx
    bucket_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    presence_entities = _presence_entities(signals)

    relevant = [
        change
        for change in changes
        if change.ts < end_ts
        and not options.is_excluded(change.entity_id)
        and (change.entity_id in presence_entities or change.domain in _DOMAIN_BUCKET)
    ]
    relevant.sort(key=lambda c: c.ts)

    # Only *edges* - a change whose active/inactive state actually differs
    # from that entity's last tracked state - move a bucket's count; a
    # `light.turn_on` while it is already tracked "on" contributes nothing,
    # the same dedup `apply()` always did. This loop is therefore already
    # over transitions, not bins, and is unavoidably sequential per entity
    # (each decision depends on that entity's own previous state) - what
    # used to also be a Python-level loop here was the *snapshotting* below,
    # writing every one of up to MAX_TRAIN_BINS bins one row at a time even
    # when nothing in it changed. A bin's level is exactly the running total
    # of every edge at or before it, which a single cumulative sum computes
    # for every bin and every bucket at once - integer running counts, so
    # this is an exact reformulation, not an approximation: `np.add.at` followed
    # by `cumsum` gives bit-for-bit (here, exactly, since counts are whole
    # numbers) the same per-bin totals the old snapshot-every-bin loop wrote.
    current: dict[str, bool] = {}
    edit_bins: list[int] = []
    edit_features: list[int] = []
    edit_deltas: list[int] = []

    def apply(entity_id: str, active: bool, bin_idx: int) -> None:
        if current.get(entity_id, False) == active:
            return
        current[entity_id] = active
        delta = 1 if active else -1
        if entity_id in presence_entities:
            edit_bins.append(bin_idx)
            edit_features.append(bucket_index["presence"])
            edit_deltas.append(delta)
        bucket = _DOMAIN_BUCKET.get(entity_id.split(".", 1)[0])
        if bucket:
            edit_bins.append(bin_idx)
            edit_features.append(bucket_index[bucket])
            edit_deltas.append(delta)

    for change in relevant:
        # The bin whose close-boundary sweep would first have reached this
        # change under the old bin-by-bin walk - clamped to bin 0 for
        # anything at or before the window's own start, which has already
        # happened by the time the very first bin closes (the old walk's
        # `pos` pointer always reached those before bin_idx=0 finished too).
        # `relevant` is filtered to `change.ts < end_ts`, so this never
        # reaches n_bins.
        bin_idx = max(int(change.ts // bin_seconds) - first_idx, 0)
        apply(change.entity_id, _is_active_state(change.state), bin_idx)

    if edit_bins:
        np.add.at(counts, (np.array(edit_bins), np.array(edit_features)), edit_deltas)
        np.cumsum(counts, axis=0, out=counts)
    return counts, first_idx


def _to_features(counts: np.ndarray) -> np.ndarray:
    """log1p - the same heavy-tailed-count treatment ``learn/ranking.py`` uses."""
    return np.log1p(np.maximum(counts, 0.0))


# ----------------------------------------------------------------------
# Gaussian HMM: forward-backward EM (Baum-Welch), pure numpy
# ----------------------------------------------------------------------
def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_gaussian_pdf(
    x: np.ndarray, means: np.ndarray, variances: np.ndarray, var_floor: float = VAR_FLOOR
) -> np.ndarray:
    """Diagonal-covariance Gaussian log-density. ``x``: (T,d) -> (T,k)."""
    variances = np.maximum(variances, var_floor)
    diff2 = (x[:, None, :] - means[None, :, :]) ** 2
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * variances), axis=1)
    quad = -0.5 * np.sum(diff2 / variances[None, :, :], axis=2)
    return quad + log_norm[None, :]


def _logsumexp_row(values: list[float]) -> float:
    """Scalar log-sum-exp over a short Python list.

    The sequential recursions below (:func:`_forward_backward`,
    :func:`_filter_states`, :func:`_causal_bin_loglik`) call this once per
    state (or state pair) *per bin* - up to several hundred thousand times
    for one fit. Profiling an early version of this module found that at
    ``k <= 5``, ``numpy``'s per-call dispatch overhead for an operation this
    tiny dwarfs the arithmetic itself: replacing the vectorised
    :func:`_logsumexp` in exactly these hot loops with plain
    ``math.exp``/``math.log`` over a Python list cut a 45-day fit from
    roughly a minute to a few seconds, with the actual numbers unchanged
    (see ``tests/test_home_mode.py``'s determinism tests, which pin that
    down). :func:`_logsumexp` itself is kept, unvectorised call sites
    unchanged, for the M-step's whole-array reductions, where the arrays are
    large enough that numpy's overhead is negligible next to the work.
    """
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def _forward_backward(
    log_b: np.ndarray, log_pi: np.ndarray, log_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard log-space forward-backward. Sequential in ``t`` by necessity.

    Pure Python inside the per-bin loop - see :func:`_logsumexp_row`.
    """
    t_len, k = log_b.shape
    log_b_list = log_b.tolist()
    log_a_list = log_a.tolist()
    log_alpha: list[list[float]] = [[0.0] * k for _ in range(t_len)]
    log_alpha[0] = [float(log_pi[j]) + log_b_list[0][j] for j in range(k)]
    for t in range(1, t_len):
        prev = log_alpha[t - 1]
        row_b = log_b_list[t]
        log_alpha[t] = [
            _logsumexp_row([prev[i] + log_a_list[i][j] for i in range(k)]) + row_b[j]
            for j in range(k)
        ]
    log_beta: list[list[float]] = [[0.0] * k for _ in range(t_len)]
    for t in range(t_len - 2, -1, -1):
        nxt_beta = log_beta[t + 1]
        nxt_b = log_b_list[t + 1]
        log_beta[t] = [
            _logsumexp_row([log_a_list[i][j] + nxt_b[j] + nxt_beta[j] for j in range(k)])
            for i in range(k)
        ]
    loglik = _logsumexp_row(log_alpha[-1])
    return np.array(log_alpha), np.array(log_beta), loglik


@dataclass
class _EMResult:
    log_likelihood: float
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    n_iter: int


# ----------------------------------------------------------------------
# Batched EM: many independent restarts of the same k, fit together
#
# Every candidate ``k`` is ever fit as a *batch* of restarts (see
# _fit_best_of_restarts below) rather than one at a time - restarts of the
# same k share nothing statistically (independent initial draws, independent
# EM trajectories), only the sequential-in-t loop structure is shared, purely
# to amortise numpy's fixed per-call dispatch overhead across all of them at
# once instead of paying it once per restart (see
# _forward_backward_batch's own docstring, and _logsumexp_row's for the
# underlying "overhead dwarfs the work at k<=5" fact this exploits the other
# direction). A batch of one restart is a valid, if pointless, special case -
# there is no separate single-chain EM function to keep in step with this
# one.
# ----------------------------------------------------------------------
def _log_gaussian_pdf_batch(
    x: np.ndarray, means: np.ndarray, variances: np.ndarray, var_floor: float
) -> np.ndarray:
    """:func:`_log_gaussian_pdf`, batched over an extra restart axis.

    ``x``: ``(T, D)`` - shared by every restart in the batch, they are all
    fitting the same data. ``means``/``variances``: ``(B, K, D)`` -> ``(T, B, K)``.
    """
    variances = np.maximum(variances, var_floor)
    diff2 = (x[:, None, None, :] - means[None, :, :, :]) ** 2
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * variances), axis=2)
    quad = -0.5 * np.sum(diff2 / variances[None, :, :, :], axis=3)
    return quad + log_norm[None, :, :]


def _forward_backward_batch(
    log_b: np.ndarray, log_pi: np.ndarray, log_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`_forward_backward`, batched over an extra restart axis ``B``.

    ``log_b``: ``(T, B, K)``, ``log_pi``: ``(B, K)``, ``log_a``: ``(B, K, K)``.
    Returns ``(log_alpha, log_beta, loglik)`` shaped ``(T, B, K)``, ``(T, B, K)``,
    ``(B,)``.

    Forward-backward is sequential in ``t`` by construction (see
    :func:`_forward_backward`'s own docstring) and nothing here changes
    that - the ``t`` loop below still runs exactly ``T`` times. What is
    batched is *width*: at ``k <= 5``, :func:`_logsumexp_row`'s own docstring
    records that replacing a *single* chain's per-``t`` numpy call with pure
    Python was a real win, because numpy's fixed per-call dispatch overhead
    dwarfs the arithmetic for an array that tiny. That overhead is paid once
    per call, not once per element - so fitting ``B`` independent restarts of
    the same ``k`` in lock-step, one shared numpy call per ``t`` computing
    all ``B`` of them at once instead of ``B`` separate calls (numpy, once
    per restart) or ``B`` separate pure-Python sweeps, amortises exactly that
    fixed cost across ``B`` restarts' worth of real work instead of paying it
    ``B`` times over. Measured on this module's own production-shaped
    benchmark (T ~ 3000-5000, k = 5, B = RESTARTS/SELECTION_RESTARTS): a
    clear win over both alternatives, growing with ``B``.

    Restarts share nothing statistically - independent initial draws,
    independent trajectories - only the loop structure is shared. Each
    batch slot's arithmetic only ever touches its own slot (elementwise
    ops, and every reduction below is over the state axis, never the batch
    axis), so this computes exactly what ``B`` independent calls to
    :func:`_forward_backward` would - not an approximation, the same
    algorithm run wider.
    """
    # `.reduce()` on the ufunc directly (`np.maximum.reduce`/`np.add.reduce`)
    # rather than the free functions `np.max`/`np.sum` - measured ~2x faster
    # per call here, purely from skipping the free functions' generic
    # keyword-argument dispatch (`_wrapreduction`); same arithmetic, same
    # result, called T times per EM iteration so its own per-call overhead is
    # exactly the kind of fixed cost this function exists to amortise.
    t_len, b, k = log_b.shape
    log_alpha = np.empty((t_len, b, k))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, t_len):
        prev = log_alpha[t - 1]  # (B, K_from)
        m = prev[:, :, None] + log_a  # (B, K_from, K_to)
        mx = np.maximum.reduce(m, axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        lse = mx[:, 0, :] + np.log(np.add.reduce(np.exp(m - mx), axis=1))
        log_alpha[t] = lse + log_b[t]

    log_beta = np.zeros((t_len, b, k))
    for t in range(t_len - 2, -1, -1):
        tail = log_b[t + 1] + log_beta[t + 1]  # (B, K_to)
        m = log_a + tail[:, None, :]  # (B, K_from, K_to)
        mx = np.maximum.reduce(m, axis=2, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        lse = mx[:, :, 0] + np.log(np.add.reduce(np.exp(m - mx), axis=2))
        log_beta[t] = lse

    mx_last = np.maximum.reduce(log_alpha[-1], axis=1, keepdims=True)
    mx_last = np.where(np.isfinite(mx_last), mx_last, 0.0)
    loglik = mx_last[:, 0] + np.log(np.add.reduce(np.exp(log_alpha[-1] - mx_last), axis=1))
    return log_alpha, log_beta, loglik


def _fit_em_batch(
    x: np.ndarray,
    k: int,
    inits: Sequence[tuple[np.random.Generator | None, _EMResult | None]],
    var_floor: float,
    max_iter: int,
) -> list[_EMResult]:
    """``len(inits)`` independent, deterministic EM runs to convergence (or
    ``max_iter``), each to exactly what running it alone would have produced
    - one per ``(rng, init)`` entry of ``inits``, initialised either from
    ``rng`` (an independent random restart - the normal case) or, when
    ``init`` is given instead, from an *existing* fit's parameters (used to
    warm-start the full-quality refit from whatever :func:`select_model`'s
    cheaper inner comparison already found - see :func:`_fit_best_of_restarts`,
    so structure that comparison already located cannot be lost purely
    because an independent random redraw on the full data happens not to
    rediscover it); exactly one of a pair's ``rng``/``init`` must be given.

    Every restart runs the same standard Baum-Welch EM step (forward-backward
    E-step, then closed-form M-step - see :func:`_forward_backward_batch` for
    what is shared across restarts and why), computed for the whole batch at
    once, but each keeps its own independent convergence check: the moment a
    restart's own ``abs(ll - prev_ll) < EM_TOL * max(1, abs(prev_ll))``
    condition fires, checked against that restart's own ``ll``/``prev_ll``
    alone, never any other restart's, its ``(initial, transition, means,
    variances)`` are frozen at exactly the values running it alone would have
    left them at (the M-step output of the iteration that tripped the check -
    every iteration always applies one more M-step before testing for
    convergence) and are never touched again, while any restart still running
    in the same batch continues. This is what makes batching restarts
    together equivalent to running them apart: a restart's own results
    depend only on its own trajectory, never on how long any other restart
    in the batch happens to keep going.
    """
    b = len(inits)
    t_len, d = x.shape
    pi = np.empty((b, k))
    a = np.empty((b, k, k))
    means = np.empty((b, k, d))
    variances = np.empty((b, k, d))
    for i, (rng, init) in enumerate(inits):
        if init is not None:
            pi[i] = init.initial
            a[i] = init.transition
            means[i] = init.means
            variances[i] = init.variances
        else:
            assert rng is not None, "_fit_em_batch needs either rng or init per entry"
            pi[i] = rng.dirichlet(np.ones(k))
            a[i] = rng.dirichlet(np.ones(k), size=k)
            means[i] = x[rng.choice(t_len, size=k, replace=False)]
            variances[i] = np.maximum(x.var(axis=0), var_floor)

    converged = np.zeros(b, dtype=bool)
    prev_ll = np.full(b, -np.inf)
    n_iter = np.zeros(b, dtype=int)

    for iteration in range(1, max_iter + 1):
        active = ~converged
        if not active.any():
            break
        n_iter = np.where(active, iteration, n_iter)

        log_pi = np.log(np.clip(pi, 1e-300, None))
        log_a = np.log(np.clip(a, 1e-300, None))
        log_b = _log_gaussian_pdf_batch(x, means, variances, var_floor)
        log_alpha, log_beta, ll = _forward_backward_batch(log_b, log_pi, log_a)

        # E-step - the same per-slot log-space normalisation as any
        # Baum-Welch EM step, just with the batch axis carried along
        # untouched.
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=2)[:, :, None]
        gamma = np.exp(log_gamma)  # (T, B, K)

        log_xi = (
            log_alpha[:-1][:, :, :, None]
            + log_a[None, :, :, :]
            + (log_b[1:] + log_beta[1:])[:, :, None, :]
        )
        flat = log_xi.reshape(t_len - 1, b, -1)
        flat = flat - _logsumexp(flat, axis=2)[:, :, None]
        xi = np.exp(flat).reshape(t_len - 1, b, k, k)

        # M-step - x has no batch axis (every restart fits the same data);
        # everything else keeps its own per-restart values.
        new_pi = gamma[0] / gamma[0].sum(axis=1, keepdims=True)
        denom = np.maximum(gamma[:-1].sum(axis=0), 1e-300)  # (B, K)
        new_a = xi.sum(axis=0) / denom[:, :, None]
        new_a = new_a / np.maximum(new_a.sum(axis=2, keepdims=True), 1e-300)
        weight = np.maximum(gamma.sum(axis=0), 1e-300)  # (B, K)
        new_means = np.einsum("tbk,td->bkd", gamma, x) / weight[:, :, None]
        diff2 = (x[:, None, None, :] - new_means[None, :, :, :]) ** 2  # (T,B,K,D)
        new_variances = np.einsum("tbk,tbkd->bkd", gamma, diff2) / weight[:, :, None]
        new_variances = np.maximum(new_variances, var_floor)

        # Only restarts still active *this* iteration are updated - one
        # already converged in an earlier iteration is frozen exactly where
        # it broke, exactly the break-before-touching-params-again
        # behaviour running it alone would have had.
        pi = np.where(active[:, None], new_pi, pi)
        a = np.where(active[:, None, None], new_a, a)
        means = np.where(active[:, None, None], new_means, means)
        variances = np.where(active[:, None, None], new_variances, variances)

        just_converged = active & (np.abs(ll - prev_ll) < EM_TOL * np.maximum(1.0, np.abs(prev_ll)))
        prev_ll = np.where(active, ll, prev_ll)
        converged = converged | just_converged

    # Recompute the final log-likelihood under the parameters actually kept,
    # same reason any single EM run does: the loop's last `ll` was computed
    # *before* the M-step that produced the params each restart was frozen at.
    log_pi = np.log(np.clip(pi, 1e-300, None))
    log_a = np.log(np.clip(a, 1e-300, None))
    log_b = _log_gaussian_pdf_batch(x, means, variances, var_floor)
    _, _, final_ll = _forward_backward_batch(log_b, log_pi, log_a)

    return [
        _EMResult(float(final_ll[i]), pi[i], a[i], means[i], variances[i], int(n_iter[i]))
        for i in range(b)
    ]


def _bic(result: _EMResult, n_bins: int, n_features: int) -> float:
    k = len(result.initial)
    n_params = (k - 1) + k * (k - 1) + 2 * k * n_features
    return float(-2.0 * result.log_likelihood + n_params * np.log(max(n_bins, 2)))


def _variance_floor(x: np.ndarray) -> float:
    """The effective floor for this dataset. See :data:`VAR_FLOOR_FRACTION`."""
    return float(max(VAR_FLOOR, VAR_FLOOR_FRACTION * np.mean(x.var(axis=0))))


#: Fraction of the (already train-only) data held back, *inside* training,
#: purely to score candidate state counts - this never touches the real
#: backtest holdout amminer.pipeline carves off; it is a second, inner split,
#: the same idea amminer.learn.ranking's own leave-one-out guard uses within
#: its own label set. See "State count" in the module docstring for why this
#: is the default selector rather than in-sample BIC.
INNER_VALIDATION_FRACTION = 0.2

#: Below this many bins, a 20% inner slice is too thin (a couple of days) to
#: score five candidate state counts against; in-sample BIC (this project's
#: other sanctioned criterion) is used instead, and every fitted model
#: records honestly which method actually picked its state count.
MIN_BINS_FOR_HELD_OUT_SELECTION = 1000

#: A candidate ``k`` is only ever fit when there are at least this many bins
#: available *per state it would need*, i.e. only when ``len(x) >= k *
#: MIN_BINS_PER_CANDIDATE_STATE`` - below that, even a perfectly even split
#: across its states would leave the smallest of them less than this many
#: bins' worth of evidence, and no restart budget rescues an EM fit asked to
#: explain more regimes than the data could ever demonstrate. 200 bins is 50
#: hours, a little over two days - not a rigorous minimum-sample bound (the
#: real answer to "is this k actually supported" is what
#: :func:`select_model`'s held-out/BIC scoring and
#: :func:`_larger_fit_adds_a_novel_state`'s separation check already exist to
#: decide), just a cheap, early "do not even spend a restart budget on this"
#: cutoff for candidates that are obviously too big for how little data is on
#: the table - the same spirit as :data:`STATE_CANDIDATES` itself never
#: trying more than 5 states in the first place.
MIN_BINS_PER_CANDIDATE_STATE = 200


def _stationary_distribution(a: np.ndarray) -> np.ndarray:
    """The transition matrix's stationary distribution, as an entry prior.

    Used only to score held-out log-likelihood on a validation slice that
    does *not* start at the very first bin of the sequence - the fitted
    ``initial`` distribution describes "the state at the start of the
    training window", which has no particular reason to describe the state
    at some arbitrary later point the validation slice happens to begin at.
    The chain's own long-run distribution is the least assumption-laden prior
    for "which state is this process in, at a moment picked without looking".
    """
    k = a.shape[0]
    eigvals, eigvecs = np.linalg.eig(a.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = np.clip(vec, 0.0, None)
    total = vec.sum()
    if total <= 0:
        return np.full(k, 1.0 / k)
    return vec / total


def _causal_bin_loglik(
    x: np.ndarray, pi: np.ndarray, a: np.ndarray,
    means: np.ndarray, variances: np.ndarray, var_floor: float,
) -> np.ndarray:
    """Per-bin ``log P(x_t | x_1..x_{t-1})`` under a fixed model - one value per bin.

    This is the normalising constant discarded at each step of
    :func:`_filter_states`'s causal recursion, recovered instead of thrown
    away: summing it gives the sequence's total log-likelihood, and keeping
    it *per bin* is what lets :func:`select_model` compare two candidate
    state counts on the same held-out bins as a paired sample, the same way
    ``amminer.learn.ranking.train_from_labels`` compares its own cross-validated
    losses pairwise rather than as two bare aggregate numbers.
    """
    t_len = len(x)
    k = len(pi)
    log_pi = np.log(np.clip(pi, 1e-300, None)).tolist()
    log_a = np.log(np.clip(a, 1e-300, None)).tolist()
    log_b = _log_gaussian_pdf(x, means, variances, var_floor).tolist()
    # Pure Python per bin - see _logsumexp_row's own docstring for why.
    per_bin = [0.0] * t_len
    log_alpha: list[list[float]] = [[0.0] * k for _ in range(t_len)]
    raw0 = [log_pi[j] + log_b[0][j] for j in range(k)]
    per_bin[0] = _logsumexp_row(raw0)
    log_alpha[0] = [v - per_bin[0] for v in raw0]
    for t in range(1, t_len):
        prev = log_alpha[t - 1]
        row_b = log_b[t]
        raw = [
            _logsumexp_row([prev[i] + log_a[i][j] for i in range(k)]) + row_b[j]
            for j in range(k)
        ]
        per_bin[t] = _logsumexp_row(raw)
        log_alpha[t] = [v - per_bin[t] for v in raw]
    return np.array(per_bin)


def _held_out_bin_loglik(result: _EMResult, x_val: np.ndarray, var_floor: float) -> np.ndarray:
    """:func:`_causal_bin_loglik` on a slice the model was not fit from.

    Entered via the chain's stationary distribution (see
    :func:`_stationary_distribution`), since ``x_val`` starts wherever the
    inner split happens to fall, not at the beginning of a sequence the
    fitted ``initial`` distribution has any reason to describe.
    """
    if len(x_val) == 0:
        return np.zeros(0)
    prior = _stationary_distribution(result.transition)
    return _causal_bin_loglik(x_val, prior, result.transition, result.means, result.variances,
                               var_floor)


#: How many standard errors a larger model's mean per-bin gain has to clear
#: before the extra states are believed. ``amminer.learn.ranking`` uses a
#: bar of exactly one standard error for its own analogous guard, but that
#: guard is a single yes/no decision (personal model vs. the prior); this one
#: is walked up to three times in a row (2 vs 3, winner vs 4, winner vs 5,
#: see :func:`select_model`), and a one-SE bar (roughly a one-sided 16%
#: false-accept rate under pure noise, borderline on its own) compounds
#: across three sequential tries into a real chance of ending up several
#: states larger than the data actually support - reproduced empirically on
#: this module's own two-regime synthetic fixture across a spread of random
#: seeds before this constant was raised from 1.0. Two standard errors per
#: step (roughly a one-sided 2-3% false-accept rate) keeps the compounded
#: risk low while still accepting a real, clearly separated extra regime
#: (see ``tests/test_home_mode.py``'s wind-down fixture, which has one).
#:
#: This bar alone is **not enough**, and cannot be made enough by raising it
#: further - see :func:`_larger_fit_adds_a_novel_state` for why a second,
#: independent condition is required alongside it.
SELECTION_Z_SCORE = 2.0

#: Below this scale-aware distance (see :func:`_pooled_distance`), two
#: states' emissions are not something an observer - or this selection
#: criterion - can tell apart. Shared by :attr:`HomeModeModel.weakly_separated`
#: (describing an already-chosen model *after* the fact) and
#: :func:`_larger_fit_adds_a_novel_state` (deciding *during* selection
#: whether a larger model earned its extra state) - the same question either
#: way: is this state actually distinguishable from that one.
SEPARATION_MIN_DISTANCE = 1.0


def _pooled_distance(
    mean_a: np.ndarray, var_a: np.ndarray, mean_b: np.ndarray, var_b: np.ndarray,
    var_floor: float = VAR_FLOOR,
) -> float:
    """Scale-aware distance between two states' emissions, in pooled-standard-
    deviation units - the same quantity :attr:`HomeModeModel.weakly_separated`
    compares against :data:`SEPARATION_MIN_DISTANCE`, generalised here to
    compare a state from one fit against a state from a *different* fit
    (:func:`_larger_fit_adds_a_novel_state`, and separately
    :func:`_align_state_order` for matching a fit against a previous run's,
    where the default floor is fine - only a *relative* ranking of
    permutations, never an absolute threshold, is ever read from it there).
    ``var_floor`` should be the effective floor the fit(s) being compared
    were actually fit under (see :data:`VAR_FLOOR_FRACTION`) wherever an
    absolute threshold *is* being read from the result, so this always
    means the same thing :attr:`HomeModeModel.weakly_separated` does.
    """
    pooled_std = np.sqrt((var_a + var_b) / 2.0)
    pooled_std = np.maximum(pooled_std, np.sqrt(var_floor))
    return float(np.linalg.norm((mean_a - mean_b) / pooled_std))


def _larger_fit_adds_a_novel_state(
    current_fit: _EMResult, candidate_fit: _EMResult, var_floor: float
) -> bool:
    """Does ``candidate_fit`` (more states than ``current_fit``) contain at
    least one state that is not a near-duplicate, by :data:`SEPARATION_MIN_DISTANCE`,
    of every state ``current_fit`` already has?

    This exists because a significance test on held-out likelihood alone
    (:func:`_prefer_more_states`) is not sufficient to decide whether a
    larger ``k`` found a genuine extra behavioural regime, and cannot be
    made sufficient by raising :data:`SELECTION_Z_SCORE` - no matter how
    high. A first-order Markov chain's dwell time in a state is
    memoryless (geometric); real activity durations routinely are not (a
    "the TV is on for about an hour" burst is much closer to constant than
    exponential). Splitting one true regime into two states with the
    *same* emission distribution but different transition dynamics lets the
    chain approximate that non-geometric dwell time - a real, substantial,
    reproducible improvement in held-out likelihood, not sampling noise, so
    it clears *any* significance bar: measured z-scores for this exact
    spurious split ranged up to 286 across ``build_two_regime_activity``'s
    seeds, comfortably exceeding the 23-53 measured for a genuine third
    regime in ``build_winddown_habit_activity`` - there is no threshold
    that keeps the second and rejects the first, because the first is not
    noise either. What reliably tells them apart is not how *significant*
    the gain is but *what the extra state actually looks like*: a
    dwell-time phase split's new state has, by construction, (numerically
    measured) **zero** distance to a state the smaller model already had,
    while a genuine extra regime's new state measures several units away
    (see ``tests/test_home_mode.py::test_state_count_stays_small_across_seeds``
    for the reproduction, and ``test_a_third_real_regime_is_still_found``
    for the case this must still accept). Requiring both this *and* the
    significance bar means a state only ever gets added when it is both a
    genuine surprise on held-out data *and* something an observer could
    point to as different - exactly what :attr:`HomeModeModel.state_summaries`
    then goes on to describe.
    """
    for j in range(len(candidate_fit.means)):
        nearest = min(
            _pooled_distance(
                candidate_fit.means[j], candidate_fit.variances[j],
                current_fit.means[i], current_fit.variances[i],
                var_floor,
            )
            for i in range(len(current_fit.means))
        )
        if nearest >= SEPARATION_MIN_DISTANCE:
            return True
    return False


def _prefer_more_states(
    current_per_bin: np.ndarray, candidate_per_bin: np.ndarray,
    current_fit: _EMResult, candidate_fit: _EMResult, var_floor: float,
) -> bool:
    """Would the larger model's per-bin gain survive being sampling noise -
    and does it actually add a state worth having?

    The *mean* paired per-bin improvement must exceed
    :data:`SELECTION_Z_SCORE` standard errors of the paired differences, not
    merely be positive - the same idea ``amminer.learn.ranking.train_from_labels``
    uses for its own "did the fancier option actually earn its complexity"
    check, at a stricter bar (see :data:`SELECTION_Z_SCORE`'s own comment for
    why). Two candidate state counts fit on the same training slice routinely
    differ by a hairline on a modest validation slice even when neither
    describes anything real about the household - see this module's own
    state-count test, which is exactly what motivated this guard over a bare
    ``argmax`` across held-out scores (an earlier version of this function
    used one, and it picked a different, arbitrary state count for a
    genuinely two-regime household depending on nothing but the random seed
    of the synthetic fixture). :func:`select_model` calls this walking
    candidates from fewest states to most, each one challenging only the
    current best - so a bigger model has to clear the bar against the best
    complexity found *so far*, not against a fixed baseline it might have
    beaten by luck alone.

    Clearing that bar is necessary but - see
    :func:`_larger_fit_adds_a_novel_state` - never sufficient on its own:
    a significant held-out gain is also exactly what a spurious dwell-time
    phase split produces, so the larger fit must also introduce a state that
    is not just a near-duplicate of one the smaller fit already had.
    """
    diffs = candidate_per_bin - current_per_bin
    n = len(diffs)
    if n == 0:
        return False
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    significant = mean_diff > SELECTION_Z_SCORE * se_diff if se_diff > 0 else mean_diff > 0
    if not significant:
        return False
    return _larger_fit_adds_a_novel_state(current_fit, candidate_fit, var_floor)


def _causal_occupied_count(em: _EMResult, x: np.ndarray, var_floor: float) -> int:
    labels = decode_labels(x, em.initial, em.transition, em.means, em.variances, var_floor)
    return len({int(label) for label in labels})


def _select_best_restart(
    candidates: list[_EMResult], x: np.ndarray, var_floor: float,
    prefer_max_occupied: bool = True,
) -> _EMResult:
    """Pick the best of ``candidates`` - by one of two different criteria for
    two different jobs, never the same one for both (see ``prefer_max_occupied``).

    ``prefer_max_occupied=True`` (the default): prefer the restart that
    causally occupies the *most* states; break ties by (smoothed)
    log-likelihood. Not "exactly ``k`` states or fall back to raw
    likelihood": that exact-or-bust rule throws away everything a restart
    found the moment none of them happens to hit ``k`` on the nose,
    including a restart that causally distinguished ``k - 1`` (or ``k - 2``)
    states - which is obviously less collapsed, and therefore obviously
    preferable, to one that only ever occupies a single state, even though
    the old rule treated both as equally disqualified and picked between
    them on likelihood alone. Reproduced concretely:
    ``build_two_regime_activity(days=365, seed=7)`` - at the
    :data:`SELECTION_RESTARTS` value in effect when this was found - had
    selection score ``k=4`` well above ``k=2``/``k=3``, but the
    full-quality refit's three restarts causally occupied only
    ``{1, 2, 2}`` states - none hit 4 - and the old rule's likelihood-only
    fallback picked the ``1``-state restart over both ``2``-state ones,
    discarding structure the selection stage had already found (see
    ``tests/test_home_mode.py::test_selection_survives_an_unlucky_restart_draw``).
    Whether ``k=4`` itself was the *right* count for that data is a separate
    question from this one - see :func:`_larger_fit_adds_a_novel_state` and
    :data:`SELECTION_RESTARTS`'s own comment, where it turned out not to be -
    but this rule's job is narrower and still exactly right regardless:
    whatever count selection settles on, the model actually kept must not
    silently be a more-collapsed one. This is the right rule for choosing
    *among restarts of one already-chosen
    k* - the actual model that gets kept - because a restart this deployment
    will actually be able to tell apart in practice is preferred over one
    that merely scores higher on an objective (smoothed likelihood) nothing
    here ever gets to use directly (see the module docstring's "The holdout
    discipline").

    ``prefer_max_occupied=False``: plain best (smoothed) log-likelihood,
    full stop - no occupied-count preference at all. This is the right (and
    only correct) rule when the candidates being compared are for
    *different* k, i.e. when the caller is ranking candidate state counts
    against each other (:func:`select_model`'s inner comparison, and its BIC
    fallback): occupied-count-first there does not fix anything, because a
    restart that only occupies one state was never in the running as the
    *best-scoring* candidate for a given k anyway. What it *does* do,
    proven wrong empirically, is systematically reward fragmentation - a
    restart that splits one true regime into several near-duplicate states
    (see the module docstring's "State count" on why a Gaussian-HMM can
    always buy this) now looks preferable to one that does not, at *every*
    k, which pulls the whole ranking toward the top of
    :data:`STATE_CANDIDATES` regardless of what the data actually supports.
    Measured: with occupied-count preference applied here too,
    ``build_two_regime_activity(days=45, seed=1..5)`` - a genuinely
    two-regime home - returned 4 or 5 states on every seed, never 2 (see
    ``tests/test_home_mode.py::test_state_count_stays_small_across_seeds``).
    Ranking must stay on the criterion that exists for exactly this job -
    held-out likelihood, or in-sample BIC on the fallback path - which
    penalises a fragmented fit's extra parameters/lost held-out
    generalisation instead of rewarding its occupied-count.
    """
    if not prefer_max_occupied:
        return max(candidates, key=lambda c: c.log_likelihood)
    occupied = [(_causal_occupied_count(c, x, var_floor), c) for c in candidates]
    max_occupied = max(n for n, _ in occupied)
    pool = [c for n, c in occupied if n == max_occupied]
    return max(pool, key=lambda c: c.log_likelihood)


def _fit_best_of_restarts(
    x: np.ndarray,
    k: int,
    var_floor: float,
    restarts: int,
    max_iter: int,
    warm_start: _EMResult | None = None,
    prefer_max_occupied: bool = True,
) -> _EMResult:
    """The best of ``restarts`` independent, seeded fits, plus (optionally) one
    warm-started from an existing fit - "best" per :func:`_select_best_restart`,
    whose ``prefer_max_occupied`` this simply forwards (default ``True``:
    occupied-count first, likelihood as tiebreak - the right choice when the
    caller has already settled on ``k`` and is only choosing which restart's
    *model* to keep; pass ``False`` when instead comparing this ``k`` against
    a *different* k, where occupied-count-first would reward fragmentation -
    see that function's own docstring for why the two jobs need different
    rules).

    ``warm_start``, when given, is an already-fit :class:`_EMResult` (from a
    *different*, typically smaller, slice of data - see
    :func:`select_model`'s full-quality refit) whose parameters seed one
    extra EM run on ``x`` instead of a random draw. This is what lets
    structure the cheaper selection stage already found survive into the
    full-quality refit even when every independently-seeded restart on the
    full data happens to redraw into something more collapsed - the
    independent restarts are still run and still compete on equal footing
    (a warm start that is actually worse than a random restart on the full
    data does not win just for being a warm start), it is only ever an
    additional candidate in the same pool, never a replacement for genuine
    restarts.

    Every candidate here is fit in one batched call (see :func:`_fit_em_batch`)
    rather than one independent EM run per candidate - restarts of the same
    ``k`` share nothing statistically, only the batching amortises numpy's
    fixed per-call overhead across all of them at once; see that function's
    own docstring for the reasoning and the measured effect.
    """
    inits: list[tuple[np.random.Generator | None, _EMResult | None]] = [
        (np.random.default_rng([_SEED_BASE, k, restart]), None) for restart in range(restarts)
    ]
    if warm_start is not None:
        inits.append((None, warm_start))
    candidates = _fit_em_batch(x, k, inits, var_floor, max_iter)
    return _select_best_restart(candidates, x, var_floor, prefer_max_occupied)


def _fit_candidates(
    x: np.ndarray,
    var_floor: float,
    max_k_exclusive: int,
    restarts: int = RESTARTS,
    max_iter: int = MAX_EM_ITERS,
    prefer_max_occupied: bool = True,
) -> dict[int, _EMResult]:
    """Best-of-``restarts`` fit for every candidate ``k`` that fits in ``x``.

    ``prefer_max_occupied`` is forwarded to :func:`_fit_best_of_restarts` for
    every candidate - pass ``False`` when the result is going to be used to
    *rank* these candidate ``k`` values against each other (see
    :func:`_select_best_restart`'s own docstring); the default ``True`` is
    right only when each candidate's fit here is the one that will actually
    be kept, with no separate anti-collapse refit downstream.
    """
    fits: dict[int, _EMResult] = {}
    n_bins = len(x)
    for k in STATE_CANDIDATES:
        if k >= max_k_exclusive:
            # Cannot have more states than data points to assign them to;
            # skip rather than fit something degenerate.
            continue
        if n_bins < k * MIN_BINS_PER_CANDIDATE_STATE:
            # Not enough bins even in principle to support k states - see
            # MIN_BINS_PER_CANDIDATE_STATE's own comment.
            continue
        fits[k] = _fit_best_of_restarts(
            x, k, var_floor, restarts, max_iter, prefer_max_occupied=prefer_max_occupied
        )
    return fits


def select_model(
    x: np.ndarray, var_floor: float | None = None
) -> tuple[int, dict[int, dict[str, Any]], str]:
    """Choose how many hidden states the data support, and fit that model.

    Returns ``(chosen_k, per_k, method)``. ``per_k[k]`` always carries
    ``"fit"`` (an :class:`_EMResult` fit on the *full* input ``x``) and
    ``"score"`` (higher-is-better); ``method`` is ``"held_out_likelihood"``
    when there was enough data to score candidates on a slice they were not
    fit from, or ``"bic"`` when there was not (see
    :data:`MIN_BINS_FOR_HELD_OUT_SELECTION`) - callers should say which was
    used, since the two are not the same claim ("generalises to bins it did
    not see" vs. "the best in-sample fit for its complexity").

    Held-out likelihood is preferred whenever there is room for it. An
    in-sample criterion like BIC assumes the emission family (diagonal
    Gaussian, here) is a reasonable description of the *true* distribution;
    real activity counts are not Gaussian, and a Gaussian mixture can always
    buy a little more in-sample likelihood by adding a component that fits
    the *shape* of a non-Gaussian count distribution rather than a genuine
    behavioural regime - a failure mode this module's own test suite
    reproduces on synthetic data with a deliberately wide, non-Gaussian count
    spread and exactly two true regimes (see
    ``tests/test_home_mode.py::test_state_count_stays_small_for_a_quiet_home``).
    Scoring candidates on a slice of bins they were never fit from does not
    share that blind spot: an extra component that only chases in-sample
    noise shape does not also predict a held-out slice better, so it stops
    winning.

    Cost: the inner comparison across up to four candidates uses
    :data:`SELECTION_RESTARTS`/:data:`SELECTION_MAX_ITERS` (cheap - it only
    has to rank four numbers against each other), and only the *winner* is
    then refit at full :data:`RESTARTS`/:data:`MAX_EM_ITERS` quality on every
    training bin. Doing that full-quality refit for all four candidates
    twice (once to rank them, again for the winner) was measured as the
    single largest cost in this module - see :data:`SELECTION_RESTARTS`'s
    own comment for the reasoning and the multiplier this avoids.
    """
    if var_floor is None:
        var_floor = _variance_floor(x)
    t_len = len(x)
    split = int(t_len * (1.0 - INNER_VALIDATION_FRACTION))
    x_fit, x_val = x[:split], x[split:]

    if len(x_fit) >= max(STATE_CANDIDATES) + 1 and t_len >= MIN_BINS_FOR_HELD_OUT_SELECTION:
        # prefer_max_occupied=False: these fits are only ever used to rank
        # candidate k values against each other on held-out likelihood below
        # - occupied-count-first here would systematically reward a restart
        # that fragments one true regime into several near-duplicate states
        # at *every* k, pulling the whole ranking toward the top of
        # STATE_CANDIDATES regardless of what the data supports (see
        # _select_best_restart's own docstring, and
        # tests/test_home_mode.py::test_state_count_stays_small_across_seeds,
        # which is exactly the regression this guards). The model actually
        # kept for chosen_k is a *separate* refit below, where
        # occupied-count-first is the right rule.
        inner_fits = _fit_candidates(
            x_fit, var_floor, len(x_fit), SELECTION_RESTARTS, SELECTION_MAX_ITERS,
            prefer_max_occupied=False,
        )
        if not inner_fits:
            raise ValueError("not enough bins to fit even the smallest candidate state count")
        per_bin = {k: _held_out_bin_loglik(f, x_val, var_floor) for k, f in inner_fits.items()}
        # Walk candidates from the fewest states up, only ever moving to a
        # larger k when its held-out gain over the current best clears the
        # noise bar in _prefer_more_states - so each step up in complexity
        # has to earn it again, rather than a bare argmax letting one noisy
        # candidate win outright. See _prefer_more_states's own docstring.
        ordered = sorted(inner_fits)
        chosen_k = ordered[0]
        for k in ordered[1:]:
            if _prefer_more_states(
                per_bin[chosen_k], per_bin[k], inner_fits[chosen_k], inner_fits[k], var_floor
            ):
                chosen_k = k
        scores = {k: float(np.mean(v)) if len(v) else -np.inf for k, v in per_bin.items()}
        method = "held_out_likelihood"
        # Only the winner is refit at full quality on every training bin -
        # the ranking above already decided k; there is nothing left for the
        # other candidates' full-quality fits to be used for. Warm-started
        # from the selection-stage winner's own parameters (fit on x_fit) so
        # that the structure which just won the held-out comparison cannot
        # be lost purely because this refit's independent random restarts on
        # the full data happen to redraw into something more collapsed - see
        # _fit_best_of_restarts's own docstring.
        winner = _fit_best_of_restarts(
            x, chosen_k, var_floor, RESTARTS, MAX_EM_ITERS, warm_start=inner_fits[chosen_k]
        )
        per_k = {
            k: {
                "score": scores[k],
                "log_likelihood": winner.log_likelihood if k == chosen_k else f.log_likelihood,
                "fit": winner if k == chosen_k else f,
            }
            for k, f in inner_fits.items()
        }
    else:
        # Same separation as the held-out branch above, and for the same
        # reason: these fits rank candidate k values against each other (via
        # BIC, which is scored from each one's log_likelihood) and must not
        # be biased by occupied-count preference, or a fragmented restart
        # would look like a better (lower-BIC) fit than a genuine one at
        # every k, for the same reason described in
        # _select_best_restart's own docstring.
        final_fits = _fit_candidates(x, var_floor, t_len, prefer_max_occupied=False)
        if not final_fits:
            raise ValueError("not enough bins to fit even the smallest candidate state count")
        per_k = {
            k: {
                "score": -_bic(f, t_len, x.shape[1]),  # higher-is-better, like the other branch
                "log_likelihood": f.log_likelihood,
                "fit": f,
            }
            for k, f in final_fits.items()
        }
        # Walked the same way as the held-out branch above, for the same
        # reason (see _larger_fit_adds_a_novel_state's own docstring): a
        # bare argmax over BIC scores is just as fooled by a dwell-time
        # phase split as a bare argmax over held-out likelihood would be -
        # BIC's in-sample likelihood term rewards the same split the
        # held-out branch does, and its parameter-count penalty has no way
        # to know the "extra" state is a near-duplicate of one already
        # there. Only ever move to a larger k when it both scores better
        # and adds a state that is not just a near-duplicate of one the
        # current best already has.
        ordered = sorted(final_fits)
        chosen_k = ordered[0]
        for k in ordered[1:]:
            if per_k[k]["score"] > per_k[chosen_k]["score"] and _larger_fit_adds_a_novel_state(
                final_fits[chosen_k], final_fits[k], var_floor
            ):
                chosen_k = k
        method = "bic"
        # As with the held-out branch: the fit actually *kept* for chosen_k
        # gets its own refit with occupied-count-first restart selection
        # (the default), warm-started from the ranking-only fit above - the
        # anti-collapse guarantee this module exists to provide applies to
        # the model that gets used, not to the fits that only ever existed
        # to compare BIC scores against each other.
        winner = _fit_best_of_restarts(
            x, chosen_k, var_floor, RESTARTS, MAX_EM_ITERS, warm_start=final_fits[chosen_k]
        )
        per_k[chosen_k]["fit"] = winner
        per_k[chosen_k]["log_likelihood"] = winner.log_likelihood
    return chosen_k, per_k, method


def _filter_states(
    x: np.ndarray, pi: np.ndarray, a: np.ndarray,
    means: np.ndarray, variances: np.ndarray, var_floor: float = VAR_FLOOR,
) -> np.ndarray:
    """Causal (forward-only, no smoothing) filtered state posterior, per bin.

    Bin ``t``'s row depends only on ``x[0..t]`` - never on activity that
    happens later.  This is what makes it safe to run across a holdout: a
    label for a holdout bin is exactly what a live system computing this in
    real time would have known at that moment, not something informed by
    activity still in the future relative to it.  See the module docstring's
    "The holdout discipline".
    """
    t_len = len(x)
    k = len(pi)
    log_pi = np.log(np.clip(pi, 1e-300, None)).tolist()
    log_a = np.log(np.clip(a, 1e-300, None)).tolist()
    log_b = _log_gaussian_pdf(x, means, variances, var_floor).tolist()
    # Pure Python per bin - see _logsumexp_row's own docstring for why.
    log_alpha: list[list[float]] = [[0.0] * k for _ in range(t_len)]
    row0 = [log_pi[j] + log_b[0][j] for j in range(k)]
    norm0 = _logsumexp_row(row0)
    log_alpha[0] = [v - norm0 for v in row0]
    for t in range(1, t_len):
        prev = log_alpha[t - 1]
        row_b = log_b[t]
        row = [
            _logsumexp_row([prev[i] + log_a[i][j] for i in range(k)]) + row_b[j]
            for j in range(k)
        ]
        norm = _logsumexp_row(row)
        log_alpha[t] = [v - norm for v in row]
    return np.array(log_alpha)


def decode_labels(
    x: np.ndarray, pi: np.ndarray, a: np.ndarray,
    means: np.ndarray, variances: np.ndarray, var_floor: float = VAR_FLOOR,
) -> np.ndarray:
    """Most likely state per bin under causal filtering. See :func:`_filter_states`."""
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    log_alpha = _filter_states(x, pi, a, means, variances, var_floor)
    return np.argmax(log_alpha, axis=1)


# ----------------------------------------------------------------------
# The fitted model
# ----------------------------------------------------------------------
@dataclass
class HomeModeModel:
    """A fitted (or explicitly-not-fitted) home-mode model."""

    fitted: bool
    n_states: int = 0
    feature_names: tuple[str, ...] = FEATURE_NAMES
    bin_seconds: float = BIN_SECONDS
    initial: np.ndarray = field(default_factory=lambda: np.zeros(0))
    transition: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    means: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    variances: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: The emission variance floor this fit actually used (see
    #: :data:`VAR_FLOOR_FRACTION`) - persisted and reused verbatim at decode
    #: time, so replaying the model later applies exactly the regularisation
    #: it was fit under rather than a different, module-default value.
    var_floor: float = VAR_FLOOR
    #: The first bin index the model was fit from (after MAX_TRAIN_BINS
    #: truncation) - decoding starts here, never earlier, because there are no
    #: fitted parameters to explain anything before it.
    first_bin_index: int = 0
    train_start_ts: float = 0.0
    train_end_ts: float = 0.0
    train_bins: int = 0
    log_likelihood: float = 0.0
    #: state count -> selection score (higher is better), for every candidate
    #: that was tried, not just the winner - see :func:`select_model`.
    score_by_states: dict[int, float] = field(default_factory=dict)
    #: ``"held_out_likelihood"`` or ``"bic"`` - which of this project's two
    #: sanctioned criteria actually picked :attr:`n_states`. See
    #: :func:`select_model`'s docstring for why the choice is not arbitrary.
    selection_method: str = ""
    #: Per-state, human-readable-but-honest description. See
    #: :meth:`describe_states`.
    state_summaries: list[dict[str, Any]] = field(default_factory=list)
    #: True when the fitted states are not well separated relative to their
    #: own spread - see the module docstring's "Honesty" section.
    weakly_separated: bool = False
    trained_ts: float = field(default_factory=time.time)
    #: Set instead of fitting when there was not enough training history.
    #: Never left implicit - every caller that skips conditioning on the mode
    #: signal because ``fitted`` is False can read exactly why.
    fallback_reason: str | None = None
    #: Set when :attr:`n_states` is *smaller* than the state count
    #: :func:`select_model` actually chose on held-out (or BIC) evidence -
    #: i.e. the full-quality refit's restarts, even warm-started from the
    #: selection winner, never causally distinguished that many states, so
    #: :func:`_drop_unoccupied_states` pruned it down. Never left implicit,
    #: for the same reason :attr:`fallback_reason` is not: a reader seeing
    #: ``n_states=1`` after selection found real evidence for more should be
    #: able to tell "this is what the data supports once actually deployed"
    #: apart from "the household plainly has only one regime" without
    #: reading this module's source.
    selection_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HOME_MODE_SCHEMA_VERSION,
            "fitted": self.fitted,
            "n_states": self.n_states,
            "feature_names": list(self.feature_names),
            "bin_seconds": self.bin_seconds,
            "initial": self.initial.tolist(),
            "transition": self.transition.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "var_floor": self.var_floor,
            "first_bin_index": self.first_bin_index,
            "train_start_ts": self.train_start_ts,
            "train_end_ts": self.train_end_ts,
            "train_bins": self.train_bins,
            "log_likelihood": self.log_likelihood,
            "score_by_states": {str(k): v for k, v in self.score_by_states.items()},
            "selection_method": self.selection_method,
            "state_summaries": self.state_summaries,
            "weakly_separated": self.weakly_separated,
            "trained_ts": self.trained_ts,
            "fallback_reason": self.fallback_reason,
            "selection_note": self.selection_note,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> HomeModeModel | None:
        """Rebuild from a persisted row, discarding it if its schema is stale.

        Same discipline as ``amminer.learn.ranking.RankingModel.from_row``: a
        model fit under an older feature vector or schema version is not
        stale data to patch around, it is meaningless.
        """
        if not row:
            return None
        if int(row.get("schema_version", -1)) != HOME_MODE_SCHEMA_VERSION:
            return None
        if not row.get("fitted"):
            return cls(fitted=False, fallback_reason=row.get("fallback_reason"))
        feature_names = tuple(row.get("feature_names") or ())
        if feature_names != FEATURE_NAMES:
            return None
        try:
            return cls(
                fitted=True,
                n_states=int(row["n_states"]),
                feature_names=feature_names,
                bin_seconds=float(row["bin_seconds"]),
                initial=np.array(row["initial"], dtype=np.float64),
                transition=np.array(row["transition"], dtype=np.float64),
                means=np.array(row["means"], dtype=np.float64),
                variances=np.array(row["variances"], dtype=np.float64),
                var_floor=float(row.get("var_floor", VAR_FLOOR)),
                first_bin_index=int(row["first_bin_index"]),
                train_start_ts=float(row["train_start_ts"]),
                train_end_ts=float(row["train_end_ts"]),
                train_bins=int(row["train_bins"]),
                log_likelihood=float(row["log_likelihood"]),
                score_by_states={
                    int(k): float(v) for k, v in (row.get("score_by_states") or {}).items()
                },
                selection_method=str(row.get("selection_method", "")),
                state_summaries=list(row.get("state_summaries") or []),
                weakly_separated=bool(row.get("weakly_separated", False)),
                trained_ts=float(row.get("trained_ts", 0.0)),
                fallback_reason=row.get("fallback_reason"),
                selection_note=row.get("selection_note"),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _mode_label(index: int) -> str:
    """The state's stable, honest value - never a name implying certainty."""
    return f"mode_{index}"


def _top_areas(
    changes: Sequence[StateChange],
    options: Options,
    resolver: Any,
    labels: np.ndarray,
    first_bin_index: int,
    bin_seconds: float,
    n_states: int,
    top_n: int = 3,
) -> list[list[str]]:
    """The areas most active in each state, for description only (no resolver -> empty)."""
    if resolver is None:
        return [[] for _ in range(n_states)]
    counts: list[dict[str, int]] = [{} for _ in range(n_states)]
    for change in changes:
        if change.cause is not Cause.HUMAN or change.domain not in _DOMAIN_BUCKET:
            continue
        if not change.is_transition or options.is_excluded(change.entity_id):
            continue
        idx = int(change.ts // bin_seconds) - first_bin_index
        if idx < 0 or idx >= len(labels):
            continue
        area = resolver.resolve(change.entity_id).area_name
        if not area:
            continue
        state = int(labels[idx])
        counts[state][area] = counts[state].get(area, 0) + 1
    return [
        [area for area, _n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
        for c in counts
    ]


def _describe_states(
    model_k: int,
    em: _EMResult,
    x_train: np.ndarray,
    first_bin_index: int,
    bin_seconds: float,
    train_changes: Sequence[StateChange],
    options: Options,
    resolver: Any,
    var_floor: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Build honest per-state summaries, and flag weak separation.

    Every number here comes from what was actually decoded during training,
    never from the model's parameters read in isolation - "typical hours" is
    the circular mean/spread of the bins the state actually won, not a
    property the Gaussian happens to have.
    """
    labels = decode_labels(x_train, em.initial, em.transition, em.means, em.variances, var_floor)
    tz = local_tz()
    bin_start_ts = (first_bin_index + np.arange(len(x_train))) * bin_seconds
    areas_by_state = _top_areas(
        train_changes, options, resolver, labels, first_bin_index, bin_seconds, model_k
    )

    summaries: list[dict[str, Any]] = []
    for state in range(model_k):
        mask = labels == state
        occupancy = float(mask.sum()) / max(len(labels), 1)
        # Local minute-of-day (not the UTC bin arithmetic above) for a
        # centre/spread that means anything to a person reading it.
        local_minutes = [
            dt.datetime.fromtimestamp(ts, tz).hour * 60 + dt.datetime.fromtimestamp(ts, tz).minute
            for ts in bin_start_ts[mask]
        ]
        if local_minutes:
            centre = circular_mean_minutes(local_minutes)
            spread = circular_std_minutes(local_minutes)
        else:
            centre, spread = 0.0, 0.0
        # Raw (non-log) mean counts, for a description a person can read
        # ("mostly light and media_player activity") rather than log-space
        # numbers nobody can eyeball.
        raw_means = np.expm1(em.means[state]) if mask.sum() else np.zeros(len(FEATURE_NAMES))
        top_domains = [
            FEATURE_NAMES[i]
            for i in np.argsort(raw_means)[::-1]
            if raw_means[i] > 0.05
        ][:3]
        summaries.append(
            {
                "state": state,
                "label": _mode_label(state),
                "occupancy_share": round(occupancy, 4),
                "typical_time": hhmm(centre) if local_minutes else None,
                "typical_spread_minutes": round(spread, 1) if local_minutes else None,
                "top_domains": top_domains,
                "top_areas": areas_by_state[state],
                # Advisory-only, filled in later by amminer.llm.home_mode_labels
                # when configured - see that module and the class docstring's
                # "Honesty about what the modes are".
                "llm_label": None,
                "llm_label_is_advisory": True,
            }
        )

    # Weak separation: compare each pair of state means against the spread of
    # both (a scale-aware distance, in raw feature-variance units) - if the
    # closest pair of states is not clearly apart relative to their own
    # variance, they are not two states an observer could actually tell
    # apart, whatever BIC's arithmetic preferred.
    weakly_separated = False
    if model_k >= 2:
        min_separation = np.inf
        for i in range(model_k):
            for j in range(i + 1, model_k):
                distance = _pooled_distance(
                    em.means[i], em.variances[i], em.means[j], em.variances[j], var_floor
                )
                min_separation = min(min_separation, distance)
        weakly_separated = bool(min_separation < SEPARATION_MIN_DISTANCE)
    return summaries, weakly_separated


def _drop_unoccupied_states(em: _EMResult, x: np.ndarray, var_floor: float) -> _EMResult:
    """Renumber away any state the decoded training data never actually uses.

    EM's state-count *selection* (:func:`select_model`) picks ``k`` from an
    inner slice of the training data; the *final* refit on all of it, run
    fresh with its own restarts, is not guaranteed to land on a solution that
    uses every one of those ``k`` components - a component with no support
    left in the data it is refit on can decay towards being empty rather
    than merging into one that already explains everything nearby. A model
    that reports ``n_states=5`` while three of them are never once decoded is
    not more honest than reporting 2 outright, it is less: the empty ones
    would list a description built from zero bins, and every reader would
    reasonably read ``n_states`` as "how many patterns", not "how many
    components a fitting procedure happened to allocate". This is the
    generalisation of "did EM silently converge to one state" the project's
    own pre-flight review asked to guard against - reported state count and
    the number of states actually distinguishable in the data must never be
    allowed to quietly disagree in the *other* direction, count too high.

    Rows/columns for dropped states are removed from the transition matrix
    and every remaining row renormalised; log-likelihood is recomputed under
    the pruned parameters so it always matches what is actually returned.
    """
    if len(em.initial) <= 1:
        return em
    labels = decode_labels(x, em.initial, em.transition, em.means, em.variances, var_floor)
    occupied = sorted({int(label) for label in labels})
    if len(occupied) == len(em.initial):
        return em
    if not occupied:
        # Nothing decoded at all (e.g. x is empty) - nothing to prune to;
        # leave the fit as-is rather than returning a zero-state model.
        return em

    initial = em.initial[occupied]
    total = initial.sum()
    initial = initial / total if total > 0 else np.full(len(occupied), 1.0 / len(occupied))
    transition = em.transition[np.ix_(occupied, occupied)]
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = np.where(
        row_sums > 0, transition / np.maximum(row_sums, 1e-300), 1.0 / len(occupied)
    )
    means = em.means[occupied]
    variances = em.variances[occupied]

    log_pi = np.log(np.clip(initial, 1e-300, None))
    log_a = np.log(np.clip(transition, 1e-300, None))
    log_b = _log_gaussian_pdf(x, means, variances, var_floor)
    _, _, log_likelihood = _forward_backward(log_b, log_pi, log_a)
    return _EMResult(log_likelihood, initial, transition, means, variances, em.n_iter)


# ----------------------------------------------------------------------
# Mode identity across runs
# ----------------------------------------------------------------------
def _align_state_order(
    previous: HomeModeModel | None, means: np.ndarray, variances: np.ndarray
) -> tuple[int, ...] | None:
    """The permutation of this fit's own state indices that best matches
    ``previous``'s, by scale-aware emission distance - or ``None`` when there
    is nothing to align to.

    EM's states are unlabelled: nothing about the fit itself prefers calling
    the sofa-and-TV cluster "state 0" over "state 1"; :func:`_fit_best_of_restarts`
    and :func:`_drop_unoccupied_states` never had any reason to settle that
    consistently across nights, because until now nothing downstream cared.
    It matters once a user (or a mined candidate that conditions on
    ``mode_2``) starts attaching meaning to a specific label across runs -
    refit nightly from scratch, tonight's "state 0" is otherwise as likely to
    be last night's "state 1" as its "state 0", purely from which restart's
    arbitrary internal ordering happened to win. Picking whichever of the
    ``k!`` equally-valid relabellings lines up best with the previous run's
    own states costs nothing (:func:`fit`'s statistical result - which
    activity got grouped with which - is completely unchanged, only which
    integer names each group) and removes that churn.

    Returns ``None`` - leaving this fit's own (arbitrary) order alone -
    whenever there is no previous *fitted* model to align to, its feature
    vocabulary does not match this build's (a stale schema :meth:`HomeModeModel.from_row`
    would already have refused), or its state count differs from this fit's:
    with a different number of states there is no bijection between the two
    sets that could preserve every identity, and guessing a partial one (drop
    an old mode here, invent a new label there) is exactly the kind of
    unforced, hard-to-explain choice this module's own "Honesty" section
    argues against - a state count change is real information (the household
    changed, or the fit found a level of structure it could not detect
    before) and is reported as such by :attr:`HomeModeModel.n_states` itself
    changing, not smoothed over by relabelling.
    """
    if (
        previous is None
        or not previous.fitted
        or previous.feature_names != FEATURE_NAMES
        or len(previous.means) != len(means)
    ):
        return None
    k = len(means)
    best_perm = tuple(range(k))
    best_cost = math.inf
    for perm in permutations(range(k)):
        cost = sum(
            _pooled_distance(previous.means[i], previous.variances[i], means[perm[i]], variances[perm[i]])
            for i in range(k)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return best_perm


def _reorder_em_result(em: _EMResult, perm: Sequence[int]) -> _EMResult:
    """Relabel ``em``'s states according to ``perm`` (state ``i`` becomes
    whatever was at ``perm[i]``) - a pure relabelling, so ``log_likelihood``
    (a property of the model, not of which integer names which state) is
    carried over unchanged rather than recomputed.
    """
    idx = np.array(perm)
    return _EMResult(
        em.log_likelihood,
        em.initial[idx],
        em.transition[np.ix_(idx, idx)],
        em.means[idx],
        em.variances[idx],
        em.n_iter,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fit(
    train_changes: Sequence[StateChange],
    signals: SignalSet,
    options: Options,
    train_window: tuple[float, float],
    resolver: Any = None,
    previous: HomeModeModel | None = None,
) -> HomeModeModel:
    """Fit a home-mode model on **training-window data only**.

    This is the one function in this module that may ever see raw
    ``StateChange`` rows to fit from, and it is the caller's responsibility -
    ``amminer.pipeline`` - to pass only ``train_changes``/``train_window``,
    never the full analysis window.  Nothing here reaches outside the window
    it is given: :func:`build_feature_matrix` filters every row to
    ``[start_ts, end_ts)`` itself, so even a caller that (by mistake) handed
    it the full ``changes`` list alongside the train window's bounds would
    still only have training-window activity counted here - the guarantee is
    "this function never learns from anything outside the window it names",
    not merely "callers are expected to be careful".

    ``previous``, when given, is the *previously persisted* model
    (``amminer.pipeline`` reads it back via ``Store.get_home_mode_model``) -
    used only to relabel this run's own states to match its state numbering
    where the two line up (see :func:`_align_state_order`), never to seed or
    otherwise influence the fit itself. It has no bearing on the holdout
    discipline above: a fixed relabelling chosen after the fact cannot leak
    holdout information into training, because it is not a function of the
    holdout at all, only of two already-fitted models' parameters.
    """
    start_ts, end_ts = train_window
    min_train_days = options.home_mode_min_train_days
    min_train_bins = int(min_train_days * 86400.0 / BIN_SECONDS)
    train_days = (end_ts - start_ts) / 86400.0
    if train_days < min_train_days:
        return HomeModeModel(
            fitted=False,
            fallback_reason=(
                f"only {train_days:.1f} days of training history (< {min_train_days}); "
                "too little to infer a household mode yet"
            ),
        )

    counts, first_idx = build_feature_matrix(train_changes, signals, options, start_ts, end_ts)
    n_bins = len(counts)
    if n_bins > MAX_TRAIN_BINS:
        # Keep the most recent MAX_TRAIN_BINS bins - see the module
        # docstring's "Bounding the work". first_idx moves forward to match.
        drop = n_bins - MAX_TRAIN_BINS
        counts = counts[drop:]
        first_idx += drop
        n_bins = MAX_TRAIN_BINS
    if n_bins < min_train_bins:
        return HomeModeModel(
            fitted=False,
            fallback_reason=(
                f"only {n_bins} activity bins available (< {min_train_bins} needed); "
                "too little to infer a household mode yet"
            ),
        )

    x = _to_features(counts)
    var_floor = _variance_floor(x)
    try:
        chosen_k, per_k, method = select_model(x, var_floor)
    except ValueError as err:
        return HomeModeModel(fitted=False, fallback_reason=str(err))

    selected_k = chosen_k
    best = _drop_unoccupied_states(per_k[chosen_k]["fit"], x, var_floor)
    chosen_k = len(best.initial)
    selection_note = None
    if chosen_k < selected_k:
        # select_model's own evidence (held-out likelihood, or BIC when there
        # was not enough history for a held-out slice) preferred selected_k
        # states, but the full-quality refit's restarts - including one
        # warm-started from that selection winner's own parameters, see
        # _fit_best_of_restarts - never causally distinguished more than
        # chosen_k of them once decoded the way this model is ever actually
        # deployed. That is itself evidence selected_k was not achievable
        # from this data, not a fit gone wrong; see this module's docstring,
        # "State count".
        selection_note = (
            f"{method.replace('_', ' ')} selection preferred {selected_k} states, but "
            f"the full-quality fit never causally distinguished more than {chosen_k} of "
            f"them; reporting the {chosen_k} the fit actually uses rather than a state "
            "count it does not."
        )
    perm = _align_state_order(previous, best.means, best.variances)
    if perm is not None:
        best = _reorder_em_result(best, perm)
    summaries, weakly_separated = _describe_states(
        chosen_k, best, x, first_idx, BIN_SECONDS, train_changes, options, resolver, var_floor
    )
    actual_train_start = first_idx * BIN_SECONDS
    return HomeModeModel(
        fitted=True,
        n_states=chosen_k,
        initial=best.initial,
        transition=best.transition,
        means=best.means,
        variances=best.variances,
        var_floor=var_floor,
        first_bin_index=first_idx,
        train_start_ts=actual_train_start,
        train_end_ts=end_ts,
        train_bins=n_bins,
        log_likelihood=best.log_likelihood,
        score_by_states={k: v["score"] for k, v in per_k.items()},
        selection_method=method,
        state_summaries=summaries,
        weakly_separated=weakly_separated,
        selection_note=selection_note,
    )


def decode_series(
    model: HomeModeModel,
    changes: Sequence[StateChange],
    signals: SignalSet,
    options: Options,
    window: tuple[float, float],
    entity_id: str = HOME_MODE_ENTITY_ID,
) -> SignalSeries | None:
    """Replay a fitted model, causally, over ``window`` as a :class:`SignalSeries`.

    ``window`` should cover at least the model's own training span through
    however far it needs to be usable (train window through the holdout end,
    when called for backtesting).  Decoding always starts at the model's own
    ``first_bin_index``, never earlier - there is no fitted state to explain
    activity before the data it was trained on.  Each bin's label is stamped
    at the bin's *close* - see the module docstring's "Bin width"/"holdout
    discipline" sections for why.
    """
    if not model.fitted:
        return None
    decode_start = model.first_bin_index * BIN_SECONDS
    end_ts = max(window[1], model.train_end_ts)
    counts, first_idx = build_feature_matrix(changes, signals, options, decode_start, end_ts)
    if first_idx != model.first_bin_index or len(counts) == 0:
        return None
    x = _to_features(counts)
    labels = decode_labels(
        x, model.initial, model.transition, model.means, model.variances, model.var_floor
    )
    series = SignalSeries(entity_id=entity_id, numeric=False, source="home_mode")
    bin_close = (first_idx + np.arange(len(labels)) + 1) * BIN_SECONDS
    # Every window in this codebase is half-open [start, end) - see
    # amminer.backtest.split_window and the invariant tests built on it - so
    # a bin whose close lands exactly on (or past) window[1] belongs to
    # whatever comes next, not to this window.  Left in, it would plant a
    # signal timestamp at-or-after a holdout boundary even though the label
    # itself was decoded causally; dropping it costs at most one BIN_SECONDS
    # bin of coverage right at the edge.
    for ts, label in zip(bin_close, labels, strict=True):
        if ts >= window[1]:
            continue
        series.add(float(ts), _mode_label(int(label)))
    return series.finalise()
