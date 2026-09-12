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

**Observation representation - per-bin counts of human activity, bucketed by a
small, fixed domain vocabulary, with diagonal-covariance Gaussian emissions on
their log1p.**  A multinomial/count emission model is a reasonable
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
complexity, and a quiet home legitimately comes back with 2. It is not,
however, a hard floor: the state count actually reported is the number of
states the *final* fit still uses after refitting on every training bin (a
component with no support left in that refit can end up empty rather than
merging into a neighbour - see :func:`_drop_unoccupied_states`), so a home
whose activity does not distinguish into separate regimes at all can
honestly come back as 1, described plainly rather than split into two
groups neither of which means anything (see "Honesty" below).

**Restarts and determinism - fixed, seeded, best-of-N.**  EM only finds a local
optimum, so each candidate ``k`` is fit :data:`RESTARTS` (or, for ranking
candidates against each other, the cheaper :data:`SELECTION_RESTARTS`) times
from independent initialisations and the best kept (see
:func:`_fit_best_of_restarts`) - "best" by log-likelihood among whichever
restarts actually distinguish all ``k`` states once decoded the same
causal way this module is ever deployed, since EM's own smoothed training
objective can otherwise prefer a restart that looks better on paper and
collapses in practice. Every restart's random state comes from
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
oversight papered over: see the module's tests for the honest refusal, and the
project notes for why building a continuously-updating publisher was left out
of this change.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
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

#: Below this many days of training history there have not been enough
#: distinct days for a regime split to mean anything - mirrors
#: ``amminer.pipeline.MIN_DAYS_FOR_SEQUENCE_MINING``'s own judgement call for
#: a different kind of pattern that also needs several full days to trust.
MIN_TRAIN_DAYS = 7
MIN_TRAIN_BINS = int(MIN_TRAIN_DAYS * 86400.0 / BIN_SECONDS)

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
SELECTION_RESTARTS = 2
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

    current: dict[str, bool] = {}
    bucket_counts = dict.fromkeys(FEATURE_NAMES, 0)

    def apply(entity_id: str, active: bool) -> None:
        if current.get(entity_id, False) == active:
            return
        current[entity_id] = active
        delta = 1 if active else -1
        if entity_id in presence_entities:
            bucket_counts["presence"] += delta
        bucket = _DOMAIN_BUCKET.get(entity_id.split(".", 1)[0])
        if bucket:
            bucket_counts[bucket] += delta

    pos = 0
    n = len(relevant)
    for bin_idx in range(n_bins):
        bin_close = (first_idx + bin_idx + 1) * bin_seconds
        while pos < n and relevant[pos].ts < bin_close:
            change = relevant[pos]
            apply(change.entity_id, _is_active_state(change.state))
            pos += 1
        for name in FEATURE_NAMES:
            counts[bin_idx, bucket_index[name]] = bucket_counts[name]
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


def _fit_em(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    var_floor: float = VAR_FLOOR,
    max_iter: int = MAX_EM_ITERS,
) -> _EMResult:
    """One deterministic (given ``rng``) EM run to convergence or ``max_iter``."""
    t_len = x.shape[0]
    pi = rng.dirichlet(np.ones(k))
    a = rng.dirichlet(np.ones(k), size=k)
    means = x[rng.choice(t_len, size=k, replace=False)].copy()
    variances = np.tile(np.maximum(x.var(axis=0), var_floor), (k, 1))

    prev_ll = -np.inf
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        log_pi = np.log(np.clip(pi, 1e-300, None))
        log_a = np.log(np.clip(a, 1e-300, None))
        log_b = _log_gaussian_pdf(x, means, variances, var_floor)
        log_alpha, log_beta, ll = _forward_backward(log_b, log_pi, log_a)

        # E-step: posterior state occupancy (gamma) and pairwise transition
        # posteriors (xi), each row-normalised independently in log-space
        # rather than by subtracting the single sequence-wide log-likelihood -
        # equivalent in exact arithmetic, but does not compound floating-point
        # error across a long sequence the way one global subtraction would.
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        log_xi = (
            log_alpha[:-1][:, :, None]
            + log_a[None, :, :]
            + (log_b[1:] + log_beta[1:])[:, None, :]
        )
        flat = log_xi.reshape(t_len - 1, -1)
        flat = flat - _logsumexp(flat, axis=1)[:, None]
        xi = np.exp(flat).reshape(t_len - 1, k, k)

        # M-step
        pi = gamma[0] / gamma[0].sum()
        denom = np.maximum(gamma[:-1].sum(axis=0), 1e-300)
        a = xi.sum(axis=0) / denom[:, None]
        a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-300)
        weight = np.maximum(gamma.sum(axis=0), 1e-300)
        means = (gamma.T @ x) / weight[:, None]
        diff2 = (x[:, None, :] - means[None, :, :]) ** 2
        variances = np.einsum("tk,tkd->kd", gamma, diff2) / weight[:, None]
        variances = np.maximum(variances, var_floor)

        if abs(ll - prev_ll) < EM_TOL * max(1.0, abs(prev_ll)):
            prev_ll = ll
            break
        prev_ll = ll

    # Recompute the final log-likelihood under the parameters actually kept
    # (the loop's `ll` was computed *before* the last M-step that produced
    # them) so `log_likelihood` always matches `(initial, transition, means,
    # variances)` on this object, which is what BIC below is scored from.
    log_pi = np.log(np.clip(pi, 1e-300, None))
    log_a = np.log(np.clip(a, 1e-300, None))
    log_b = _log_gaussian_pdf(x, means, variances, var_floor)
    _, _, final_ll = _forward_backward(log_b, log_pi, log_a)
    return _EMResult(final_ll, pi, a, means, variances, n_iter)


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
SELECTION_Z_SCORE = 2.0


def _prefer_more_states(
    current_per_bin: np.ndarray, candidate_per_bin: np.ndarray
) -> bool:
    """Would the larger model's per-bin gain survive being sampling noise?

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
    """
    diffs = candidate_per_bin - current_per_bin
    n = len(diffs)
    if n == 0:
        return False
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se_diff > 0:
        return mean_diff > SELECTION_Z_SCORE * se_diff
    return mean_diff > 0


def _causal_occupied_count(em: _EMResult, x: np.ndarray, var_floor: float) -> int:
    labels = decode_labels(x, em.initial, em.transition, em.means, em.variances, var_floor)
    return len({int(label) for label in labels})


def _fit_best_of_restarts(
    x: np.ndarray, k: int, var_floor: float, restarts: int, max_iter: int
) -> _EMResult:
    """The best of ``restarts`` independent, seeded fits - preferring one that
    actually uses all ``k`` states under *causal* decoding.

    EM's own fitting objective is the smoothed (forward-backward) likelihood,
    which can legitimately prefer a restart that, decoded the way this
    module is actually ever deployed (:func:`decode_labels`, forward-only,
    see the module docstring's "The holdout discipline"), collapses onto a
    single dominant state even though its *smoothed* training likelihood is
    higher than a restart that does distinguish all ``k`` states causally - a
    sufficiently "sticky" fitted transition matrix can make one state win
    every causal decision even when the fit meaningfully uses a second one
    with the benefit of hindsight. Since production never has that
    hindsight, a restart this model will actually be able to tell apart in
    practice is preferred over one that merely scores higher on an objective
    nothing here ever gets to use directly - falling back to plain
    best-likelihood only when *every* restart collapses, since that is then
    the honest answer about what this data supports once actually deployed
    (see :func:`_drop_unoccupied_states`, which prunes whatever is left).
    """
    candidates = [
        _fit_em(x, k, np.random.default_rng([_SEED_BASE, k, restart]), var_floor, max_iter)
        for restart in range(restarts)
    ]
    fully_occupied = [c for c in candidates if _causal_occupied_count(c, x, var_floor) == k]
    pool = fully_occupied or candidates
    best = max(pool, key=lambda c: c.log_likelihood)
    return best


def _fit_candidates(
    x: np.ndarray,
    var_floor: float,
    max_k_exclusive: int,
    restarts: int = RESTARTS,
    max_iter: int = MAX_EM_ITERS,
) -> dict[int, _EMResult]:
    """Best-of-``restarts`` fit for every candidate ``k`` that fits in ``x``."""
    fits: dict[int, _EMResult] = {}
    for k in STATE_CANDIDATES:
        if k >= max_k_exclusive:
            # Cannot have more states than data points to assign them to;
            # skip rather than fit something degenerate.
            continue
        fits[k] = _fit_best_of_restarts(x, k, var_floor, restarts, max_iter)
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
        inner_fits = _fit_candidates(
            x_fit, var_floor, len(x_fit), SELECTION_RESTARTS, SELECTION_MAX_ITERS
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
            if _prefer_more_states(per_bin[chosen_k], per_bin[k]):
                chosen_k = k
        scores = {k: float(np.mean(v)) if len(v) else -np.inf for k, v in per_bin.items()}
        method = "held_out_likelihood"
        # Only the winner is refit at full quality on every training bin -
        # the ranking above already decided k; there is nothing left for the
        # other candidates' full-quality fits to be used for.
        winner = _fit_best_of_restarts(x, chosen_k, var_floor, RESTARTS, MAX_EM_ITERS)
        per_k = {
            k: {
                "score": scores[k],
                "log_likelihood": winner.log_likelihood if k == chosen_k else f.log_likelihood,
                "fit": winner if k == chosen_k else f,
            }
            for k, f in inner_fits.items()
        }
    else:
        final_fits = _fit_candidates(x, var_floor, t_len)
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
        chosen_k = max(per_k, key=lambda k: (per_k[k]["score"], -k))
        method = "bic"
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
                pooled_std = np.sqrt((em.variances[i] + em.variances[j]) / 2.0)
                pooled_std = np.maximum(pooled_std, np.sqrt(var_floor))
                distance = np.linalg.norm((em.means[i] - em.means[j]) / pooled_std)
                min_separation = min(min_separation, distance)
        weakly_separated = bool(min_separation < 1.0)
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
# Public API
# ----------------------------------------------------------------------
def fit(
    train_changes: Sequence[StateChange],
    signals: SignalSet,
    options: Options,
    train_window: tuple[float, float],
    resolver: Any = None,
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
    """
    start_ts, end_ts = train_window
    train_days = (end_ts - start_ts) / 86400.0
    if train_days < MIN_TRAIN_DAYS:
        return HomeModeModel(
            fitted=False,
            fallback_reason=(
                f"only {train_days:.1f} days of training history (< {MIN_TRAIN_DAYS}); "
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
    if n_bins < MIN_TRAIN_BINS:
        return HomeModeModel(
            fitted=False,
            fallback_reason=(
                f"only {n_bins} activity bins available (< {MIN_TRAIN_BINS} needed); "
                "too little to infer a household mode yet"
            ),
        )

    x = _to_features(counts)
    var_floor = _variance_floor(x)
    try:
        chosen_k, per_k, method = select_model(x, var_floor)
    except ValueError as err:
        return HomeModeModel(fitted=False, fallback_reason=str(err))

    best = _drop_unoccupied_states(per_k[chosen_k]["fit"], x, var_floor)
    chosen_k = len(best.initial)
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
    for ts, label in zip(bin_close, labels, strict=True):
        series.add(float(ts), _mode_label(int(label)))
    return series.finalise()
