"""Learn what a household does next, directly, instead of asking it many
narrow questions.

Every miner in this project is really a hand-specified projection of one
underlying question: *what does this household do next, given what just
happened?* ``time_of_day.py`` asks it of the clock alone.  ``association.py``
asks it of co-occurring pairs, capped at two items because the search space
of larger itemsets is combinatorial.  ``sequence.py`` (the PrefixSpan miner)
asks it of ordered subsequences, but only ones that repeat verbatim.
``conditional.py`` asks it of one external signal at a time, tested
independently of every other signal.  None of them can see an interaction
the others handle, and none of them can see three things interacting at
once - "the hallway light turns on, in the evening, only when the door
sensor tripped less than a minute earlier" is invisible to all four, each
for a different structural reason.

This module asks the underlying question directly.  A small sequence model
is trained on the household's own tokenised event stream to predict the next
*human* action from recent context, and wherever it predicts one with real
confidence *and* that exact context/outcome pair has real support in
training history, a :class:`~amminer.miners.base.Candidate` is emitted -
exactly like any other miner's - and sent through the same backtest and
conflict machinery, unchanged.  The model is never trusted on its own; see
"Honesty" below.

Architecture, and why
----------------------

**A compact GRU over a short, fixed context window - not a causal-attention
block, and not a full unbounded-length RNN.**  Both a GRU and single-layer
causal attention are reasonable choices for this; the deciding factor here
is what "pure numpy, no autodiff library, hand-derived backward pass, bounded
wall-clock on a Raspberry Pi" makes cheapest to get *right*.  A GRU's
backward pass (backprop-through-time) is textbook and, at the window lengths
this module uses (:data:`WINDOW` = 6), only six sequential steps - each step
is one matmul-sized piece of work across the whole batch, not a per-example
Python loop.  A causal-attention block would have been no worse to train,
but its backward pass (through a masked softmax over query/key products) is
a strictly larger surface for a hand-derived-gradient bug to hide in, for no
accuracy benefit at a context length this short - attention's real
advantage is modelling *long* dependencies cheaply, and this module
deliberately never looks further back than :data:`WINDOW` events.  Every
gradient here is checked against finite differences in
``tests/test_sequence_model.py::test_gradients_match_finite_differences``,
which is the actual argument for correctness, not the architecture choice
by itself.

**Tokenisation - (entity, state, Δt-bucket).**  Each usable state change
becomes one token: which entity, what state, and a small, bounded bucket for
how long it had been since the *previous* usable event (never how long until
the action being predicted - that would leak the future into the token
itself).  Item identity and timing are embedded separately
(:data:`EMBED_ITEM_DIM`, :data:`EMBED_DT_DIM`) and concatenated, so the model
can learn "the same item, but the timing matters" without an explosion in
vocabulary size the way a joint (item, Δt) vocabulary would need.

**Bounded everything.**  Context length, vocabulary, embedding/hidden size,
training examples, gradient steps and wall-clock are all capped by module
constants below. Measured on ordinary x86_64 desktop/server hardware (see
``tests/test_sequence_model.py`` and this project's PR description), a
realistic synthetic year of household history (tens of thousands of state
changes) trains in a few seconds. Profiling that benchmark found the
dominant cost at this scale is numpy's own fixed per-call overhead across
the many small elementwise operations a recurrent update needs - not raw
FLOPs, and not (as tried and measured out - see ``_forward``'s own comment)
batching the non-recurrent matmuls - so wall-clock here is expected to
track single-core clock speed and interpreter overhead more than matmul
throughput. This project has no ARM hardware to measure directly;
:func:`capability`'s CPU/memory floors and :data:`MAX_TRAIN_SECONDS`'s hard
wall-clock cutoff are this module's answer to that uncertainty - see
"Hardware gating".

Candidate extraction, and honesty
----------------------------------

The model's own softmax probability is *never* written into a candidate's
``evidence.confidence`` - that field, like every other miner's, is a plainly
counted ratio: of the times this exact (trigger, conditions) context
actually occurred in training history, how many times did this exact action
follow.  The model's own predicted probability is reported alongside,
separately, in ``evidence.extra["model_probability"]`` and in a note that
says plainly this candidate is model-derived - never blended into a number
that looks like every other miner's counted evidence.  A candidate is
proposed only when *both* bars clear
(``sequence_model_min_confidence`` for the counted ratio,
``sequence_model_min_model_probability`` for the model's own probability) -
either alone is not enough, matching this module's own docstring warning in
``amminer/miners/base.py`` that these numbers are not interchangeable.

The candidates this module proposes carry no special status once built: they
pass through ``amminer.backtest`` and ``amminer.conflicts`` exactly like any
other miner's, with no code path that reads ``candidate.miner`` to change
what those modules do.  A confident, well-supported-looking prediction from
this model that still fails its holdout backtest is rejected, the same as a
confident-looking association rule would be.

Holdout discipline
-------------------

:func:`fit` and :func:`build_examples` may only ever be called with
``train_changes``/``train_window`` - never the full analysis window - for
exactly the reason ``amminer.learn.home_mode`` gives at length in its own
docstring: fitting a model that then *proposes* candidates on data a
candidate is later graded against is the leak
``amminer.backtest.split_window`` exists to prevent.  The candidates this
module emits are graded on the *whole* window (training and holdout both) by
the ordinary backtest step in ``amminer.pipeline``, exactly like every other
miner's - never re-validated by this module itself.

Hardware gating
-----------------

This is the one non-LLM miner in the project that is off by default
(``sequence_model_enabled``) and gated behind an explicit capability check
(:func:`capability`) that runs on every attempt, whether or not the feature
is enabled, and declines with a specific, honest reason rather than
attempting a fit it cannot afford - see that function's own docstring for
what it measures and why. Training and extraction never run when the gate
fails; nothing here silently reduces its own cost when memory is tight, it
simply does not run, the same "attempt it properly or say plainly why not"
posture ``amminer.miners.motif`` already takes with ``stumpy``.

No persistence
----------------

Unlike ``amminer.learn.home_mode`` and ``amminer.learn.ranking``, nothing
here is written to the store. Both of those modules persist because they
have a genuine reason to: ``home_mode`` needs its previous fit to keep mode
numbering stable across runs, and ``ranking`` needs its previous fit because
today's model has to reflect *all* accept/dismiss history, not just
tonight's run. This module has neither need - it is refit from scratch,
deterministically, from each run's own training window, and its candidates
are already persisted (as ordinary suggestions) the same way every other
miner's are. Persisting the weights themselves as well would be pure
duplication with a migration surface and nothing to show for it; the
per-run stats below already reach the Status page through
``RunReport.sequence_model``, which is written into ``runs.stats`` like the
rest of that report.
"""

from __future__ import annotations

import logging
import os
import time as time_module
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ACTIONABLE_DOMAINS, Options
from ..miners.base import Action, Candidate, Condition, Evidence, Trigger
from ..miners.time_of_day import service_for
from ..recorderdb.models import Cause, StateChange

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware gate
# ---------------------------------------------------------------------------

#: A single-core host almost certainly needs every cycle it has for Home
#: Assistant itself, the recorder, and every other miner already running in
#: the same nightly pass - this feature is the one thing in that pass that is
#: allowed to simply not run rather than compete for them.
MIN_CPU_COUNT = 2

#: Conservative on purpose: chosen to comfortably protect a Raspberry Pi Zero
#: or an early Pi 3 running Home Assistant OS with 512 MB-1 GB total memory,
#: where this add-on is one tenant among many (the recorder, the supervisor,
#: every other integration). This module's own peak arrays are tiny (see the
#: module docstring's "Bounded everything") - the floor exists for headroom
#: on the host, not because this feature itself needs 512 MB.
MIN_AVAILABLE_MEMORY_BYTES = 512 * 1024 * 1024


def _available_memory_bytes() -> int | None:
    """Best-effort available memory, or ``None`` when it cannot be read.

    ``/proc/meminfo``'s ``MemAvailable`` (Linux - what every add-on actually
    runs on, in a container) is preferred because it already accounts for
    reclaimable cache, which is what "can I actually allocate this much"
    means; ``os.sysconf`` is a portable fallback for local development off
    Linux. ``None`` - not zero, not a guess - is returned when neither
    works, and :func:`capability` treats that as "cannot check", not "not
    enough": a host this cannot be measured on is not silently blocked, only
    honestly noted as unmeasured (the CPU-count check still applies).
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        pass
    return None


@dataclass(frozen=True)
class Capability:
    """Whether this host can afford to train the sequence model tonight."""

    ok: bool
    reason: str | None
    cpu_count: int | None
    available_memory_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "cpu_count": self.cpu_count,
            "available_memory_mb": (
                round(self.available_memory_bytes / (1024 * 1024), 1)
                if self.available_memory_bytes is not None
                else None
            ),
        }


def capability(
    cpu_count: int | None = None, available_memory_bytes: int | None = None
) -> Capability:
    """Detect whether the host can afford this feature, honestly.

    Called every time :func:`fit` runs, whether or not
    ``sequence_model_enabled`` is set - the point is to *never* attempt a fit
    and fail, and never to quietly train a smaller/cheaper model instead:
    either the host clears both floors or nothing is attempted at all, with
    the specific reason it was declined surfaced to the run report and the
    UI. ``cpu_count``/``available_memory_bytes`` are accepted as overrides
    purely so tests can exercise both branches without patching
    ``os.cpu_count``/``/proc/meminfo`` globally.
    """
    cpu = os.cpu_count() if cpu_count is None else cpu_count
    mem = _available_memory_bytes() if available_memory_bytes is None else available_memory_bytes
    reasons: list[str] = []
    if cpu is not None and cpu < MIN_CPU_COUNT:
        reasons.append(f"only {cpu} CPU core(s) detected (need at least {MIN_CPU_COUNT})")
    if mem is not None and mem < MIN_AVAILABLE_MEMORY_BYTES:
        reasons.append(
            f"only {mem / (1024 * 1024):.0f} MB of available memory detected "
            f"(need at least {MIN_AVAILABLE_MEMORY_BYTES // (1024 * 1024)} MB)"
        )
    if reasons:
        return Capability(False, "; ".join(reasons), cpu, mem)
    return Capability(True, None, cpu, mem)


# ---------------------------------------------------------------------------
# Bounded model/training constants
# ---------------------------------------------------------------------------

#: How many recent tokens of context the model ever sees.  Short on purpose -
#: see the module docstring's architecture section.
WINDOW = 6
#: The predicted-action vocabulary is capped; anything outside the most
#: frequent :data:`MAX_VOCAB` (entity, state) pairs in the training window is
#: never a prediction target (it can still appear as context - see
#: :data:`OOV_INDEX`). Bounded so the embedding table and output layer have a
#: fixed size regardless of how large a household's entity inventory is.
MAX_VOCAB = 48
#: Context tokens are drawn from a wider set of domains than a target ever
#: can be: ``ACTIONABLE_DOMAINS`` plus ``binary_sensor``. A binary sensor
#: (motion, a door, a contact switch) is exactly the kind of proximal cue
#: "the hallway light turns on ... only when the door sensor tripped less
#: than a minute earlier" needs, and its small, discrete state space (on/off,
#: or a handful of device-class values) fits this module's (entity, exact
#: state) tokenisation the same way an actionable domain's does. A general
#: numeric ``sensor`` is deliberately excluded: its exact reading almost
#: never repeats, which would defeat the whole point of a bounded,
#: frequency-ranked vocabulary of *repeated* tokens - amminer.miners.motif
#: already covers numeric-threshold-driven rules on that kind of signal.
#: ``service_for`` has no mapping for ``binary_sensor``, so it is included
#: here for context only and can never become a target - no separate check
#: is needed to keep it out of the predicted vocabulary.
CONTEXT_DOMAINS: tuple[str, ...] = ACTIONABLE_DOMAINS + ("binary_sensor",)

#: Context/target token index reserved for "no event here" (left-padding a
#: window shorter than WINDOW). Its embedding row is pinned at zero for the
#: whole fit - see ``_zero_pad_row``.
PAD_INDEX = 0
#: Context token index for an item outside the target vocabulary. It still
#: gets its own (shared) embedding, so the model can use "something happened,
#: just not a top item" as a signal, without growing the vocabulary further.
OOV_INDEX = 1
#: Seconds-since-previous-event bucket edges. 8 buckets total (7 edges).
DT_BUCKET_EDGES: tuple[float, ...] = (60.0, 300.0, 900.0, 3600.0, 14400.0, 43200.0, 86400.0)
N_DT_BUCKETS = len(DT_BUCKET_EDGES) + 1
#: One extra bucket for "this is the very first usable event in the window,
#: there is no previous event to measure a gap from" and one more for
#: "this slot is left-padding, not a real event" - kept distinct from the
#: real gap buckets so the model is never told a fabricated gap.
DT_NO_PRIOR_INDEX = N_DT_BUCKETS
DT_PAD_INDEX = N_DT_BUCKETS + 1
DT_VOCAB_SIZE = N_DT_BUCKETS + 2

EMBED_ITEM_DIM = 10
EMBED_DT_DIM = 4
HIDDEN_SIZE = 16

#: One cap, used for both training and candidate extraction (see the module
#: docstring's "No persistence" - and, more directly, the honesty concern
#: that training-only subsampling must never quietly shrink the denominator
#: candidates are graded against). Above this many eligible examples, a
#: deterministic seeded subsample is taken rather than growing the batch
#: further. See ``tests/test_sequence_model.py`` for the measured training
#: time this keeps a full year of realistic history under.
MAX_TRAIN_EXAMPLES = 2500
#: Below this many eligible examples, there is not enough to fit or to count
#: occurrences from meaningfully - reported honestly as too little history,
#: the same bar ``amminer.learn.ranking.MIN_LABELS_TO_FIT`` sets for its own
#: much smaller feature space.
MIN_TRAIN_EXAMPLES = 40

MAX_TRAIN_STEPS = 120
#: A hard wall-clock ceiling on top of the step cap - the safety valve for
#: "bounded cost" when the step/example caps above turn out not to be enough
#: on some host. Training stops early (not fails) if this is hit; the model
#: is still used, with fewer steps than requested, and that is reported
#: honestly in ``SequenceModel.steps_run`` vs ``MAX_TRAIN_STEPS``.
MAX_TRAIN_SECONDS = 30.0

LEARNING_RATE = 0.15
MOMENTUM = 0.9
#: L2 weight decay on the recurrent/output weight matrices only (not
#: embeddings or biases) - see ``_backward``.
L2 = 1e-3

#: Fixed, never wall-clock, never Python's global ``random`` module - see
#: ``amminer.learn.home_mode``'s own docstring on why, which applies
#: identically here: the same input must fit to the same weights in a fresh
#: process. Two distinct constants (never reused for two different purposes
#: in the same fit) keep initialisation and example-subsampling independent.
INIT_SEED = 0x5EA0E7CE
SUBSAMPLE_SEED = 0x5EA0E7CF

#: A candidate's trigger plus at most this many extra conditions - this is
#: the concrete answer to "an interaction between three things at once":
#: one proximal trigger and up to two condition entities, each a real
#: ``state`` condition ``amminer.backtest`` already knows how to replay.
CANDIDATE_MAX_CONDITIONS = 2


# ---------------------------------------------------------------------------
# Tokenising the event stream
# ---------------------------------------------------------------------------


def _token_key(entity_id: str, state: str) -> str:
    return f"{entity_id}={state.lower()}"


def _usable_events(
    changes: list[StateChange], options: Options, window: tuple[float, float]
) -> list[StateChange]:
    """The event stream this model tokenises: real transitions on actionable
    or binary-sensor entities (see :data:`CONTEXT_DOMAINS`), human- or
    machine-caused, excluding whatever the household has excluded. Includes
    automated changes and sensor trips as *context* (a light an automation
    just turned on, a door that just opened, are both part of what happened
    next), never as a *target* - see ``build_examples``.

    Filtered to ``[window[0], window[1])`` here, not merely trusted from the
    caller - the same defensive stance ``amminer.learn.home_mode.
    build_feature_matrix`` takes on its own ``start_ts``/``end_ts``: a caller
    that (by mistake) passed rows beyond the training window alongside the
    right window bounds must still never have them counted, so the guarantee
    is "this never learns from anything outside the window it is given", not
    merely "callers are expected to be careful" - see the module docstring's
    "Holdout discipline".
    """
    start_ts, end_ts = window
    events = [
        c
        for c in changes
        if start_ts <= c.ts < end_ts
        and c.is_transition
        and c.domain in CONTEXT_DOMAINS
        and not options.is_excluded(c.entity_id)
        and (c.state or "").lower() not in ("unknown", "unavailable", "")
        and c.cause in (Cause.HUMAN, Cause.DEVICE, Cause.AUTOMATION, Cause.SCRIPT)
    ]
    events.sort(key=lambda c: c.ts)
    return events


def _build_vocab(events: list[StateChange]) -> dict[str, int]:
    """The most frequent ``MAX_VOCAB`` (entity, state) pairs, deterministically
    ordered (ties broken by key, never by dict/set iteration order) so the
    same events produce the same vocabulary - and so the same label indices -
    in any process."""
    counts: Counter[str] = Counter()
    for event in events:
        counts[_token_key(event.entity_id, event.state)] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_VOCAB]
    return {key: index + 2 for index, (key, _count) in enumerate(ranked)}


def _dt_bucket(delta_seconds: float) -> int:
    for index, edge in enumerate(DT_BUCKET_EDGES):
        if delta_seconds < edge:
            return index
    return len(DT_BUCKET_EDGES)


_PAD_SLOT_ENTITY = ""


@dataclass(frozen=True)
class _Slot:
    item_idx: int
    dt_idx: int
    entity_id: str
    state: str
    in_vocab: bool


_PAD_SLOT = _Slot(item_idx=PAD_INDEX, dt_idx=DT_PAD_INDEX, entity_id="", state="", in_vocab=False)


@dataclass(frozen=True)
class Example:
    """One training/extraction example: a fixed-length context window
    (oldest first, left-padded with :data:`_PAD_SLOT`) and the human action
    that followed it."""

    context: tuple[_Slot, ...]
    label_idx: int
    target_entity_id: str
    target_state: str
    ts: float


def build_examples(
    train_changes: list[StateChange],
    vocab: dict[str, int],
    options: Options,
    train_window: tuple[float, float],
) -> list[Example]:
    """Walk the training-window event stream once, building one example per
    eligible human action.

    Only rows inside ``train_window`` are ever read here - see
    ``_usable_events`` and the module docstring's "Holdout discipline". A
    human action is eligible when it maps to a real service call
    (``service_for``) and its (entity, state) is inside the bounded target
    vocabulary; every other actionable transition (human or not) still
    becomes *context* for whatever comes after it.
    """
    events = _usable_events(train_changes, options, train_window)
    examples: list[Example] = []
    slots: list[_Slot] = []
    prev_ts: float | None = None
    for event in events:
        key = _token_key(event.entity_id, event.state)
        in_vocab = key in vocab
        item_idx = vocab[key] if in_vocab else OOV_INDEX
        dt_idx = _dt_bucket(event.ts - prev_ts) if prev_ts is not None else DT_NO_PRIOR_INDEX
        prev_ts = event.ts
        slot = _Slot(
            item_idx=item_idx,
            dt_idx=dt_idx,
            entity_id=event.entity_id,
            state=(event.state or "").lower(),
            in_vocab=in_vocab,
        )

        state = (event.state or "").lower()
        if (
            event.cause is Cause.HUMAN
            and in_vocab
            and slots
            and service_for(event.entity_id, state) is not None
        ):
            context = tuple(slots[-WINDOW:])
            pad_needed = WINDOW - len(context)
            padded = (_PAD_SLOT,) * pad_needed + context
            examples.append(
                Example(
                    context=padded,
                    label_idx=item_idx - 2,
                    target_entity_id=event.entity_id,
                    target_state=state,
                    ts=event.ts,
                )
            )
        slots.append(slot)
    return examples


def _cap_examples(examples: list[Example], cap: int) -> list[Example]:
    """Deterministically subsample down to ``cap`` examples, preserving
    chronological order in the kept subset. The same seed, applied to the
    same (already-deterministic) example list, picks the same subset in any
    process."""
    if len(examples) <= cap:
        return examples
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    keep = rng.choice(len(examples), size=cap, replace=False)
    keep.sort()
    return [examples[i] for i in keep]


def _design_matrices(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(examples)
    item_idx = np.zeros((n, WINDOW), dtype=np.int64)
    dt_idx = np.zeros((n, WINDOW), dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    for row, example in enumerate(examples):
        for col, slot in enumerate(example.context):
            item_idx[row, col] = slot.item_idx
            dt_idx[row, col] = slot.dt_idx
        labels[row] = example.label_idx
    return item_idx, dt_idx, labels


# ---------------------------------------------------------------------------
# The model: a compact GRU, hand-derived forward/backward in pure numpy
# ---------------------------------------------------------------------------

_REGULARISED_PARAMS: tuple[str, ...] = ("Wz", "Wr", "Wn", "Uz", "Ur", "Un", "Wo")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _init_params(rng: np.random.Generator, vocab_size: int) -> dict[str, np.ndarray]:
    """Glorot/Xavier-uniform init for every weight matrix, zeros for biases.

    The PAD row of ``item_emb`` (index :data:`PAD_INDEX`) is zeroed and its
    gradient is zeroed every step (``_backward``), so a padded slot
    contributes nothing to the model, forever - not merely "starts at
    nothing and is free to drift".
    """

    def glorot(fan_in: int, fan_out: int) -> np.ndarray:
        limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))

    d_in = EMBED_ITEM_DIM + EMBED_DT_DIM
    params = {
        "item_emb": glorot(vocab_size + 2, EMBED_ITEM_DIM),
        "dt_emb": glorot(DT_VOCAB_SIZE, EMBED_DT_DIM),
        "Wz": glorot(d_in, HIDDEN_SIZE),
        "Uz": glorot(HIDDEN_SIZE, HIDDEN_SIZE),
        "bz": np.zeros(HIDDEN_SIZE),
        "Wr": glorot(d_in, HIDDEN_SIZE),
        "Ur": glorot(HIDDEN_SIZE, HIDDEN_SIZE),
        "br": np.zeros(HIDDEN_SIZE),
        "Wn": glorot(d_in, HIDDEN_SIZE),
        "Un": glorot(HIDDEN_SIZE, HIDDEN_SIZE),
        "bn": np.zeros(HIDDEN_SIZE),
        "Wo": glorot(HIDDEN_SIZE, vocab_size),
        "bo": np.zeros(vocab_size),
    }
    params["item_emb"][PAD_INDEX, :] = 0.0
    return params


def _forward(params: dict[str, np.ndarray], item_idx: np.ndarray, dt_idx: np.ndarray) -> dict[str, Any]:
    """One forward pass over a whole batch, ``WINDOW`` steps, GRU update:
    ``h_t = (1 - z_t) * n_t + z_t * h_{t-1}``.

    Every quantity needed by :func:`_backward` is cached per step - this is
    the one function whose output shape that one depends on.

    An earlier version of this function batched the input-side projections
    (``x_t @ Wz`` etc.) into three large matmuls computed once before the
    loop, on the theory that fewer, larger numpy calls would beat many small
    ones. Measured against this module's own benchmark (a realistic
    synthetic year - see ``tests/test_sequence_model.py`` and this
    project's PR description), it made no measurable difference: at this
    model's size the per-step cost is dominated by the elementwise
    ``sigmoid``/``tanh`` gates and reductions themselves, not by matmul call
    count, so the simpler, more obviously-correct form below was kept
    instead of complexity that did not earn its keep.
    """
    n = item_idx.shape[0]
    hidden = np.zeros((n, HIDDEN_SIZE))
    steps: list[dict[str, Any]] = []
    for t in range(WINDOW):
        item_vec = params["item_emb"][item_idx[:, t]]
        dt_vec = params["dt_emb"][dt_idx[:, t]]
        x = np.concatenate([item_vec, dt_vec], axis=1)

        z_pre = x @ params["Wz"] + hidden @ params["Uz"] + params["bz"]
        z = _sigmoid(z_pre)
        r_pre = x @ params["Wr"] + hidden @ params["Ur"] + params["br"]
        r = _sigmoid(r_pre)
        h_un = hidden @ params["Un"]
        n_pre = x @ params["Wn"] + r * h_un + params["bn"]
        n_gate = np.tanh(n_pre)

        h_prev = hidden
        hidden = (1.0 - z) * n_gate + z * h_prev
        steps.append(
            {
                "x": x,
                "z": z,
                "r": r,
                "n": n_gate,
                "h_prev": h_prev,
                "h_un": h_un,
                "item_idx": item_idx[:, t],
                "dt_idx": dt_idx[:, t],
            }
        )
    logits = hidden @ params["Wo"] + params["bo"]
    return {"steps": steps, "h_final": hidden, "logits": logits}


def _loss(logits: np.ndarray, labels: np.ndarray, params: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    probs = _softmax(logits)
    n = logits.shape[0]
    eps = 1e-12
    true_probs = np.clip(probs[np.arange(n), labels], eps, 1.0)
    cross_entropy = float(-np.mean(np.log(true_probs)))
    reg = sum(float(np.sum(params[name] ** 2)) for name in _REGULARISED_PARAMS)
    return cross_entropy + 0.5 * L2 * reg, probs


def _backward(
    params: dict[str, np.ndarray],
    cache: dict[str, Any],
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Backprop-through-time, matching :func:`_forward` step for step.

    ``tests/test_sequence_model.py::test_gradients_match_finite_differences``
    checks every parameter's gradient against a numeric finite-difference
    estimate on a tiny model - the actual evidence this is correct, not the
    derivation in this comment.
    """
    steps = cache["steps"]
    h_final = cache["h_final"]
    logits = cache["logits"]
    n = logits.shape[0]

    probs = _softmax(logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    dlogits = (probs - onehot) / n

    grads = {name: np.zeros_like(value) for name, value in params.items()}
    grads["Wo"] += h_final.T @ dlogits
    grads["bo"] += dlogits.sum(axis=0)
    d_hidden = dlogits @ params["Wo"].T

    for step in reversed(steps):
        x, z, r, n_gate = step["x"], step["z"], step["r"], step["n"]
        h_prev, h_un = step["h_prev"], step["h_un"]

        # hidden = (1 - z) * n_gate + z * h_prev
        d_z = d_hidden * (h_prev - n_gate)
        d_n = d_hidden * (1.0 - z)
        d_h_prev = d_hidden * z  # direct path through z * h_prev

        d_n_pre = d_n * (1.0 - n_gate**2)
        grads["Wn"] += x.T @ d_n_pre
        grads["bn"] += d_n_pre.sum(axis=0)
        d_r = d_n_pre * h_un
        d_h_un = d_n_pre * r
        grads["Un"] += h_prev.T @ d_h_un
        d_h_prev = d_h_prev + d_h_un @ params["Un"].T

        d_r_pre = d_r * r * (1.0 - r)
        grads["Wr"] += x.T @ d_r_pre
        grads["br"] += d_r_pre.sum(axis=0)
        grads["Ur"] += h_prev.T @ d_r_pre
        d_h_prev = d_h_prev + d_r_pre @ params["Ur"].T

        d_z_pre = d_z * z * (1.0 - z)
        grads["Wz"] += x.T @ d_z_pre
        grads["bz"] += d_z_pre.sum(axis=0)
        grads["Uz"] += h_prev.T @ d_z_pre
        d_h_prev = d_h_prev + d_z_pre @ params["Uz"].T

        d_x = d_n_pre @ params["Wn"].T + d_r_pre @ params["Wr"].T + d_z_pre @ params["Wz"].T
        d_item = d_x[:, :EMBED_ITEM_DIM]
        d_dt = d_x[:, EMBED_ITEM_DIM:]
        np.add.at(grads["item_emb"], step["item_idx"], d_item)
        np.add.at(grads["dt_emb"], step["dt_idx"], d_dt)

        d_hidden = d_h_prev

    for name in _REGULARISED_PARAMS:
        grads[name] += L2 * params[name]
    grads["item_emb"][PAD_INDEX, :] = 0.0
    return grads


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class SequenceModel:
    """A fitted (or declined) sequence model, plus everything the run report
    needs to say honestly what happened."""

    fitted: bool
    fallback_reason: str | None = None
    vocab: dict[str, int] = field(default_factory=dict)
    params: dict[str, np.ndarray] | None = None
    window: int = WINDOW
    hidden_size: int = HIDDEN_SIZE
    n_examples: int = 0
    n_vocab: int = 0
    steps_run: int = 0
    train_loss: float | None = None
    train_accuracy: float | None = None
    training_seconds: float = 0.0
    capability: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "fallback_reason": self.fallback_reason,
            "window": self.window,
            "hidden_size": self.hidden_size,
            "n_examples": self.n_examples,
            "n_vocab": self.n_vocab,
            "steps_run": self.steps_run,
            "train_loss": round(self.train_loss, 4) if self.train_loss is not None else None,
            "train_accuracy": (
                round(self.train_accuracy, 4) if self.train_accuracy is not None else None
            ),
            "training_seconds": round(self.training_seconds, 3),
            "capability": self.capability,
        }


def fit(
    train_changes: list[StateChange], options: Options, train_window: tuple[float, float]
) -> tuple[SequenceModel, list[Example]]:
    """Fit a sequence model on **training-window data only**.

    Mirrors ``amminer.learn.home_mode.fit``'s contract: this is the one
    function that may see raw ``StateChange`` rows to fit from, and it is
    ``amminer.pipeline``'s responsibility to pass only
    ``train_changes``/``train_window``, never the full analysis window -
    see the module docstring's "Holdout discipline". Whether
    ``sequence_model_enabled`` is set is the caller's decision, exactly like
    ``home_mode_enabled`` above it; this function does not read it.

    Returns the model *and* the (possibly capped) example list it was built
    from, so :func:`extract_candidates` counts occurrences/opportunities
    against exactly the examples the model actually saw - never a second,
    differently-capped rebuild that could silently disagree with it.
    """
    cap = capability()
    if not cap.ok:
        return SequenceModel(
            fitted=False,
            fallback_reason=f"this host cannot afford it: {cap.reason}",
            capability=cap.as_dict(),
        ), []

    train_days = (train_window[1] - train_window[0]) / 86400.0
    if train_days < options.sequence_model_min_train_days:
        return SequenceModel(
            fitted=False,
            fallback_reason=(
                f"only {train_days:.1f} days of training history (< "
                f"{options.sequence_model_min_train_days} needed)"
            ),
            capability=cap.as_dict(),
        ), []

    events = _usable_events(train_changes, options, train_window)
    vocab = _build_vocab(events)
    if not vocab:
        return SequenceModel(
            fitted=False,
            fallback_reason="no actionable entities with enough activity to learn from",
            capability=cap.as_dict(),
        ), []

    examples = build_examples(train_changes, vocab, options, train_window)
    if len(examples) < MIN_TRAIN_EXAMPLES:
        return SequenceModel(
            fitted=False,
            fallback_reason=(
                f"only {len(examples)} usable (context, action) examples "
                f"(< {MIN_TRAIN_EXAMPLES} needed)"
            ),
            capability=cap.as_dict(),
        ), []

    examples = _cap_examples(examples, MAX_TRAIN_EXAMPLES)
    item_idx, dt_idx, labels = _design_matrices(examples)

    rng = np.random.default_rng(INIT_SEED)
    params = _init_params(rng, vocab_size=len(vocab))
    velocity = {name: np.zeros_like(value) for name, value in params.items()}

    start = time_module.monotonic()
    steps_run = 0
    for _ in range(MAX_TRAIN_STEPS):
        if time_module.monotonic() - start > MAX_TRAIN_SECONDS:
            break
        cache = _forward(params, item_idx, dt_idx)
        grads = _backward(params, cache, labels)
        for name in params:
            velocity[name] = MOMENTUM * velocity[name] - LEARNING_RATE * grads[name]
            params[name] = params[name] + velocity[name]
        params["item_emb"][PAD_INDEX, :] = 0.0
        steps_run += 1
    training_seconds = time_module.monotonic() - start

    final_cache = _forward(params, item_idx, dt_idx)
    final_loss, final_probs = _loss(final_cache["logits"], labels, params)
    accuracy = float(np.mean(np.argmax(final_probs, axis=1) == labels))

    model = SequenceModel(
        fitted=True,
        fallback_reason=None,
        vocab=vocab,
        params=params,
        window=WINDOW,
        hidden_size=HIDDEN_SIZE,
        n_examples=len(examples),
        n_vocab=len(vocab),
        steps_run=steps_run,
        train_loss=final_loss,
        train_accuracy=accuracy,
        training_seconds=training_seconds,
        capability=cap.as_dict(),
    )
    _LOGGER.info(
        "sequence model: fitted on %d examples (%d-word vocab) in %d steps, %.2fs, "
        "train accuracy %.0f%%",
        model.n_examples,
        model.n_vocab,
        model.steps_run,
        model.training_seconds,
        model.train_accuracy * 100.0,
    )
    return model, examples


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def _signature(
    context: tuple[_Slot, ...], target_entity_id: str
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    """The (trigger, conditions) this context reduces to, or ``None`` when it
    cannot be expressed as one (no in-vocabulary trigger, or the trigger is
    the same entity the action targets - "when the light turns on, turn the
    light on" is not a rule).

    The trigger is always the *most recent* context slot; up to
    :data:`CANDIDATE_MAX_CONDITIONS` further, distinct-entity, in-vocabulary
    slots (scanning backwards from just before the trigger) become
    ``state`` conditions. This is what turns a context the model actually
    saw into the concrete, replayable shape ``amminer.backtest`` already
    knows how to simulate - see the module docstring's "Candidate
    extraction, and honesty".
    """
    trigger = context[-1]
    if not trigger.in_vocab or not trigger.entity_id or trigger.entity_id == target_entity_id:
        return None
    conditions: list[tuple[str, str]] = []
    seen = {trigger.entity_id, target_entity_id}
    for slot in reversed(context[:-1]):
        if len(conditions) >= CANDIDATE_MAX_CONDITIONS:
            break
        if not slot.in_vocab or not slot.entity_id or slot.entity_id in seen:
            continue
        seen.add(slot.entity_id)
        conditions.append((slot.entity_id, slot.state))
    return trigger.entity_id, trigger.state, tuple(sorted(conditions))


@dataclass(frozen=True)
class _Rule:
    """One (trigger, conditions) -> action pairing that cleared both
    confidence bars, before redundancy pruning."""

    signature: tuple[str, str, tuple[tuple[str, str], ...]]
    target_key: tuple[str, str]
    rows: tuple[int, ...]
    occurrences: int
    opportunities: int
    confidence: float
    model_confidence: float


#: How much higher a rule with an extra condition has to score before it is
#: kept alongside (or instead of) the simpler rule it extends. Without this,
#: a condition that happened to be true by coincidence for a handful of the
#: same occurrences produces a near-duplicate, equally "confident" candidate
#: for every incidental entity that was also on at the time - see the
#: module's own test for a synthetic case that reproduces exactly this.
#: Mirrors ``amminer.miners.sequence``'s own redundant-pattern margin
#: (``DAY_RESTRICTION_MARGIN`` in ``time_of_day.py`` is the same idea again).
REDUNDANCY_MARGIN = 0.02


def _drop_redundant_supersets(rules: list[_Rule]) -> list[_Rule]:
    """Prefer the simplest rule that explains a (trigger, action) pairing.

    Grouped by (trigger, action) - never across different triggers or
    different actions, which are genuinely different claims - then kept in
    order of how few conditions they add: a rule is dropped only when an
    already-kept rule's conditions are a subset of its own *and* it is not
    meaningfully more confident than that simpler rule.
    """
    by_head: dict[tuple[str, str, tuple[str, str]], list[_Rule]] = defaultdict(list)
    for rule in rules:
        trigger_entity, trigger_state, _conditions = rule.signature
        by_head[(trigger_entity, trigger_state, rule.target_key)].append(rule)

    kept: list[_Rule] = []
    for group in by_head.values():
        group.sort(key=lambda r: len(r.signature[2]))
        accepted: list[_Rule] = []
        for rule in group:
            cond_set = set(rule.signature[2])
            redundant = any(
                set(other.signature[2]) <= cond_set
                and rule.confidence <= other.confidence + REDUNDANCY_MARGIN
                for other in accepted
            )
            if not redundant:
                accepted.append(rule)
        kept.extend(accepted)
    return kept


def extract_candidates(
    model: SequenceModel,
    examples: list[Example],
    options: Options,
    window: tuple[float, float],
    resolver: Any = None,
) -> list[Candidate]:
    """Turn a fitted model's confident, well-supported predictions into
    candidates.

    For every distinct (trigger, conditions) signature actually observed in
    ``examples`` (never a context the model merely imagines - see the module
    docstring), this counts how many times it occurred at all
    (``opportunities``) and how many of those times each possible action
    followed (``occurrences`` for the most common one); a candidate is
    proposed only when the counted ratio *and* the model's own average
    predicted probability for that outcome both clear their configured
    bars. Both are reported - never blended into one number. Rules that only
    restate a simpler rule with an incidental extra condition are dropped -
    see :func:`_drop_redundant_supersets`.
    """
    if not model.fitted or not model.params or not examples:
        return []

    item_idx, dt_idx, labels = _design_matrices(examples)
    cache = _forward(model.params, item_idx, dt_idx)
    probs = _softmax(cache["logits"])
    model_probs = probs[np.arange(len(examples)), labels]

    groups: dict[tuple[str, str, tuple[tuple[str, str], ...]], dict[tuple[str, str], list[int]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for index, example in enumerate(examples):
        signature = _signature(example.context, example.target_entity_id)
        if signature is None:
            continue
        target_key = (example.target_entity_id, example.target_state)
        groups[signature][target_key].append(index)

    rules: list[_Rule] = []
    for signature, outcomes in sorted(groups.items()):
        opportunities = sum(len(rows) for rows in outcomes.values())
        if opportunities < options.sequence_model_min_occurrences:
            continue
        target_key, rows = max(sorted(outcomes.items()), key=lambda kv: len(kv[1]))
        occurrences = len(rows)
        if occurrences < options.sequence_model_min_occurrences:
            continue
        empirical_confidence = occurrences / opportunities
        if empirical_confidence < options.sequence_model_min_confidence:
            continue
        model_confidence = float(np.mean(model_probs[rows]))
        if model_confidence < options.sequence_model_min_model_probability:
            continue
        rules.append(
            _Rule(
                signature=signature,
                target_key=target_key,
                rows=tuple(rows),
                occurrences=occurrences,
                opportunities=opportunities,
                confidence=empirical_confidence,
                model_confidence=model_confidence,
            )
        )

    start_ts, end_ts = window
    candidates: list[Candidate] = []
    for rule in _drop_redundant_supersets(rules):
        signature = rule.signature
        target_key = rule.target_key
        rows = list(rule.rows)
        occurrences = rule.occurrences
        opportunities = rule.opportunities
        empirical_confidence = rule.confidence
        model_confidence = rule.model_confidence

        trigger_entity, trigger_state, conditions = signature
        target_entity, target_state = target_key
        service = service_for(target_entity, target_state)
        if service is None:
            continue
        service_name, service_data = service

        trigger = Trigger(kind="state", entity_id=trigger_entity, to_state=trigger_state)
        condition_objs = [
            Condition(kind="state", entity_id=entity, state=state, source="sequence model context")
            for entity, state in conditions
        ]
        action = Action(service=service_name, entity_id=target_entity, data=dict(service_data))

        t_name = resolver.name_of(trigger_entity) if resolver else trigger_entity
        a_name = resolver.name_of(target_entity) if resolver else target_entity
        cond_text = ""
        if condition_objs:
            parts = [
                f"{resolver.name_of(c.entity_id) if resolver else c.entity_id} is '{c.state}'"
                for c in condition_objs
            ]
            cond_text = " while " + " and ".join(parts)

        verb = service_name.split(".", 1)[-1].replace("_", " ")
        samples = [examples[i].ts for i in rows][:50]

        candidates.append(
            Candidate(
                miner="sequence_model",
                title=f"{verb.capitalize()} {a_name} after {t_name} becomes '{trigger_state}'{cond_text}",
                description=(
                    f"A sequence model trained on your history predicted this after {t_name} "
                    f"becomes '{trigger_state}'{cond_text}: {occurrences} of the {opportunities} "
                    "times this exact context occurred, this was the next thing you did."
                ),
                triggers=[trigger],
                conditions=condition_objs,
                actions=[action],
                evidence=Evidence(
                    occurrences=occurrences,
                    opportunities=opportunities,
                    confidence=empirical_confidence,
                    window_start_ts=start_ts,
                    window_end_ts=end_ts,
                    window_days=(end_ts - start_ts) / 86400.0,
                    samples=samples,
                    notes=[
                        "Model-derived: proposed by a compact sequence model trained on your "
                        "activity (amminer.learn.sequence), not hand-written. The occurrence and "
                        "opportunity counts above are measured directly from your history, the "
                        "same as every other miner's evidence - the model only chose which "
                        "context to look at.",
                        f"The model's own predicted probability for this outcome, averaged over "
                        f"the {occurrences} times it happened, was {model_confidence:.0%}.",
                    ],
                    extra={
                        "model": "gru",
                        "model_probability": round(model_confidence, 4),
                        "hidden_size": model.hidden_size,
                        "window": model.window,
                        "n_vocab": model.n_vocab,
                    },
                ),
                score=round(empirical_confidence * model_confidence, 4),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    _LOGGER.info("sequence model produced %d candidates", len(candidates))
    return candidates
