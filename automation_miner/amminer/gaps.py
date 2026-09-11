"""Integration and hardware gap suggestions.

Each rule looks for a *detected capability gap* - something the user's own
behaviour shows they want, that their current setup cannot express well - and
names a concrete addition plus the expected benefit.  Nothing here is a generic
"you could buy more sensors" recommendation; every finding cites the evidence
that produced it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .enrich.detect import SignalSet
from .miners.base import Candidate
from .recorderdb.models import Cause, StateChange

_LOGGER = logging.getLogger(__name__)


@dataclass
class GapSuggestion:
    """One recommended integration or piece of hardware."""

    kind: str
    title: str
    gap: str
    recommendation: str
    benefit: str
    #: What has to be true about the user's life for this to be worth doing.
    #:
    #: Every gap here is inferred from entities, and entities cannot see a
    #: contract, a roof, or a commute.  "Add a dynamic price sensor" is sound
    #: advice for someone on a variable tariff and useless for someone on a
    #: fixed one, and the suggestion has no way to tell which.  Saying so is the
    #: difference between a recommendation and a guess presented as one.
    requires: str = ""
    evidence: list[str] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    score: float = 0.5
    #: Set when a model proposed this rather than a detector rule.
    source: str = "detector"

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.kind}|{self.title}".encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "gap": self.gap,
            "recommendation": self.recommendation,
            "benefit": self.benefit,
            "requires": self.requires,
            "evidence": self.evidence,
            "links": self.links,
            "score": round(self.score, 3),
            "source": self.source,
        }


def human_actions_by_domain(changes: Sequence[StateChange]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for change in changes:
        if change.cause is Cause.HUMAN and change.is_transition:
            counts[change.domain] = counts.get(change.domain, 0) + 1
    return counts


def suggest(
    resolver,
    signals: SignalSet,
    changes: Sequence[StateChange],
    candidates: Sequence[Candidate] = (),
    recorder_info=None,
) -> list[GapSuggestion]:
    """Derive every applicable gap suggestion for this instance."""
    out: list[GapSuggestion] = []
    actions = human_actions_by_domain(changes)
    light_actions = actions.get("light", 0) + actions.get("switch", 0)

    # --- room presence -------------------------------------------------
    if light_actions >= 20 and not signals.room_presence:
        if signals.motion and not signals.occupancy:
            out.append(
                GapSuggestion(
                    kind="hardware",
                    title="Add mmWave presence sensing for lights that stay on",
                    gap=(
                        f"You have {len(signals.motion)} motion (PIR) sensors but no room-presence "
                        "sensor. PIR only sees movement, so lights switch off while you sit still."
                    ),
                    recommendation=(
                        "Add an mmWave presence sensor (Aqara FP2, Everything Presence One/Lite, "
                        "or an LD2410/LD2450 board), or set up ESPresense / Bermuda BLE room "
                        "presence using ESP32 nodes you may already have."
                    ),
                    benefit=(
                        "Presence-driven lighting that does not turn off on a still person, and "
                        "per-room occupancy conditions for the rules mined here."
                    ),
                    evidence=[
                        f"{light_actions} manual light/switch actions in the analysis window.",
                        f"Motion sensors found: {', '.join(signals.motion[:5])}.",
                        "No occupancy/room-presence entity detected.",
                    ],
                    score=0.8,
                )
            )
        elif not signals.motion and not signals.occupancy:
            out.append(
                GapSuggestion(
                    kind="hardware",
                    title="Add presence sensing so lighting can be automated at all",
                    gap=(
                        f"You switch lights by hand {light_actions} times, but there is no motion "
                        "or presence sensor anywhere in the instance."
                    ),
                    recommendation=(
                        "Start with one mmWave presence sensor (Aqara FP2, Everything Presence) "
                        "in the room you control most often."
                    ),
                    benefit="Turns most of your manual light switching into reliable automation.",
                    evidence=[f"{light_actions} manual light/switch actions, zero presence entities."],
                    score=0.75,
                )
            )

    # --- energy: dynamic tariff + deferrable load ----------------------
    has_price = bool(signals.energy_price or signals.price_level)
    deferrable = list(signals.deferrable_loads) + list(signals.ev_charger)
    if deferrable and not has_price:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Add a dynamic electricity price sensor to shift these loads",
                gap=(
                    f"You have deferrable loads ({', '.join(deferrable[:4])}) but no dynamic "
                    "price signal, so nothing can decide when running them is cheap."
                ),
                recommendation=(
                    "Add Tibber (all-in price plus a cheap/normal/expensive level), Nordpool, "
                    "ENTSO-e (sensor.current_electricity_market_price), aWATTar/EPEX or Energi "
                    "Data Service - whichever matches your country and tariff."
                ),
                benefit=(
                    "Automation Miner can then propose costed load-shifting rules, and EMHASS "
                    "becomes an option for full optimisation."
                ),
                requires=(
                    "an electricity contract whose price actually varies through the day - "
                    "a spot, hourly or time-of-use tariff. On a fixed-price contract there is "
                    "nothing to shift loads towards and this is not worth doing."
                ),
                evidence=[f"Deferrable loads detected: {', '.join(deferrable[:6])}."],
                score=0.7,
            )
        )
    elif has_price and deferrable:
        shifting = any(c.miner == "energy_shift" for c in candidates)
        if not shifting:
            out.append(
                GapSuggestion(
                    kind="integration",
                    title="You have prices and deferrable loads but no load shifting",
                    gap=(
                        "A dynamic price signal and deferrable loads are both present, but no "
                        "automation moves those loads into cheap windows."
                    ),
                    recommendation=(
                        "Add a price-triggered automation (Automation Miner can generate one), or "
                        "install EMHASS for whole-home optimisation against price and solar."
                    ),
                    benefit="Direct bill reduction with no change in comfort for deferrable loads.",
                    requires=(
                        "that the price sensor reflects what you are actually billed. If it "
                        "tracks a market you are not exposed to, shifting saves nothing."
                    ),
                    evidence=[
                        f"Price signal: {', '.join((signals.energy_price + signals.price_level)[:3])}.",
                        f"Deferrable loads: {', '.join(deferrable[:4])}.",
                    ],
                    score=0.65,
                )
            )

    # --- solar production without a forecast ---------------------------
    if signals.solar_production and not signals.solar_forecast:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Add a solar production forecast",
                gap=(
                    "You have a solar inverter reporting production, but nothing forecasts what "
                    "it will produce, so loads cannot be scheduled against expected sunshine."
                ),
                recommendation=(
                    "Add Forecast.Solar (core integration, no account needed) or Solcast via HACS "
                    "(BJReplay/ha-solcast-solar; call solcast_solar.update_forecasts on a schedule)."
                ),
                benefit=(
                    "Lets rules run heavy loads on the sunny part of tomorrow instead of "
                    "reacting only to current production."
                ),
                evidence=[f"Solar production entities: {', '.join(signals.solar_production[:4])}."],
                requires=(
                    "solar panels on this property. The forecast is a prediction of what your "
                    "own array will produce; without one there is nothing to forecast."
                ),
                score=0.6,
            )
        )

    # --- climate without outdoor temperature ---------------------------
    if signals.thermostat and not signals.outdoor_temperature and not signals.weather:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Add outdoor temperature for your heating rules",
                gap=(
                    f"You have {len(signals.thermostat)} climate entities but no outdoor "
                    "temperature source, so heating rules cannot know it is mild outside."
                ),
                recommendation=(
                    "Add the Met.no integration (free, no key) for a weather entity, or an "
                    "outdoor temperature sensor for a local reading."
                ),
                benefit=(
                    "Turns 'heat the living room at 18:00' into 'heat it at 18:00 when it is "
                    "below 12 C outside' - far fewer pointless heating cycles."
                ),
                evidence=[f"Climate entities: {', '.join(signals.thermostat[:4])}."],
                score=0.7,
            )
        )

    # --- calendar / workday --------------------------------------------
    weekday_rules = [
        c
        for c in candidates
        if any(cond.kind == "time" and cond.weekday for cond in c.conditions)
    ]
    if weekday_rules and not signals.workday:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Add the Workday binary sensor",
                gap=(
                    f"{len(weekday_rules)} mined rules distinguish weekdays from weekends by "
                    "day-of-week alone, which is wrong on public holidays and your days off."
                ),
                recommendation=(
                    "Add the Workday integration (country + optional province) and, if you use "
                    "one, a Local Calendar or CalDAV calendar for holidays."
                ),
                benefit="Weekday rules stop firing on public holidays and booked leave.",
                requires=(
                    "a schedule that follows public holidays - shift work, retirement or an "
                    "irregular week make a workday sensor say the wrong thing."
                ),
                evidence=[f"Weekday-conditioned rules: {len(weekday_rules)}."],
                score=0.55,
            )
        )

    # --- presence tracking ----------------------------------------------
    if not signals.person and not signals.device_tracker:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Set up presence detection",
                gap="No person or device_tracker entity exists, so no rule can know if anyone is home.",
                recommendation=(
                    "Add the Home Assistant companion app (per-person device_tracker), or router "
                    "based tracking - the AVM FRITZ!Box integration polls home/not_home about "
                    "every 30 s - then create a person for each household member."
                ),
                benefit="Unlocks arrive/leave automations, which are usually the highest-value ones.",
                requires=(
                    "that everyone whose arrival should matter carries a phone with the "
                    "companion app, or a tracked device. Presence is only as good as what it "
                    "can see leave the house."
                ),
                evidence=["No person.* or device_tracker.* entities found."],
                score=0.85,
            )
        )

    # --- carbon intensity ------------------------------------------------
    if deferrable and not signals.carbon_intensity and has_price:
        out.append(
            GapSuggestion(
                kind="integration",
                title="Add grid carbon intensity",
                gap="You already shift loads on price; carbon intensity is a second free signal.",
                recommendation=(
                    "Add Electricity Maps (co2signal) for "
                    "sensor.electricity_maps_carbon_intensity in gCO2eq/kWh."
                ),
                benefit="Lets the same deferrable loads also run when the grid is cleanest.",
                requires=(
                    "that running loads on cleaner grid power is something you want to "
                    "optimise for. It is a preference, not a saving - the cleanest hour and "
                    "the cheapest hour are often different ones."
                ),
                evidence=[f"Deferrable loads: {', '.join(deferrable[:4])}."],
                score=0.4,
            )
        )

    # --- retention -------------------------------------------------------
    if recorder_info is not None and recorder_info.is_sqlite and recorder_info.history_days < 14:
        out.append(
            GapSuggestion(
                kind="infrastructure",
                title="Move the recorder to MariaDB for longer history",
                gap=(
                    f"Only {recorder_info.history_days:.1f} days of raw history are available "
                    f"(purge_keep_days is {recorder_info.purge_keep_days}). Sequence and "
                    "association mining need weeks of data to be trustworthy."
                ),
                recommendation=(
                    "Install the MariaDB add-on and point the recorder at it:\n"
                    "recorder:\n"
                    "  db_url: mysql://user:pw@core-mariadb/homeassistant?charset=utf8mb4\n"
                    "  purge_keep_days: 60"
                ),
                benefit=(
                    "Unlocks every miner, makes backtests statistically meaningful, and removes "
                    "SQLite write contention."
                ),
                evidence=[
                    f"Raw history: {recorder_info.history_days:.1f} days.",
                    f"Recorder dialect: {recorder_info.dialect}.",
                ],
                score=0.9,
            )
        )

    out.sort(key=lambda g: g.score, reverse=True)
    _LOGGER.info("gap analysis produced %d suggestions", len(out))
    return out
