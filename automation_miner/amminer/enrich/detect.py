"""Detect which external signals this Home Assistant instance actually has.

Nothing here is ever *required*.  Each detector returns the entity ids it found
(possibly none) and the pipeline simply mines fewer conditions when a signal is
missing.  Detection is by domain + device_class + platform + entity-id
heuristics, so it works for core integrations and HACS ones alike.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

_LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _bounded(pattern: str) -> re.Pattern[str]:
    """Compile *pattern* so it can only start at a word boundary.

    These patterns are matched against entity ids and friendly names, where a
    bare substring search finds a lot of things it did not mean: ``go_?e``
    matches "man**go_e**thanol", ``sma_`` matches "pla**sma_**display", and
    ``ev_?charg`` matches "pr**ev_charg**e".  A separator - anything that is not
    a letter or a digit - must come first.  Nothing is added to the end, because
    several of these are real prefixes of longer names (``go_echarger``).
    """
    prefix = r"(?<![a-z0-9])" if pattern[:1].isalnum() else ""
    return re.compile(prefix + pattern)


def _match_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(_bounded(pattern).search(lowered) for pattern in patterns)


@dataclass
class SignalSet:
    """Every external signal we found, grouped by role."""

    sun: list[str] = field(default_factory=list)
    weather: list[str] = field(default_factory=list)
    outdoor_temperature: list[str] = field(default_factory=list)
    illuminance: list[str] = field(default_factory=list)
    workday: list[str] = field(default_factory=list)
    holiday: list[str] = field(default_factory=list)
    calendar: list[str] = field(default_factory=list)
    person: list[str] = field(default_factory=list)
    device_tracker: list[str] = field(default_factory=list)
    room_presence: list[str] = field(default_factory=list)
    occupancy: list[str] = field(default_factory=list)
    motion: list[str] = field(default_factory=list)
    energy_price: list[str] = field(default_factory=list)
    price_level: list[str] = field(default_factory=list)
    solar_forecast: list[str] = field(default_factory=list)
    solar_production: list[str] = field(default_factory=list)
    carbon_intensity: list[str] = field(default_factory=list)
    weather_warning: list[str] = field(default_factory=list)
    power: list[str] = field(default_factory=list)
    energy: list[str] = field(default_factory=list)
    deferrable_loads: list[str] = field(default_factory=list)
    ev_charger: list[str] = field(default_factory=list)
    thermostat: list[str] = field(default_factory=list)
    #: The inferred household mode (amminer.learn.home_mode), when a model
    #: was fit this run. Not found by this module's own regex heuristics
    #: below like every other field here - amminer.pipeline sets it directly
    #: once home_mode.fit succeeds, after detect_signals() has already run.
    #: Kept on SignalSet anyway (rather than threaded through every miner
    #: call separately) so amminer.miners.conditional's condition_entities()
    #: can offer it to the same discriminative-signal search every other
    #: signal already goes through, as either a trigger or a condition.
    #:
    #: One honest caveat that is *not* enforced here, deliberately: unlike
    #: every other entry on this dataclass, nothing in Home Assistant's own
    #: registry backs this id (see amminer.learn.home_mode's module
    #: docstring, "What this deliberately does not do") - so a candidate
    #: that ends up conditioned on it is real, mined, backtested evidence
    #: about the household, but amminer.llm.validate.validate_references
    #: will (correctly) refuse to let it be applied as a live automation
    #: until a follow-up feature actually publishes the current mode into
    #: Home Assistant as an entity. That refusal is the right, existing
    #: safety net for this - not a bug to route around here.
    home_mode: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {k: v for k, v in self.__dict__.items() if v}

    def present(self) -> set[str]:
        return {k for k, v in self.__dict__.items() if v}

    @property
    def all_entities(self) -> list[str]:
        seen: list[str] = []
        for values in self.__dict__.values():
            for entity_id in values:
                if entity_id not in seen:
                    seen.append(entity_id)
        return seen

    def summary(self) -> str:
        found = sorted(self.present())
        return ", ".join(found) if found else "none"


# --- entity-id / platform heuristics ---------------------------------
_PRICE_PATTERNS = (
    r"tibber.*price",
    r"electricity_price",
    r"electricity_market_price",
    r"nordpool",
    r"energi_data_service",
    r"awattar",
    r"epex",
    r"entsoe",
    r"current_electricity",
    r"spot_price",
    r"energy_price",
)
_PRICE_LEVEL_PATTERNS = (r"price_level", r"electricity_price_level", r"tibber.*level")
_SOLAR_FORECAST_PATTERNS = (
    r"solcast",
    r"forecast_solar",
    r"energy_production_today",
    r"energy_next_hour",
    r"pv_forecast",
    r"estimated_energy",
)
_SOLAR_PRODUCTION_PATTERNS = (r"pv_power", r"solar_power", r"inverter", r"solaredge", r"fronius", r"huawei_solar", r"sma_")
_CARBON_PATTERNS = (r"carbon_intensity", r"co2_intensity", r"co2_signal", r"electricity_maps")
_WARNING_PATTERNS = (r"dwd_weather_warnings", r"weather_warning", r"warning_level", r"meteoalarm")
_WORKDAY_PATTERNS = (r"workday",)
_HOLIDAY_PATTERNS = (r"holiday", r"schulferien", r"feiertag")
_ROOM_PRESENCE_PATTERNS = (r"espresense", r"bermuda", r"room_presence", r"ble_?tracker", r"_area$")
_DEFERRABLE_PATTERNS = (
    r"dishwasher",
    r"washing_machine",
    r"washer",
    r"dryer",
    r"tumble",
    r"geschirrsp",
    r"waschmaschine",
    r"water_heater",
    r"boiler",
    r"heat_pump",
    r"pool_pump",
)
_EV_PATTERNS = (r"wallbox", r"charger", r"easee", r"go_?e", r"keba", r"openevse", r"zaptec", r"ev_?charg")
_MMWAVE_MODELS = (r"fp2", r"fp1", r"mmwave", r"presence.?sensor", r"everything.?presence", r"ld2410", r"ld2450")


def detect_signals(resolver) -> SignalSet:
    """Walk the resolved entity set and classify every useful external signal."""
    signals = SignalSet()

    for info in resolver.entities.values():
        entity_id = info.entity_id
        domain = info.domain
        device_class = (info.device_class or "").lower()
        platform = (info.platform or "").lower()
        # Deliberately not the platform.  It says which integration produced
        # the entity, not what the entity measures, and putting it here made
        # every `template` sensor in a garden an outdoor thermometer, because
        # "template" contains "temp".  Where a platform really does identify a
        # signal (workday, dwd_weather_warnings) it is checked by name below.
        haystack = f"{entity_id} {info.friendly_name or ''} {info.model or ''}"

        if domain == "sun":
            signals.sun.append(entity_id)
        elif domain == "weather":
            signals.weather.append(entity_id)
        elif domain == "calendar":
            signals.calendar.append(entity_id)
        elif domain == "person":
            signals.person.append(entity_id)
        elif domain == "device_tracker":
            signals.device_tracker.append(entity_id)
            if _match_any(haystack, _ROOM_PRESENCE_PATTERNS):
                signals.room_presence.append(entity_id)
        elif domain == "climate":
            signals.thermostat.append(entity_id)
        elif domain == "binary_sensor":
            if device_class == "occupancy" or _match_any(haystack, (r"occupancy", r"presence")):
                signals.occupancy.append(entity_id)
                if _match_any(haystack, _MMWAVE_MODELS):
                    signals.room_presence.append(entity_id)
            if device_class == "motion" or _match_any(haystack, (r"motion",)):
                signals.motion.append(entity_id)
            if _match_any(haystack, _WORKDAY_PATTERNS) or platform == "workday":
                signals.workday.append(entity_id)
            elif _match_any(haystack, _HOLIDAY_PATTERNS):
                signals.holiday.append(entity_id)
        elif domain == "sensor":
            outdoor_words = (r"outdoor", r"outside", r"external", r"garden", r"aussen", r"balcony")
            is_outdoor = _match_any(haystack, outdoor_words)
            # The device_class is authoritative, but it is unavailable when only
            # the recorder could be read - fall back to the name then.
            if is_outdoor and (
                device_class == "temperature" or _match_any(haystack, (r"temp",))
            ):
                signals.outdoor_temperature.append(entity_id)
            elif device_class == "illuminance" or _match_any(haystack, (r"lux",)):
                signals.illuminance.append(entity_id)
            elif device_class == "power":
                signals.power.append(entity_id)
            elif device_class == "energy":
                signals.energy.append(entity_id)
            if _match_any(haystack, _PRICE_LEVEL_PATTERNS):
                signals.price_level.append(entity_id)
            elif _match_any(haystack, _PRICE_PATTERNS):
                signals.energy_price.append(entity_id)
            if _match_any(haystack, _SOLAR_FORECAST_PATTERNS):
                signals.solar_forecast.append(entity_id)
            elif _match_any(haystack, _SOLAR_PRODUCTION_PATTERNS):
                signals.solar_production.append(entity_id)
            if _match_any(haystack, _CARBON_PATTERNS):
                signals.carbon_intensity.append(entity_id)
            if _match_any(haystack, _WARNING_PATTERNS) or platform == "dwd_weather_warnings":
                signals.weather_warning.append(entity_id)
            if _match_any(haystack, _ROOM_PRESENCE_PATTERNS):
                signals.room_presence.append(entity_id)

        if domain in ("switch", "sensor", "binary_sensor", "vacuum", "water_heater"):
            if _match_any(haystack, _DEFERRABLE_PATTERNS):
                signals.deferrable_loads.append(entity_id)
            if _match_any(haystack, _EV_PATTERNS):
                signals.ev_charger.append(entity_id)

        # Weather integrations expose the outdoor temperature as an attribute;
        # a matching sensor is preferable but this is a usable fallback.
        if domain == "weather" and info.attributes.get("temperature") is not None:
            attr_entity = f"{entity_id}#temperature"
            if not signals.outdoor_temperature:
                signals.outdoor_temperature.append(attr_entity)

    for key, values in list(signals.__dict__.items()):
        # Stable, de-duplicated ordering keeps candidate ids reproducible.
        signals.__dict__[key] = sorted(dict.fromkeys(values))

    _LOGGER.info("External signals detected: %s", signals.summary())
    return signals
