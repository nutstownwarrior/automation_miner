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

_LOGGER = logging.getLogger(__name__)


def _match_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


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
        haystack = f"{entity_id} {info.friendly_name or ''} {platform} {info.model or ''}"

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
            if device_class == "motion" or "motion" in haystack:
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
            if is_outdoor and (device_class == "temperature" or "temp" in haystack):
                signals.outdoor_temperature.append(entity_id)
            elif device_class == "illuminance" or "lux" in haystack:
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
