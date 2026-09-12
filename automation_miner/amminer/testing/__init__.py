"""Test helpers shipped with the package: synthetic recorder DBs, dataset maps."""

from .synthetic import (
    SyntheticRecorder,
    TwoRegimeFixture,
    WindDownFixture,
    build_default_fixture,
    build_two_regime_activity,
    build_winddown_habit_activity,
)

__all__ = [
    "SyntheticRecorder",
    "TwoRegimeFixture",
    "WindDownFixture",
    "build_default_fixture",
    "build_two_regime_activity",
    "build_winddown_habit_activity",
]
