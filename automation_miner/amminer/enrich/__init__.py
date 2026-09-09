"""Auto-detection and joining of external signals (sun, weather, price, ...)."""

from .detect import SignalSet, detect_signals
from .signals import SignalSeries, SignalStore, build_signal_store

__all__ = [
    "SignalSet",
    "SignalSeries",
    "SignalStore",
    "build_signal_store",
    "detect_signals",
]
