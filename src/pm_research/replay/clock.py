"""Deterministic UTC simulated clock for backtesting and historical replay."""

from __future__ import annotations

from datetime import datetime, timedelta

from pm_research.utils import ensure_utc


class SimulatedClock:
    """Deterministic simulated clock enforcing strictly monotonic forward time in UTC."""

    def __init__(self, initial_time: datetime) -> None:
        self._current_time: datetime = ensure_utc(initial_time)

    def now(self) -> datetime:
        """Get the current simulated UTC time."""
        return self._current_time

    def set_time(self, new_time: datetime) -> None:
        """Advance simulated clock to a specific timestamp.

        Raises ValueError if new_time is earlier than current simulated time (no time travel backwards).
        """
        utc_new = ensure_utc(new_time)
        if utc_new < self._current_time:
            raise ValueError(
                f"Clock cannot move backwards: current simulated time is {self._current_time.isoformat()}, "
                f"attempted to set {utc_new.isoformat()}"
            )
        self._current_time = utc_new

    def advance(self, duration: timedelta) -> datetime:
        """Advance simulated clock forward by a given timedelta duration."""
        if duration < timedelta(0):
            raise ValueError(f"Duration cannot be negative: {duration}")
        self._current_time += duration
        return self._current_time

    def __repr__(self) -> str:
        return f"SimulatedClock({self._current_time.isoformat()})"
