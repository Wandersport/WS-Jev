"""Unit tests for SimulatedClock temporal progression and UTC invariants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pm_research.replay.clock import SimulatedClock


def test_clock_initialization_and_utc_enforcement() -> None:
    """Clock initializes in UTC and reports now() accurately."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = SimulatedClock(t0)
    assert clock.now() == t0
    assert clock.now().tzinfo == timezone.utc


def test_clock_cannot_move_backwards() -> None:
    """Clock raises ValueError if an attempt is made to move backwards in time."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = SimulatedClock(t0)

    # Forward is allowed
    t1 = t0 + timedelta(hours=1)
    clock.set_time(t1)
    assert clock.now() == t1

    # Backwards must raise ValueError
    t_past = t0 - timedelta(minutes=5)
    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.set_time(t_past)

    # Remains at t1
    assert clock.now() == t1


def test_clock_advance_duration() -> None:
    """Advancing clock by positive duration increases time monotonically."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = SimulatedClock(t0)

    clock.advance(timedelta(minutes=30))
    assert clock.now() == datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="Duration cannot be negative"):
        clock.advance(timedelta(minutes=-10))
