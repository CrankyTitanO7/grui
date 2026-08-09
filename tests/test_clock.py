"""Tests for the shared monotonic session clock."""

import time

import pytest

from recorder.clock import SessionClock


def test_now_starts_near_zero_and_increases():
    clock = SessionClock()
    assert clock.now() >= 0
    time.sleep(0.02)
    assert clock.now() >= 0.015
    a = clock.now()
    b = clock.now()
    assert b >= a


def test_now_matches_nanosecond_math():
    clock = SessionClock()
    expected = (clock.now_ns() - clock.start_ns) / 1e9
    assert clock.now() == pytest.approx(expected, abs=1e-4)


def test_relative_times_are_consistent_across_calls():
    clock = SessionClock()
    t0 = clock.now()
    time.sleep(0.01)
    t1 = clock.now()
    assert 0.005 < (t1 - t0) < 0.2
