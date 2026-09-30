"""Tests for deterministic peer sync backoff and bounded draining."""

import unittest

from core.nexus.sync_daemon import SyncReport
from core.nexus.sync_scheduler import (
    ScheduleDisposition,
    SyncScheduler,
)


class FakeDaemon:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def sync_peer(self, peer_id, transport):
        self.calls.append((peer_id, transport))
        if not self.reports:
            raise AssertionError("unexpected sync call")
        return self.reports.pop(0)


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def report(*, peer="dell", selected=1, acked=1, remaining=0, error=None):
    return SyncReport(
        peer_id=peer,
        batch_id="batch" if selected else None,
        selected=selected,
        acked=acked,
        remaining=remaining,
        attempts_recorded=selected,
        error=error,
    )


class SyncSchedulerTests(unittest.TestCase):
    def test_failure_uses_exponential_backoff_and_skips_early_tick(self):
        daemon = FakeDaemon([
            report(acked=0, remaining=1, error="ConnectionError"),
            report(acked=0, remaining=1, error="ConnectionError"),
        ])
        clock = Clock()
        scheduler = SyncScheduler(
            daemon,
            base_backoff_seconds=2,
            max_backoff_seconds=30,
            monotonic=clock,
        )

        first = scheduler.sync_if_ready("dell", object())
        self.assertEqual(first.retry_in_seconds, 2)

        blocked = scheduler.sync_if_ready("dell", object())
        self.assertEqual(blocked.disposition, ScheduleDisposition.BACKOFF)
        self.assertEqual(len(daemon.calls), 1)

        clock.advance(2)
        second = scheduler.sync_if_ready("dell", object())
        self.assertEqual(second.retry_in_seconds, 4)
        self.assertEqual(scheduler.state("dell").consecutive_failures, 2)

    def test_success_resets_failure_counter(self):
        daemon = FakeDaemon([
            report(acked=0, remaining=1, error="ConnectionError"),
            report(acked=1, remaining=0, error=None),
        ])
        clock = Clock()
        scheduler = SyncScheduler(daemon, monotonic=clock)

        first = scheduler.sync_if_ready("dell", object())
        clock.advance(first.retry_in_seconds)
        second = scheduler.sync_if_ready("dell", object())

        self.assertEqual(second.disposition, ScheduleDisposition.SENT)
        self.assertEqual(scheduler.state("dell").consecutive_failures, 0)

    def test_remaining_work_uses_short_drain_delay(self):
        daemon = FakeDaemon([report(acked=1, remaining=4)])
        clock = Clock()
        scheduler = SyncScheduler(
            daemon,
            success_drain_delay_seconds=0.5,
            idle_interval_seconds=30,
            monotonic=clock,
        )

        result = scheduler.sync_if_ready("dell", object())

        self.assertEqual(result.retry_in_seconds, 0.5)
        self.assertEqual(result.disposition, ScheduleDisposition.SENT)

    def test_no_work_uses_idle_interval(self):
        daemon = FakeDaemon([report(selected=0, acked=0, remaining=0)])
        clock = Clock()
        scheduler = SyncScheduler(
            daemon,
            idle_interval_seconds=20,
            monotonic=clock,
        )

        result = scheduler.sync_if_ready("dell", object())

        self.assertEqual(result.disposition, ScheduleDisposition.IDLE)
        self.assertEqual(result.retry_in_seconds, 20)

    def test_run_ready_is_bounded_to_one_call_per_peer(self):
        daemon = FakeDaemon([
            report(peer="dell", remaining=2),
            report(peer="phone", remaining=0),
        ])
        clock = Clock()
        scheduler = SyncScheduler(daemon, monotonic=clock)

        results = scheduler.run_ready({"phone": object(), "dell": object()})

        self.assertEqual(len(results), 2)
        self.assertEqual([call[0] for call in daemon.calls], ["dell", "phone"])

    def test_backoff_caps_at_maximum(self):
        daemon = FakeDaemon([
            report(acked=0, remaining=1, error="x")
            for _ in range(6)
        ])
        clock = Clock()
        scheduler = SyncScheduler(
            daemon,
            base_backoff_seconds=2,
            max_backoff_seconds=8,
            monotonic=clock,
        )

        delays = []
        for _ in range(6):
            result = scheduler.sync_if_ready("dell", object())
            delays.append(result.retry_in_seconds)
            clock.advance(result.retry_in_seconds)

        self.assertEqual(delays, [2, 4, 8, 8, 8, 8])


if __name__ == "__main__":
    unittest.main()
