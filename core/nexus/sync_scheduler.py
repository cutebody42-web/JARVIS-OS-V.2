"""Deterministic peer sync scheduling with bounded exponential backoff.

This module contains no background thread and no sleeps. The host application
calls run_ready() on its own cadence. Each ready peer performs at most one
SyncDaemon.sync_peer() batch per tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Mapping, Protocol, runtime_checkable

from core.nexus.sync_daemon import SyncReport, SyncTransport, _validate_peer_id


@runtime_checkable
class SyncPeerDriver(Protocol):
    def sync_peer(self, peer_id: str, transport: SyncTransport) -> SyncReport:
        ...


class ScheduleDisposition(str, Enum):
    SENT = "sent"
    IDLE = "idle"
    BACKOFF = "backoff"


@dataclass(frozen=True)
class PeerSchedule:
    peer_id: str
    consecutive_failures: int
    next_attempt_at: float
    last_report: SyncReport | None = None


@dataclass(frozen=True)
class ScheduledResult:
    peer_id: str
    disposition: ScheduleDisposition
    report: SyncReport | None
    retry_in_seconds: float


class SyncScheduler:
    def __init__(
        self,
        daemon: SyncPeerDriver,
        *,
        base_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 300.0,
        success_drain_delay_seconds: float = 0.25,
        idle_interval_seconds: float = 30.0,
        monotonic=None,
    ):
        if not isinstance(daemon, SyncPeerDriver):
            raise TypeError("daemon must implement sync_peer(peer_id, transport)")
        for name, value in (
            ("base_backoff_seconds", base_backoff_seconds),
            ("max_backoff_seconds", max_backoff_seconds),
            ("success_drain_delay_seconds", success_drain_delay_seconds),
            ("idle_interval_seconds", idle_interval_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be positive")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")

        self._daemon = daemon
        self._base_backoff = float(base_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._drain_delay = float(success_drain_delay_seconds)
        self._idle_interval = float(idle_interval_seconds)
        self._clock = monotonic or time.monotonic
        self._states: dict[str, PeerSchedule] = {}

    def state(self, peer_id: str) -> PeerSchedule:
        _validate_peer_id(peer_id)
        return self._states.get(
            peer_id,
            PeerSchedule(
                peer_id=peer_id,
                consecutive_failures=0,
                next_attempt_at=0.0,
                last_report=None,
            ),
        )

    def reset_peer(self, peer_id: str) -> None:
        _validate_peer_id(peer_id)
        self._states.pop(peer_id, None)

    def _backoff_for(self, failures: int) -> float:
        if failures <= 0:
            return 0.0
        return min(
            self._max_backoff,
            self._base_backoff * (2 ** (failures - 1)),
        )

    def sync_if_ready(
        self,
        peer_id: str,
        transport: SyncTransport,
    ) -> ScheduledResult:
        _validate_peer_id(peer_id)
        now = float(self._clock())
        current = self.state(peer_id)
        if now < current.next_attempt_at:
            return ScheduledResult(
                peer_id=peer_id,
                disposition=ScheduleDisposition.BACKOFF,
                report=current.last_report,
                retry_in_seconds=max(0.0, current.next_attempt_at - now),
            )

        report = self._daemon.sync_peer(peer_id, transport)

        if report.error:
            failures = current.consecutive_failures + 1
            delay = self._backoff_for(failures)
            self._states[peer_id] = PeerSchedule(
                peer_id=peer_id,
                consecutive_failures=failures,
                next_attempt_at=now + delay,
                last_report=report,
            )
            return ScheduledResult(
                peer_id=peer_id,
                disposition=ScheduleDisposition.SENT,
                report=report,
                retry_in_seconds=delay,
            )

        if report.remaining > 0:
            delay = self._drain_delay
            disposition = ScheduleDisposition.SENT
        else:
            delay = self._idle_interval
            disposition = (
                ScheduleDisposition.SENT
                if report.selected > 0
                else ScheduleDisposition.IDLE
            )

        self._states[peer_id] = PeerSchedule(
            peer_id=peer_id,
            consecutive_failures=0,
            next_attempt_at=now + delay,
            last_report=report,
        )
        return ScheduledResult(
            peer_id=peer_id,
            disposition=disposition,
            report=report,
            retry_in_seconds=delay,
        )

    def run_ready(
        self,
        transports: Mapping[str, SyncTransport],
    ) -> tuple[ScheduledResult, ...]:
        """Run at most one bounded batch per ready peer."""
        results: list[ScheduledResult] = []
        for peer_id in sorted(transports):
            results.append(self.sync_if_ready(peer_id, transports[peer_id]))
        return tuple(results)
