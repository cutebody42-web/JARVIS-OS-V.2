"""Bounded retry/backoff scheduler for durable peer sync."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from core.nexus.peer_auth import PeerRegistry
from core.nexus.signed_transport import SignedHTTPSyncTransport
from core.nexus.sync_daemon import SyncDaemon, SyncReport


@dataclass(frozen=True)
class PeerSchedule:
    peer_id: str
    failures: int
    next_due: float
    last_report: SyncReport | None = None


class SyncScheduler:
    """Drive one bounded SyncDaemon batch per eligible peer per tick."""

    def __init__(
        self,
        daemon: SyncDaemon,
        registry: PeerRegistry,
        transport: SignedHTTPSyncTransport,
        *,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 300.0,
        idle_seconds: float = 30.0,
        drain_seconds: float = 0.25,
        clock=time.monotonic,
    ):
        if retry_base_seconds <= 0 or retry_max_seconds < retry_base_seconds:
            raise ValueError("invalid retry backoff bounds")
        if idle_seconds <= 0 or drain_seconds < 0:
            raise ValueError("invalid scheduler intervals")
        self.daemon = daemon
        self.registry = registry
        self.transport = transport
        self.retry_base = float(retry_base_seconds)
        self.retry_max = float(retry_max_seconds)
        self.idle_seconds = float(idle_seconds)
        self.drain_seconds = float(drain_seconds)
        self._clock = clock
        self._state: dict[str, PeerSchedule] = {}

    def state(self, peer_id: str) -> PeerSchedule:
        now = self._clock()
        return self._state.get(peer_id, PeerSchedule(peer_id, 0, now, None))

    def _delay_after(self, report: SyncReport, failures: int) -> tuple[int, float]:
        if report.error:
            failures += 1
            delay = min(self.retry_max, self.retry_base * (2 ** (failures - 1)))
            return failures, delay
        if report.remaining > 0:
            return 0, self.drain_seconds
        return 0, self.idle_seconds

    def tick(self) -> tuple[SyncReport, ...]:
        now = self._clock()
        reports: list[SyncReport] = []
        active_ids = {peer.peer_id for peer in self.registry.active_sync_peers()}

        # Revoked peers are removed from future scheduling immediately.
        for peer_id in tuple(self._state):
            if peer_id not in active_ids:
                self._state.pop(peer_id, None)

        for peer_id in sorted(active_ids):
            previous = self._state.get(peer_id)
            if previous is not None and previous.next_due > now:
                continue

            report = self.daemon.sync_peer(peer_id, self.transport)
            prior_failures = previous.failures if previous else 0
            failures, delay = self._delay_after(report, prior_failures)
            self._state[peer_id] = PeerSchedule(
                peer_id,
                failures,
                now + delay,
                report,
            )
            reports.append(report)

        return tuple(reports)

    def run(self, stop_event: threading.Event, *, poll_seconds: float = 0.25) -> None:
        if not isinstance(stop_event, threading.Event):
            raise TypeError("stop_event must be threading.Event")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        while not stop_event.is_set():
            self.tick()
            stop_event.wait(poll_seconds)
