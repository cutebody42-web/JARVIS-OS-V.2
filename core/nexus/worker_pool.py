"""Bounded ephemeral NEXUS worker pool.

Workers execute only host-registered job types. Model/user payloads are JSON data,
never Python callables, import paths, shell snippets or persistence instructions.
Handlers receive a copied payload and a cancellation event, not the pool itself,
so a worker cannot recursively create workers through this API.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import threading
import time
from typing import Any, Callable, Mapping
from uuid import uuid4


WorkerHandler = Callable[[dict[str, Any], threading.Event], Any]
_JOB_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("worker payload must be an object")
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ValueError("worker payload exceeds 64 KiB")
    return json.loads(encoded)


@dataclass(frozen=True)
class WorkerLease:
    job_id: str
    job_type: str
    submitted_at: str
    deadline_monotonic: float


@dataclass(frozen=True)
class WorkerSnapshot:
    job_id: str
    job_type: str
    state: str
    submitted_at: str
    done: bool
    cancelled: bool


class WorkerPoolClosed(RuntimeError):
    pass


class WorkerPoolBusy(RuntimeError):
    pass


class BoundedWorkerPool:
    """Trusted host-side registry plus bounded ephemeral execution."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_queued: int = 16,
        default_timeout_seconds: float = 120.0,
        clock=time.monotonic,
    ):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if isinstance(max_queued, bool) or not isinstance(max_queued, int) or max_queued < 0:
            raise ValueError("max_queued must be a non-negative integer")
        if isinstance(default_timeout_seconds, bool) or default_timeout_seconds <= 0:
            raise ValueError("default timeout must be positive")

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="nexus-worker",
        )
        self._capacity = threading.BoundedSemaphore(max_workers + max_queued)
        self._handlers: dict[str, WorkerHandler] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}
        self._leases: dict[str, WorkerLease] = {}
        self._guard = threading.RLock()
        self._closed = False
        self._clock = clock
        self._default_timeout = float(default_timeout_seconds)

    def register(self, job_type: str, handler: WorkerHandler) -> None:
        if not isinstance(job_type, str) or not _JOB_TYPE_RE.fullmatch(job_type):
            raise ValueError("invalid worker job type")
        if not callable(handler):
            raise TypeError("worker handler must be callable")
        with self._guard:
            if self._closed:
                raise WorkerPoolClosed("worker pool is closed")
            if job_type in self._handlers:
                raise ValueError("worker job type is already registered")
            self._handlers[job_type] = handler

    @property
    def registered_job_types(self) -> tuple[str, ...]:
        with self._guard:
            return tuple(sorted(self._handlers))

    def _run(
        self,
        job_id: str,
        handler: WorkerHandler,
        payload: dict[str, Any],
        cancel_event: threading.Event,
    ) -> Any:
        if cancel_event.is_set():
            raise RuntimeError("worker job cancelled before execution")
        return handler(payload, cancel_event)

    def _release_capacity(self, future: Future) -> None:
        # Completion callbacks run for success, failure and queued cancellation,
        # so every accepted submission releases exactly one capacity slot.
        self._capacity.release()

    def submit(
        self,
        job_type: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> WorkerLease:
        if not isinstance(job_type, str) or not _JOB_TYPE_RE.fullmatch(job_type):
            raise ValueError("invalid worker job type")
        safe_payload = _json_copy(payload)
        timeout = self._default_timeout if timeout_seconds is None else timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        with self._guard:
            if self._closed:
                raise WorkerPoolClosed("worker pool is closed")
            handler = self._handlers.get(job_type)
            if handler is None:
                raise PermissionError("worker job type is not registered")

        if not self._capacity.acquire(blocking=False):
            raise WorkerPoolBusy("worker capacity is exhausted")

        job_id = uuid4().hex
        cancel_event = threading.Event()
        lease = WorkerLease(
            job_id=job_id,
            job_type=job_type,
            submitted_at=_utc_now(),
            deadline_monotonic=self._clock() + float(timeout),
        )

        try:
            future = self._executor.submit(
                self._run,
                job_id,
                handler,
                safe_payload,
                cancel_event,
            )
            future.add_done_callback(self._release_capacity)
        except Exception:
            self._capacity.release()
            raise

        with self._guard:
            self._cancel[job_id] = cancel_event
            self._futures[job_id] = future
            self._leases[job_id] = lease
        return lease

    def cancel(self, job_id: str) -> bool:
        with self._guard:
            event = self._cancel.get(job_id)
            future = self._futures.get(job_id)
            if event is None or future is None:
                return False
            event.set()
            future.cancel()
            return True

    def expired_jobs(self) -> tuple[str, ...]:
        now = self._clock()
        with self._guard:
            return tuple(
                job_id
                for job_id, lease in self._leases.items()
                if lease.deadline_monotonic <= now
                and not self._futures[job_id].done()
            )

    def cancel_expired(self) -> tuple[str, ...]:
        expired = self.expired_jobs()
        for job_id in expired:
            self.cancel(job_id)
        return expired

    def result(self, job_id: str, *, timeout: float | None = None) -> Any:
        with self._guard:
            future = self._futures.get(job_id)
            if future is None:
                raise KeyError("worker job not found")
        return future.result(timeout=timeout)

    def snapshot(self) -> tuple[WorkerSnapshot, ...]:
        with self._guard:
            rows = []
            for job_id, future in self._futures.items():
                event = self._cancel[job_id]
                lease = self._leases[job_id]
                if future.cancelled() or event.is_set():
                    state = "cancelled"
                elif future.done():
                    state = "done"
                else:
                    state = "running"
                rows.append(
                    WorkerSnapshot(
                        job_id,
                        lease.job_type,
                        state,
                        lease.submitted_at,
                        future.done(),
                        event.is_set(),
                    )
                )
            return tuple(rows)

    def prune(self) -> int:
        """Forget completed metadata; does not affect running work."""
        removed = 0
        with self._guard:
            for job_id in tuple(self._futures):
                if self._futures[job_id].done():
                    self._futures.pop(job_id, None)
                    self._cancel.pop(job_id, None)
                    self._leases.pop(job_id, None)
                    removed += 1
        return removed

    def close(self, *, cancel_pending: bool = True) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            if cancel_pending:
                for event in self._cancel.values():
                    event.set()
                for future in self._futures.values():
                    future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
