"""Bounded middleware pipeline for NEXUS runtime operations.

Inspired by production agent middleware patterns, but intentionally smaller and
owner-controlled. Middleware is registered by application code, never discovered
from model output. Each layer may invoke the next layer at most once, preventing
accidental duplicate side effects.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
import math
import re
import time
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol


_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,79}")
_OPERATION = re.compile(r"[a-z][a-z0-9_.:/-]{0,159}")
_TERMINAL_STATE_ORDER = {"not_started": 0, "started": 1, "completed": 2}

_MAX_METADATA_TOP_LEVEL_ITEMS = 64
_MAX_METADATA_COLLECTION_ITEMS = 256
_MAX_METADATA_DEPTH = 10
_MAX_METADATA_NODES = 2048
_MAX_METADATA_BYTES = 32768


class MiddlewareError(RuntimeError):
    """Pipeline failure with explicit terminal retry safety."""

    def __init__(self, message: str, *, terminal_state: str = "not_started") -> None:
        if terminal_state not in _TERMINAL_STATE_ORDER:
            raise ValueError("Invalid middleware terminal state.")
        self._base_message = message
        self.terminal_state = terminal_state
        self._refresh_message()

    @property
    def terminal_started(self) -> bool:
        return self.terminal_state != "not_started"

    @property
    def terminal_completed(self) -> bool:
        return self.terminal_state == "completed"

    @property
    def retry_safe(self) -> bool:
        return not self.terminal_started

    def _mark_terminal_state(self, terminal_state: str) -> None:
        if _TERMINAL_STATE_ORDER[terminal_state] > _TERMINAL_STATE_ORDER[self.terminal_state]:
            self.terminal_state = terminal_state
            self._refresh_message()

    def _refresh_message(self) -> None:
        suffix = ""
        if self.terminal_state == "started":
            suffix = " The terminal started; automatic retry is unsafe."
        elif self.terminal_state == "completed":
            suffix = " The terminal completed; automatic retry is unsafe."
        RuntimeError.__init__(self, self._base_message + suffix)


class MiddlewareExecutionError(MiddlewareError):
    def __init__(self, stage: str, error_type: str, *, terminal_state: str = "not_started") -> None:
        self.stage = stage
        self.error_type = error_type
        super().__init__(
            f"NEXUS middleware stage '{stage}' failed ({error_type}).",
            terminal_state=terminal_state,
        )


class MiddlewareCancelledError(asyncio.CancelledError):
    """Cancellation that preserves asyncio semantics and terminal retry safety."""

    def __init__(self, stage: str, *, terminal_state: str = "not_started") -> None:
        if terminal_state not in _TERMINAL_STATE_ORDER:
            raise ValueError("Invalid middleware terminal state.")
        self.stage = stage
        self.terminal_state = terminal_state
        super().__init__(f"NEXUS middleware stage '{stage}' was cancelled.")

    @property
    def terminal_started(self) -> bool:
        return self.terminal_state != "not_started"

    @property
    def terminal_completed(self) -> bool:
        return self.terminal_state == "completed"

    @property
    def retry_safe(self) -> bool:
        return not self.terminal_started

    def _mark_terminal_state(self, terminal_state: str) -> None:
        if _TERMINAL_STATE_ORDER[terminal_state] > _TERMINAL_STATE_ORDER[self.terminal_state]:
            self.terminal_state = terminal_state


def _copy_metadata_value(
    value: Any,
    *,
    depth: int,
    active: set[int],
    nodes: list[int],
    byte_budget: list[int],
) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("Middleware metadata is too deeply nested.")
    nodes[0] += 1
    if nodes[0] > _MAX_METADATA_NODES:
        raise ValueError("Middleware metadata has too many values.")
    byte_budget[0] -= 4
    if byte_budget[0] < 0:
        raise ValueError("Middleware metadata is too large.")

    if isinstance(value, str):
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("Middleware metadata contains invalid text.") from None
        byte_budget[0] -= size
        if byte_budget[0] < 0:
            raise ValueError("Middleware metadata is too large.")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > 4096:
            raise ValueError("Middleware metadata contains an oversized integer.")
        byte_budget[0] -= len(str(value))
        if byte_budget[0] < 0:
            raise ValueError("Middleware metadata is too large.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Middleware metadata contains a non-finite number.")
        byte_budget[0] -= 32
        if byte_budget[0] < 0:
            raise ValueError("Middleware metadata is too large.")
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("Middleware metadata contains a cycle.")
        if len(value) > _MAX_METADATA_COLLECTION_ITEMS:
            raise ValueError("Middleware metadata collection is too large.")
        active.add(identity)
        try:
            copied: dict[str, Any] = {}
            for key, nested in value.items():
                if (
                    not isinstance(key, str)
                    or len(key) > 256
                    or any(ord(character) < 0x20 or ord(character) == 0x7F for character in key)
                ):
                    raise ValueError("Invalid middleware metadata key.")
                try:
                    key_size = len(key.encode("utf-8"))
                except UnicodeEncodeError:
                    raise ValueError("Invalid middleware metadata key.") from None
                byte_budget[0] -= key_size
                if byte_budget[0] < 0:
                    raise ValueError("Middleware metadata is too large.")
                copied[key] = _copy_metadata_value(
                    nested,
                    depth=depth + 1,
                    active=active,
                    nodes=nodes,
                    byte_budget=byte_budget,
                )
            return copied
        finally:
            active.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("Middleware metadata contains a cycle.")
        if len(value) > _MAX_METADATA_COLLECTION_ITEMS:
            raise ValueError("Middleware metadata collection is too large.")
        active.add(identity)
        try:
            return [
                _copy_metadata_value(
                    nested,
                    depth=depth + 1,
                    active=active,
                    nodes=nodes,
                    byte_budget=byte_budget,
                )
                for nested in value
            ]
        finally:
            active.remove(identity)

    raise ValueError("Middleware metadata must contain only JSON values.")


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_metadata_value(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(nested) for nested in value)
    return value


def _bounded_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Middleware metadata must be a mapping.")
    if len(metadata) > _MAX_METADATA_TOP_LEVEL_ITEMS:
        raise ValueError("Middleware metadata is too large.")
    copied = _copy_metadata_value(
        metadata,
        depth=0,
        active=set(),
        nodes=[0],
        byte_budget=[_MAX_METADATA_BYTES],
    )
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError):
        raise ValueError("Middleware metadata is not valid JSON.") from None
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError("Middleware metadata is too large.")
    return _freeze_metadata_value(copied)


def _normalize_identity(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid middleware {label}.")
    normalized = value.strip()
    if not normalized or not normalized.isascii():
        raise ValueError(f"Invalid middleware {label}.")
    normalized = normalized.lower()
    if not pattern.fullmatch(normalized):
        raise ValueError(f"Invalid middleware {label}.")
    return normalized


@dataclass(frozen=True)
class MiddlewareContext:
    operation: str
    route: str
    task_id: str = ""
    step_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _normalize_identity(self.operation, "operation", _OPERATION))
        object.__setattr__(self, "route", _normalize_identity(self.route, "route", _NAME))
        for label, value in (("task_id", self.task_id), ("step_id", self.step_id)):
            if (
                not isinstance(value, str)
                or len(value) > 160
                or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
            ):
                raise ValueError(f"Invalid middleware {label}.")
        clean_metadata = _bounded_metadata(self.metadata)
        for key in clean_metadata:
            if not isinstance(key, str) or not _NAME.fullmatch(key):
                raise ValueError("Invalid middleware metadata key.")
        object.__setattr__(self, "metadata", clean_metadata)


NextCall = Callable[[], Awaitable[Any]]
MiddlewareHandler = Callable[[MiddlewareContext, NextCall], Awaitable[Any] | Any]
TerminalHandler = Callable[[MiddlewareContext], Awaitable[Any] | Any]


@dataclass(frozen=True)
class MiddlewareSpec:
    name: str
    handler: MiddlewareHandler
    allow_short_circuit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("Invalid middleware name.")
        if not callable(self.handler):
            raise ValueError("Middleware handler must be callable application code.")
        if not isinstance(self.allow_short_circuit, bool):
            raise ValueError("allow_short_circuit must be boolean.")


@dataclass(frozen=True)
class MiddlewareTrace:
    stage: str
    started_at: str
    duration_ms: float
    outcome: str
    called_next: bool


class MiddlewarePipeline:
    """Immutable bounded middleware chain.

    The chain is constructed from application-owned callables. There is no
    import-by-name, eval, plugin string, or model-controlled registration path.
    """

    MAX_LAYERS = 32

    def __init__(self, specs: tuple[MiddlewareSpec, ...] = ()) -> None:
        if len(specs) > self.MAX_LAYERS:
            raise ValueError(f"NEXUS middleware is limited to {self.MAX_LAYERS} layers.")
        names = [item.name for item in specs]
        if len(names) != len(set(names)):
            raise ValueError("Middleware names must be unique.")
        self._specs = tuple(specs)

    @property
    def specs(self) -> tuple[MiddlewareSpec, ...]:
        return self._specs

    async def run(self, context: MiddlewareContext, terminal: TerminalHandler) -> tuple[Any, tuple[MiddlewareTrace, ...]]:
        if not isinstance(context, MiddlewareContext):
            raise ValueError("A typed MiddlewareContext is required.")
        if not callable(terminal):
            raise ValueError("terminal must be callable application code.")
        traces: list[MiddlewareTrace] = []
        terminal_state = "not_started"
        latched_error: MiddlewareError | MiddlewareCancelledError | None = None

        def latch_error(
            error: MiddlewareError | MiddlewareCancelledError,
        ) -> MiddlewareError | MiddlewareCancelledError:
            nonlocal latched_error
            error._mark_terminal_state(terminal_state)
            if latched_error is None or (
                isinstance(error, MiddlewareCancelledError)
                and not isinstance(latched_error, MiddlewareCancelledError)
            ):
                latched_error = error
            else:
                latched_error._mark_terminal_state(terminal_state)
            return latched_error

        async def maybe_await(value):
            return await value if inspect.isawaitable(value) else value

        async def dispatch(index: int):
            nonlocal terminal_state
            if index >= len(self._specs):
                started_wall = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                terminal_state = "started"
                try:
                    result = await maybe_await(terminal(context))
                except asyncio.CancelledError:
                    traces.append(MiddlewareTrace(
                        "terminal", started_wall, (time.perf_counter() - started) * 1000.0,
                        "cancelled", False,
                    ))
                    raise latch_error(MiddlewareCancelledError(
                        "terminal", terminal_state=terminal_state,
                    )) from None
                except Exception as exc:
                    traces.append(MiddlewareTrace(
                        "terminal", started_wall, (time.perf_counter() - started) * 1000.0,
                        "error", False,
                    ))
                    error = MiddlewareExecutionError(
                        "terminal",
                        type(exc).__name__,
                        terminal_state=terminal_state,
                    )
                    raise latch_error(error) from None
                terminal_state = "completed"
                traces.append(MiddlewareTrace(
                    "terminal", started_wall, (time.perf_counter() - started) * 1000.0,
                    "ok", False,
                ))
                return result

            spec = self._specs[index]
            started_wall = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            next_called = False
            handler_active = True
            owner_task = asyncio.current_task()

            async def call_next():
                nonlocal next_called, handler_active
                if not handler_active:
                    error = MiddlewareError(
                        f"Middleware '{spec.name}' attempted to call next outside its active handler.",
                        terminal_state=terminal_state,
                    )
                    raise latch_error(error)
                if asyncio.current_task() is not owner_task:
                    error = MiddlewareError(
                        f"Middleware '{spec.name}' attempted to call next from a detached task.",
                        terminal_state=terminal_state,
                    )
                    raise latch_error(error)
                if next_called:
                    error = MiddlewareError(
                        f"Middleware '{spec.name}' attempted to call next more than once.",
                        terminal_state=terminal_state,
                    )
                    raise latch_error(error)
                next_called = True
                return await dispatch(index + 1)

            try:
                try:
                    result = await maybe_await(spec.handler(context, call_next))
                finally:
                    handler_active = False
                if latched_error is not None:
                    raise latched_error
                if not next_called and not spec.allow_short_circuit:
                    raise MiddlewareError(
                        f"Middleware '{spec.name}' returned without calling next and is not allowed to short-circuit.",
                        terminal_state=terminal_state,
                    )
            except asyncio.CancelledError as exc:
                error = exc if isinstance(exc, MiddlewareCancelledError) else MiddlewareCancelledError(
                    spec.name, terminal_state=terminal_state,
                )
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "cancelled", next_called,
                ))
                raise latch_error(error) from None
            except MiddlewareExecutionError as exc:
                error = latch_error(exc)
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "downstream_error", next_called,
                ))
                raise error
            except MiddlewareError as exc:
                error = latch_error(exc)
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "contract_error", next_called,
                ))
                raise error
            except Exception as exc:
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "error", next_called,
                ))
                error = MiddlewareExecutionError(
                    spec.name,
                    type(exc).__name__,
                    terminal_state=terminal_state,
                )
                raise latch_error(error) from None

            traces.append(MiddlewareTrace(
                spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                "short_circuit" if not next_called else "ok", next_called,
            ))
            return result

        result = await dispatch(0)
        return result, tuple(traces)
