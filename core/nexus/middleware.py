"""Bounded middleware pipeline for NEXUS runtime operations.

Inspired by production agent middleware patterns, but intentionally smaller and
owner-controlled. Middleware is registered by application code, never discovered
from model output. Each layer may invoke the next layer at most once, preventing
accidental duplicate side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import re
import time
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol


_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,79}")


class MiddlewareError(RuntimeError):
    pass


class MiddlewareExecutionError(MiddlewareError):
    def __init__(self, stage: str, error_type: str) -> None:
        self.stage = stage
        self.error_type = error_type
        super().__init__(f"NEXUS middleware stage '{stage}' failed ({error_type}).")


@dataclass(frozen=True)
class MiddlewareContext:
    operation: str
    route: str
    task_id: str = ""
    step_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value, limit in (
            ("operation", self.operation, 160),
            ("route", self.route, 80),
            ("task_id", self.task_id, 160),
            ("step_id", self.step_id, 160),
        ):
            if not isinstance(value, str) or len(value) > limit or "\x00" in value:
                raise ValueError(f"Invalid middleware {label}.")
        if not self.operation.strip() or not self.route.strip():
            raise ValueError("operation and route are required.")
        clean = dict(self.metadata)
        if len(clean) > 64:
            raise ValueError("Middleware metadata is too large.")
        for key in clean:
            if not isinstance(key, str) or not _NAME.fullmatch(key):
                raise ValueError("Invalid middleware metadata key.")
        object.__setattr__(self, "metadata", MappingProxyType(clean))


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

        async def maybe_await(value):
            return await value if inspect.isawaitable(value) else value

        async def dispatch(index: int):
            if index >= len(self._specs):
                started_wall = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                try:
                    result = await maybe_await(terminal(context))
                except Exception as exc:
                    traces.append(MiddlewareTrace(
                        "terminal", started_wall, (time.perf_counter() - started) * 1000.0,
                        "error", False,
                    ))
                    raise MiddlewareExecutionError("terminal", type(exc).__name__) from None
                traces.append(MiddlewareTrace(
                    "terminal", started_wall, (time.perf_counter() - started) * 1000.0,
                    "ok", False,
                ))
                return result

            spec = self._specs[index]
            started_wall = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            next_called = False

            async def call_next():
                nonlocal next_called
                if next_called:
                    raise MiddlewareError(f"Middleware '{spec.name}' attempted to call next more than once.")
                next_called = True
                return await dispatch(index + 1)

            try:
                result = await maybe_await(spec.handler(context, call_next))
                if not next_called and not spec.allow_short_circuit:
                    raise MiddlewareError(
                        f"Middleware '{spec.name}' returned without calling next and is not allowed to short-circuit."
                    )
            except MiddlewareExecutionError:
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "downstream_error", next_called,
                ))
                raise
            except MiddlewareError:
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "contract_error", next_called,
                ))
                raise
            except Exception as exc:
                traces.append(MiddlewareTrace(
                    spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                    "error", next_called,
                ))
                raise MiddlewareExecutionError(spec.name, type(exc).__name__) from None

            traces.append(MiddlewareTrace(
                spec.name, started_wall, (time.perf_counter() - started) * 1000.0,
                "short_circuit" if not next_called else "ok", next_called,
            ))
            return result

        result = await dispatch(0)
        return result, tuple(traces)
