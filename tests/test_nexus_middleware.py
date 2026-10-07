import asyncio

import pytest

from core.nexus.middleware import (
    MiddlewareContext,
    MiddlewareError,
    MiddlewareExecutionError,
    MiddlewarePipeline,
    MiddlewareSpec,
)


def test_middleware_wraps_terminal_in_order_and_traces_once():
    events = []

    async def first(context, call_next):
        events.append("first:before")
        value = await call_next()
        events.append("first:after")
        return value + "!"

    async def second(context, call_next):
        events.append("second:before")
        value = await call_next()
        events.append("second:after")
        return value.upper()

    async def terminal(context):
        events.append("terminal")
        return "jarvis"

    pipeline = MiddlewarePipeline((
        MiddlewareSpec("first", first),
        MiddlewareSpec("second", second),
    ))
    result, trace = asyncio.run(pipeline.run(MiddlewareContext("model.reply", "standard"), terminal))

    assert result == "JARVIS!"
    assert events == ["first:before", "second:before", "terminal", "second:after", "first:after"]
    assert {item.stage for item in trace} == {"first", "second", "terminal"}
    assert all(item.duration_ms >= 0 for item in trace)


def test_middleware_cannot_call_next_twice():
    async def bad(context, call_next):
        await call_next()
        return await call_next()

    pipeline = MiddlewarePipeline((MiddlewareSpec("bad", bad),))
    with pytest.raises(MiddlewareError):
        asyncio.run(pipeline.run(MiddlewareContext("tool.call", "mission"), lambda context: "ok"))


def test_short_circuit_requires_explicit_application_registration():
    async def stop(context, call_next):
        return "denied"

    strict = MiddlewarePipeline((MiddlewareSpec("guard", stop),))
    with pytest.raises(MiddlewareError):
        asyncio.run(strict.run(MiddlewareContext("tool.call", "mission"), lambda context: "unsafe"))

    guarded = MiddlewarePipeline((MiddlewareSpec("guard", stop, allow_short_circuit=True),))
    result, trace = asyncio.run(guarded.run(MiddlewareContext("tool.call", "mission"), lambda context: "unsafe"))
    assert result == "denied"
    assert trace[-1].outcome == "short_circuit"
    assert all(item.stage != "terminal" for item in trace)


def test_handler_exception_is_redacted_and_identifies_stage():
    async def explode(context, call_next):
        raise RuntimeError("secret provider payload")

    pipeline = MiddlewarePipeline((MiddlewareSpec("telemetry", explode),))
    with pytest.raises(MiddlewareExecutionError) as caught:
        asyncio.run(pipeline.run(MiddlewareContext("model.reply", "fast"), lambda context: "ok"))
    assert caught.value.stage == "telemetry"
    assert caught.value.error_type == "RuntimeError"
    assert "secret provider payload" not in str(caught.value)


def test_pipeline_is_bounded_and_names_are_unique():
    async def passthrough(context, call_next):
        return await call_next()

    with pytest.raises(ValueError):
        MiddlewarePipeline(tuple(MiddlewareSpec(f"m{i}", passthrough) for i in range(33)))
    with pytest.raises(ValueError):
        MiddlewarePipeline((MiddlewareSpec("same", passthrough), MiddlewareSpec("same", passthrough)))


def test_context_metadata_is_immutable_and_bounded():
    source = {"owner": "local"}
    context = MiddlewareContext("model.reply", "standard", metadata=source)
    source["owner"] = "changed"
    assert context.metadata["owner"] == "local"
    with pytest.raises(TypeError):
        context.metadata["owner"] = "x"
